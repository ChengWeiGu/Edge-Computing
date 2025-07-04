// cls_infer.cpp  ── 1-D CNN 時序分類 TFLite 推論
// 讀取 CSV，如超出模型需求長度(500)時，自動取最後 500 點
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <cmath>

/*********************
 *  CSV utilities    *
 *********************/
std::vector<float> read_csv(const std::string& path) {
    std::vector<float> data;
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "Cannot open CSV: " << path << "\n"; std::exit(1); }
    std::string line;
    while (std::getline(f, line)) {
        try { data.push_back(std::stof(line)); }
        catch (...) { std::cerr << "Bad value in CSV: " << line << "\n"; }
    }
    return data;
}

void write_csv(const std::string& path, int cls, float prob) {
    std::ofstream f(path);
    if (!f.is_open()) { std::cerr << "Cannot write CSV: " << path << "\n"; std::exit(1); }
    f << cls << ", " << prob << '\n';
}

/*********************
 *  Main             *
 *********************/
int main(int argc, char* argv[]) {
    std::string model_path, input_path, output_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const std::string& flag) {
            if (i + 1 >= argc) { std::cerr << flag << " needs value\n"; std::exit(1); }
            return std::string(argv[++i]);
        };
        if (arg == "-m" || arg == "--model")      model_path  = next(arg);
        else if (arg == "-i" || arg == "--input") input_path  = next(arg);
        else if (arg == "-o" || arg == "--output")output_path = next(arg);
        else if (arg.rfind("--model=",0)==0)      model_path  = arg.substr(8);
        else if (arg.rfind("--input=",0)==0)      input_path  = arg.substr(8);
        else if (arg.rfind("--output=",0)==0)     output_path = arg.substr(9);
    }
    if (model_path.empty() || input_path.empty() || output_path.empty()) {
        std::cerr << "Usage: ./cls_infer -m model.tflite -i sample.csv -o result.csv\n";
        return 1;
    }

    /* ------------------------------------------------------------- *
     * 1) 讀取資料                                                    *
     * ------------------------------------------------------------- */
    std::vector<float> series = read_csv(input_path);

    /* ------------------------------------------------------------- *
     * 2) 載入 TFLite 模型                                           *
     * ------------------------------------------------------------- */
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) { std::cerr << "Load model failed\n"; return 1; }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) { std::cerr << "Create interpreter failed\n"; return 1; }
    if (interpreter->AllocateTensors() != kTfLiteOk) { std::cerr << "AllocateTensors failed\n"; return 1; }

    /* ------------------------------------------------------------- *
     * 3) 準備輸入                                                    *
     * ------------------------------------------------------------- */
    const int in_idx = interpreter->inputs()[0];
    TfLiteTensor* in_tensor = interpreter->tensor(in_idx);

    if (in_tensor->type != kTfLiteFloat32 && in_tensor->type != kTfLiteInt8) {
        std::cerr << "Only float32 / int8 input supported\n"; return 1;
    }

    // 模型的時間維度 (通常 500)
    int time_dim = in_tensor->dims->data[in_tensor->dims->size - 2];

    // 若資料過長 → 取最後 time_dim 點
    if (series.size() > static_cast<size_t>(time_dim)) {
        series.assign(series.end() - time_dim, series.end());
        std::cout << "Input longer than "<<time_dim<<" → truncated to last "<<time_dim<<" points\n";
    }

    // 若資料不足 → 報錯
    if (series.size() != static_cast<size_t>(time_dim)) {
        std::cerr << "CSV length ("<<series.size()<<") != model time dimension ("<<time_dim<<")\n";
        return 1;
    }

    // 寫入 Tensor
    if (in_tensor->type == kTfLiteFloat32) {
        float* in = interpreter->typed_tensor<float>(in_idx);
        std::copy(series.begin(), series.end(), in);
    } else { // int8 量化
        int8_t* in = interpreter->typed_tensor<int8_t>(in_idx);
        float scale = in_tensor->params.scale;
        int32_t zp  = in_tensor->params.zero_point;
        for (int t = 0; t < time_dim; ++t) {
            int32_t q = static_cast<int32_t>(std::round(series[t] / scale) + zp);
            q = std::min<int32_t>(std::max<int32_t>(q, std::numeric_limits<int8_t>::min()),
                                  std::numeric_limits<int8_t>::max());
            in[t] = static_cast<int8_t>(q);
        }
    }

    /* ------------------------------------------------------------- *
     * 4) 推論                                                        *
     * ------------------------------------------------------------- */
    if (interpreter->Invoke() != kTfLiteOk) { std::cerr << "Invoke failed\n"; return 1; }

    /* ------------------------------------------------------------- *
     * 5) 解析輸出，找最大 softmax 機率                                *
     * ------------------------------------------------------------- */
    const int out_idx = interpreter->outputs()[0];
    TfLiteTensor* out_tensor = interpreter->tensor(out_idx);

    std::vector<float> probs;
    if (out_tensor->type == kTfLiteFloat32) {
        const float* out = interpreter->typed_tensor<float>(out_idx);
        probs.assign(out, out + out_tensor->dims->data[out_tensor->dims->size - 1]);
    } else if (out_tensor->type == kTfLiteInt8) {
        const int8_t* out = interpreter->typed_tensor<int8_t>(out_idx);
        float scale = out_tensor->params.scale;
        int32_t zp  = out_tensor->params.zero_point;
        int num_cls = out_tensor->dims->data[out_tensor->dims->size - 1];
        probs.reserve(num_cls);
        for (int i = 0; i < num_cls; ++i)
            probs.push_back((static_cast<int32_t>(out[i]) - zp) * scale);
    } else {
        std::cerr << "Unsupported output type\n"; return 1;
    }

    int   pred_class = 0;
    float pred_prob  = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) {
        if (probs[i] > pred_prob) { pred_prob = probs[i]; pred_class = static_cast<int>(i); }
    }

    /* ------------------------------------------------------------- *
     * 6) 輸出結果                                                    *
     * ------------------------------------------------------------- */
    write_csv(output_path, pred_class, pred_prob);
    std::cout << "Prediction  : class = " << pred_class
              << ", prob = " << pred_prob << '\n'
              << "Saved to    : " << output_path << '\n';
    return 0;
}
