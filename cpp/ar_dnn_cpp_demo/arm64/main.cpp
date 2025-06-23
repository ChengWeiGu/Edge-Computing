#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <limits>



/******************************
 *  Utilities                 *
 ******************************/
// 讀取一維數據 CSV (每行一個浮點數)
std::vector<float> read_csv(const std::string& csv_path) {
    std::vector<float> data;
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Cannot open the CSV file: " << csv_path << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(file, line)) {
        try {
            data.push_back(std::stof(line));
        } catch (const std::invalid_argument&) {
            std::cerr << "CSV file contains an invalid data: " << line << std::endl;
        }
    }
    return data;
}

// 將浮點數向量寫入 CSV (每行一個值)
void write_csv(const std::string& output_path, const std::vector<float>& data) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "Cannot output CSV file: " << output_path << std::endl;
        exit(1);
    }
    for (const auto& v : data) file << v << '\n';
}

struct Stats { float mean{0.f}; float std{1.f}; };

// 讀取 stats_path，格式： key,value  (std,<val>\n mean,<val>)
Stats read_stats(const std::string& csv_path) {
    Stats s;
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Cannot open stats CSV: " << csv_path << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string key, val_str;
        if (!std::getline(ss, key, ',')) continue;
        if (!std::getline(ss, val_str)) continue;
        try {
            float v = std::stof(val_str);
            if (key == "mean") s.mean = v;
            else if (key == "std") s.std = v;
        } catch (const std::invalid_argument&) {
            std::cerr << "Stats CSV contains invalid data: " << line << std::endl;
        }
    }
    if (s.std == 0.f) {
        std::cerr << "std in stats file must not be 0" << std::endl;
        exit(1);
    }
    return s;
}

/******************************
 *  Main                      *
 ******************************/
int main(int argc, char* argv[]) {
    std::string model_path, input_path, output_path, stats_path;
    int n_steps = 25;

    // --- CLI 參數解析 --- //
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto read_next = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << flag << " requires a value" << std::endl;
                exit(1);
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model")            model_path = read_next(arg);
        else if (arg.rfind("--model=",0)==0)            model_path = arg.substr(8);
        else if (arg == "-i" || arg == "--input")       input_path = read_next(arg);
        else if (arg.rfind("--input=",0)==0)            input_path = arg.substr(8);
        else if (arg == "-o" || arg == "--output")      output_path = read_next(arg);
        else if (arg.rfind("--output=",0)==0)           output_path = arg.substr(9);
        else if (arg == "-s" || arg == "--stats_path")  stats_path = read_next(arg);
        else if (arg.rfind("--stats_path=",0)==0)       stats_path = arg.substr(13);
        else if (arg == "-n" || arg == "--n_steps")      n_steps = std::stoi(read_next(arg));
        else if (arg.rfind("--n_steps=",0)==0)          n_steps = std::stoi(arg.substr(10));
    }

    if (model_path.empty() || input_path.empty() || output_path.empty() || stats_path.empty()) {
        std::cerr << "Usage: ./run_model -m <model.tflite> -i <history.csv> -o <preds.csv> -s <stats.csv> -n <steps>\n";
        return 1;
    }

    std::cout << "model_path   : " << model_path << "\n"
              << "input_path   : " << input_path << "\n"
              << "output_path  : " << output_path << "\n"
              << "stats_path   : " << stats_path << "\n"
              << "n_steps      : " << n_steps << "\n";

    // --- 讀取歷史數據與統計量 --- //
    std::vector<float> history = read_csv(input_path);
    Stats stats = read_stats(stats_path);
    std::cout << "mean=" << stats.mean << ", std=" << stats.std << "\n";

    // --- 載入 TFLite 模型 --- //
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) { std::cerr << "Failed to load model\n"; return 1; }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) { std::cerr << "Failed to construct interpreter\n"; return 1; }
    if (interpreter->AllocateTensors() != kTfLiteOk) { std::cerr << "AllocateTensors failed\n"; return 1; }

    // --- 取得輸入 tensor 長度 (滑動窗口大小) --- //
    const int input_index = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_index);
    if (input_tensor->dims->size < 1) { std::cerr << "Invalid input tensor dims\n"; return 1; }
    const int input_len = input_tensor->dims->data[input_tensor->dims->size - 1];
    if (input_len <= 0) { std::cerr << "Input length must be > 0\n"; return 1; }

    // --- 歷史數據需足夠 --- //
    if (history.size() < static_cast<size_t>(input_len)) {
        std::cerr << "History size (" << history.size() << ") is smaller than input_len (" << input_len << ")\n";
        return 1;
    }

    // --- 初始化滑動窗口 (最後 input_len 筆) --- //
    std::vector<float> window(history.end() - input_len, history.end());
    std::vector<float> predictions;

    // --- 預測 n_steps 次 --- //
    for (int step = 0; step < n_steps; ++step) {
        // 1) 將 window 標準化後寫入輸入張量
        if (input_tensor->type == kTfLiteFloat32) {
            float* in = interpreter->typed_tensor<float>(input_index);
            for (int j = 0; j < input_len; ++j) {
                in[j] = (window[j] - stats.mean) / stats.std;
            }
        } else if (input_tensor->type == kTfLiteInt8) {
            int8_t* in = interpreter->typed_tensor<int8_t>(input_index);
            float scale = input_tensor->params.scale;
            int32_t zero_pt = input_tensor->params.zero_point;
            for (int j = 0; j < input_len; ++j) {
                float norm_val = (window[j] - stats.mean) / stats.std;           // 標準化
                int32_t quant = static_cast<int32_t>(std::round(norm_val / scale) + zero_pt);
                quant = std::min<int32_t>(std::max<int32_t>(quant, std::numeric_limits<int8_t>::min()), std::numeric_limits<int8_t>::max());
                in[j] = static_cast<int8_t>(quant);
            }
        } else {
            std::cerr << "Unsupported input tensor type\n"; return 1; }

        // 2) 執行推理
        if (interpreter->Invoke() != kTfLiteOk) { std::cerr << "Inference failed\n"; return 1; }

        // 3) 讀取輸出 (標準化值) 並還原
        const int output_index = interpreter->outputs()[0];
        TfLiteTensor* out_tensor = interpreter->tensor(output_index);
        float pred_std = 0.f;
        if (out_tensor->type == kTfLiteFloat32) {
            pred_std = interpreter->typed_tensor<float>(output_index)[0];
        } else if (out_tensor->type == kTfLiteInt8) {
            int8_t q = interpreter->typed_tensor<int8_t>(output_index)[0];
            pred_std = (static_cast<int32_t>(q) - out_tensor->params.zero_point) * out_tensor->params.scale;
        } else {
            std::cerr << "Unsupported output tensor type\n"; return 1; }

        float pred = pred_std * stats.std + stats.mean;  // 反標準化
        predictions.push_back(pred);

        // 4) 更新滑動窗口
        window.erase(window.begin());      // 移除最舊值
        window.push_back(pred);            // 加入新預測 (原始尺度)
    }

    // --- 寫入結果 --- //
    write_csv(output_path, predictions);
    std::cout << "Predictions saved to: " << output_path << std::endl;
    return 0;
}
