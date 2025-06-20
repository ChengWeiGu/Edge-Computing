#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <string>

// 讀取 ARIMA 權重
std::unordered_map<std::string, double> load_arima_model(const std::string& filename) {
    std::unordered_map<std::string, double> model;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        size_t comma = line.find(',');
        if (comma == std::string::npos) continue;
        std::string key = line.substr(0, comma);
        double val = std::stod(line.substr(comma + 1));
        model[key] = val;
    }
    return model;
}

// 讀取歷史資料
std::vector<double> load_history(const std::string& filename) {
    std::vector<double> vals;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        // 跳過空行
        if (line.empty()) continue;
        // 處理若有header，若第一行不是數字則跳過
        try {
            vals.push_back(std::stod(line));
        } catch (...) {
            continue;
        }
    }
    return vals;
}


//------------------------------------------------------------
// ARIMA(p,d,q) rolling forecast ‒ 支援 d = 0/1/…、p 或 q = 0
//------------------------------------------------------------
void arima_forecast(
    const std::unordered_map<std::string, double>& model,
    const std::vector<double>& history,
    int n_steps,
    std::vector<double>& out_forecast
) {
    /* ---------- 0. 讀模型參數 ---------- */
    const int p = int(model.at("order_p"));
    const int d = int(model.at("order_d"));
    const int q = int(model.at("order_q"));
    const double mu = model.count("mu") ? model.at("mu") : 0.0;

    if (p < 0 || d < 0 || q < 0) {                 // ★ 允許 p = 0，但不得為負
        std::cerr << "Error: p/d/q must be ≥ 0\n";
        return;
    }

    /* ---------- 1. 檢查並裁剪歷史 ---------- */
    const int need = p + d;                        // 公式最少需要 p+d 筆 y
    if (int(history.size()) < need) {              // ★ 修正括號 bug
        std::cerr << "Error: history length < p + d = " << need << '\n';
        return;
    }
    std::vector<double> hist(history.end() - need, history.end());

    /* ---------- 2. 讀 φ / θ / ε ---------- */
    std::vector<double> phi(p, 0.0), theta(q, 0.0), eps(q, 0.0);
    for (int i = 0; i < p; ++i)
        if (auto it = model.find("phi" + std::to_string(i + 1)); it != model.end())
            phi[i] = it->second;

    for (int j = 0; j < q; ++j) {
        auto tkey = "theta" + std::to_string(j + 1);
        auto ekey = "eps"   + std::to_string(j + 1);
        if (auto it = model.find(tkey); it != model.end()) theta[j] = it->second;
        if (auto it = model.find(ekey); it != model.end()) eps[j]   = it->second;
    }

    /* ---------- 3. 差分序列 ---------- */
    std::vector<std::vector<double>> diff(d);      // diff[k] = Δ^{k+1}y
    if (d > 0) {
        diff[0].reserve(hist.size() - 1);
        for (size_t i = 1; i < hist.size(); ++i)
            diff[0].push_back(hist[i] - hist[i - 1]);

        for (int k = 1; k < d; ++k) {
            const auto& prev = diff[k - 1];
            diff[k].reserve(prev.size() - 1);
            for (size_t i = 1; i < prev.size(); ++i)
                diff[k].push_back(prev[i] - prev[i - 1]);
        }
    }
    /* ★ d==0 時 diff_d 指向 hist；d>0 指向 Δ^d y 序列 */
    std::vector<double>& diff_d = (d == 0) ? hist : diff.back();

    /* ---------- 4. 最新 y_t ---------- */
    double last_level = hist.back();               // 原始尺度最新值

    /* ---------- 5. Forecast loop ---------- */
    for (int step = 0; step < n_steps; ++step) {

        /* 5-1  AR 部分：φ ⋅ (y or Δ^d y) */
        double ar_sum = 0.0;
        for (int i = 0; i < p; ++i)
            ar_sum += phi[i] * diff_d[diff_d.size() - 1 - i];

        /* 5-2  MA 部分：θ ⋅ ε */
        double ma_sum = 0.0;
        for (int j = 0; j < q; ++j)
            ma_sum += theta[j] * eps[j];

        /* 5-3  差分層級預測值 */
        double diff_d_hat = mu + ar_sum + ma_sum;

        /* 5-4  把 diff_d_hat 推回原始尺度 ---------------------- */
        double y_hat;
        if (d == 0) {                              // ★ 無差分：直接相加
            y_hat = last_level + diff_d_hat;
        } else {
            /* 逐層累加：new_diff[k] = diff[k].back() + carry */
            double carry = diff_d_hat;             // 從 Δ^d ŷ 開始
            for (int k = d - 1; k >= 1; --k) {     // k = d-1 … 1
                double new_diff = diff[k - 1].back() + carry;
                diff[k - 1].push_back(new_diff);   // 追加到對應差分序列
                carry = new_diff;                  // 傳給下一層
            }
            y_hat = last_level + carry;            // 最後 carry = Δŷ
        }

        /* 5-5  更新狀態 / 序列 / 殘差 ------------------------ */
        last_level = y_hat;
        diff_d.push_back((d == 0) ? y_hat : diff_d_hat); // AR 需要

        if (q > 0) {                           // ε_{t+1} 期望 0 → 入首
            eps.insert(eps.begin(),0.0);
            eps.pop_back();
        }

        out_forecast.push_back(y_hat);
    }
}


// 寫出預測結果
void write_forecast(const std::string& filename, const std::vector<double>& forecast) {
    std::ofstream file(filename);
    for (const auto& y : forecast) file << y << "\n";
}

int main(int argc, char* argv[]) {
    std::string model_path, input_path, output_path;
    int n_steps = 25;
    // 參數解析
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) model_path = argv[++i];
        else if (arg.find("--model=") == 0) model_path = arg.substr(8);

        else if ((arg == "--input" || arg == "-i") && i + 1 < argc) input_path = argv[++i];
        else if (arg.find("--input=") == 0) input_path = arg.substr(8);

        else if ((arg == "--output" || arg == "-o") && i + 1 < argc) output_path = argv[++i];
        else if (arg.find("--output=") == 0) output_path = arg.substr(9);

        else if ((arg == "--n_steps" || arg == "-n") && i + 1 < argc) n_steps = std::stoi(argv[++i]);
        else if (arg.find("--n_steps=") == 0) n_steps = std::stoi(arg.substr(10));
    }

    if (model_path.empty() || input_path.empty() || output_path.empty()) {
        std::cerr << "Usage: ./run_model --model=<model_file_path> --input=<input_file_path> --output=<output_file_path> --n_steps=<num_preds_points>\nOR ./run_model -m <model_file_path> -i <input_file_path> -o <output_file_path> -n <num_preds_points>\n";
        return 1;
    }

    std::cout << "model_path: " << model_path << std::endl;
    std::cout << "input_path: " << input_path << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "n_steps: " << n_steps << std::endl;

    auto model = load_arima_model(model_path);
    auto history = load_history(input_path);
    std::vector<double> forecast;
    arima_forecast(model, history, n_steps, forecast);
    write_forecast(output_path, forecast);
    std::cout << "Done" << std::endl;
    return 0;
}

