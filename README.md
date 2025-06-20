# EdgeCompute
Here shows some steps to achieve edge compute with python and c++   

# Platform
- x86: wsl ubuntu   
```bash
wsl.exe -d Ubuntu
```
- arm64: aarch64:dunfell 進入交叉環境   
```bash
source /opt/weintek-sdk/dunfell-20250307-git/environment-setup-aarch64-weintek-linux 
```

# ARIMA(p,d,q)
Training Scripts and Data Visualization of UTSD-Energy Wind Farm Data:   
- `train_arima_energyfarm - p5d1q0.ipynb`   
- `train_arima_energyfarm - p5d2q2.ipynb`   

Thus, we can output trained weights `model_XXX.csv` and data `input-XXX.csv` for purpose of prediction via cpp.   

According to your system platform (x86 or arm64), please use the following ways to compile main.cpp:   
- For x86, execute `cpp/arima_cpp_demo/x86/compile.sh`.   
- For arm64, execute `cpp/arima_cpp_demo/arm64/build.sh` to compile main.cpp with `cpp/arima_cpp_demo/arm64/CMakeLists.txt`.   

Finally, your can manage your folder like this   
```pgsql
ar_wind_farm_arima_exe_file/
├── arm64-setable_preds/
│   ├── run_model
│   ├── model.csv
│   ├── run_model.sh
│   └── input.csv
├── x86-setable-preds/
│   ├── run_model
│   ├── model.csv
│   ├── run_model.sh
│   └── input.csv
└── data_samples_28k.csv
```

you can execute `run_model.sh` for inference.   

# AR-like Dense NN
Training Scripts and Data Visualization of UTSD-Energy Wind Farm Data:   
- TBD
- ...

# Classifier 1D-FCN
Training Scripts and Data Visualization of FordA:   
- TBD
- ...
