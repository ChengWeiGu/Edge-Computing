# EdgeCompute
Here shows some steps to achieve edge compute with python and c++   

# ARIMA(p,d,q)
Training Script and Data Visualization of UTSD-EnergyWindFarmData:   
- `train_arima_energyfarm - p5d1q0.ipynb`   
- `train_arima_energyfarm - p5d2q2.ipynb`   

Thus, we can output trained weights `model_XXX.csv` and data `input-XXX.csv` for purpose of prediction via cpp   

According to your system platform (x86 or arm64), please use the following ways to compile your cpp:   
- For x86, execute `arima_cpp_demo/x86/compile.sh`, and then execute `arima_cpp_demo/x86/run_model.sh` for inference.   
- For arm64, execute `arima_cpp_demo/arm64/build.sh` to compile main.cpp with `arima_cpp_demo/arm64/CMakeLists.txt`.

Finally, your can manage your folder like this   
```pgsql
ar_wind_farm_arima_exe_file/
├── arm64-setable_preds/
│   ├── run_model
│   ├── model.csv
│   ├── run_model_sh
│   └── input.csv
├── x86-setable-preds/
│   ├── run_model
│   ├── model.csv
│   ├── run_model_sh
│   └── input.csv
└── data_samples_28k.csv
```
