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

# Training Environment
- Windows 11, Anaconda, Python 3.11.11   
- Install packages:   
```bash
pip install -r requirements.txt
```

# ARIMA(p,d,q)
Training Scripts and Data Visualization of UTSD-Energy Wind Farm Data:   
- `train_arima_energyfarm - p5d1q0.ipynb`   
- `train_arima_energyfarm - p5d2q2.ipynb`   

Thus, we can output trained weights `model_XXX.csv` and data `input-XXX.csv` for purpose of prediction via cpp.   

According to your system platform (x86 or arm64), please use the following ways to compile main.cpp:   
- For x86, execute `cpp/arima_cpp_demo/x86/compile.sh`.   
- For arm64, execute `cpp/arima_cpp_demo/arm64/build.sh` with `cpp/arima_cpp_demo/arm64/CMakeLists.txt`.   

Finally, you can manage your folder like this   
```markdown
ar_wind_farm_arima_exe_file/
├── arm64-setable_preds/
│   ├── run_model
│   ├── run_model.sh
│   ├── model.csv
│   └── input.csv
├── x86-setable-preds/
│   ├── run_model
│   ├── run_model.sh
│   ├── model.csv
│   └── input.csv
└── data_samples_28k.csv
```

you can execute `run_model.sh` for inference.   

# AR-like Dense NN
Training Scripts and Data Visualization of UTSD-Energy Wind Farm Data:   
- `train_ar_dnn_energyfarm.ipynb`   

Thus, we can output `*.tflite` and standardization info `*-std-mean.csv` for purpose of prediction via cpp.    

Then, use the following ways to compile main.cpp:   
- For x86, execute `cpp/ar_dnn_cpp_demo/x86/compile.sh`.   
- For arm64, execute `cpp/ar_dnn_cpp_demo/arm64/build.sh` with `cpp/ar_dnn_cpp_demo/arm64/CMakeLists.txt`.   

Finally, manage your folder   
```markdown
ar_wind_farm_dnn_exe_file/
├── arm64-setable_preds/
│   ├── run_model
│   ├── run_model.sh
│   ├── input-dnn.csv
│   ├── ar_dnn-w10-l16-l32-l16_windfarm_0620.tflite
│   ├── ar_dnn-w10-l16-l32-l16_windfarm_0620-std-mean.csv
│   └── lib/*.so # the shared library (all dependency, quite large) for tensorflow compiled under arm64
└── x86-setable-preds/
    ├── run_model
    ├── run_model.sh
    ├── input-dnn.csv
    ├── ar_dnn-w10-l16-l32-l16_windfarm_0620.tflite
    ├── ar_dnn-w10-l16-l32-l16_windfarm_0620-std-mean.csv
    └── libtensorflow-lite.so # the shared library for tensorflow compiled under x86 (small)
```

Note `lib/*.so` in arm64 is too large. one should compile tf before execute `run_model.sh`   

# Classifier 1D-FCN
Training Scripts and Data Visualization of FordA:   
- TBD
- ...
