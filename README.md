# EdgeCompute
Here shows some steps to achieve edge compute with python and c++   

# ARIMA

- Training Script and Data Visualization of UTSD-EnergyWindFarmData
`train_arima_energyfarm - p5d1q0.ipynb` and `train_arima_energyfarm - p5d2q2.ipynb`

- We can output trained weights `model_XXX.csv` and to-be-predicted data `input-XXX.csv` in order to do prediciton via cpp   

- According to your system platform (x86 or arm64), use different way to compile your cpp, for examples,   
  1. For x86, execute `arima_cpp_demo/x86/compile.sh`, and then execute `arima_cpp_demo/x86/run_model.sh` for inference.
  2. For arm64, execute `arima_cpp_demo/arm64/build.sh` to compile main.cpp with `arima_cpp_demo/arm64/CMakeLists.txt`.   
