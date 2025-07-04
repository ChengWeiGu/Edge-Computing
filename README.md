# EdgeCompute
Here shows some steps to achieve edge compute with python and c++   

# Platform
- x86 wsl ubuntu:   
```bash
# 1. install and enter wsl in powershell
wsl --install
wsl.exe -d Ubuntu

# 2. update libs
sudo apt update && sudo apt upgrade -y

# 3. install necessary tools
sudo apt install git cmake g++ wget unzip -y

# 4. go to root and install TensorFlow Lite C++
cd ~
git clone https://github.com/tensorflow/tensorflow.git

# 5. Build libtensorflow-lite.so under build-share
cd ~/tensorflow
rm -rf build-shared                # make sure clean

cmake -S tensorflow/lite -B build-shared \
  -DCMAKE_BUILD_TYPE=Release       \
  -DBUILD_SHARED_LIBS=ON           \   # CMake 全域開關
  -DTFLITE_ENABLE_SHARED=ON        \   # TFLite 自己的開關
  -DTFLITE_ENABLE_XNNPACK=ON           # 建議打開，效能較好

cmake --build build-shared -j$(nproc)

# 6. check if the build is successful
ls -lh libtensorflow-lite.so
```

**Additional troubleshooting**
```bash
#----------- if build failed, please checkout to old ver-----------
# 1. get all tags
git fetch --tags

# 2. lookup what tags begins with v2.15.x (or other stable ver)
git tag -l | grep v2.15

# 3. switch to a stable ver where CMake compiling is OK（e.g. 2.15.0）
git checkout v2.15.0 

# 4. clear all failed build and restart build
```

- arm64: aarch64:dunfell 進入交叉環境   
```bash
# 1. install and enter env (take weintek toolchain for a instance)
bash weintek-sdk-x86_64-meta-toolchain-weintek-aarch64-toolchain-dunfell-20250307-git.sh
source /opt/weintek-sdk/dunfell-20250307-git/environment-setup-aarch64-weintek-linux

# 2. Use CMake to cross-compile tf so, and use smaller cpu 8 --> 2 to avoid `OOM-killer`
cd ~/tensorflow
rm -rf build_aarch64 && mkdir build_aarch64  # 乾淨環境

cmake -S tensorflow/lite -B build_aarch64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DTFLITE_ENABLE_SHARED=ON \
  -DTFLITE_ENABLE_XNNPACK=ON \
  -DXNNPACK_ENABLE_ARM_I8MM=OFF \
  -DXNNPACK_ENABLE_ARM_BF16=OFF \
  -DCMAKE_CXX_STANDARD=17

cmake --build build_aarch64 -j2 # use smaller cpu to compile (Wait for a period of time)
```

**Additional troubleshooting**
```bash
#----------- if build failed, please make sure the CMake flags are correct-----------
#-DBUILD_SHARED_LIBS=ON => 全域開共享庫
#-DTFLITE_ENABLE_SHARED=ON => TFLite 自己的 switch
#-DTFLITE_ENABLE_XNNPACK=ON => 打開 XNNPACK 提升推論速度
#-DTFLITE_ENABLE_ARM_I8MM=OFF => 直接關掉 I8MM 避免編譯出錯
#-DCMAKE_CXX_FLAGS => 讓某一類warning不要被視為錯誤而導致編譯失敗
#-DCMAKE_C_FLAGS="-Werror=return-type"  => 讓某一類warning不要被視為錯誤而導致編譯失敗
#-DCMAKE_CXX_FLAGS="-Werror=return-type"  => 讓某一類warning不要被視為錯誤而導致編譯失敗

#----------- error: control reaches end of non-void function [-Werror=return-type]-----------
# go tointerop
cd ~/tensorflow/tensorflow/lite/core/async/interop
# back up 
cp variant.cc variant.cc.bak
# revise variant.cc at line 40: switch case, and insert
default:
      return false;

# go tokernels
cd ~/tensorflow/tensorflow/lite/kernels
# backup
cp conv3d.cc conv3d.cc.bak
# revise conv3d.cc at line 244 L244, and insert
default:{
      return kTfLiteOk;
    }

# backup
cp reduce_window.cc reduce_window.cc.bak
# revise reduce_window.cc at line 327 and insert 
default:
	return kTfLiteOk;
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

# Conv1D on FordA
Training Scripts and Data Visualization of FordA:   
- `train_cls_conv1d_fordA.ipynb`

From the training script, we can prepare `*.tflite` file before compiling cpp.   
- For x86, execute `cpp/conv1d_cpp_demo/x86/compile.sh`.
- For arm64, execute `cpp/conv1d_cpp_demo/arm64/build.sh` with `cpp/conv1d_cpp_demo/arm64/CMakeLists.txt`.   

Finally, manage your folder   
```markdown
cls_fordA_conv1d_exe_file/
├── arm64/
│   ├── run_model
│   ├── run_model.sh
│   ├── sample_idx_*.csv # there are 6 test samples
│   ├── cls_1dcnn_forda_0612.tflite
│   └── lib/*.so # the shared library (all dependency, quite large) for tensorflow compiled under arm64
└── x86/
    ├── run_model
    ├── run_model.sh
    ├── sample_idx_*.csv # there are 6 test samples
    ├── cls_1dcnn_forda_0612.tflite
    └── libtensorflow-lite.so # the shared library for tensorflow compiled under x86 (small)
```

After executing `run_model.sh`, predicted results for each test sample will be generated. If you open one of the result files, you will see something like:   
```csv
1, 0.95272
```
Here, `1` represents the predicted class, followed by its probability `0.95272`.   
