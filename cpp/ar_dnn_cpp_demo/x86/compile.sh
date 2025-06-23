#!/bin/bash

CPP_FILE="main.cpp"
OUT_FILE="run_model"

# 共享庫 libtensorflow-lite.so 需在當前資料夾
g++ -std=c++17 $CPP_FILE -o "${OUT_FILE}" \
  -I$HOME/tensorflow                 \
  -L$HOME/tensorflow/build-shared    \
  -ltensorflow-lite -lpthread \
  -Wl,-rpath=.