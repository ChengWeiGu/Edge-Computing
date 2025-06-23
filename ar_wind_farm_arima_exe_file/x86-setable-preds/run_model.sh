#!/bin/bash

MOD_FILE="model.csv"
INP_FILE="input.csv"
OUT_FILE="output.csv"
N_STEPS=25

# ./run_model -m $MOD_FILE -i $INP_FILE -o $OUT_FILE -n $N_STEPS
./run_model --model=$MOD_FILE --input=$INP_FILE --output=$OUT_FILE --n_steps=$N_STEPS

# 將預測資料拋到本機資料夾中
# MNT_PATH="/mnt/d/Projects/Regression_train_inference/checkpoints/arima"
# cp $OUT_FILE $MNT_PATH
