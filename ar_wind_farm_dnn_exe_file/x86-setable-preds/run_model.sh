#!/bin/bash

MOD_FILE="ar_dnn-w10-l16-l32-l16_windfarm_0620.tflite"
STATS_FILE="ar_dnn-w10-l16-l32-l16_windfarm_0620-std-mean.csv"
INP_FILE="input-dnn.csv"
OUT_FILE="output-dnn.csv"
N_STEPS=25


time ./run_model -m $MOD_FILE -i $INP_FILE -o $OUT_FILE -s $STATS_FILE -n $N_STEPS
# ./run_model --model=$MOD_FILE --input=$INP_FILE --output=$OUT_FILE --stats_path=$STATS_FILE --n_steps=$N_STEPS


# 將預測資料拋到本機資料夾中
# MNT_PATH="/mnt/d/Projects/Regression_train_inference/checkpoints/ar_dnn"
# cp $OUT_FILE $MNT_PATH