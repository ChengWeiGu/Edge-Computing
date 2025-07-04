#!/bin/bash

# 定義輸入和輸出文件
MOD_FILE="cls_1dcnn_forda_0612.tflite"
INP_FILES=("sample_idx_200_lab_1.csv" "sample_idx_300_lab_0.csv" "sample_idx_600_lab_1.csv" "sample_idx_700_lab_0.csv" "sample_idx_1000_lab_0.csv" "sample_idx_1200_lab_1.csv")
OUT_FILES=("result_idx_200.csv" "result_idx_300.csv" "result_idx_600.csv" "result_idx_700.csv" "result_idx_1000.csv" "result_idx_1200.csv")


# 檢查長度是否一致
if [ ${#INP_FILES[@]} -ne ${#OUT_FILES[@]} ]; then
    echo "Error: The number of input files does not match the number of output files."
    exit 1
fi


# scan 所有文件
for i in "${!INP_FILES[@]}"; do
    INP_FILE="${INP_FILES[$i]}"
    OUT_FILE="${OUT_FILES[$i]}"

    # execute cmd
    echo "--------------------------------------------------------"
    echo "Process input file : ${INP_FILE}"
    echo
    time ./run_model -m $MOD_FILE -i $INP_FILE -o $OUT_FILE
    # ./run_model --model=$MOD_FILE --input=$INP_FILE --output=$OUT_FILE
    echo "--------------------------------------------------------"
done