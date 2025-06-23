#!/bin/bash

# 錯誤即中止（Error Exit）的模式
set -e

# 用於系統提示
MACHINE=aarch64
# 生成run_model所在的資料夾
BUILD_DIR=build_aarch64
# 專案內放依賴 .so 的目錄
SRC_LIB_DIR=lib
# 輸出內也建同名目錄
DEST_LIB_DIR=${BUILD_DIR}/lib
# 共享庫所在的目錄
TFLITE_BUILD_DIR="$HOME/tensorflow/build_aarch64"

echo "You select $MACHINE"
echo "Start build $MACHINE on $BUILD_DIR folder"

rm -rf "${BUILD_DIR}" && mkdir ${BUILD_DIR} && mkdir "${DEST_LIB_DIR}"

# 自動同步所有 .so 到 lib
# 只複製新檔 (-u)；找出 build_aarch64 內所有 lib*.so
mkdir -p "${SRC_LIB_DIR}"
find "${TFLITE_BUILD_DIR}" -name 'lib*.so' -exec cp -u {} "${SRC_LIB_DIR}" \;

# 執行 CMake 編譯
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

# 執行期檔案一次帶走
echo "Copying libs / model / data …"
cp "${SRC_LIB_DIR}"/*.so "${DEST_LIB_DIR}"
cp ./*.tflite ./*-std-mean.csv ./input-dnn.csv "${BUILD_DIR}"

echo "✔ 交叉編譯完成 → ${BUILD_DIR}/run_model"
echo "✔ 依賴 .so 已複製至 ${DEST_LIB_DIR}，執行檔會在 ./lib 找到它們"