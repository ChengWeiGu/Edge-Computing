#!/usr/bin/env bash
set -e

BUILD_DIR=build_aarch64

echo "▶ 清理並建立 $BUILD_DIR"
rm -rf "$BUILD_DIR"

cmake -S . -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "✅ 完成！可執行檔在 $BUILD_DIR/run_model"
