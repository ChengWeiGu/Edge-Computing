cmake_minimum_required(VERSION 3.9)

project(dense_nn_arm64)

# ─── 基本編譯選項 ──────────────────────────────────────────────
# 建議補上 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Header 路徑 ───────────────────────────────────────────────
# 你的 TensorFlow Lite header (拿 x86 clone 的即可)
# include_directories(${CMAKE_SOURCE_DIR}/tensorflow)
message("Use specified tensorflow/flatbuffers dirs under HOME")

set(TENSORFLOW_DIR $ENV{HOME}/tensorflow)
set(TFLITE_BUILD_DIR $ENV{HOME}/tensorflow/build_aarch64)
set(FLATBUFFER_DIR ${TFLITE_BUILD_DIR}/flatbuffers/include)

include_directories(
    ${TENSORFLOW_DIR}
    ${FLATBUFFER_DIR}
)

# ─── 動態庫所在資料夾 ───────────────────────────────────────────
# lib/：你專案裡集中放依賴 .so 的地方
# TFLITE_BUILD_DIR ： 官方 build 產生的所有 .so
set(PROJECT_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)

# 把這兩個目錄都丟給 linker / rpath-search
link_directories(${PROJECT_LIB_DIR} ${TFLITE_BUILD_DIR})

# link 時找不到符號 → 再去這兩個目錄掃一次
add_link_options(
    "-Wl,-rpath-link,${PROJECT_LIB_DIR}"
    "-Wl,-rpath-link,${TFLITE_BUILD_DIR}"
)

# ─── 可執行檔 ──────────────────────────────────────────────────
# 建可執行檔
add_executable(run_model main.cpp)

# 只需顯式鏈 libtensorflow-lite.so，其餘交給 -rpath-link
target_link_libraries(run_model
    ${PROJECT_LIB_DIR}/libtensorflow-lite.so
    pthread)

# 執行期 rpath：可執行檔所在目錄
set_target_properties(run_model PROPERTIES
    BUILD_RPATH "$ORIGIN/lib"
    INSTALL_RPATH "$ORIGIN/lib")


