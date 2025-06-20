cmake_minimum_required(VERSION 3.9)

project(arima_forecast)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(run_model main.cpp)         # 僅編 main.cpp
