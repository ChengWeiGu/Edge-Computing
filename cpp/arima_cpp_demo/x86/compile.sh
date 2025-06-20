#!/bin/bash

CPP_FILE="main.cpp"
OUT_FILE="run_model"

g++ -std=c++17 -O3 $CPP_FILE -o $OUT_FILE