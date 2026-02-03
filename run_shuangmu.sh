#!/bin/bash

# 设置脚本遇到错误即停止
set -e

echo "=========================================================="
echo "   Starting Stereo Calibration Experiments (MuJoCo)"
echo "=========================================================="
# 为了防止 WSL 图形库冲突，加这一行保险（可选，推荐保留）
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
echo "Date: $(date)"

# 确保输出目录存在
mkdir -p /home/zkj/camera_calibration_sim/zhudongbiaoding/output/stereo_greedy
mkdir -p /home/zkj/camera_calibration_sim/zhudongbiaoding/output/stereo_random

# 1. 运行 Active (Proposed) 实验
echo ""
echo "[1/2] Running PROPOSED ACTIVE method..."
echo "---------------------------------------"
python3 greedy_shuangmu.py

# 2. 运行 Random (Baseline) 实验
echo ""
echo "[2/2] Running RANDOM BASELINE method..."
echo "---------------------------------------"
python3 random_shuangmu.py

echo ""
echo "=========================================================="
echo "   All Experiments Finished Successfully!"
echo "=========================================================="
echo "Results are saved in:"
echo "  -> output/stereo_active/stereo_results.csv"
echo "  -> output/stereo_random/random_stereo_results.csv"
echo "Please use these CSV files for plotting paper figures."