#!/bin/bash

# BGE-M3 3:1 模型测试脚本
# 使用 BGE-M3_1575_525 checkpoint 进行预测和可视化

echo "========================================================================"
echo "BGE-M3 3:1 模型测试"
echo "========================================================================"
echo "配置: 1575 -> 525 (3:1 比例)"
echo "Checkpoint: /home/user/WangS/Time-LLM-main/checkpoints/BGE-M3_1575_525/checkpoint"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

cd /home/user/WangS/Time-LLM-main

/home/user/miniconda3/envs/time-llm311/bin/python VID.py \
  --checkpoint_path /home/user/WangS/Time-LLM-main/checkpoints/BGE-M3_1575_525/checkpoint \
  --model_id Spectral_3_1_BGE_M3 \
  --model_comment TimeLLM-Spectral-BGE-M3-3-1 \
  --seq_len 1575 \
  --pred_len 525 \
  --llm_model BGE-M3 \
  --llm_dim 1024 \
  --llm_layers 24 \
  --wavelength_range 400 2499 \
  --data_path spectral_soil_test.csv \
  --root_path /home/user/WangS/Time-LLM-main/dataset/spectral/ \
  --save_dir /home/user/WangS/Time-LLM-main/visual_bge_m3_3_1 \
  --features S \
  --num_samples 10

echo "========================================================================"
echo "测试完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo "生成的可视化文件保存在: /home/user/WangS/Time-LLM-main/visual_bge_m3_3_1/"
echo "========================================================================"
