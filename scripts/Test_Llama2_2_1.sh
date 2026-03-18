#!/bin/bash

# Llama2 2:1 模型测试脚本
# 使用 llama2_1400_700 checkpoint 进行预测和可视化

echo "========================================================================"
echo "Llama2 2:1 模型测试"
echo "========================================================================"
echo "配置: 1400 -> 700 (2:1 比例)"
echo "Checkpoint: /home/user/WangS/Time-LLM-main/scripts/checkpoints/llama2_1400_700/checkpoint"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

cd /home/user/WangS/Time-LLM-main

/home/user/miniconda3/envs/time-llm311/bin/python VID.py \
  --checkpoint_path /home/user/WangS/Time-LLM-main/scripts/checkpoints/llama2_1400_700/checkpoint \
  --model_id Spectral_2_1_Llama2 \
  --model_comment TimeLLM-Spectral-Llama2-2-1 \
  --seq_len 1400 \
  --pred_len 700 \
  --llm_model LLAMA \
  --llm_dim 4096 \
  --llm_layers 32 \
  --wavelength_range 400 2499 \
  --data_path spectral_soil_test.csv \
  --root_path /home/user/WangS/Time-LLM-main/dataset/spectral/ \
  --save_dir /home/user/WangS/Time-LLM-main/visual_llama2_2_1 \
  --features S \
  --num_samples 10

echo "========================================================================"
echo "测试完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo "生成的可视化文件保存在: /home/user/WangS/Time-LLM-main/visual_llama2_2_1/"
echo "========================================================================"
