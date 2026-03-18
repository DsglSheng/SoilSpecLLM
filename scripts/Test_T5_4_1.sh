#!/bin/bash

echo "========================================================================"
echo "T5 4:1 模型测试 (1680 -> 420)"
echo "========================================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

/home/user/miniconda3/envs/time-llm311/bin/python /home/user/WangS/Time-LLM-main/VID.py \
  --checkpoint_path /home/user/WangS/Time-LLM-main/scripts/checkpoints/long_term_forecast_Spectral_4_1_T5_TimeLLM_Spectral_ftS_sl1680_ll0_pl420_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Exp_0-TimeLLM-Spectral-T5-4_1/checkpoint \
  --model_id Spectral_4_1_T5 \
  --model_comment TimeLLM-Spectral-T5-4-1 \
  --seq_len 1680 \
  --pred_len 420 \
  --llm_model T5 \
  --llm_dim 768 \
  --llm_layers 12 \
  --wavelength_range 400 2499 \
  --data_path spectral_soil_test.csv \
  --root_path /home/user/WangS/Time-LLM-main/dataset/spectral/ \
  --save_dir /home/user/WangS/Time-LLM-main/visual_t5_4_1 \
  --features S \
  --num_samples 200

echo "========================================================================"
echo "测试完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo "生成的可视化文件保存在: /home/user/WangS/Time-LLM-main/visual_t5_4_1/"
echo "========================================================================"
