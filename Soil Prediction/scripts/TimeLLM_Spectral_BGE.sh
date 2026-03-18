#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

accelerate launch \
  --multi_gpu \
  --mixed_precision bf16 \
  --num_processes 1 \
  --main_process_port 29510 \
  "${PROJECT_ROOT}/run_main.py" \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "${PROJECT_ROOT}/dataset/spectral/" \
  --data_path spectral_soil_test.csv \
  --model_id SoilSpecLLM_BGE \
  --model TimeLLM \
  --data Spectral \
  --features S \
  --seq_len 1575 \
  --label_len 0 \
  --pred_len 525 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 2 \
  --learning_rate 0.0005 \
  --llm_layers 12 \
  --llm_model BGE-M3 \
  --llm_dim 1024 \
  --train_epochs 2 \
  --wavelength_range 400 2499 \
  --model_comment SoilSpecLLM-BGE
