#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL_NAME="SoilSpecLLM"
TRAIN_EPOCHS=2
LEARNING_RATE=0.0005
LLM_LAYERS=8
MASTER_PORT=29500
NUM_PROCESS=1
BATCH_SIZE=2
WAVELENGTH_MIN=400
WAVELENGTH_MAX=2499

accelerate launch \
  --multi_gpu \
  --mixed_precision bf16 \
  --num_processes "${NUM_PROCESS}" \
  --main_process_port "${MASTER_PORT}" \
  "${PROJECT_ROOT}/run_main.py" \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "${PROJECT_ROOT}/dataset/spectral/" \
  --data_path spectral_soil_test.csv \
  --model_id SoilSpecLLM_Spectral \
  --model "${MODEL_NAME}" \
  --data Spectral \
  --features S \
  --seq_len 1575 \
  --label_len 0 \
  --pred_len 525 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --llm_layers "${LLM_LAYERS}" \
  --train_epochs "${TRAIN_EPOCHS}" \
  --wavelength_range "${WAVELENGTH_MIN}" "${WAVELENGTH_MAX}" \
  --model_comment SoilSpecLLM-Spectral

