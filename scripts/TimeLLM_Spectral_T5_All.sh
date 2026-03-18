#!/bin/bash

# T5 模型光谱预测训练脚本 - 所有比例 (1:1, 2:1, 3:1, 4:1)
# 使用 T5-base 预训练模型进行光谱数据预测

echo "========================================================================"
echo "T5 模型光谱预测训练 - 所有比例"
echo "========================================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

# 公共参数
MODEL_NAME="T5"
LLM_DIM=768
LLM_LAYERS=12
BATCH_SIZE=2
LEARNING_RATE=0.0005
TRAIN_EPOCHS=2
PATIENCE=3
NUM_PROCESS=4

# 数据参数
DATA_PATH="spectral_data.csv"
ROOT_PATH="/home/user/WangS/Time-LLM-main/dataset/spectral/"
FEATURES="S"
TARGET="OL"
FREQ="h"

# 模型参数
D_MODEL=16
D_FF=32
NUM_HEADS=8
E_LAYERS=2
D_LAYERS=1
FACTOR=3
PATCH_LEN=16
STRIDE=8

# 定义不同的比例配置
# 格式: "比例:seq_len:pred_len:端口"
declare -a CONFIGS=(
    "1:1:1050:1050:29600"
    "2:1:1400:700:29700"
    "3:1:1575:525:29800"
    "4:1:1680:420:29900"
)

# 循环执行每个配置
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r -a parts <<< "$config"
    RATIO="${parts[0]}_${parts[1]}"
    SEQ_LEN="${parts[2]}"
    PRED_LEN="${parts[3]}"
    PORT="${parts[4]}"
    
    echo ""
    echo "========================================================================"
    echo "训练配置: ${RATIO} (seq_len=${SEQ_LEN}, pred_len=${PRED_LEN})"
    echo "========================================================================"
    echo "端口: ${PORT}"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 设置模型ID和注释
    MODEL_ID="Spectral_${RATIO}_${MODEL_NAME}"
    MODEL_COMMENT="TimeLLM-Spectral-${MODEL_NAME}-${RATIO}"
    
    # 运行训练
    /home/user/miniconda3/envs/time-llm311/bin/python -m accelerate.commands.launch --multi_gpu --mixed_precision bf16 --num_processes ${NUM_PROCESS} --main_process_port ${PORT} /home/user/WangS/Time-LLM-main/run_main.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --model_id ${MODEL_ID} \
        --model_comment ${MODEL_COMMENT} \
        --model TimeLLM \
        --data Spectral \
        --features ${FEATURES} \
        --seq_len ${SEQ_LEN} \
        --label_len 0 \
        --pred_len ${PRED_LEN} \
        --e_layers ${E_LAYERS} \
        --d_layers ${D_LAYERS} \
        --factor ${FACTOR} \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --llm_model ${MODEL_NAME} \
        --llm_dim ${LLM_DIM} \
        --llm_layers ${LLM_LAYERS} \
        --train_epochs ${TRAIN_EPOCHS} \
        --patience ${PATIENCE} \
        --des 'Exp' \
        --itr 1 \
        --patch_len ${PATCH_LEN} \
        --stride ${STRIDE} \
        --n_heads ${NUM_HEADS}
    
    echo "配置 ${RATIO} 训练完成: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"
    
    # 短暂休息，避免GPU过热
    echo "休息 10 秒..."
    sleep 10
done

echo ""
echo "========================================================================"
echo "所有 T5 模型训练完成！"
echo "========================================================================"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""
echo "训练的模型配置："
echo "  1. T5 1:1 (1050 -> 1050)"
echo "  2. T5 2:1 (1400 -> 700)"
echo "  3. T5 3:1 (1575 -> 525)"
echo "  4. T5 4:1 (1680 -> 420)"
echo ""
echo "Checkpoint 保存位置: ./checkpoints/"
echo "========================================================================"
