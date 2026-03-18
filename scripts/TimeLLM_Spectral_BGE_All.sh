#!/bin/bash

# BGE-M3 模型多比例自动训练脚本
# 训练顺序: 4:1 -> 2:1 -> 1:1 (3:1 已完成)

model_name=TimeLLM
train_epochs=2
learning_rate=0.0005
llama_layers=32

num_process=4
batch_size=2
d_model=16
d_ff=32

# 光谱数据特定参数
wavelength_min=400
wavelength_max=2499

# 定义训练配置数组
# 格式: "比例标识:seq_len:pred_len:端口号"
configs=(
    "4_1:1680:420:29700"
    "2_1:1400:700:29800"
    "1_1:1050:1050:29900"
)

echo "========================================================================"
echo "BGE-M3 多比例自动训练脚本"
echo "========================================================================"
echo "训练配置:"
echo "  - 模型: BGE-M3 (1024 维)"
echo "  - 训练轮数: ${train_epochs} epochs"
echo "  - GPU: ${num_process} 卡并行"
echo "  - 学习率: ${learning_rate}"
echo "========================================================================"
echo ""

# 循环执行每个配置
for config in "${configs[@]}"; do
    # 解析配置
    IFS=':' read -r ratio seq_len pred_len port <<< "$config"
    
    echo "========================================================================"
    echo "开始训练 ${ratio} 比例配置"
    echo "========================================================================"
    echo "  seq_len: ${seq_len}"
    echo "  pred_len: ${pred_len}"
    echo "  端口: ${port}"
    echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------------------"
    
    # 执行训练
    accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $port /home/user/WangS/Time-LLM-main/run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path /home/user/WangS/Time-LLM-main/dataset/spectral/ \
      --data_path spectral_data.csv \
      --model_id Spectral_${ratio}_BGE_M3 \
      --model $model_name \
      --data Spectral \
      --features S \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --llm_model BGE-M3 \
      --llm_dim 1024 \
      --llm_layers $llama_layers \
      --train_epochs $train_epochs \
      --wavelength_range $wavelength_min $wavelength_max \
      --model_comment TimeLLM-Spectral-BGE-M3-${ratio}
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo "------------------------------------------------------------------------"
        echo "✓ ${ratio} 比例训练完成"
        echo "  完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================================"
        echo ""
    else
        echo "------------------------------------------------------------------------"
        echo "✗ ${ratio} 比例训练失败，退出码: $?"
        echo "  失败时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================================"
        echo ""
        # 可以选择继续或退出
        # exit 1  # 取消注释以在失败时停止
    fi
    
    # 短暂休息，避免端口冲突
    echo "等待 10 秒后开始下一个训练..."
    sleep 10
    echo ""
done

echo "========================================================================"
echo "所有训练任务完成！"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""
echo "训练结果汇总:"
echo "  ✓ 3:1 (1575->525) - 已完成 (端口 29600)"
echo "  ✓ 4:1 (1680->420) - 端口 29700"
echo "  ✓ 2:1 (1400->700) - 端口 29800"
echo "  ✓ 1:1 (1050->1050) - 端口 29900"
echo "========================================================================"
