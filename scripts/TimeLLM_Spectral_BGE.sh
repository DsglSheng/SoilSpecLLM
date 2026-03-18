model_name=TimeLLM
train_epochs=2
learning_rate=0.0005
llama_layers=32

master_port=29600
num_process=4
batch_size=2
d_model=16
d_ff=32

# 光谱数据特定参数
wavelength_min=400
wavelength_max=2499
comment='TimeLLM-Spectral-BGE-M3'

# 使用 BGE-M3 模型进行光谱预测任务 (3:1 比例: 1575->525)
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port /home/user/WangS/Time-LLM-main/run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/user/WangS/Time-LLM-main/dataset/spectral/ \
  --data_path spectral_data.csv \
  --model_id Spectral_3_1_BGE_M3 \
  --model $model_name \
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
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model BGE-M3 \
  --llm_dim 1024 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --wavelength_range $wavelength_min $wavelength_max \
  --model_comment $comment
