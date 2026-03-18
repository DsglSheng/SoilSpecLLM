model_name=TimeLLM
train_epochs=2
learning_rate=0.0005
llama_layers=32

master_port=29500
num_process=4
batch_size=2
d_model=16
d_ff=32

# 光谱数据特定参数
wavelength_min=400
wavelength_max=2499
comment='TimeLLM-Spectral'

# 光谱预测任务(总共4个训练任务，1：1(1050-1050),2:1(1400:700),3:1(1575-525),4:1(1680-420),改seq_len和pred_len,注意测试的时候要改VID文件对应的长度)
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port /home/user/WangS/Time-LLM-main/run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/user/WangS/Time-LLM-main/dataset/spectral/ \
  --data_path spectral_data.csv \
  --model_id Spectral_3_1_xr \
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
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --wavelength_range $wavelength_min $wavelength_max \
  --model_comment $comment

## 光谱预测任务 - 中等波长范围预测 (300->200)
#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port /home/user/WangS/Time-LLM-main/run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path /home/user/WangS/Time-LLM-main/dataset/spectral/ \
#  --data_path spectral_data.csv \
#  --model_id Spectral_300_200 \
#  --model $model_name \
#  --data Spectral \
#  --features M \
#  --seq_len 300 \
#  --pred_len 200 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --batch_size $batch_size \
#  --learning_rate $learning_rate \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --wavelength_range $wavelength_min $wavelength_max \
#  --model_comment $comment
#
## 光谱预测任务 - 大波长范围预测 (500->300)
#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port /home/user/WangS/Time-LLM-main/run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path /home/user/WangS/Time-LLM-main/dataset/spectral/ \
#  --data_path spectral_data.csv \
#  --model_id Spectral_500_300 \
#  --model $model_name \
#  --data Spectral \
#  --features M \
#  --seq_len 500 \
#  --pred_len 300 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --batch_size $batch_size \
#  --learning_rate $learning_rate \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --wavelength_range $wavelength_min $wavelength_max \
#  --model_comment $comment
#
## 光谱预测任务 - 超大波长范围预测 (800->500)
#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port /home/user/WangS/Time-LLM-main/run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path /home/user/WangS/Time-LLM-main/dataset/spectral/ \
#  --data_path spectral_data.csv \
#  --model_id Spectral_800_500 \
#  --model $model_name \
#  --data Spectral \
#  --features M \
#  --seq_len 800 \
#  --pred_len 500 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --batch_size $batch_size \
#  --learning_rate $learning_rate \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --wavelength_range $wavelength_min $wavelength_max \
#  --model_comment $comment