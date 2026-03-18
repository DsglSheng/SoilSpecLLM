# T5 模型训练执行指令

## ✅ 问题已修复

已成功修复 T5 训练脚本的两个问题：

1. **accelerate 命令找不到** → 使用完整 Python 路径调用 accelerate 模块
2. **run_main.py 路径错误** → 使用绝对路径

训练脚本现已正常运行！

---

## 🚀 在服务器上执行 T5 训练

### **方式一：后台运行（推荐）**

```bash
# 1. SSH 登录服务器
ssh user@192.168.190.48

# 2. 进入 scripts 目录
cd /home/user/WangS/Time-LLM-main/scripts

# 3. 后台运行训练并保存日志
nohup bash TimeLLM_Spectral_T5_All.sh > t5_training.log 2>&1 &

# 4. 查看进程
ps aux | grep TimeLLM_Spectral_T5

# 5. 实时查看日志
tail -f t5_training.log

# 6. 退出日志查看（不会停止训练）
# 按 Ctrl+C
```

### **方式二：使用 screen（推荐）**

```bash
# 1. SSH 登录服务器
ssh user@192.168.190.48

# 2. 创建 screen 会话
screen -S t5_training

# 3. 进入目录并运行
cd /home/user/WangS/Time-LLM-main/scripts
bash TimeLLM_Spectral_T5_All.sh

# 4. 分离 screen（训练继续运行）
# 按 Ctrl+A，然后按 D

# 5. 重新连接查看进度
screen -r t5_training

# 6. 查看所有 screen 会话
screen -ls

# 7. 终止 screen 会话（在 screen 内）
exit
```

### **方式三：使用 tmux**

```bash
# 1. SSH 登录服务器
ssh user@192.168.190.48

# 2. 创建 tmux 会话
tmux new -s t5_training

# 3. 进入目录并运行
cd /home/user/WangS/Time-LLM-main/scripts
bash TimeLLM_Spectral_T5_All.sh

# 4. 分离 tmux（训练继续运行）
# 按 Ctrl+B，然后按 D

# 5. 重新连接
tmux attach -t t5_training

# 6. 查看所有会话
tmux ls

# 7. 终止会话（在 tmux 内）
exit
```

---

## 📊 监控训练进度

### **查看 GPU 使用情况**
```bash
watch -n 1 nvidia-smi
```

### **查看训练日志**
```bash
# 实时查看（如果使用 nohup）
tail -f /home/user/WangS/Time-LLM-main/scripts/t5_training.log

# 查看最后 100 行
tail -n 100 /home/user/WangS/Time-LLM-main/scripts/t5_training.log

# 搜索特定信息
grep "训练配置" /home/user/WangS/Time-LLM-main/scripts/t5_training.log
grep "完成" /home/user/WangS/Time-LLM-main/scripts/t5_training.log
grep -i "error" /home/user/WangS/Time-LLM-main/scripts/t5_training.log
```

### **查看 checkpoint**
```bash
# 查看所有 T5 checkpoint
ls -lht /home/user/WangS/Time-LLM-main/checkpoints/ | grep T5

# 查看 checkpoint 详情
ls -lh /home/user/WangS/Time-LLM-main/checkpoints/long_term_forecast_Spectral_*_T5_*/
```

---

## 📋 训练配置详情

### **4 个训练配置**

| 配置 | 比例 | seq_len | pred_len | 模型ID | 预计时间 |
|------|------|---------|----------|--------|----------|
| 1 | 1:1 | 1050 | 1050 | Spectral_1_1_T5 | 1-3 小时 |
| 2 | 2:1 | 1400 | 700 | Spectral_2_1_T5 | 1-3 小时 |
| 3 | 3:1 | 1575 | 525 | Spectral_3_1_T5 | 1-3 小时 |
| 4 | 4:1 | 1680 | 420 | Spectral_4_1_T5 | 1-3 小时 |

**总预计时间**: 4-12 小时

### **T5 模型参数**
- 模型：T5-base
- 维度：768
- 层数：12
- 路径：`/home/user/WangS/ChatTime-main/t5-base`

### **训练参数**
- 批次大小：8
- 学习率：0.001
- 训练轮数：10
- 早停耐心：3
- 混合精度：bf16
- GPU：1 卡
- 端口：29501

### **数据集**
- 训练集：115,751 样本
- 验证集：16,537 样本
- 测试集：33,072 样本
- 波长点数：2,100 (400-2499 nm)

---

## 📁 输出文件位置

训练完成后，checkpoint 将保存在：

```
/home/user/WangS/Time-LLM-main/checkpoints/
├── long_term_forecast_Spectral_1_1_T5_TimeLLM_Spectral_ftS_sl1050_ll0_pl1050_...
├── long_term_forecast_Spectral_2_1_T5_TimeLLM_Spectral_ftS_sl1400_ll0_pl700_...
├── long_term_forecast_Spectral_3_1_T5_TimeLLM_Spectral_ftS_sl1575_ll0_pl525_...
└── long_term_forecast_Spectral_4_1_T5_TimeLLM_Spectral_ftS_sl1680_ll0_pl420_...
```

每个 checkpoint 目录包含：
- `checkpoint.pth` - 模型权重
- `config.json` - 配置文件
- 训练日志和指标

---

## 🔧 故障排查

### **问题 1: 训练中断**
```bash
# 查看最后的日志
tail -n 50 /home/user/WangS/Time-LLM-main/scripts/t5_training.log

# 查看是否有进程在运行
ps aux | grep python | grep run_main

# 重新启动训练
cd /home/user/WangS/Time-LLM-main/scripts
nohup bash TimeLLM_Spectral_T5_All.sh > t5_training_restart.log 2>&1 &
```

### **问题 2: GPU 内存不足**
如果出现 CUDA out of memory 错误，需要减小批次大小：

```bash
# 编辑脚本
nano /home/user/WangS/Time-LLM-main/scripts/TimeLLM_Spectral_T5_All.sh

# 将 BATCH_SIZE=8 改为 BATCH_SIZE=4 或 BATCH_SIZE=2
```

### **问题 3: 查看特定配置的训练状态**
```bash
# 查看 1:1 配置的训练日志
grep -A 20 "训练配置: 1_1" /home/user/WangS/Time-LLM-main/scripts/t5_training.log

# 查看所有配置的完成状态
grep "配置.*训练完成" /home/user/WangS/Time-LLM-main/scripts/t5_training.log
```

---

## ⚠️ 重要提示

1. **确保使用后台运行或 screen/tmux**，避免 SSH 断开导致训练中断
2. **定期检查日志**，确认训练正常进行
3. **监控 GPU 使用**，确保资源充分利用
4. **预留足够磁盘空间**，每个 checkpoint 约 1.5-2GB
5. **训练期间避免手动中断**，除非确实需要停止

---

## 📞 快速命令参考

```bash
# 启动训练
cd /home/user/WangS/Time-LLM-main/scripts && nohup bash TimeLLM_Spectral_T5_All.sh > t5_training.log 2>&1 &

# 查看日志
tail -f /home/user/WangS/Time-LLM-main/scripts/t5_training.log

# 查看 GPU
nvidia-smi

# 查看进程
ps aux | grep TimeLLM

# 查看 checkpoint
ls -lht /home/user/WangS/Time-LLM-main/checkpoints/ | grep T5

# 停止训练（谨慎使用）
pkill -f TimeLLM_Spectral_T5_All.sh
```

---

**现在可以登录服务器开始 T5 训练了！** 🎉
