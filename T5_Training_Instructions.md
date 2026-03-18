# T5 模型光谱预测训练指令

## 📋 概述

已成功添加 T5 模型支持并上传所有必要文件到服务器。T5 训练脚本将自动执行 4 个不同比例的训练：
- **1:1** (1050 → 1050)
- **2:1** (1400 → 700)
- **3:1** (1575 → 525)
- **4:1** (1680 → 420)

---

## ✅ 已完成的修改

### 1. **TimeLLM.py** - 添加 T5 模型支持
- ✅ 导入 `T5Config`, `T5EncoderModel`, `T5Tokenizer`
- ✅ 添加 T5 模型加载逻辑
- ✅ 模型路径: `/home/user/WangS/ChatTime-main/t5-base`
- ✅ 配置: dim=768, layers=12

### 2. **run_main.py** - 更新配置
- ✅ `llm_model` 参数添加 T5 选项
- ✅ `llm_dim` 注释添加 T5-base:768

### 3. **TimeLLM_Spectral_T5_All.sh** - 训练脚本
- ✅ 自动循环训练 4 个比例
- ✅ 使用 accelerate 进行混合精度训练
- ✅ 每个配置训练完成后自动休息 10 秒

---

## 🚀 执行指令

### 方式一：直接在服务器上执行（推荐）

```bash
# 1. SSH 登录服务器
ssh user@192.168.190.48

# 2. 进入项目目录
cd /home/user/WangS/Time-LLM-main

# 3. 激活 conda 环境
conda activate time-llm311

# 4. 进入 scripts 目录
cd scripts

# 5. 执行 T5 训练脚本
bash TimeLLM_Spectral_T5_All.sh
```

### 方式二：后台执行（推荐用于长时间训练）

```bash
# 1. SSH 登录服务器
ssh user@192.168.190.48

# 2. 进入项目目录
cd /home/user/WangS/Time-LLM-main

# 3. 激活 conda 环境
conda activate time-llm311

# 4. 进入 scripts 目录
cd scripts

# 5. 后台执行并保存日志
nohup bash TimeLLM_Spectral_T5_All.sh > t5_training.log 2>&1 &

# 6. 查看进程
ps aux | grep TimeLLM_Spectral_T5_All.sh

# 7. 实时查看日志
tail -f t5_training.log

# 8. 查看最近 100 行日志
tail -n 100 t5_training.log
```

### 方式三：使用 screen 或 tmux（推荐）

```bash
# 使用 screen
screen -S t5_training
cd /home/user/WangS/Time-LLM-main
conda activate time-llm311
cd scripts
bash TimeLLM_Spectral_T5_All.sh

# 分离 screen: Ctrl+A 然后按 D
# 重新连接: screen -r t5_training

# 或使用 tmux
tmux new -s t5_training
cd /home/user/WangS/Time-LLM-main
conda activate time-llm311
cd scripts
bash TimeLLM_Spectral_T5_All.sh

# 分离 tmux: Ctrl+B 然后按 D
# 重新连接: tmux attach -t t5_training
```

---

## 📊 训练配置详情

### T5 模型参数
- **模型**: T5-base
- **维度**: 768
- **层数**: 12
- **模型路径**: `/home/user/WangS/ChatTime-main/t5-base`

### 训练参数
- **批次大小**: 8
- **学习率**: 0.001
- **训练轮数**: 10
- **早停耐心**: 3
- **混合精度**: bf16
- **优化器**: Adam

### 数据参数
- **数据集**: spectral_data.csv
- **特征**: S (单变量)
- **目标**: OL
- **数据路径**: `/home/user/WangS/Time-LLM-main/dataset/spectral/`

### 4 个训练配置

| 配置 | 比例 | seq_len | pred_len | 模型ID |
|------|------|---------|----------|--------|
| 1 | 1:1 | 1050 | 1050 | Spectral_1_1_T5 |
| 2 | 2:1 | 1400 | 700 | Spectral_2_1_T5 |
| 3 | 3:1 | 1575 | 525 | Spectral_3_1_T5 |
| 4 | 4:1 | 1680 | 420 | Spectral_4_1_T5 |

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

---

## 🔍 监控训练进度

### 查看 GPU 使用情况
```bash
watch -n 1 nvidia-smi
```

### 查看训练日志
```bash
# 如果使用 nohup
tail -f /home/user/WangS/Time-LLM-main/scripts/t5_training.log

# 查看特定配置的训练进度
grep "训练配置" t5_training.log
grep "完成" t5_training.log
```

### 检查 checkpoint 文件
```bash
ls -lh /home/user/WangS/Time-LLM-main/checkpoints/ | grep T5
```

---

## ⏱️ 预计训练时间

根据之前的 BGE-M3 训练经验：
- **单个配置**: 约 1-3 小时
- **4 个配置总计**: 约 4-12 小时

实际时间取决于：
- GPU 性能
- 数据集大小
- 早停触发时间

---

## ⚠️ 注意事项

1. **确保 T5 模型已下载**
   ```bash
   ls -lh /home/user/WangS/ChatTime-main/t5-base/
   ```

2. **检查 conda 环境**
   ```bash
   conda activate time-llm311
   python -c "from transformers import T5EncoderModel; print('T5 import OK')"
   ```

3. **确保有足够的磁盘空间**
   ```bash
   df -h /home/user/WangS/Time-LLM-main/
   ```
   每个 checkpoint 约 1.5-2GB

4. **训练期间避免中断**
   - 使用 screen 或 tmux
   - 或使用 nohup 后台运行

---

## 🐛 故障排查

### 问题 1: T5 模型未找到
```bash
# 检查模型路径
ls -lh /home/user/WangS/ChatTime-main/t5-base/

# 如果不存在，需要下载
# 请联系管理员或使用 huggingface-cli 下载
```

### 问题 2: CUDA 内存不足
```bash
# 减小批次大小
# 编辑脚本，将 BATCH_SIZE=8 改为 BATCH_SIZE=4
nano /home/user/WangS/Time-LLM-main/scripts/TimeLLM_Spectral_T5_All.sh
```

### 问题 3: 训练中断
```bash
# 查看日志找到最后完成的配置
tail -n 50 t5_training.log

# 手动运行剩余配置
# 编辑脚本，注释掉已完成的配置
```

---

## 📞 联系信息

如有问题，请检查：
1. 日志文件: `t5_training.log`
2. Checkpoint 目录
3. GPU 状态: `nvidia-smi`

---

**祝训练顺利！** 🎉
