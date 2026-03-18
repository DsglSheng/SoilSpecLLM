# test_deepspeed_complete.py
import torch
import deepspeed
import os

print("=== DeepSpeed完整功能测试 ===")

# 设置环境变量
os.environ['CUDA_HOME'] = '/home/user/miniconda3/envs/time-llm'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

# 1. 基础信息
print(f"PyTorch版本: {torch.__version__}")
print(f"DeepSpeed版本: {deepspeed.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    计算能力: {props.major}.{props.minor}")
        print(f"    显存: {props.total_memory / 1024 ** 3:.1f} GB")

# 2. 测试基本DeepSpeed功能
print("\n测试基本DeepSpeed功能...")


# 创建简单模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1000, 500)
        self.linear2 = torch.nn.Linear(500, 100)

    def forward(self, x):
        return self.linear2(self.linear1(x))


model = SimpleModel()

# DeepSpeed配置
config = {
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    "fp16": {
        "enabled": False  # 先测试没有fp16
    },
    "zero_optimization": {
        "stage": 1
    },
    "steps_per_print": 10,
}

try:
    # 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=config
    )
    print("✅ DeepSpeed初始化成功")

    # 测试前向传播
    print("\n测试前向传播...")
    input_data = torch.randn(8, 1000)
    if torch.cuda.is_available():
        input_data = input_data.cuda()

    with torch.no_grad():
        output = model_engine(input_data)
        print(f"输入形状: {input_data.shape}")
        print(f"输出形状: {output.shape}")

    # 测试训练步骤（如果有GPU）
    if torch.cuda.is_available():
        print("\n测试GPU训练步骤...")
        target = torch.randn(8, 100).cuda()
        loss = torch.nn.functional.mse_loss(output, target)

        model_engine.backward(loss)
        model_engine.step()
        print("✅ GPU训练步骤成功")

    print("\n🎉 所有测试通过！")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback

    traceback.print_exc()