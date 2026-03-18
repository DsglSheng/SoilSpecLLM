import torch
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import mean_squared_error, r2_score
import gc


def generation_from_net(diffused_model, net, device, text_embed, condition, num_channels=4, dim=263):
    net.eval()
    # xi = torch.randn(batch_size, num_channels, dim).to(device)
    actual_batch_size = text_embed.shape[0]  # 动态确定 batch size
    xi = torch.randn(actual_batch_size, num_channels, dim).to(device)
    timesteps = tqdm(diffused_model.timesteps)
    for _, i in enumerate(timesteps):
        t = i * torch.ones(actual_batch_size, dtype=torch.long).to(device)
        with torch.no_grad():
            if condition:
                noise_predict = net(xi, t, text_embed, condition)
            else:
                noise_predict = net(xi, t, text_embed)

            xi = diffused_model.step(model_output=noise_predict,
                                     timestep=i,
                                     sample=xi)['prev_sample']
    return xi


def rmse(y_true, y_pred):
    mask = (y_true > 0) & (y_true < 1)
    squared_errors = (y_true[mask] - y_pred[mask]) ** 2
    return torch.sqrt(torch.mean(squared_errors))


def getRMSE(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same shape.")
    return torch.stack([rmse(tensor1[i], tensor2[i]) for i in range(tensor1.shape[0])])


def getR(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same shape.")
    corr = []
    for i in range(tensor1.shape[0]):
        mask = (tensor1[i] > 0) & (tensor1[i] < 1)
        r = np.corrcoef(tensor1[i][mask], tensor2[i][mask])[0, 1]
        corr.append(r)
    return torch.tensor(corr)


def batch_generate_ECG(settings, unet, batch, dataloader, diffused_model, decoder, condition):
    save_path = settings['save_path']
    os.makedirs(save_path, exist_ok=True)

    device = torch.device(settings['device'] if torch.cuda.is_available() else "cpu")
    unet.to(device)
    decoder.to(device)
    all_rmse = []
    all_r2 = []
    all_sam = []
    features_file_content = {}

    # 可视化参数配置
    COLORS = {'true': '#1f77b4', 'pred': '#ff7f0e'}
    LINESTYLES = {'true': '-', 'pred': '--'}
    LINE_WIDTH = 1.5
    FONT_CONFIG = {'family': 'DejaVu Sans', 'size': 12}
    FIG_SIZE = (14, 6)
    DPI = 300
    TITLE_FONTSIZE = 14

    for batch_idx, (data, label) in enumerate(dataloader):
        text_embed = label['text_embed']
        true_spectra = label['raw_spectra'].numpy()
        text_embed = np.array(text_embed).squeeze()

        # 处理text_embed的维度
        if text_embed.ndim == 1:
            # 若为 1D，则 reshape 为 [1, 1, D]
            text_embed = text_embed[np.newaxis, np.newaxis, :]
        elif text_embed.ndim == 2:
            # 若为 [B, D]，则加一个中间维度，变为 [B, 1, D]
            text_embed = text_embed[:, np.newaxis, :]
        # 如果已经是 [B, 1, D]，则无需处理
        text_embed = torch.Tensor(text_embed).to(device).detach()

        condition_dict = None

        batch_s = text_embed.shape[0]
        print(text_embed.shape[0])

        # 处理条件信息
        if condition:
            pH_CaCl2 = label['pH(CaCl2)']
            pH_H2O = label['pH(H2O)']
            OC = label['OC']
            CaCO3 = label['CaCO3']
            P = label['P']
            N = label['N']
            K = label['K']
            condition_dict = {
                'pH(CaCl2)': pH_CaCl2,
                'pH(H2O)': pH_H2O,
                'OC': OC,
                'CaCO3': CaCO3,
                'P': P,
                'N': N,
                'K': K
            }

            # 保存条件信息到文件内容中
            for key in condition_dict:
                tensor_value = condition_dict[key]
                if isinstance(tensor_value, torch.Tensor):
                    # 提取 CPU 上的值，转换为 float 或 list
                    tensor_value = tensor_value.squeeze().detach().cpu().tolist()
                features_file_content[key] = tensor_value

            # 处理条件字典中的数据类型和维度
            for key in condition_dict:
                val = condition_dict[key]  # 原始值，可能是 numpy、float、int 或 tensor

                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).float()
                elif not isinstance(val, torch.Tensor):
                    val = torch.tensor([val] * batch_s, dtype=torch.float32)
                elif isinstance(val, torch.Tensor) and val.ndim == 1 and val.shape[0] == batch_s:
                    # 如果已经是 batch 大小的 tensor，不用重复
                    val = val.float()
                else:
                    val = torch.tensor([val] * batch_s, dtype=torch.float32)

                val = val.view(batch_s, 1, 1).to(device)  # ✅ reshape 并转到 device
                condition_dict[key] = val

        # 生成潜在变量
        with torch.no_grad():
            latent = generation_from_net(
                diffused_model=diffused_model,
                net=unet,
                device=device,
                text_embed=text_embed,
                condition=condition_dict
            )
            latent = latent.detach()

        # 解码生成光谱
        with torch.no_grad():
            gen_cure = decoder(latent, 2100, 4)
            pred_spectra = gen_cure.detach().cpu()

        if pred_spectra.ndim == 3:
            pred_spectra = pred_spectra.squeeze(2)

        # ✅ 使用自定义 RMSE 和 R 计算函数
        true_spectra = torch.tensor(true_spectra)

        # 为了支持变长的光谱数据生成，我添加了以下代码-------------------------------
        # 获取true_spectra的长度
        true_length = true_spectra.shape[-1]  # 假设最后一个维度是光谱长度
        # 从pred_spectra的后面截取与true_spectra相同长度的部分（从2500nm倒序）
        if pred_spectra.shape[-1] > true_length:
            pred_spectra = pred_spectra[..., -true_length:]  # 取后true_length个波段
            print(f"📏 预测光谱已从后面截取到 {true_length} 个波段")
        # 确保维度完全匹配
        assert pred_spectra.shape == true_spectra.shape, f"维度不匹配: pred_spectra {pred_spectra.shape} vs true_spectra {true_spectra.shape}"
        # 计算对应的波长范围（从2500nm倒序的true_length长度）
        end_wavelength = 2500
        start_wavelength = end_wavelength - true_length + 1  # +1是因为包含终点
        wavelengths = np.arange(start_wavelength, end_wavelength + 1)
        print(f"📊 波长范围: {start_wavelength} - {end_wavelength} nm ({len(wavelengths)} 个点)")

        batch_rmse_tensor = getRMSE(true_spectra, pred_spectra)  # shape: [B]
        batch_r_tensor = getR(true_spectra, pred_spectra)  # shape: [B]

        batch_rmse = batch_rmse_tensor.tolist()
        batch_r2 = (batch_r_tensor ** 2).tolist()  # R² = R^2

        # ✅ 向量化计算 SAM（光谱角）
        true_np = true_spectra.numpy()
        pred_np = pred_spectra.numpy()
        epsilon = 1e-8

        # dot product: [B]
        dot_product = np.sum(true_np * pred_np, axis=1)
        norm_true = np.linalg.norm(true_np, axis=1)
        norm_pred = np.linalg.norm(pred_np, axis=1)
        cos_theta = dot_product / (norm_true * norm_pred + epsilon)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        batch_sam = np.arccos(cos_theta).tolist()

        # ✅ 可视化和保存每个样本的指标
        for j in range(pred_np.shape[0]):
            true_vec = true_np[j]
            pred_vec = pred_np[j]

            # 创建光谱对比图
            plt.figure(figsize=FIG_SIZE)
            plt.plot(wavelengths, 100 * true_vec, color=COLORS['true'],
                     linestyle=LINESTYLES['true'], linewidth=LINE_WIDTH, label='True Spectrum')
            plt.plot(wavelengths, 100 * pred_vec, color=COLORS['pred'],
                     linestyle=LINESTYLES['pred'], linewidth=LINE_WIDTH, label='Predicted Spectrum')

            # plt.plot(np.arange(400, 2500), 100 * true_vec, color=COLORS['true'],
            #          linestyle=LINESTYLES['true'], linewidth=LINE_WIDTH, label='True Spectrum')
            # plt.plot(np.arange(400, 2500), 100 * pred_vec, color=COLORS['pred'],
            #          linestyle=LINESTYLES['pred'], linewidth=LINE_WIDTH, label='Predicted Spectrum')
            plt.title(
                f'Spectral Comparison (RMSE: {batch_rmse[j]:.4e}, R²: {batch_r2[j]:.4f}, SAM: {batch_sam[j]:.4f})',
                fontsize=TITLE_FONTSIZE, pad=20)
            plt.xlabel('Wavelength (nm)', fontdict=FONT_CONFIG)
            plt.ylabel('Reflectance (%)', fontdict=FONT_CONFIG)

            plt.xlim(start_wavelength, end_wavelength)

            # plt.xlim(400, 2500)
            plt.ylim(0, 100)
            plt.legend(loc='upper right', frameon=True, fontsize=FONT_CONFIG['size'] - 2, borderaxespad=0.5)
            plt.grid(True, alpha=0.3, linestyle=':')
            plt.tight_layout()

            # 保存图像
            img_path = os.path.join(save_path, f'sample_{batch_idx}_{j}_spectra.png')
            plt.savefig(img_path, dpi=DPI, bbox_inches='tight')
            plt.close('all')

            # 保存样本信息
            features_file_content[f'sample_{batch_idx}_{j}'] = {
                'metrics': {
                    'rmse': float(f"{batch_rmse[j]:.5f}"),
                    'r2': float(f"{batch_r2[j]:.5f}"),
                    'sam': float(f"{batch_sam[j]:.5f}")
                },
                'visualization': img_path
            }

        # ✅ 汇总指标
        all_rmse.extend(batch_rmse)
        all_r2.extend(batch_r2)
        all_sam.extend(batch_sam)

        # 内存清理
        torch.cuda.empty_cache()
        gc.collect()

    # 计算最终指标
    final_rmse = np.mean(all_rmse)
    final_r2 = np.mean(all_r2)
    final_sam = np.degrees(np.mean(all_sam))

    print(f'\nEvaluation Results:')
    print(f'Mean RMSE: {final_rmse:.4f}')
    print(f'Mean R²: {final_r2:.4f}')
    print(f'Mean SAM: {final_sam:.4f}')
    print(f'Total samples processed: {len(all_rmse)}')
    print(f'Visualizations saved to: {save_path}')

    # 保存评估报告
    metrics_report = {
        'global_metrics': {
            'mean_rmse': float(f"{final_rmse:.4f}"),
            'mean_r2': float(f"{final_r2:.4f}"),
            'mean_sam_degree': float(f"{final_sam:.4f}"),
            'samples_count': len(all_rmse)
        },
        'samples': features_file_content
    }

    # 保存JSON报告
    with open(os.path.join(save_path, 'evaluation_report.json'), 'w') as f:
        json.dump(metrics_report, f, indent=2)

    # 创建指标分布统计图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # RMSE分布
    axes[0].hist(all_rmse, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(final_rmse, color='red', linestyle='--', linewidth=2, label=f'Mean: {final_rmse:.4f}')
    axes[0].set_xlabel('RMSE')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('RMSE Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # R²分布
    axes[1].hist(all_r2, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].axvline(final_r2, color='red', linestyle='--', linewidth=2, label=f'Mean: {final_r2:.4f}')
    axes[1].set_xlabel('R²')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('R² Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # SAM分布
    sam_degrees = np.degrees(all_sam)
    axes[2].hist(sam_degrees, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[2].axvline(final_sam, color='red', linestyle='--', linewidth=2, label=f'Mean: {final_sam:.4f}°')
    axes[2].set_xlabel('SAM (degrees)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('SAM Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Metrics distribution plot saved to: {os.path.join(save_path, "metrics_distribution.png")}')

# 注释：历史结果记录
# Evaluation Results:
# Mean RMSE: 0.1990
# Mean R²: 0.3001
# Mean SAM: 12.9102

# 采用LUCAS0915——OSSL——BSSL的bge编码的结果如下：用的1000step,74_bge
# Evaluation Results:
# Mean RMSE: 0.2005
# Mean R²: 0.3068
# Mean SAM: 12.8311

# 采用LUCAS0915——OSSL——BSSL的Clip编码的结果如下：用的300step
# Evaluation Results:
# Mean RMSE: 0.2110
# Mean R²: 0.2503
# Mean SAM: 13.3406
# Mean RMSE: 0.1983
# Mean R²: 0.3557
# Mean SAM: 12.7189
# 采用LUCAS0915的clip编码的结果如下：用的300step,75_clip
# Evaluation Results:
# Mean RMSE: 0.1090
# Mean R²: 0.4961
# Mean SAM: 10.7152
# 采用LUCAS0915的bge编码的结果如下：1000轮all_76
# Mean RMSE: 0.1185
# Mean R²: 0.4247
# Mean SAM: 11.5367
# 采用LUCAS0915的bge编码的结果如下：300轮all_77
# Evaluation Results:
# Mean RMSE: 0.1149
# Mean R²: 0.6234
# Mean SAM: 8.9428
# 采用contion的K_no编码结果：300轮all_80_0915bge_c
# Evaluation Results:
# Mean RMSE: 0.0756
# Mean R²: 0.8671
# Mean SAM: 5.1809
# 用clip进行条件编码k_no的结果，300轮all_82:
# Mean RMSE: 0.1227
# Mean R²: 0.6399
# Mean SAM: 11.6363

# 用bge编码后的0915_OSSl数据结果:
# Mean RMSE: 0.0804
# Mean R²: 0.9341
# Mean SAM: 4.3002
# 用clip编码后的0915_OSSl数据结果：
# Evaluation Results:
# Mean RMSE: 0.0814
# Mean R²: 0.8989
# Mean SAM: 4.8467

# 新结果
# Evaluation Results:
# Mean RMSE: 0.0810
# Mean R²: 0.9349
# Mean SAM: 4.3350
# Total samples processed: 1064
# Visualizations saved to: ./test_sample_all_86_bge_png_unce

#在forest数据上的BGE的结果:
# Mean RMSE: 0.1058
# Mean R²: 0.9086
# Mean SAM: 4.1356
# Total samples processed: 46
#或者是
# Mean RMSE: 0.0855
# Mean R²: 0.8909
# Mean SAM: 4.2191
# Total samples processed: 46
# Visualizations saved to: ./test_sample_all_86_forsoil_bge_png_unce
# 在forest数据上，用CLIP的结果
# Mean RMSE: 0.1608
# Mean R²: 0.7334
# Mean SAM: 6.9171
# Total samples processed: 46
# Visualizations saved to: ./test_sample_all_88_forsoil_clip_png_unce

#在bablet上的CLIP的结果
# Evaluation Results:
# Mean RMSE: 0.1577
# Mean R²: 0.4535
# Mean SAM: 16.4430
# Total samples processed: 72
# Visualizations saved to: ./test_sample_all_88_bablet_clip_png_unce

#在bablet上的BGE的结果

# Barthes的BGE结果
# Evaluation Results:
# Mean RMSE: 0.0925
# Mean R²: 0.3522
# Mean SAM: 6.1349
# Total samples processed: 406
# Visualizations saved to: ./test_sample_all_86_Barthes_bge_png_unce
# Metrics distribution plot saved to: ./test_sample_all_86_Barthes_bge_png_unce\metrics_distribution.png

# 'total phosphorus : 55.2 g/kg', 'total nitrogen : 1.4 g/kg', 'extractable potassium : 343.4 g/kg']]
# 原始的训练、测试数据出错了，1和3的单位是mg

# 8月14日：补充对比实验：对照
# 不同的属性对他的影响如何？去掉一些属性，然后跑实验，判断不同属性对他的影响；或者是预测的时候加一些新的属性，判断该属性对他的影响；
# 相同属性，值不一样，对光谱曲线的影响如何：主要的属性有
# OSSLJ数据：
