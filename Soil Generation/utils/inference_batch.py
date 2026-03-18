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

# def batch_generate_ECG(settings, unet, batch, dataloader, diffused_model, decoder, condition):
#     save_path = settings['save_path']
#     os.makedirs(save_path, exist_ok=True)
#
#     device = torch.device(settings['device'] if torch.cuda.is_available() else "cpu")
#     unet.to(device)
#     decoder.to(device)
#     condition_dict = None
#
#     all_rmse, all_r2, all_sam = [], [], []
#     features_file_content = {}
#
#     COLORS = {'true': '#1f77b4', 'pred': '#ff7f0e'}
#     LINESTYLES = {'true': '-', 'pred': '--'}
#     LINE_WIDTH = 1.5
#     FONT_CONFIG = {'family': 'DejaVu Sans', 'size': 12}
#     FIG_SIZE = (14, 6)
#     DPI = 300
#     TITLE_FONTSIZE = 14
#
#     for batch_idx, (data, label) in enumerate(dataloader):
#         text_embed = torch.tensor(label['text_embed']).float().to(device)
#         true_spectra = label['raw_spectra'].float()  # shape: [B, 2100]
#
#         with torch.no_grad():
#             latent = generation_from_net(
#                 diffused_model=diffused_model,
#                 net=unet,
#                 device=device,
#                 text_embed=text_embed,
#                 condition=condition_dict
#             )
#             latent = latent.detach()
#
#         with torch.no_grad():
#             gen_cure = decoder(latent, 2100, 4)
#             pred_spectra = gen_cure.detach().cpu()
#
#         if pred_spectra.ndim == 3:
#             pred_spectra = pred_spectra.squeeze(2)
#
#         batch_rmse_tensor = getRMSE(true_spectra, pred_spectra)
#         batch_r_tensor = getR(true_spectra, pred_spectra)
#         batch_r2_tensor = batch_r_tensor ** 2
#         batch_rmse = batch_rmse_tensor.tolist()
#         batch_r2 = batch_r2_tensor.tolist()
#
#         batch_sam = []
#         epsilon = 1e-8
#
#         for j in range(pred_spectra.shape[0]):
#             true_vec = true_spectra[j].numpy()
#             pred_vec = pred_spectra[j].numpy()
#             dot_product = np.dot(true_vec, pred_vec)
#             norm_true = np.linalg.norm(true_vec)
#             norm_pred = np.linalg.norm(pred_vec)
#             cos_theta = dot_product / (norm_true * norm_pred + epsilon)
#             sam = np.arccos(np.clip(cos_theta, -1.0, 1.0))
#             batch_sam.append(sam)
#
#             # plt.figure(figsize=FIG_SIZE)
#             # plt.plot(np.arange(400, 2500), 100 * true_vec, color=COLORS['true'], linestyle=LINESTYLES['true'],
#             #          linewidth=LINE_WIDTH, label='True Spectrum')
#             # plt.plot(np.arange(400, 2500), 100 * pred_vec, color=COLORS['pred'], linestyle=LINESTYLES['pred'],
#             #          linewidth=LINE_WIDTH, label='Predicted Spectrum')
#             # plt.title(f'Spectral Comparison (RMSE: {batch_rmse[j]:.4e}, R²: {batch_r2[j]:.4f}, SAM: {sam:.4f})',
#             #           fontsize=TITLE_FONTSIZE, pad=20)
#             # plt.xlabel('Wavelength (nm)', fontdict=FONT_CONFIG)
#             # plt.ylabel('Reflectance (%)', fontdict=FONT_CONFIG)
#             # plt.xlim(400, 2500)
#             # plt.ylim(0, 100)
#             # plt.legend(loc='upper right', frameon=True, fontsize=FONT_CONFIG['size'] - 2, borderaxespad=0.5)
#             # plt.grid(True, alpha=0.3, linestyle=':')
#             # plt.tight_layout()
#             # img_path = os.path.join(save_path, f'sample_{batch_idx}_{j}_spectra.png')
#             # plt.savefig(img_path, dpi=DPI, bbox_inches='tight')
#             # plt.close('all')
#
#             features_file_content[f'sample_{batch_idx}_{j}'] = {
#                 'metrics': {
#                     'rmse': float(f"{batch_rmse[j]:.5f}"),
#                     'r2': float(f"{batch_r2[j]:.5f}")
#                 }
#                 # 'visualization': img_path
#             }
#
#         all_rmse.extend(batch_rmse)
#         all_r2.extend(batch_r2)
#         all_sam.extend(batch_sam)
#
#         torch.cuda.empty_cache()
#         gc.collect()
#
#     final_rmse = np.mean(all_rmse)
#     final_r2 = np.mean(all_r2)
#     final_sam = np.degrees(np.mean(all_sam))
#
#     print(f'\nEvaluation Results:')
#     print(f'Mean RMSE: {final_rmse:.4f}')
#     print(f'Mean R²: {final_r2:.4f}')
#     print(f'Mean SAM: {final_sam:.4f}')
#
#     metrics_report = {
#         'global_metrics': {
#             'mean_rmse': float(f"{final_rmse:.4f}"),
#             'mean_r2': float(f"{final_r2:.4f}"),
#             'mean_sam_degree': float(f"{final_sam:.4f}"),
#             'samples_count': len(all_rmse)
#         },
#         'samples': features_file_content
#     }
#
#     with open(os.path.join(save_path, 'evaluation_report.json'), 'w') as f:
#         json.dump(metrics_report, f, indent=2)
#

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
        # text_embed = np.repeat(text_embed[:, np.newaxis, :], 1, axis=1)
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
            # for key in condition_dict:
            #     features_file_content.update({key: condition_dict[key]})
            for key in condition_dict:
                tensor_value = condition_dict[key]
                if isinstance(tensor_value, torch.Tensor):
                    # 提取 CPU 上的值，转换为 float 或 list
                    tensor_value = tensor_value.squeeze().detach().cpu().tolist()
                features_file_content[key] = tensor_value

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

        with torch.no_grad():
            # latent = generation_from_net(diffused_model, unet, batch_size=batch, device=device, text_embed=text_embed, condition=condition_dict)
            latent = generation_from_net(
                diffused_model=diffused_model,
                net=unet,
                device=device,
                text_embed=text_embed,
                condition=condition_dict
            )

            latent = latent.detach()

        with torch.no_grad():
            gen_cure = decoder(latent, 2100, 4)
            pred_spectra = gen_cure.detach().cpu()

        if pred_spectra.ndim == 3:
            pred_spectra = pred_spectra.squeeze(2)

        # ✅ 使用自定义 RMSE 和 R 计算函数
        true_spectra = torch.tensor(true_spectra)
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

        # ✅ 保存每个样本的指标
        for j in range(pred_np.shape[0]):
            features_file_content[f'sample_{batch_idx}_{j}'] = {
                'metrics': {
                    'rmse': float(f"{batch_rmse[j]:.5f}"),
                    'r2': float(f"{batch_r2[j]:.5f}")
                }
            }

        # ✅ 汇总指标
        all_rmse.extend(batch_rmse)
        all_r2.extend(batch_r2)
        all_sam.extend(batch_sam)

        torch.cuda.empty_cache()
        gc.collect()

        # batch_rmse, batch_r2, batch_sam = [], [], []
        # epsilon = 1e-8

        # for j in range(pred_spectra.shape[0]):
        #     true_vec = true_spectra[j]
        #     pred_vec = pred_spectra[j]
        #     mse = mean_squared_error(true_vec, pred_vec)
        #     rmse = np.sqrt(mse)
        #     r2 = r2_score(true_vec, pred_vec)
        #     dot_product = np.dot(true_vec, pred_vec)
        #     norm_true = np.linalg.norm(true_vec)
        #     norm_pred = np.linalg.norm(pred_vec)
        #     cos_theta = dot_product / (norm_true * norm_pred + epsilon)
        #     sam = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        #
        #     batch_rmse.append(rmse)
        #     batch_r2.append(r2)
        #     batch_sam.append(sam)

            # plt.figure(figsize=FIG_SIZE)
            # plt.plot(np.arange(400, 2500), 100 * true_vec, color=COLORS['true'], linestyle=LINESTYLES['true'],
            #          linewidth=LINE_WIDTH, label='True Spectrum')
            # plt.plot(np.arange(400, 2500), 100 * pred_vec, color=COLORS['pred'], linestyle=LINESTYLES['pred'],
            #          linewidth=LINE_WIDTH, label='Predicted Spectrum')
            # plt.title(f'Spectral Comparison (RMSE: {rmse:.4e}, R²: {r2:.4f}, SAM: {sam:.4f})',
            #           fontsize=TITLE_FONTSIZE, pad=20)
            # plt.xlabel('Wavelength (nm)', fontdict=FONT_CONFIG)
            # plt.ylabel('Reflectance (%)', fontdict=FONT_CONFIG)
            # plt.xlim(400, 2500)
            # plt.ylim(0, 100)
            # plt.legend(loc='upper right', frameon=True, fontsize=FONT_CONFIG['size'] - 2, borderaxespad=0.5)
            # plt.grid(True, alpha=0.3, linestyle=':')
            # plt.tight_layout()
            # img_path = os.path.join(save_path, f'sample_{batch_idx}_{j}_spectra.png')
            # plt.savefig(img_path, dpi=DPI, bbox_inches='tight')
            # plt.close('all')

        #     features_file_content[f'sample_{batch_idx}_{j}'] = {
        #         'metrics': {
        #             'rmse': float(f"{rmse:.5f}"),
        #             'r2': float(f"{r2:.5f}")
        #         }
        #         # 'visualization': img_path
        #     }
        #
        # all_rmse.extend(batch_rmse)
        # all_r2.extend(batch_r2)
        # all_sam.extend(batch_sam)
        #
        # torch.cuda.empty_cache()
        # gc.collect()

    final_rmse = np.mean(all_rmse)
    final_r2 = np.mean(all_r2)
    final_sam = np.degrees(np.mean(all_sam))

    print(f'\nEvaluation Results:')
    print(f'Mean RMSE: {final_rmse:.4f}')
    print(f'Mean R²: {final_r2:.4f}')
    print(f'Mean SAM: {final_sam:.4f}')

    metrics_report = {
        'global_metrics': {
            'mean_rmse': float(f"{final_rmse:.4f}"),
            'mean_r2': float(f"{final_r2:.4f}"),
            'mean_sam_degree': float(f"{final_sam:.4f}"),
            'samples_count': len(all_rmse)
        },
        'samples': features_file_content
    }

    with open(os.path.join(save_path, 'evaluation_report.json'), 'w') as f:
        json.dump(metrics_report, f, indent=2)

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