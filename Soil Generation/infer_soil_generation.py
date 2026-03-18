import torch
import argparse
import json
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from dataset.soil_gen_dataset import DictDataset


def parse_arg():
    parser = argparse.ArgumentParser(description='SoilSpecLLM Spectral Generation Inference')
    parser.add_argument('--config', default='config/soil_generation.json', help='Root of training configuration')
    parser.add_argument('--uncertainty', action='store_true', default=True, help='Enable uncertainty analysis for samples 16 and 17')
    parser.add_argument('--uncertainty_runs', type=int, default=10,
                        help='Number of generations for uncertainty analysis')

    args = parser.parse_args()
    return args


def generation_from_net_uncertainty(diffused_model, net, device, text_embed, condition, num_channels=4, dim=263):
    """涓嶇‘瀹氭€у垎鏋愪笓鐢ㄧ殑鐢熸垚鍑芥暟"""
    net.eval()
    actual_batch_size = text_embed.shape[0]
    xi = torch.randn(actual_batch_size, num_channels, dim).to(device)
    timesteps = tqdm(diffused_model.timesteps, desc="Generating", leave=False)
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


def run_uncertainty_analysis(settings, unet, dataloader, diffused_model, decoder, condition,
                             target_indices=[0, 2, 16, 17], num_generations=10):
    """
    杩愯涓嶇‘瀹氭€у垎鏋?

    Args:
        settings: 璁剧疆瀛楀吀
        unet: U-Net妯″瀷
        dataloader: 鏁版嵁鍔犺浇鍣?
        diffused_model: 鎵╂暎妯″瀷
        decoder: 瑙ｇ爜鍣?
        condition: 鏄惁浣跨敤鏉′欢
        target_indices: 鐩爣鏍锋湰绱㈠紩鍒楄〃 [16, 17]
        num_generations: 鐢熸垚娆℃暟锛岄粯璁?0娆?
    """
    save_path = os.path.join(settings['save_path'], 'uncertainty_analysis')
    os.makedirs(save_path, exist_ok=True)

    device = torch.device(settings['device'] if torch.cuda.is_available() else "cpu")
    unet.to(device)
    decoder.to(device)

    print(f"\n{'=' * 60}")
    print(f"Starting Uncertainty Analysis")
    print(f"Target samples: {target_indices}")
    print(f"Number of generations per sample: {num_generations}")
    print(f"Save path: {save_path}")
    print(f"{'=' * 60}")

    # 鎵惧埌鐩爣鏍锋湰鏁版嵁
    target_samples = {}
    current_idx = 0

    for batch_idx, (data, label) in enumerate(dataloader):
        batch_size = len(label['text_embed'])

        for i in range(batch_size):
            if current_idx in target_indices:
                # 淇濆瓨鐩爣鏍锋湰鐨勬暟鎹?
                sample_data = {
                    'text_embed': np.array(label['text_embed'][i]).squeeze(),
                    'true_spectra': label['raw_spectra'][i].numpy() if torch.is_tensor(label['raw_spectra'][i]) else
                    label['raw_spectra'][i]
                }

                # 濡傛灉鏈夋潯浠朵俊鎭紝涔熶繚瀛?
                if condition:
                    sample_data['conditions'] = {
                        'pH(CaCl2)': label['pH(CaCl2)'][i] if hasattr(label['pH(CaCl2)'], '__getitem__') else label[
                            'pH(CaCl2)'],
                        'pH(H2O)': label['pH(H2O)'][i] if hasattr(label['pH(H2O)'], '__getitem__') else label[
                            'pH(H2O)'],
                        'OC': label['OC'][i] if hasattr(label['OC'], '__getitem__') else label['OC'],
                        'CaCO3': label['CaCO3'][i] if hasattr(label['CaCO3'], '__getitem__') else label['CaCO3'],
                        'P': label['P'][i] if hasattr(label['P'], '__getitem__') else label['P'],
                        'N': label['N'][i] if hasattr(label['N'], '__getitem__') else label['N'],
                        'K': label['K'][i] if hasattr(label['K'], '__getitem__') else label['K']
                    }

                target_samples[current_idx] = sample_data
                print(f"Found target sample {current_idx}")

            current_idx += 1

        # 濡傛灉宸茬粡鎵惧埌鎵€鏈夌洰鏍囨牱鏈紝鍙互鎻愬墠閫€鍑?
        if len(target_samples) == len(target_indices):
            break

    print(f"Successfully located {len(target_samples)} target samples: {list(target_samples.keys())}")

    # 瀛樺偍鎵€鏈夋牱鏈殑缁撴灉鐢ㄤ簬鍚堝苟鍥?
    all_results = {}

    # 瀵规瘡涓洰鏍囨牱鏈繘琛屼笉纭畾鎬у垎鏋?
    for sample_idx, sample_data in target_samples.items():
        print(f"\nProcessing sample {sample_idx} for uncertainty analysis...")

        # 鍑嗗杈撳叆鏁版嵁
        text_embed = sample_data['text_embed']
        true_spectrum = sample_data['true_spectra']

        # 澶勭悊text_embed缁村害
        if text_embed.ndim == 1:
            text_embed = text_embed[np.newaxis, np.newaxis, :]
        elif text_embed.ndim == 2:
            text_embed = text_embed[:, np.newaxis, :]
        text_embed = torch.Tensor(text_embed).to(device)

        # 鍑嗗鏉′欢鏁版嵁
        condition_dict = None
        if condition and 'conditions' in sample_data:
            condition_dict = {}
            for key, val in sample_data['conditions'].items():
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).float()
                elif not isinstance(val, torch.Tensor):
                    val = torch.tensor([val], dtype=torch.float32)
                else:
                    val = val.float()

                val = val.view(1, 1, 1).to(device)
                condition_dict[key] = val

        # 澶氭鐢熸垚
        generated_spectra = []

        for gen_i in range(num_generations):
            print(f"  Generation {gen_i + 1}/{num_generations} for sample {sample_idx}")

            with torch.no_grad():
                # 鐢熸垚娼滃湪鍙橀噺
                latent = generation_from_net_uncertainty(
                    diffused_model=diffused_model,
                    net=unet,
                    device=device,
                    text_embed=text_embed,
                    condition=condition_dict
                )

                # 瑙ｇ爜鐢熸垚鍏夎氨
                gen_spectrum = decoder(latent, 2100, 4)
                gen_spectrum = gen_spectrum.detach().cpu().numpy()

                if gen_spectrum.ndim == 3:
                    gen_spectrum = gen_spectrum.squeeze()
                if gen_spectrum.ndim == 2:
                    gen_spectrum = gen_spectrum[0]

                generated_spectra.append(gen_spectrum)

            # 娓呯悊GPU鍐呭瓨
            torch.cuda.empty_cache()
            gc.collect()

        # 杞崲涓簄umpy鏁扮粍杩涜缁熻鍒嗘瀽
        generated_spectra = np.array(generated_spectra)  # shape: [num_generations, 2100]

        # 璁＄畻缁熻閲?
        mean_spectrum = np.mean(generated_spectra, axis=0)
        std_spectrum = np.std(generated_spectra, axis=0)

        # 鍒涘缓娉㈤暱鏁扮粍 (400-2500 nm, 2100涓偣)
        wavelengths = np.linspace(400, 2500, 2100)

        # 璁＄畻涓嶇‘瀹氭€ф寚鏍?
        relative_uncertainty = np.mean(std_spectrum / (mean_spectrum + 1e-8)) * 100
        print(f"  Sample {sample_idx} - Mean relative uncertainty: {relative_uncertainty:.2f}%")

        # 淇濆瓨缁撴灉鐢ㄤ簬鍚庣画鍚堝苟鍥?
        all_results[sample_idx] = {
            'wavelengths': wavelengths,
            'true_spectrum': true_spectrum,
            'mean_spectrum': mean_spectrum,
            'std_spectrum': std_spectrum,
            'relative_uncertainty': relative_uncertainty,
            'generated_spectra': generated_spectra
        }

        # 淇濆瓨鍗曠嫭鐨勫垎鏋愬浘
        single_fig, single_ax = plt.subplots(1, 1, figsize=(12, 7))
        single_ax.plot(wavelengths, 100 * mean_spectrum, color='#5882b8', linewidth=2.5,
                       label='Generation mean')
        single_ax.fill_between(wavelengths,
                               100 * (mean_spectrum - std_spectrum),
                               100 * (mean_spectrum + std_spectrum),
                               color='#5882b8', alpha=0.3,
                               label='Generation standard deviation')
        single_ax.plot(wavelengths, 100 * true_spectrum, color='red', linewidth=2.5,
                       label='Real spectrum')
        single_ax.set_xlabel('Wavelength (nm)', fontsize=14)
        single_ax.set_ylabel('Reflectance (%)', fontsize=14)
        single_ax.set_title(f'Uncertainty Analysis - Sample {sample_idx}\n'
                            f'Mean Relative Uncertainty: {relative_uncertainty:.2f}%',
                            fontsize=16, pad=20)
        single_ax.legend(fontsize=12, loc='upper right')
        single_ax.grid(True, alpha=0.3, linestyle=':')
        single_ax.set_xlim(400, 2500)
        y_max = max(100 * np.max(true_spectrum), 100 * np.max(mean_spectrum + std_spectrum)) + 5
        single_ax.set_ylim(0, y_max)

        # 淇濆瓨鍗曠嫭鍥捐〃
        single_fig.tight_layout()
        single_fig.savefig(os.path.join(save_path, f'uncertainty_analysis_sample_{sample_idx}.png'),
                           dpi=300, bbox_inches='tight')
        plt.close(single_fig)

        # 淇濆瓨涓嶇‘瀹氭€ф暟鎹?
        uncertainty_data = {
            'sample_index': int(sample_idx),
            'num_generations': num_generations,
            'wavelengths': wavelengths.tolist(),
            'true_spectrum': true_spectrum.tolist(),
            'generated_mean': mean_spectrum.tolist(),
            'generated_std': std_spectrum.tolist(),
            'relative_uncertainty_percent': float(relative_uncertainty),
            'all_generations': generated_spectra.tolist()
        }

        with open(os.path.join(save_path, f'uncertainty_data_sample_{sample_idx}.json'), 'w') as f:
            json.dump(uncertainty_data, f, indent=2)

        print(f"  Completed analysis for sample {sample_idx}")

    # 鍒涘缓鍚堝苟鐨勫浘琛紙濡傛灉鏈変袱涓牱鏈級
    if len(all_results) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        sample_indices = sorted(all_results.keys())
        axes = [ax1, ax2]
        titles = ['(a)', '(b)']

        for i, sample_idx in enumerate(sample_indices):
            result = all_results[sample_idx]
            ax = axes[i]

            # 缁樺埗鐢熸垚鍏夎氨鐨勫潎鍊煎拰鏍囧噯宸?
            ax.plot(result['wavelengths'], 100 * result['mean_spectrum'],
                    color='#5882b8', linewidth=2.5, label='Generation mean')
            ax.fill_between(result['wavelengths'],
                            100 * (result['mean_spectrum'] - result['std_spectrum']),
                            100 * (result['mean_spectrum'] + result['std_spectrum']),
                            color='#5882b8', alpha=0.3,
                            label='Generation standard deviation')

            # 缁樺埗鐪熷疄鍏夎氨
            ax.plot(result['wavelengths'], 100 * result['true_spectrum'],
                    color='red', linewidth=2.5, label='Real spectrum')

            # 璁剧疆鍥捐〃灞炴€?
            ax.set_xlabel('Wavelength (nm)', fontsize=14)
            ax.set_ylabel('Reflectance (%)', fontsize=14)
            ax.set_title(f'{titles[i]}', fontsize=16, pad=20)
            ax.legend(fontsize=12, loc='upper right')
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_xlim(400, 2500)
            y_max = max(100 * np.max(result['true_spectrum']),
                        100 * np.max(result['mean_spectrum'] + result['std_spectrum'])) + 5
            ax.set_ylim(0, y_max)

        fig.suptitle('Uncertainty Analysis of Spectral Generation', fontsize=18, y=0.98)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'uncertainty_analysis_combined.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"\n{'=' * 60}")
    print(f"Uncertainty Analysis Completed!")
    print(f"Results saved to: {save_path}")
    print(f"Files generated:")
    if len(all_results) == 2:
        print(f"  - uncertainty_analysis_combined.png")
    for idx in all_results.keys():
        print(f"  - uncertainty_analysis_sample_{idx}.png")
        print(f"  - uncertainty_data_sample_{idx}.json")
    print(f"{'=' * 60}")


def main():
    args = parse_arg()

    with open(args.config, 'r') as f:
        config = json.load(f)

    settings = config['inference_setting']
    h_ = config['hyper_para']

    condition = config['meta']['condition']
    use_vae_latent = config['meta']['vae_latent']
    settings['device'] = config['meta']['device']
    batch = settings['gen_batch']
    test_dataset = DictDataset(settings['dataset_path'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch)

    n_channels = 4 if use_vae_latent else 12
    if condition:
        from unet.unet_conditional import ECGconditional
        unet = ECGconditional(h_['num_train_steps'], kernel_size=h_['unet_kernel_size'],
                              num_levels=h_['unet_num_level'], n_channels=n_channels)
    else:
        from unet.unet_nocondition import ECGnocondition
        unet = ECGnocondition(h_['num_train_steps'], kernel_size=h_['unet_kernel_size'],
                              num_levels=h_['unet_num_level'], n_channels=n_channels)

    unet_path = settings['unet_path']
    unet.load_state_dict(torch.load(unet_path, map_location='cpu'))

    diffused_model = DDPMScheduler(num_train_timesteps=h_['num_train_steps'],
                                   beta_start=h_['beta_start'], beta_end=h_['beta_end'])
    diffused_model.set_timesteps(settings['inference_timestep'])

    if use_vae_latent:
        from utils.inference_batch_png import batch_generate_ECG
        from vae.vae import VAE_Decoder

        decoder = VAE_Decoder()
        vae_path = config['dependencies']['vae_path']
        checkpoint = torch.load(vae_path, map_location='cpu')
        decoder.load_state_dict(checkpoint['decoder'])

        # 鎵ц姝ｅ父鐨勬壒閲忕敓鎴?
        print("Starting batch generation and visualization...")
        # batch_generate_ECG(settings=settings,
        #                    unet=unet,
        #                    batch=batch,
        #                    dataloader=test_dataloader,
        #                    diffused_model=diffused_model,
        #                    decoder=decoder,
        #                    condition=condition)

        # 濡傛灉鍚敤浜嗕笉纭畾鎬у垎鏋?
        if args.uncertainty:
            print("\nStarting uncertainty analysis...")
            run_uncertainty_analysis(
                settings=settings,
                unet=unet,
                dataloader=test_dataloader,
                diffused_model=diffused_model,
                decoder=decoder,
                condition=condition,
                target_indices=[0, 1, 2, 3,4,5,6,7,8,9,10,12,16,17],
                num_generations=args.uncertainty_runs
            )
        else:
            print("\nSkipping uncertainty analysis. Use --uncertainty flag to enable.")

    else:
        from utils.inference_novae import batch_generate_ECG_novae

        batch_generate_ECG_novae(settings=settings,
                                 unet=unet,
                                 diffused_model=diffused_model,
                                 condition=condition)

        if args.uncertainty:
            print("Warning: Uncertainty analysis is currently only supported with VAE latent mode.")


if __name__ == "__main__":
    main()

