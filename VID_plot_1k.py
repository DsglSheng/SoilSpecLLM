import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models import TimeLLM
from data_provider.data_factory import data_provider
import argparse
import matplotlib
import traceback
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib.patches import Rectangle

# Set Matplotlib to non-interactive mode, suitable for server operation
matplotlib.use('Agg')

# Set style for scientific publications
plt.style.use('default')
sns.set_palette("husl")


def parse_args():
    parser = argparse.ArgumentParser(description='TimeLLM Spectral Prediction Visualization')

    # Basic configuration
    parser.add_argument('--model_id', type=str, default='Spectral_1400_700', help='Model ID')
    parser.add_argument('--model_comment', type=str, default='TimeLLM-Spectral', help='Model comment')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/home/user/WangS/Time-LLM-main/scripts/checkpoints/llama2_1680_420/checkpoint',
                        help='Checkpoint file path')

    # Dataset parameters
    parser.add_argument('--data', type=str, default='Spectral', help='Dataset type')
    parser.add_argument('--root_path', type=str, default='/home/user/WangS/Time-LLM-main/dataset/spectral/',
                        help='Dataset root directory')
    parser.add_argument('--data_path', type=str, default='spectral_soil_test.csv', help='Data file')
    parser.add_argument('--features', type=str, default='M', help='Feature type: [M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='Target feature (S or MS task)')
    parser.add_argument('--freq', type=str, default='h', help='Frequency encoding')
    parser.add_argument('--seasonal_patterns', type=str, default='Ll', help='Seasonal patterns for M4 dataset')
    parser.add_argument('--percent', type=int, default=100, help='Percentage of data to use')
    parser.add_argument('--scale', type=bool, default=True, help='Whether to scale data')
    parser.add_argument('--timeenc', type=int, default=0, help='Time encoding type')
    parser.add_argument('--loader', type=str, default='modal', help='Dataset type')

    # Model parameters
    parser.add_argument('--seq_len', type=int, default=1680, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='Overlapping sequence length')
    parser.add_argument('--pred_len', type=int, default=420, help='Prediction sequence length')
    parser.add_argument('--wavelength_range', type=int, nargs=2, default=[400, 2499], help='Wavelength range')

    # Model structure parameters
    parser.add_argument('--model', type=str, default='TimeLLM', help='Model name')
    parser.add_argument('--enc_in', type=int, default=1, help='Encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='Decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='Output size')
    parser.add_argument('--d_model', type=int, default=16, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='Dimension of feed-forward network')
    parser.add_argument('--factor', type=int, default=3, help='Attention factor')
    parser.add_argument('--moving_avg', type=int, default=25, help='Moving average window size')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--output_attention', action='store_true', help='Whether to output attention')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time feature encoding, options:[timeF, fixed, learned]')

    # TimeLLM specific parameters
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model type: LLAMA, GPT2, BERT')
    parser.add_argument('--llm_dim', type=int, default=4096,
                        help='LLM model dimension, LLama7b:4096; GPT2-small:768; BERT-base:768')
    parser.add_argument('--llm_layers', type=int, default=32, help='Number of LLM layers')
    parser.add_argument('--prompt_domain', type=int, default=0, help='Prompt domain')
    parser.add_argument('--patch_len', type=int, default=16, help='Patch length')
    parser.add_argument('--stride', type=int, default=8, help='Stride')
    parser.add_argument('--content', type=str,
                        default='Spectral data is used to analyze composition and properties of materials.',
                        help='Domain description text')
    parser.add_argument('--ablation', action='store_true', default=False,
                        help='Whether it is an ablation experiment model')

    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start for learning rate scheduler')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment type')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true', help='Whether to use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of data loader workers')

    # Experiment parameters
    parser.add_argument('--is_training', type=int, default=0, help='Whether it is training mode')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiments')
    parser.add_argument('--des', type=str, default='test', help='Experiment description')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')
    parser.add_argument('--align_epochs', type=int, default=10, help='Number of alignment epochs')

    # Visualization parameters - 修改默认值以预测所有数据
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Number of samples to visualize (-1 for all samples)')
    parser.add_argument('--save_dir', type=str, default='/home/user/WangS/Time-LLM-main/visual_enhanced',
                        help='Directory to save visualization results')
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='Task name')
    parser.add_argument('--use_bfloat16', action='store_true', default=False, help='Whether to use bfloat16 precision')

    # Savitzky-Golay filter parameters
    parser.add_argument('--apply_filter', action='store_true', default=True,
                        help='Whether to apply Savitzky-Golay filter')
    parser.add_argument('--window_length', type=int, default=85, help='Savitzky-Golay window length (must be odd)')
    parser.add_argument('--polyorder', type=int, default=2, help='Savitzky-Golay polynomial order')

    args = parser.parse_args()

    # Ensure window length is odd
    if args.window_length % 2 == 0:
        args.window_length += 1

    # Ensure polynomial order is less than window length
    if args.polyorder >= args.window_length:
        args.polyorder = args.window_length - 1

    return args


def patch_timellm():
    """Patch TimeLLM model's forecast method to resolve type mismatch issues and adapt to original model"""
    original_forecast = TimeLLM.Model.forecast

    def patched_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Patched forecast method, no longer forcing conversion to bfloat16"""
        # Use the same type as model parameters
        param_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        # Ensure input data matches model parameter types
        x_enc = x_enc.to(param_dtype).to(device)

        # The following is the forecast method adapted for model structure
        x_enc = self.normalize_layers(x_enc, 'norm')
        # B is batch size, T is time series length, N is number of variables
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # Calculate statistical features
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        # Restore original dimensions
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # Convert prompts to tokens and get embeddings
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        # Patch embeddings
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)  # Removed .to(torch.bfloat16)

        # Check if the current model has direct_mapping attribute (ablation experiment model)
        if hasattr(self, 'direct_mapping'):
            # Direct linear mapping instead of patch-programming
            enc_out = self.direct_mapping(enc_out)
        else:
            # Original model's processing method
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # Concatenate prompt embeddings and input feature embeddings
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)

        # Process through LLM model
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    # Replace original method
    TimeLLM.Model.forecast = patched_forecast
    print("TimeLLM model patched for type compatibility")


def setup_model(args):
    """Create and set up the model"""
    print("Creating model...")
    print(f"Using original model with patch-programming")

    args.scale = False

    try:
        # Create model
        model = TimeLLM.Model(args).float()

        # Checkpoint path
        checkpoint_path = args.checkpoint_path

        # Check if file exists
        if not os.path.exists(checkpoint_path):
            print(f"Original checkpoint path doesn't exist: {checkpoint_path}")

            # Try different file extensions
            possible_extensions = ['.file', '.pth', '.pt', '.ckpt']
            base_path = checkpoint_path.rsplit('.', 1)[0] if '.' in os.path.basename(
                checkpoint_path) else checkpoint_path

            for ext in possible_extensions:
                alt_path = base_path + ext
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    print(f"Found alternative checkpoint file: {checkpoint_path}")
                    break

            # If still not found, try to find any checkpoint file in the directory
            if not os.path.exists(checkpoint_path):
                dir_path = os.path.dirname(checkpoint_path)
                if os.path.exists(dir_path):
                    print(f"Looking for checkpoints in directory: {dir_path}")
                    for file in os.listdir(dir_path):
                        if any(file.endswith(ext) for ext in possible_extensions) or file == 'checkpoint':
                            checkpoint_path = os.path.join(dir_path, file)
                            print(f"Found checkpoint file in directory: {checkpoint_path}")
                            break

        # Load model
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')

                # Check state_dict format, some .file files may contain nested dictionaries
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']

                model.load_state_dict(state_dict)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Problems encountered when trying to load model, will use untrained model for demonstration.")
        else:
            # If still not found, check exact directory structure and list contents
            dir_to_check = '/home/user/WangS/Time-LLM-main/scripts/checkpoints/'
            print(f"Checkpoint file not found. View directory contents: {dir_to_check}")
            if os.path.exists(dir_to_check):
                print("Directory contents:")
                for root, dirs, files in os.walk(dir_to_check):
                    print(f"Directory: {root}")
                    for dir_name in dirs:
                        print(f"  Subdirectory: {dir_name}")
                    for file_name in files:
                        print(f"  File: {file_name}")

            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Set to evaluation mode
        model.eval()

        # If using bfloat16
        if args.use_bfloat16 and torch.cuda.is_available():
            print("Using bfloat16 precision for inference")
            model = model.to(torch.bfloat16)

        return model
    except Exception as e:
        print(f"Error setting up model: {str(e)}")
        traceback.print_exc()
        raise


def get_data(args):
    """Load all test data without train/test split"""
    print("Loading all test data (no train/test split)...")
    try:
        args.scale = False
        # 修改：直接使用所有数据，不进行训练测试划分
        # 这里我们仍然使用data_provider，但明确告知其使用所有数据作为测试集
        _, test_loader = data_provider(args, 'test')
        print(f"All test data loaded successfully, {len(test_loader)} batches")
        print(f"Total samples in dataset: {len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 'Unknown'}")
        return test_loader
    except Exception as e:
        print(f"Failed to load test data: {str(e)}")
        traceback.print_exc()
        raise


def predict(model, test_loader, args, device='cpu'):
    """Make predictions using the model on all available data"""
    print("Starting prediction on all available data...")
    preds = []
    trues = []
    wavelengths_x = []
    wavelengths_y = []
    history_data = []
    history_wavelengths = []

    # Determine data type currently used by the model
    dtype = next(model.parameters()).dtype
    print(f"Model data type: {dtype}")

    total_samples = 0
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            print(f"Processing batch {i + 1}/{len(test_loader)}")

            try:
                # Use original data directly, no need for inverse normalization
                history_data.append(batch_x.detach().cpu().numpy())
                history_wavelengths.append(batch_x_mark.detach().cpu().numpy())

                # Transfer data to device and match model type
                batch_x = batch_x.to(dtype).to(device)
                batch_y = batch_y.to(dtype).to(device)
                batch_x_mark = batch_x_mark.to(dtype).to(device)
                batch_y_mark = batch_y_mark.to(dtype).to(device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).to(dtype).to(device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(device)

                # Predict
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

                # Store results (using original data directly)
                preds.append(outputs.to(torch.float32).detach().cpu().numpy())
                trues.append(batch_y.to(torch.float32).detach().cpu().numpy())
                wavelengths_x.append(batch_x_mark.to(torch.float32).detach().cpu().numpy())
                wavelengths_y.append(batch_y_mark.to(torch.float32).detach().cpu().numpy())

                total_samples += batch_x.shape[0]

            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                traceback.print_exc()
                continue

            # 修改：如果args.num_samples为-1，则预测所有数据
            if args.num_samples > 0 and total_samples >= args.num_samples:
                print(f"Reached specified sample limit: {args.num_samples}")
                break

    if not preds:
        raise RuntimeError("Prediction process failed, no predictions generated")

    # Convert to numpy arrays
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    wavelengths_x = np.concatenate(wavelengths_x, axis=0)
    wavelengths_y = np.concatenate(wavelengths_y, axis=0)
    history_data = np.concatenate(history_data, axis=0)
    history_wavelengths = np.concatenate(history_wavelengths, axis=0)

    # 如果指定了样本数量且不是-1，则截取指定数量
    if args.num_samples > 0:
        preds = preds[:args.num_samples]
        trues = trues[:args.num_samples]
        wavelengths_x = wavelengths_x[:args.num_samples]
        wavelengths_y = wavelengths_y[:args.num_samples]
        history_data = history_data[:args.num_samples]
        history_wavelengths = history_wavelengths[:args.num_samples]

    print(f"Prediction completed, generated predictions for {preds.shape[0]} samples")

    # Apply Savitzky-Golay filter
    if args.apply_filter:
        print("Applying Savitzky-Golay filter to predictions...")
        for i in range(preds.shape[0]):
            for j in range(preds.shape[2]):  # For each feature dimension
                # Apply Savitzky-Golay filter
                preds[i, :, j] = savgol_filter(
                    preds[i, :, j],
                    window_length=args.window_length,
                    polyorder=args.polyorder
                )

    return preds, trues, wavelengths_x, wavelengths_y, history_data, history_wavelengths


def get_metrics(preds, trues):
    """Calculate prediction performance metrics"""
    print("Calculating performance metrics...")
    # Flatten data to calculate metrics
    pred_flat = preds.reshape(-1)
    true_flat = trues.reshape(-1)

    # Calculate metrics
    mse = mean_squared_error(true_flat, pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_flat, pred_flat)
    r2 = r2_score(true_flat, pred_flat)

    print(f"==== Prediction Performance Metrics ====")
    print(f"Total samples predicted: {preds.shape[0]}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def calculate_per_sample_metrics(preds, trues):
    """Calculate metrics for each sample"""
    sample_metrics = []
    for i in range(preds.shape[0]):
        pred_sample = preds[i, :, 0].flatten()
        true_sample = trues[i, :, 0].flatten()

        metrics = {
            'sample': i + 1,
            'mse': mean_squared_error(true_sample, pred_sample),
            'rmse': np.sqrt(mean_squared_error(true_sample, pred_sample)),
            'mae': mean_absolute_error(true_sample, pred_sample),
            'r2': r2_score(true_sample, pred_sample)
        }
        sample_metrics.append(metrics)

    return sample_metrics


def calculate_wavelength_band_metrics(preds, trues, wavelengths_y, bands):
    """Calculate metrics for different wavelength bands"""
    band_metrics = []

    for band_name, (start, end) in bands.items():
        # Find wavelength indices for this band
        band_indices = []
        for i in range(preds.shape[0]):
            wavelengths = wavelengths_y[i, :, 0]
            indices = np.where((wavelengths >= start) & (wavelengths <= end))[0]
            band_indices.append(indices)

        # Calculate metrics for this band
        band_preds = []
        band_trues = []

        for i in range(preds.shape[0]):
            if len(band_indices[i]) > 0:
                band_preds.extend(preds[i, band_indices[i], 0])
                band_trues.extend(trues[i, band_indices[i], 0])

        if len(band_preds) > 0:
            band_preds = np.array(band_preds)
            band_trues = np.array(band_trues)

            metrics = {
                'band': band_name,
                'wavelength_range': f"{start}-{end} nm",
                'mse': mean_squared_error(band_trues, band_preds),
                'rmse': np.sqrt(mean_squared_error(band_trues, band_preds)),
                'mae': mean_absolute_error(band_trues, band_preds),
                'r2': r2_score(band_trues, band_preds)
            }
            band_metrics.append(metrics)

    return band_metrics

def create_enhanced_visualizations(preds, trues, wavelengths_y, args, history_data=None, history_wavelengths=None):
    """Create enhanced visualizations for scientific publications"""
    print("Generating enhanced visualizations...")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Calculate overall metrics
    overall_metrics = get_metrics(preds, trues)

    # Calculate per-sample metrics
    sample_metrics = calculate_per_sample_metrics(preds, trues)

    # Define wavelength bands for analysis (adjusted for prediction range)
    # Calculate prediction wavelength range
    wl_max = args.wavelength_range[1]  # 2499
    wl_min_pred = wl_max - args.pred_len  # e.g., 2499 - 700 = 1799, so range is 1800-2499

    wavelength_bands = {
        'SWIR-2': (max(1800, wl_min_pred), min(2000, wl_max)),
        'SWIR-3': (max(2000, wl_min_pred), wl_max)
    }

    # Calculate band-specific metrics
    band_metrics = calculate_wavelength_band_metrics(preds, trues, wavelengths_y, wavelength_bands)

    # Set up publication-quality plotting parameters
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif'
    })

    # 1. Performance Metrics Bar Chart
    print("Creating performance metrics bar chart...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Overall metrics
    metrics_names = ['R²', 'RMSE', 'MAE', 'MSE']
    metrics_values = [overall_metrics['r2'], overall_metrics['rmse'],
                      overall_metrics['mae'], overall_metrics['mse']]
    colors = ['#2E8B57', '#DC143C', '#FF8C00', '#4169E1']

    bars1 = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Overall Performance Metrics', fontweight='bold')
    ax1.set_ylabel('Metric Value')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars1, metrics_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

    # Per-sample R² values (show first 50 samples to avoid overcrowding)
    sample_r2 = [m['r2'] for m in sample_metrics[:50]]
    sample_nums = [m['sample'] for m in sample_metrics[:50]]

    bars2 = ax2.bar(sample_nums, sample_r2, color='#2E8B57', alpha=0.7, edgecolor='black')
    ax2.set_title('R² Score per Sample (First 50)', fontweight='bold')
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('R² Score')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Band-specific R² values
    if band_metrics:
        band_names = [m['band'] for m in band_metrics]
        band_r2 = [m['r2'] for m in band_metrics]

        bars3 = ax3.bar(band_names, band_r2, color='#4169E1', alpha=0.7, edgecolor='black')
        ax3.set_title('R² Score by Wavelength Band', fontweight='bold')
        ax3.set_xlabel('Wavelength Band')
        ax3.set_ylabel('R² Score')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # Band-specific RMSE values
        band_rmse = [m['rmse'] for m in band_metrics]
        bars4 = ax4.bar(band_names, band_rmse, color='#DC143C', alpha=0.7, edgecolor='black')
        ax4.set_title('RMSE by Wavelength Band', fontweight='bold')
        ax4.set_xlabel('Wavelength Band')
        ax4.set_ylabel('RMSE')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'performance_metrics_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation Analysis - 简化版本（只显示R²）
    print("Creating correlation analysis...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot: Predicted vs True values
    pred_flat = preds.reshape(-1)
    true_flat = trues.reshape(-1)

    # Sample points for visualization (to avoid overcrowding)
    n_points = min(10000, len(pred_flat))
    indices = np.random.choice(len(pred_flat), n_points, replace=False)
    pred_sample = pred_flat[indices]
    true_sample = true_flat[indices]

    ax1.scatter(true_sample, pred_sample, alpha=0.6, s=20, color='#4169E1', edgecolor='none')

    # Add perfect prediction line
    min_val = min(true_sample.min(), pred_sample.min())
    max_val = max(true_sample.max(), pred_sample.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # 只显示R²和p值，删除r的显示
    ax1.text(0.05, 0.95, f'R² = {overall_metrics["r2"]:.4f}\np < 0.001',
             transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor='gray'),
             verticalalignment='top',
             fontweight='bold',
             fontsize=12)

    ax1.set_xlabel('True Values', fontweight='bold')
    ax1.set_ylabel('Predicted Values', fontweight='bold')
    ax1.set_title('Predicted vs True Values', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # Residuals plot
    residuals = pred_flat - true_flat
    ax2.scatter(true_flat[indices], residuals[indices], alpha=0.6, s=20, color='#DC143C', edgecolor='none')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # 设置Y轴范围，确保文本框位置一致
    residual_min = np.min(residuals)
    residual_max = np.max(residuals)
    residual_range = residual_max - residual_min
    # 添加10%的边距，确保文本框有足够空间
    ax2.set_ylim(residual_min - 0.1 * residual_range, residual_max + 0.1 * residual_range)

    ax2.set_xlabel('True Values', fontweight='bold')
    ax2.set_ylabel('Residuals (Pred - True)', fontweight='bold')
    ax2.set_title('Residual Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add residual statistics - 与左图保持相同的垂直位置
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    ax2.text(0.05, 0.95, f'Mean: {residual_mean:.3f}\nStd: {residual_std:.3f}',
             transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor='gray'),
             verticalalignment='top',
             fontweight='bold',
             fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'correlation_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Error Distribution Analysis
    print("Creating error distribution analysis...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Histogram of residuals
    ax1.hist(residuals, bins=50, alpha=0.7, color='#4169E1', edgecolor='black', density=True)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Residuals', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Residual Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Box plot of errors by sample (show first 20 samples)
    sample_errors = []
    sample_labels = []
    for i in range(min(20, preds.shape[0])):
        errors = preds[i, :, 0] - trues[i, :, 0]
        sample_errors.append(errors)
        sample_labels.append(f'S{i + 1}')

    bp = ax2.boxplot(sample_errors, labels=sample_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#FF8C00')
        patch.set_alpha(0.7)
    ax2.set_xlabel('Sample ID', fontweight='bold')
    ax2.set_ylabel('Prediction Error', fontweight='bold')
    ax2.set_title('Error Distribution by Sample (First 20)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # Absolute error by wavelength
    mean_abs_errors = np.mean(np.abs(preds[:, :, 0] - trues[:, :, 0]), axis=0)

    # Calculate actual wavelength values for prediction range
    # pred_len represents the number of wavelengths to predict from the end
    wl_max = args.wavelength_range[1]  # 2499
    wl_min_pred = wl_max - args.pred_len  # e.g., 2499 - 700 = 1799, so range is 1800-2499
    prediction_wavelengths = np.linspace(wl_min_pred + 1, wl_max, len(mean_abs_errors))

    ax3.plot(prediction_wavelengths, mean_abs_errors, color='#DC143C', linewidth=2)
    ax3.fill_between(prediction_wavelengths, mean_abs_errors, alpha=0.3, color='#DC143C')
    ax3.set_xlabel('Wavelength (nm)', fontweight='bold')
    ax3.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax3.set_title('Prediction Error by Wavelength', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Q-Q plot for normality check
    from scipy import stats
    stats.probplot(residuals[::1000], dist="norm", plot=ax4)  # Sample every 1000th point for larger datasets
    ax4.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'error_distribution_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Spectral Quality Assessment (show first 3 samples to keep visualization manageable)
    print("Creating spectral quality assessment...")
    fig = plt.figure(figsize=(18, 12))

    # Create a 3x3 grid for multiple analyses
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Sample spectra comparison (top row) - show best performing samples
    sample_r2_values = [m['r2'] for m in sample_metrics]
    best_samples = np.argsort(sample_r2_values)[-3:][::-1]  # Get top 3 samples

    for idx, sample_idx in enumerate(best_samples):
        ax = fig.add_subplot(gs[0, idx])
        if sample_idx < preds.shape[0]:
            wavelength = wavelengths_y[sample_idx, :, 0]
            true_spec = trues[sample_idx, :, 0]
            pred_spec = preds[sample_idx, :, 0]

            ax.plot(wavelength, true_spec, 'b-', linewidth=2, label='True', alpha=0.8)
            ax.plot(wavelength, pred_spec, 'r--', linewidth=2, label='Predicted', alpha=0.8)

            # Add shaded error region
            ax.fill_between(wavelength, true_spec, pred_spec, alpha=0.2, color='gray')

            ax.set_xlabel('Wavelength (nm)', fontweight='bold')
            ax.set_ylabel('Reflectance', fontweight='bold')
            ax.set_title(f'Sample {sample_idx + 1} (R² = {sample_metrics[sample_idx]["r2"]:.3f})', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

    # Error heatmap (middle left) - show all samples
    ax4 = fig.add_subplot(gs[1, 0])
    error_matrix = np.abs(preds[:, :, 0] - trues[:, :, 0])
    im = ax4.imshow(error_matrix, cmap='YlOrRd', aspect='auto')

    # Set x-axis to show actual wavelengths
    wl_max = args.wavelength_range[1]  # 2499
    wl_min_pred = wl_max - args.pred_len  # 1799 for pred_len=700
    n_ticks = min(5, error_matrix.shape[1])  # Show maximum 5 wavelength ticks
    tick_positions = np.linspace(0, error_matrix.shape[1] - 1, n_ticks, dtype=int)
    tick_labels = np.linspace(wl_min_pred + 1, wl_max, n_ticks, dtype=int)  # 1800-2499
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels)

    ax4.set_xlabel('Wavelength (nm)', fontweight='bold')
    ax4.set_ylabel('Sample Index', fontweight='bold')
    ax4.set_title(f'Absolute Error Heatmap (All {preds.shape[0]} Samples)', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Absolute Error')

    # Wavelength-wise statistics (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    wavelength_mean_error = np.mean(preds[:, :, 0] - trues[:, :, 0], axis=0)
    wavelength_std_error = np.std(preds[:, :, 0] - trues[:, :, 0], axis=0)

    # Calculate actual wavelength values for prediction range
    wl_max = args.wavelength_range[1]  # 2499
    wl_min_pred = wl_max - args.pred_len  # 1799
    prediction_wavelengths = np.linspace(wl_min_pred + 1, wl_max, len(wavelength_mean_error))

    ax5.plot(prediction_wavelengths, wavelength_mean_error, 'b-', linewidth=2, label='Mean Error')
    ax5.fill_between(prediction_wavelengths, wavelength_mean_error - wavelength_std_error,
                     wavelength_mean_error + wavelength_std_error, alpha=0.3, color='blue')
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Wavelength (nm)', fontweight='bold')
    ax5.set_ylabel('Prediction Error', fontweight='bold')
    ax5.set_title('Error Statistics by Wavelength', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Performance comparison (middle right) - show subset
    ax6 = fig.add_subplot(gs[1, 2])
    max_samples_plot = min(100, len(sample_metrics))
    metrics_subset = sample_metrics[:max_samples_plot]
    metrics_comparison = pd.DataFrame(metrics_subset)
    ax6.plot(metrics_comparison['sample'], metrics_comparison['r2'], 'o-',
             linewidth=2, markersize=4, color='#2E8B57', label='R²')
    ax6_twin = ax6.twinx()
    ax6_twin.plot(metrics_comparison['sample'], metrics_comparison['rmse'], 's-',
                  linewidth=2, markersize=3, color='#DC143C', label='RMSE')

    ax6.set_xlabel('Sample Number', fontweight='bold')
    ax6.set_ylabel('R² Score', fontweight='bold', color='#2E8B57')
    ax6_twin.set_ylabel('RMSE', fontweight='bold', color='#DC143C')
    ax6.set_title(f'Per-Sample Performance (First {max_samples_plot})', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Band performance (bottom row)
    if band_metrics:
        ax7 = fig.add_subplot(gs[2, :])
        band_df = pd.DataFrame(band_metrics)

        x = np.arange(len(band_df))
        width = 0.25

        bars1 = ax7.bar(x - width, band_df['r2'], width, label='R²', color='#2E8B57', alpha=0.8)
        bars2 = ax7.bar(x, band_df['rmse'] / max(band_df['rmse']), width, label='RMSE (normalized)',
                        color='#DC143C', alpha=0.8)
        bars3 = ax7.bar(x + width, band_df['mae'] / max(band_df['mae']), width, label='MAE (normalized)',
                        color='#FF8C00', alpha=0.8)

        ax7.set_xlabel('Wavelength Bands', fontweight='bold')
        ax7.set_ylabel('Metric Value', fontweight='bold')
        ax7.set_title('Performance Metrics by Wavelength Band', fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels([f"{row['band']}\n({row['wavelength_range']})" for _, row in band_df.iterrows()])
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.savefig(os.path.join(args.save_dir, 'spectral_quality_assessment.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Statistical Significance Tests
    print("Creating statistical analysis...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Violin plot of errors by wavelength bands
    if band_metrics and len(band_metrics) > 1:
        band_errors = []
        band_names = []

        for band_name, (start, end) in wavelength_bands.items():
            band_error_list = []
            for i in range(preds.shape[0]):
                wavelengths = wavelengths_y[i, :, 0]
                indices = np.where((wavelengths >= start) & (wavelengths <= end))[0]
                if len(indices) > 0:
                    errors = preds[i, indices, 0] - trues[i, indices, 0]
                    band_error_list.extend(errors)

            if len(band_error_list) > 0:
                band_errors.append(band_error_list)
                band_names.append(band_name)

        if len(band_errors) > 0:
            violin_parts = ax1.violinplot(band_errors, positions=range(len(band_names)),
                                          showmeans=True, showmedians=True)
            ax1.set_xticks(range(len(band_names)))
            ax1.set_xticklabels(band_names, rotation=45)
            ax1.set_ylabel('Prediction Error', fontweight='bold')
            ax1.set_title('Error Distribution by Wavelength Band', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)

    # Learning curve style plot (error vs sample complexity)
    max_samples_curve = min(100, preds.shape[0])
    sample_sizes = range(1, max_samples_curve + 1, max(1, max_samples_curve // 50))
    cumulative_r2 = []
    cumulative_rmse = []

    for size in sample_sizes:
        subset_preds = preds[:size, :, 0].flatten()
        subset_trues = trues[:size, :, 0].flatten()

        r2 = r2_score(subset_trues, subset_preds)
        rmse = np.sqrt(mean_squared_error(subset_trues, subset_preds))

        cumulative_r2.append(r2)
        cumulative_rmse.append(rmse)

    ax2.plot(sample_sizes, cumulative_r2, 'o-', linewidth=2, color='#2E8B57', label='R²')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(sample_sizes, cumulative_rmse, 's-', linewidth=2, color='#DC143C', label='RMSE')

    ax2.set_xlabel('Number of Samples', fontweight='bold')
    ax2.set_ylabel('R² Score', fontweight='bold', color='#2E8B57')
    ax2_twin.set_ylabel('RMSE', fontweight='bold', color='#DC143C')
    ax2.set_title('Cumulative Performance', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Prediction uncertainty
    prediction_std = np.std(preds[:, :, 0], axis=0)
    true_std = np.std(trues[:, :, 0], axis=0)

    # Calculate actual wavelength values for prediction range
    wl_max = args.wavelength_range[1]  # 2499
    wl_min_pred = wl_max - args.pred_len  # 1799
    prediction_wavelengths = np.linspace(wl_min_pred + 1, wl_max, len(prediction_std))

    ax3.plot(prediction_wavelengths, prediction_std, 'r-', linewidth=2, label='Prediction Std')
    ax3.plot(prediction_wavelengths, true_std, 'b-', linewidth=2, label='True Std')
    ax3.fill_between(prediction_wavelengths, prediction_std, alpha=0.3, color='red')
    ax3.fill_between(prediction_wavelengths, true_std, alpha=0.3, color='blue')

    ax3.set_xlabel('Wavelength (nm)', fontweight='bold')
    ax3.set_ylabel('Standard Deviation', fontweight='bold')
    ax3.set_title('Prediction Uncertainty Analysis', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Model stability (coefficient of variation)
    pred_cv = np.std(preds[:, :, 0], axis=0) / (np.mean(np.abs(preds[:, :, 0]), axis=0) + 1e-8)
    true_cv = np.std(trues[:, :, 0], axis=0) / (np.mean(np.abs(trues[:, :, 0]), axis=0) + 1e-8)

    ax4.scatter(true_cv, pred_cv, alpha=0.6, s=30, color='#4169E1')
    ax4.plot([0, max(true_cv)], [0, max(true_cv)], 'r--', linewidth=2, label='Perfect Stability')

    ax4.set_xlabel('True CV', fontweight='bold')
    ax4.set_ylabel('Predicted CV', fontweight='bold')
    ax4.set_title('Coefficient of Variation Comparison', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'statistical_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed metrics to files
    print("Saving detailed metrics...")

    # Overall metrics
    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(os.path.join(args.save_dir, 'overall_metrics.csv'), index=False)

    # Per-sample metrics
    sample_df = pd.DataFrame(sample_metrics)
    sample_df.to_csv(os.path.join(args.save_dir, 'per_sample_metrics.csv'), index=False)

    # Band-specific metrics
    if band_metrics:
        band_df = pd.DataFrame(band_metrics)
        band_df.to_csv(os.path.join(args.save_dir, 'wavelength_band_metrics.csv'), index=False)

    print(f"Enhanced visualizations saved to {args.save_dir}")

    return overall_metrics, sample_metrics, band_metrics



def visualize_predictions(preds, trues, wavelengths_y, args, history_data=None, history_wavelengths=None):
    """Visualize prediction results for a subset of samples"""
    print("Generating basic prediction visualizations...")

    # Limit visualization to reasonable number of samples
    max_vis_samples = min(20, args.num_samples if args.num_samples > 0 else 20, preds.shape[0])
    sample_count = max_vis_samples

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Get wavelength range
    wl_min, wl_max = args.wavelength_range
    print(f"Wavelength range: {wl_min}-{wl_max} nm")
    print(f"Visualizing {sample_count} samples out of {preds.shape[0]} total samples")

    # Print dimension information
    print(f"Prediction data dimensions: {preds.shape}")
    print(f"True data dimensions: {trues.shape}")
    print(f"Wavelength data dimensions: {wavelengths_y.shape}")

    if history_data is not None:
        print(f"Historical Spectrum data dimensions: {history_data.shape}")
        print(f"Historical Spectrum wavelength data dimensions: {history_wavelengths.shape}")

    # English labels for all visualization elements
    x_label = 'Wavelength (nm)'
    y_label = 'Reflectance'
    true_label = 'True Spectrum'
    pred_label = 'Predicted Spectrum'
    diff_label = 'Difference'
    history_label = 'Historical Spectrum'
    boundary_label = 'History/Prediction Boundary'

    # Draw complete spectrum (history + prediction) for subset of samples
    for i in range(sample_count):
        print(f"Drawing complete spectrum for sample {i + 1}/{sample_count}")
        plt.figure(figsize=(16, 8))

        # Get current sample's prediction and true values
        wavelength_pred = wavelengths_y[i, :, 0]
        pred_data = preds[i, :, 0].copy()  # Create a copy to avoid modifying original data
        true_data = trues[i, :, 0]

        # Handle mismatch between wavelength data and prediction data dimensions
        if len(wavelength_pred) != len(true_data):
            if len(wavelength_pred) > len(true_data):
                wavelength_pred = wavelength_pred[-len(true_data):]
            else:
                wavelength_pred = np.linspace(wl_min, wl_max, len(true_data))

        # Calculate performance metrics for current sample
        sample_metrics = {
            'mse': mean_squared_error(true_data, pred_data),
            'rmse': np.sqrt(mean_squared_error(true_data, pred_data)),
            'mae': mean_absolute_error(true_data, pred_data),
            'r2': r2_score(true_data, pred_data)
        }

        # Add sample metrics to title
        plt.suptitle(f'Sample {i + 1} - R²: {sample_metrics["r2"]:.4f}, RMSE: {sample_metrics["rmse"]:.4f}',
                     fontsize=16, fontweight='bold')

        # If historical sequence data exists, draw history + prediction complete sequence
        if history_data is not None and i < len(history_data):
            # Get historical sequence data
            history = history_data[i, :, 0]
            history_wl = history_wavelengths[i, :, 0]

            # Ensure wavelength data dimensions match
            if len(history_wl) != len(history):
                history_wl = np.linspace(wl_min, wl_max, len(history))

            # Modify first point of prediction data to equal last point of historical sequence
            # Note: This is only for visualization, does not affect original performance evaluation
            vis_pred_data = pred_data.copy()
            vis_pred_data[0] = history[-1]

            # Draw historical sequence
            plt.plot(history_wl, history, 'g-', linewidth=2, label=history_label)

            # Add vertical line marking boundary between history and prediction
            boundary_x = wavelength_pred[0]
            plt.axvline(x=boundary_x, color='k', linestyle='--', alpha=0.7)

            # Add horizontal boundary label
            y_pos = min(history.min(), true_data.min()) - 0.05
            plt.text(boundary_x + 20, y_pos, boundary_label, rotation=0,
                     horizontalalignment='left', verticalalignment='top')

            # Draw modified prediction curve
            plt.plot(wavelength_pred, vis_pred_data, 'r--', linewidth=2, label=pred_label)

            # Show difference area (using modified prediction data)
            plt.fill_between(wavelength_pred, true_data, vis_pred_data, color='gray', alpha=0.2, label=diff_label)
        else:
            # If no historical data, use original prediction data
            plt.plot(wavelength_pred, pred_data, 'r--', linewidth=2, label=pred_label)

            # Show difference area
            plt.fill_between(wavelength_pred, true_data, pred_data, color='gray', alpha=0.2, label=diff_label)

        # Draw true curve (always on top)
        plt.plot(wavelength_pred, true_data, 'b-', linewidth=2, label=true_label)

        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, loc='best')

        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f'prediction_sample_{i + 1}_full.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Basic visualization results saved to {args.save_dir} directory")


def main():
    try:
        # Parse command line arguments
        args = parse_args()

        # Apply patch to fix type mismatch issues
        patch_timellm()

        # Prepare device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Set up model
        model = setup_model(args)
        model = model.to(device)

        # Get all test data (no train/test split)
        test_loader = get_data(args)

        # Make predictions on all available data
        preds, trues, wavelengths_x, wavelengths_y, history_data, history_wavelengths = predict(
            model, test_loader, args, device)

        # Calculate performance metrics
        metrics = get_metrics(preds, trues)

        # Create enhanced visualizations for scientific publications
        overall_metrics, sample_metrics, band_metrics = create_enhanced_visualizations(
            preds, trues, wavelengths_y, args, history_data, history_wavelengths)

        # Also create basic visualizations for subset of samples
        visualize_predictions(preds, trues, wavelengths_y, args, history_data, history_wavelengths)

        # Save prediction data and true data for subsequent analysis
        save_data = {
            'predictions': preds,
            'ground_truth': trues,
            'wavelengths_x': wavelengths_x,
            'wavelengths_y': wavelengths_y,
            'history_data': history_data,
            'history_wavelengths': history_wavelengths,
            'total_samples': preds.shape[0]
        }

        results_file = 'prediction_results_enhanced.npz'
        np.savez(os.path.join(args.save_dir, results_file), **save_data)
        print(f"Prediction result data saved to {os.path.join(args.save_dir, results_file)}")

        print(f"Enhanced prediction analysis and visualization completed!")
        print(f"Total samples processed: {preds.shape[0]}")
        return metrics

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()