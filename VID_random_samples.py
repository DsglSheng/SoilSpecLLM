#!/usr/bin/env python3
"""
TimeLLM Spectral Prediction Visualization with Random Sample Selection
Modified to randomly select samples for visualization
"""

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
from matplotlib.patches import Rectangle
import random

# Set Matplotlib to non-interactive mode
matplotlib.use('Agg')

# Set style
plt.style.use('default')


def parse_args():
    parser = argparse.ArgumentParser(description='TimeLLM Spectral Prediction Visualization')

    # Basic configuration
    parser.add_argument('--model_id', type=str, default='Spectral_1400_700', help='Model ID')
    parser.add_argument('--model_comment', type=str, default='TimeLLM-Spectral', help='Model comment')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/home/user/WangS/Time-LLM-main/scripts/checkpoints/llama2_1400_700/checkpoint',
                        help='Checkpoint file path')

    # Dataset parameters
    parser.add_argument('--data', type=str, default='Spectral', help='Dataset type')
    parser.add_argument('--root_path', type=str, default='/home/user/WangS/Time-LLM-main/dataset/spectral/',
                        help='Dataset root directory')
    parser.add_argument('--data_path', type=str, default='spectral_data.csv', help='Data file')
    parser.add_argument('--features', type=str, default='S', help='Feature type: [M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='Target feature')
    parser.add_argument('--freq', type=str, default='h', help='Frequency encoding')
    parser.add_argument('--seasonal_patterns', type=str, default='Ll', help='Seasonal patterns')
    parser.add_argument('--percent', type=int, default=100, help='Percentage of data to use')
    parser.add_argument('--scale', type=bool, default=False, help='Whether to scale data')
    parser.add_argument('--timeenc', type=int, default=0, help='Time encoding type')
    parser.add_argument('--loader', type=str, default='modal', help='Dataset type')

    # Model parameters
    parser.add_argument('--seq_len', type=int, default=1400, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='Overlapping sequence length')
    parser.add_argument('--pred_len', type=int, default=700, help='Prediction sequence length')
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
    parser.add_argument('--embed', type=str, default='timeF', help='Time feature encoding')

    # TimeLLM specific parameters
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model type')
    parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')
    parser.add_argument('--llm_layers', type=int, default=32, help='Number of LLM layers')
    parser.add_argument('--prompt_domain', type=int, default=0, help='Prompt domain')
    parser.add_argument('--patch_len', type=int, default=16, help='Patch length')
    parser.add_argument('--stride', type=int, default=8, help='Stride')
    parser.add_argument('--content', type=str,
                        default='Spectral data is used to analyze composition and properties of materials.',
                        help='Domain description text')

    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true', help='Whether to use mixed precision')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of data loader workers')

    # Experiment parameters
    parser.add_argument('--is_training', type=int, default=0, help='Whether it is training mode')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiments')
    parser.add_argument('--des', type=str, default='test', help='Experiment description')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')
    parser.add_argument('--align_epochs', type=int, default=10, help='Number of alignment epochs')

    # Visualization parameters
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='/home/user/WangS/Time-LLM-main/visual_llama2_2_1',
                        help='Directory to save visualization results')
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='Task name')
    parser.add_argument('--random_samples', action='store_true', default=True, 
                        help='Whether to randomly select samples for visualization')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for sample selection')

    args = parser.parse_args()
    return args


def setup_model(args):
    """Create and load the model"""
    print("Creating model...")
    args.scale = False

    try:
        model = TimeLLM.Model(args).float()
        checkpoint_path = args.checkpoint_path

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            possible_extensions = ['.pth', '.pt', '.ckpt']
            base_path = checkpoint_path.rsplit('.', 1)[0] if '.' in os.path.basename(checkpoint_path) else checkpoint_path
            
            for ext in possible_extensions:
                alt_path = base_path + ext
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    print(f"Found alternative checkpoint: {checkpoint_path}")
                    break

        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            model.load_state_dict(state_dict)
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        model.eval()
        return model
    except Exception as e:
        print(f"Error setting up model: {str(e)}")
        traceback.print_exc()
        raise


def get_data(args):
    """Load test data"""
    print("Loading test data...")
    try:
        args.scale = False
        _, test_loader = data_provider(args, 'test')
        print(f"Test data loaded successfully, {len(test_loader)} batches")
        return test_loader
    except Exception as e:
        print(f"Failed to load test data: {str(e)}")
        traceback.print_exc()
        raise


def predict(model, test_loader, device='cpu'):
    """Make predictions"""
    print("Starting prediction...")
    preds = []
    trues = []
    wavelengths_y = []

    dtype = next(model.parameters()).dtype
    print(f"Model data type: {dtype}")

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            print(f"Processing batch {i + 1}/{len(test_loader)}")

            batch_x = batch_x.to(dtype).to(device)
            batch_y = batch_y.to(dtype).to(device)
            batch_x_mark = batch_x_mark.to(dtype).to(device)
            batch_y_mark = batch_y_mark.to(dtype).to(device)

            dec_inp = torch.zeros_like(batch_y[:, -model.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :model.label_len, :], dec_inp], dim=1).to(dtype).to(device)

            outputs = model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -model.pred_len:, :]

            pred = outputs.detach().cpu().numpy()
            true = batch_y[:, -model.pred_len:, :].detach().cpu().numpy()
            wavelength_y = batch_y_mark[:, -model.pred_len:, :].detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)
            wavelengths_y.append(wavelength_y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    wavelengths_y = np.concatenate(wavelengths_y, axis=0)

    print(f"Prediction complete: {preds.shape[0]} samples")
    return preds, trues, wavelengths_y


def visualize_with_random_samples(preds, trues, wavelengths_y, args):
    """Generate visualizations with random sample selection"""
    
    print("Creating spectral quality assessment with random samples...")
    
    # Calculate metrics for all samples
    sample_metrics = []
    for i in range(preds.shape[0]):
        pred_sample = preds[i, :, 0].flatten()
        true_sample = trues[i, :, 0].flatten()
        
        metrics = {
            'sample': i,
            'mse': mean_squared_error(true_sample, pred_sample),
            'rmse': np.sqrt(mean_squared_error(true_sample, pred_sample)),
            'mae': mean_absolute_error(true_sample, pred_sample),
            'r2': r2_score(true_sample, pred_sample)
        }
        sample_metrics.append(metrics)
    
    # Randomly select 3 samples for visualization
    total_samples = preds.shape[0]
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
    
    selected_indices = random.sample(range(total_samples), min(3, total_samples))
    selected_indices.sort()
    
    print(f"Randomly selected samples for visualization: {selected_indices}")
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Sample spectra comparison (top row) - with random samples
    for plot_idx, sample_idx in enumerate(selected_indices):
        ax = fig.add_subplot(gs[0, plot_idx])
        
        wavelength = wavelengths_y[sample_idx, :, 0]
        true_spec = trues[sample_idx, :, 0]
        pred_spec = preds[sample_idx, :, 0]
        
        ax.plot(wavelength, true_spec, 'b-', linewidth=2, label='True', alpha=0.8)
        ax.plot(wavelength, pred_spec, 'r--', linewidth=2, label='Predicted', alpha=0.8)
        ax.fill_between(wavelength, true_spec, pred_spec, alpha=0.2, color='gray')
        
        ax.set_xlabel('Wavelength (nm)', fontweight='bold')
        ax.set_ylabel('Reflectance', fontweight='bold')
        ax.set_title(f'Sample {sample_idx + 1} (R² = {sample_metrics[sample_idx]["r2"]:.3f})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Error heatmap (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    error_matrix = np.abs(preds[:min(20, preds.shape[0]), :, 0] - trues[:min(20, preds.shape[0]), :, 0])
    im = ax4.imshow(error_matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xlabel('Wavelength Point Index', fontweight='bold')
    ax4.set_ylabel('Sample Index', fontweight='bold')
    ax4.set_title('Absolute Error Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Absolute Error')
    
    # Wavelength-wise statistics (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    wavelength_mean_error = np.mean(preds[:, :, 0] - trues[:, :, 0], axis=0)
    wavelength_std_error = np.std(preds[:, :, 0] - trues[:, :, 0], axis=0)
    wavelength_pos = np.arange(len(wavelength_mean_error))
    
    ax5.plot(wavelength_pos, wavelength_mean_error, 'b-', linewidth=2, label='Mean Error')
    ax5.fill_between(wavelength_pos, wavelength_mean_error - wavelength_std_error,
                     wavelength_mean_error + wavelength_std_error, alpha=0.3, color='blue')
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Wavelength Position Index', fontweight='bold')
    ax5.set_ylabel('Prediction Error', fontweight='bold')
    ax5.set_title('Error Statistics by Wavelength', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Performance comparison (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    metrics_df = pd.DataFrame(sample_metrics)
    ax6.plot(metrics_df['sample'], metrics_df['r2'], 'o-',
             linewidth=2, markersize=8, color='#2E8B57', label='R²')
    ax6_twin = ax6.twinx()
    ax6_twin.plot(metrics_df['sample'], metrics_df['rmse'], 's-',
                  linewidth=2, markersize=6, color='#DC143C', label='RMSE')
    
    ax6.set_xlabel('Sample Number', fontweight='bold')
    ax6.set_ylabel('R² Score', fontweight='bold', color='#2E8B57')
    ax6_twin.set_ylabel('RMSE', fontweight='bold', color='#DC143C')
    ax6.set_title('Per-Sample Performance', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Overall statistics (bottom row)
    ax7 = fig.add_subplot(gs[2, :])
    overall_metrics = {
        'MSE': np.mean([m['mse'] for m in sample_metrics]),
        'RMSE': np.mean([m['rmse'] for m in sample_metrics]),
        'MAE': np.mean([m['mae'] for m in sample_metrics]),
        'R²': np.mean([m['r2'] for m in sample_metrics])
    }
    
    metrics_names = list(overall_metrics.keys())
    metrics_values = list(overall_metrics.values())
    
    bars = ax7.bar(metrics_names, metrics_values, color=['#2E8B57', '#DC143C', '#FF8C00', '#4169E1'], alpha=0.8)
    ax7.set_ylabel('Metric Value', fontweight='bold')
    ax7.set_title('Overall Performance Metrics', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add text annotation about selected samples
    fig.text(0.5, 0.02, f'Randomly selected samples: {[s+1 for s in selected_indices]}', 
             ha='center', fontsize=12, style='italic')
    
    plt.savefig(os.path.join(args.save_dir, 'spectral_quality_assessment.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {args.save_dir}/spectral_quality_assessment.png")
    print(f"Selected samples: {[s+1 for s in selected_indices]}")
    
    return overall_metrics


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*70)
    print("TimeLLM Spectral Prediction Visualization (Random Sample Selection)")
    print("="*70)
    print(f"Model: {args.llm_model}")
    print(f"Configuration: {args.seq_len} -> {args.pred_len}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Save directory: {args.save_dir}")
    print(f"Random sample selection: {args.random_samples}")
    if args.random_seed:
        print(f"Random seed: {args.random_seed}")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = setup_model(args)
    model = model.to(device)
    
    # Load data
    test_loader = get_data(args)
    
    # Predict
    preds, trues, wavelengths_y = predict(model, test_loader, device)
    
    # Visualize
    metrics = visualize_with_random_samples(preds, trues, wavelengths_y, args)
    
    print("\n" + "="*70)
    print("Overall Performance Metrics:")
    print("="*70)
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")
    print("="*70)
    print("Visualization complete!")


if __name__ == '__main__':
    main()
