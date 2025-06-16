"""
Compare Original and Retrained Models
==================================

This script generates comparison visualizations and metrics between
the original and retrained models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_results(model_dir: str):
    """Load model predictions and metrics."""
    model_dir = Path(model_dir)
    
    # Load predictions and true values
    y_pred = np.load(model_dir / 'evaluation' / 'predictions.npy')
    y_true = np.load(model_dir / 'evaluation' / 'y_test.npy')
    
    # Load metrics
    metrics = pd.read_csv(model_dir / 'evaluation' / 'metrics.csv')
    
    return y_pred, y_true, metrics

def plot_comparison(original_dir: str, retrained_dir: str, save_dir: str):
    """Generate comparison plots between original and retrained models."""
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    original_pred, original_true, original_metrics = load_model_results(original_dir)
    retrained_pred, retrained_true, retrained_metrics = load_model_results(retrained_dir)
    
    # 1. Compare metrics
    plt.figure(figsize=(12, 6))
    metrics_comparison = pd.DataFrame({
        'Metric': original_metrics['Metric'],
        'Original': original_metrics['Value'],
        'Retrained': retrained_metrics['Value']
    })
    
    # Plot metrics comparison
    metrics_comparison.plot(x='Metric', y=['Original', 'Retrained'], kind='bar')
    plt.title('Metrics Comparison: Original vs Retrained Model')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path / 'metrics_comparison.png')
    plt.close()
    
    # 2. Compare predictions
    plt.figure(figsize=(15, 6))
    
    # Original model predictions
    plt.subplot(1, 2, 1)
    plt.scatter(original_true[:, 0], original_true[:, 1], c='blue', label='True', alpha=0.6)
    plt.scatter(original_pred[:, 0], original_pred[:, 1], c='red', label='Predicted', alpha=0.6)
    plt.title('Original Model Predictions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    
    # Retrained model predictions
    plt.subplot(1, 2, 2)
    plt.scatter(retrained_true[:, 0], retrained_true[:, 1], c='blue', label='True', alpha=0.6)
    plt.scatter(retrained_pred[:, 0], retrained_pred[:, 1], c='red', label='Predicted', alpha=0.6)
    plt.title('Retrained Model Predictions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'predictions_comparison.png')
    plt.close()
    
    # 3. Compare error distributions
    plt.figure(figsize=(15, 6))
    
    # Calculate errors
    original_errors = np.sqrt(np.sum((original_true - original_pred) ** 2, axis=1))
    retrained_errors = np.sqrt(np.sum((retrained_true - retrained_pred) ** 2, axis=1))
    
    # Plot error distributions
    plt.subplot(1, 2, 1)
    sns.histplot(original_errors, kde=True, label='Original')
    plt.axvline(np.mean(original_errors), color='r', linestyle='--', 
                label=f'Mean: {np.mean(original_errors):.2f}')
    plt.title('Original Model Error Distribution')
    plt.xlabel('Error Distance')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(retrained_errors, kde=True, label='Retrained')
    plt.axvline(np.mean(retrained_errors), color='r', linestyle='--', 
                label=f'Mean: {np.mean(retrained_errors):.2f}')
    plt.title('Retrained Model Error Distribution')
    plt.xlabel('Error Distance')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'error_distribution_comparison.png')
    plt.close()
    
    # Save comparison metrics to CSV
    comparison_df = pd.DataFrame({
        'Metric': original_metrics['Metric'],
        'Original': original_metrics['Value'],
        'Retrained': retrained_metrics['Value'],
        'Improvement': retrained_metrics['Value'] - original_metrics['Value']
    })
    comparison_df.to_csv(save_path / 'comparison_metrics.csv', index=False)
    
    logger.info(f"Comparison results saved to {save_path}")

def main():
    """Main function to generate comparison plots."""
    # Set paths
    original_dir = Path('Indoor-Localization-main/models/original')
    retrained_dir = Path('Indoor-Localization-main/models/retrained')
    save_dir = Path('Indoor-Localization-main/models/comparison_original_retrained')
    
    # Generate comparison plots
    plot_comparison(original_dir, retrained_dir, save_dir)

if __name__ == "__main__":
    main() 