"""
Model Comparison and Visualization
================================

This script compares the performance of the original and dual-branch models,
generating side-by-side visualizations of their predictions, error distributions,
and metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_results(model_dir: str):
    """Load model predictions and metrics."""
    model_dir = Path(model_dir)
    
    # Load predictions if available
    predictions_file = model_dir / 'evaluation' / 'predictions.npy'
    if predictions_file.exists():
        predictions = np.load(predictions_file)
    else:
        predictions = None
    
    # Load metrics
    metrics_file = model_dir / 'evaluation' / 'metrics.csv'
    if metrics_file.exists():
        metrics = pd.read_csv(metrics_file)
    else:
        metrics = None
    
    return predictions, metrics

def plot_comparison(original_dir: str, dual_branch_dir: str, save_dir: str):
    """Generate comparison plots between models."""
    original_dir = Path(original_dir)
    dual_branch_dir = Path(dual_branch_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    orig_metrics = pd.read_csv(original_dir / 'metrics.csv')
    dual_metrics = pd.read_csv(dual_branch_dir / 'evaluation' / 'metrics.csv')
    
    # Create metrics comparison plot
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    metrics_df = pd.DataFrame({
        'Model': ['Original'] * len(orig_metrics) + ['Dual-Branch'] * len(dual_metrics),
        'Metric': list(orig_metrics['Metric']) + list(dual_metrics['Metric']),
        'Value': list(orig_metrics['Value']) + list(dual_metrics['Value'])
    })
    
    # Plot metrics
    sns.barplot(data=metrics_df, x='Metric', y='Value', hue='Model')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison.png')
    plt.close()
    
    # Load and plot predictions
    orig_pred = np.load(original_dir / 'predictions.npy')
    dual_pred = np.load(dual_branch_dir / 'evaluation' / 'predictions.npy')
    y_true = np.load(original_dir / 'y_test.npy')
    
    # Create prediction comparison plot
    plt.figure(figsize=(15, 6))
    
    # Original model predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y_true[:, 0], y_true[:, 1], c='blue', label='True', alpha=0.6)
    plt.scatter(orig_pred[:, 0], orig_pred[:, 1], c='red', label='Predicted', alpha=0.6)
    plt.title('Original Model Predictions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    
    # Dual-branch model predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_true[:, 0], y_true[:, 1], c='blue', label='True', alpha=0.6)
    plt.scatter(dual_pred[:, 0], dual_pred[:, 1], c='red', label='Predicted', alpha=0.6)
    plt.title('Dual-Branch Model Predictions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions_comparison.png')
    plt.close()
    
    # Create error distribution comparison
    plt.figure(figsize=(12, 6))
    
    # Calculate errors
    orig_errors = np.sqrt(np.sum((y_true - orig_pred) ** 2, axis=1))
    dual_errors = np.sqrt(np.sum((y_true - dual_pred) ** 2, axis=1))
    
    # Plot error distributions
    sns.kdeplot(data=orig_errors, label='Original Model', fill=True)
    sns.kdeplot(data=dual_errors, label='Dual-Branch Model', fill=True)
    plt.title('Error Distribution Comparison')
    plt.xlabel('Error Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'error_distribution_comparison.png')
    plt.close()
    
    # Create spatial error comparison
    plt.figure(figsize=(15, 6))
    
    # Original model spatial errors
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(y_true[:, 0], y_true[:, 1], c=orig_errors, cmap='viridis')
    plt.colorbar(scatter, label='Error Distance')
    plt.title('Original Model Spatial Errors')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    
    # Dual-branch model spatial errors
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(y_true[:, 0], y_true[:, 1], c=dual_errors, cmap='viridis')
    plt.colorbar(scatter, label='Error Distance')
    plt.title('Dual-Branch Model Spatial Errors')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'spatial_errors_comparison.png')
    plt.close()
    
    # Save comparison metrics to CSV
    comparison_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAE_X', 'MAE_Y'],
        'Original': orig_metrics['Value'].values,
        'Dual-Branch': dual_metrics['Value'].values,
        'Improvement': (orig_metrics['Value'].values - dual_metrics['Value'].values) / orig_metrics['Value'].values * 100
    })
    comparison_df.to_csv(save_dir / 'comparison_metrics.csv', index=False)
    
    logger.info(f"Comparison plots and metrics saved to {save_dir}")

def main():
    """Main function to generate comparison plots."""
    # Set paths
    original_dir = Path('Indoor-Localization-main/models/original')
    dual_branch_dir = Path('Indoor-Localization-main/models/dual_branch')
    save_dir = Path('Indoor-Localization-main/models/comparison')
    
    # Generate comparison plots
    plot_comparison(original_dir, dual_branch_dir, save_dir)

if __name__ == "__main__":
    main() 