"""
Model Comparison Script
=====================

This script compares the performance of all models:
1. Original Model
2. Dual Branch Model
3. Enhanced Dual Branch Model
4. Attention-Based Model

The script generates comparative visualizations and metrics.
"""

import numpy as np
from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_metrics(results_dir: Path, model_name: str):
    """Load metrics for a specific model."""
    try:
        metrics_df = pd.read_csv(results_dir / f'{model_name}_metrics.csv')
        return metrics_df
    except Exception as e:
        logger.error(f"Error loading metrics for {model_name}: {e}")
        return None

def plot_metrics_comparison(metrics_dict: dict, save_path: Path):
    """Plot comparison of metrics across all models."""
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    metrics = ['MAE', 'RMSE', 'MAE_X', 'MAE_Y']
    model_names = list(metrics_dict.keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(model_names):
        values = [metrics_dict[model_name].loc[metrics_dict[model_name]['Metric'] == m, 'Value'].values[0] 
                 for m in metrics]
        plt.bar(x + i * width, values, width, label=model_name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * (len(model_names) - 1) / 2, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'metrics_comparison.png')
    plt.close()

def plot_predictions_comparison(models_dict: dict, test_data: tuple, save_path: Path):
    """Plot predictions comparison for all models."""
    plt.figure(figsize=(15, 10))
    
    # Plot true locations
    plt.scatter(test_data[2][:, 0], test_data[2][:, 1], 
               c='black', label='True Locations', alpha=0.6, marker='x')
    
    # Plot predictions for each model
    colors = ['red', 'blue', 'green', 'purple']
    for (model_name, model), color in zip(models_dict.items(), colors):
        y_pred = model.predict([test_data[0], test_data[1]])
        plt.scatter(y_pred[:, 0], y_pred[:, 1], 
                   c=color, label=f'{model_name} Predictions', alpha=0.6)
    
    plt.title('Location Predictions Comparison')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'predictions_comparison.png')
    plt.close()

def plot_error_distribution_comparison(models_dict: dict, test_data: tuple, save_path: Path):
    """Plot error distribution comparison for all models."""
    plt.figure(figsize=(12, 6))
    
    for model_name, model in models_dict.items():
        y_pred = model.predict([test_data[0], test_data[1]])
        errors = np.sqrt(np.sum((test_data[2] - y_pred) ** 2, axis=1))
        sns.kdeplot(errors, label=model_name)
    
    plt.title('Error Distribution Comparison')
    plt.xlabel('Error Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'error_distribution_comparison.png')
    plt.close()

def main():
    # Set paths
    results_dir = Path('results')
    comparison_dir = results_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all models
    logger.info("Loading models...")
    models = {
        'Original': tf.keras.models.load_model(results_dir / 'original/best_model.keras'),
        'Dual Branch': tf.keras.models.load_model(results_dir / 'dual_branch/best_model.keras'),
        'Enhanced Dual Branch': tf.keras.models.load_model(results_dir / 'enhanced_models/enhanced_dual_branch_best.keras'),
        'Attention-Based': tf.keras.models.load_model(results_dir / 'enhanced_models/attention_based_best.keras')
    }
    
    # Load test data
    logger.info("Loading test data...")
    test_data = (
        np.load('data/processed/test_csi.npy'),
        np.load('data/processed/test_rssi.npy'),
        np.load('data/processed/test_labels.npy')
    )
    
    # Load metrics for all models
    logger.info("Loading metrics...")
    metrics_dict = {}
    for model_name in models.keys():
        metrics = load_model_metrics(results_dir / model_name.lower().replace(' ', '_'), model_name)
        if metrics is not None:
            metrics_dict[model_name] = metrics
    
    # Generate comparison plots
    logger.info("Generating comparison plots...")
    plot_metrics_comparison(metrics_dict, comparison_dir)
    plot_predictions_comparison(models, test_data, comparison_dir)
    plot_error_distribution_comparison(models, test_data, comparison_dir)
    
    # Save comparison metrics
    comparison_metrics = pd.DataFrame()
    for model_name, metrics in metrics_dict.items():
        comparison_metrics[model_name] = metrics.set_index('Metric')['Value']
    comparison_metrics.to_csv(comparison_dir / 'comparison_metrics.csv')
    
    logger.info("Comparison completed. Results saved in %s", comparison_dir)

if __name__ == '__main__':
    main() 