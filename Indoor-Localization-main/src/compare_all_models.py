"""
Model Comparison Script
=====================

This script compares the performance of all models:
1. Original Model
2. Dual Branch Model
3. Enhanced Dual Branch Model
4. Attention-Based Model

The script generates comprehensive visualizations and metrics for comparison.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_results(model_dir: str):
    """
    Load model predictions and metrics.
    
    Args:
        model_dir: Path to the model directory
    
    Returns:
        Tuple of (predictions, metrics, y_test)
    """
    model_dir = Path(model_dir)
    
    # Load predictions
    predictions = np.load(model_dir / 'predictions.npy')
    
    # Load metrics
    metrics = pd.read_csv(model_dir / 'metrics.csv')
    
    # Load test data
    y_test = np.load(model_dir / 'y_test.npy')
    
    return predictions, metrics, y_test

def plot_metrics_comparison(metrics_dict: dict, save_path: Path):
    """
    Plot comparison of metrics across all models.
    
    Args:
        metrics_dict: Dictionary of model names and their metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    metrics_data = []
    for model_name, metrics in metrics_dict.items():
        for _, row in metrics.iterrows():
            metrics_data.append({
                'Model': model_name,
                'Metric': row['Metric'],
                'Value': row['Value']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot metrics
    sns.barplot(data=metrics_df, x='Metric', y='Value', hue='Model')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path / 'metrics_comparison.png')
    plt.close()

def plot_predictions_comparison(predictions_dict: dict, y_test: np.ndarray, save_path: Path):
    """
    Plot comparison of predictions across all models.
    
    Args:
        predictions_dict: Dictionary of model names and their predictions
        y_test: True test labels
        save_path: Path to save the plot
    """
    n_models = len(predictions_dict)
    plt.figure(figsize=(15, 4 * n_models))
    
    for i, (model_name, predictions) in enumerate(predictions_dict.items(), 1):
        plt.subplot(n_models, 1, i)
        plt.scatter(y_test[:, 0], y_test[:, 1], c='blue', label='True', alpha=0.6)
        plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='Predicted', alpha=0.6)
        
        for j in range(len(y_test)):
            plt.plot([y_test[j, 0], predictions[j, 0]], 
                    [y_test[j, 1], predictions[j, 1]], 
                    'k--', alpha=0.2)
        
        plt.title(f'{model_name} - True vs Predicted Locations')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'predictions_comparison.png')
    plt.close()

def plot_error_distribution_comparison(predictions_dict: dict, y_test: np.ndarray, save_path: Path):
    """
    Plot comparison of error distributions across all models.
    
    Args:
        predictions_dict: Dictionary of model names and their predictions
        y_test: True test labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, predictions in predictions_dict.items():
        errors = np.sqrt(np.sum((y_test - predictions) ** 2, axis=1))
        sns.kdeplot(data=errors, label=model_name, fill=True)
    
    plt.title('Error Distribution Comparison')
    plt.xlabel('Error Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / 'error_distribution_comparison.png')
    plt.close()

def plot_spatial_errors_comparison(predictions_dict: dict, y_test: np.ndarray, save_path: Path):
    """
    Plot comparison of spatial errors across all models.
    
    Args:
        predictions_dict: Dictionary of model names and their predictions
        y_test: True test labels
        save_path: Path to save the plot
    """
    n_models = len(predictions_dict)
    plt.figure(figsize=(15, 4 * n_models))
    
    for i, (model_name, predictions) in enumerate(predictions_dict.items(), 1):
        plt.subplot(n_models, 1, i)
        errors = np.sqrt(np.sum((y_test - predictions) ** 2, axis=1))
        scatter = plt.scatter(y_test[:, 0], y_test[:, 1], c=errors, cmap='viridis')
        plt.colorbar(scatter, label='Error Distance')
        plt.title(f'{model_name} - Spatial Error Distribution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'spatial_errors_comparison.png')
    plt.close()

def main():
    """Main function to compare all models."""
    # Set paths
    base_dir = Path('Indoor-Localization-main/models')
    save_dir = base_dir / 'comparison'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results for each model
    model_dirs = {
        'Original': base_dir / 'original',
        'Dual Branch': base_dir / 'dual_branch',
        'Enhanced Dual Branch': base_dir / 'enhanced' / 'enhanced_dual_branch',
        'Attention-Based': base_dir / 'enhanced' / 'attention_based'
    }
    
    predictions_dict = {}
    metrics_dict = {}
    y_test = None
    
    for model_name, model_dir in model_dirs.items():
        try:
            predictions, metrics, y_test = load_model_results(model_dir)
            predictions_dict[model_name] = predictions
            metrics_dict[model_name] = metrics
            logger.info(f"Loaded results for {model_name}")
        except Exception as e:
            logger.warning(f"Could not load results for {model_name}: {str(e)}")
    
    if not predictions_dict:
        logger.error("No model results found!")
        return
    
    # Generate comparison plots
    logger.info("Generating comparison plots...")
    plot_metrics_comparison(metrics_dict, save_dir)
    plot_predictions_comparison(predictions_dict, y_test, save_dir)
    plot_error_distribution_comparison(predictions_dict, y_test, save_dir)
    plot_spatial_errors_comparison(predictions_dict, y_test, save_dir)
    
    # Save comparison metrics to CSV
    comparison_data = []
    for model_name, metrics in metrics_dict.items():
        for _, row in metrics.iterrows():
            comparison_data.append({
                'Model': model_name,
                'Metric': row['Metric'],
                'Value': row['Value']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(save_dir / 'comparison_metrics.csv', index=False)
    
    logger.info(f"Comparison results saved to {save_dir}")

if __name__ == "__main__":
    main() 