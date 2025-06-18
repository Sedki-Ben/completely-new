"""
Training Script for Enhanced Indoor Localization Models
====================================================

This script trains and evaluates both enhanced models:
1. Enhanced Dual Branch Model
2. Attention-Based Model

The script includes data loading, preprocessing, training, and visualization.
"""

import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.enhanced_models import EnhancedDualBranchModel, AttentionBasedModel, train_and_evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_training_history(history, model_name, save_path):
    """Plot training history for a model."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / f'{model_name}_training_history.png')
    plt.close()

def plot_predictions(y_true, y_pred, model_name, save_path):
    """Plot true vs predicted locations."""
    plt.figure(figsize=(10, 8))
    
    # Plot true locations
    plt.scatter(y_true[:, 0], y_true[:, 1], c='blue', label='True Locations', alpha=0.6)
    
    # Plot predicted locations
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', label='Predicted Locations', alpha=0.6)
    
    # Draw lines between true and predicted locations
    for i in range(len(y_true)):
        plt.plot([y_true[i, 0], y_pred[i, 0]], 
                [y_true[i, 1], y_pred[i, 1]], 
                'k--', alpha=0.2)
    
    plt.title(f'{model_name} - True vs Predicted Locations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / f'{model_name}_predictions.png')
    plt.close()

def plot_error_distribution(y_true, y_pred, model_name, save_path):
    """Plot error distribution."""
    errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(np.mean(errors), color='r', linestyle='--', 
                label=f'Mean: {np.mean(errors):.2f}m')
    plt.title(f'{model_name} - Error Distribution')
    plt.xlabel('Error Distance (m)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(save_path / f'{model_name}_error_distribution.png')
    plt.close()

def plot_metrics_comparison(metrics_dict, save_path):
    """Plot comparison of metrics between models."""
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

def load_and_preprocess_data(data_dir: str):
    """
    Load and preprocess the data for training.
    
    Args:
        data_dir: Path to the data directory
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    data_dir = Path(data_dir)
    
    # Load CSI data
    csi_features = np.load(data_dir / 'csi_features.npy')
    csi_phase = np.load(data_dir / 'csi_phase.npy')
    
    # Load RSSI data
    rssi = np.load(data_dir / 'rssi.npy')
    
    # Load locations
    locations = np.load(data_dir / 'locations.npy')
    
    # Reshape CSI phase data to match features
    if len(csi_phase.shape) == 2:
        csi_phase = csi_phase[..., np.newaxis]
    
    # Combine CSI features
    X_csi = np.concatenate([csi_features, csi_phase], axis=-1)
    
    # Reshape RSSI data
    X_rssi = rssi.reshape(-1, 1)
    
    # Split data into train, validation, and test sets
    X_csi_train, X_csi_temp, X_rssi_train, X_rssi_temp, y_train, y_temp = train_test_split(
        X_csi, X_rssi, locations, test_size=0.3, random_state=42
    )
    
    X_csi_val, X_csi_test, X_rssi_val, X_rssi_test, y_val, y_test = train_test_split(
        X_csi_temp, X_rssi_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Scale the data
    csi_scaler = StandardScaler()
    rssi_scaler = StandardScaler()
    
    # Reshape CSI data for scaling
    X_csi_train_reshaped = X_csi_train.reshape(-1, X_csi_train.shape[-1])
    X_csi_val_reshaped = X_csi_val.reshape(-1, X_csi_val.shape[-1])
    X_csi_test_reshaped = X_csi_test.reshape(-1, X_csi_test.shape[-1])
    
    # Scale CSI data
    X_csi_train_scaled = csi_scaler.fit_transform(X_csi_train_reshaped)
    X_csi_val_scaled = csi_scaler.transform(X_csi_val_reshaped)
    X_csi_test_scaled = csi_scaler.transform(X_csi_test_reshaped)
    
    # Reshape back to original shape
    X_csi_train = X_csi_train_scaled.reshape(X_csi_train.shape)
    X_csi_val = X_csi_val_scaled.reshape(X_csi_val.shape)
    X_csi_test = X_csi_test_scaled.reshape(X_csi_test.shape)
    
    # Scale RSSI data
    X_rssi_train = rssi_scaler.fit_transform(X_rssi_train)
    X_rssi_val = rssi_scaler.transform(X_rssi_val)
    X_rssi_test = rssi_scaler.transform(X_rssi_test)
    
    return (
        (X_csi_train, X_rssi_train, y_train),
        (X_csi_val, X_rssi_val, y_val),
        (X_csi_test, X_rssi_test, y_test)
    )

def main():
    """Main function to train and evaluate both models."""
    # Set paths
    data_dir = Path('Indoor-Localization-main/data')
    save_dir = Path('Indoor-Localization-main/models/enhanced')
    vis_dir = save_dir / 'visualizations'
    
    # Create directories
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    train_data, val_data, test_data = load_and_preprocess_data(data_dir)
    
    # Train Enhanced Dual Branch Model
    logger.info("Training Enhanced Dual Branch Model...")
    dual_branch_model = EnhancedDualBranchModel(
        input_shape_csi=train_data[0].shape[1:],
        input_shape_rssi=train_data[1].shape[1:]
    )
    dual_branch_history, dual_branch_metrics = train_and_evaluate_model(
        dual_branch_model,
        train_data,
        val_data,
        test_data,
        save_dir,
        'enhanced_dual_branch'
    )
    
    # Train Attention-Based Model
    logger.info("Training Attention-Based Model...")
    attention_model = AttentionBasedModel(
        input_shape_csi=train_data[0].shape[1:],
        input_shape_rssi=train_data[1].shape[1:]
    )
    attention_history, attention_metrics = train_and_evaluate_model(
        attention_model,
        train_data,
        val_data,
        test_data,
        save_dir,
        'attention_based'
    )
    
    # Generate visualizations for both models
    logger.info("Generating visualizations...")
    
    # Plot training history
    plot_training_history(dual_branch_history, 'Enhanced Dual Branch', vis_dir)
    plot_training_history(attention_history, 'Attention-Based', vis_dir)
    
    # Get predictions for visualization
    dual_branch_pred = dual_branch_model.predict([test_data[0], test_data[1]])
    attention_pred = attention_model.predict([test_data[0], test_data[1]])
    
    # Plot predictions
    plot_predictions(test_data[2], dual_branch_pred, 'Enhanced Dual Branch', vis_dir)
    plot_predictions(test_data[2], attention_pred, 'Attention-Based', vis_dir)
    
    # Plot error distributions
    plot_error_distribution(test_data[2], dual_branch_pred, 'Enhanced Dual Branch', vis_dir)
    plot_error_distribution(test_data[2], attention_pred, 'Attention-Based', vis_dir)
    
    # Plot metrics comparison
    metrics_dict = {
        'Enhanced Dual Branch': dual_branch_metrics,
        'Attention-Based': attention_metrics
    }
    plot_metrics_comparison(metrics_dict, vis_dir)
    
    # Compare models
    logger.info("\nModel Comparison:")
    logger.info("\nEnhanced Dual Branch Model Metrics:")
    logger.info(dual_branch_metrics)
    logger.info("\nAttention-Based Model Metrics:")
    logger.info(attention_metrics)
    
    logger.info(f"\nResults and visualizations saved to {save_dir}")

if __name__ == "__main__":
    main() 