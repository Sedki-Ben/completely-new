"""
Evaluation and Visualization Script for Enhanced Indoor Localization Models
==========================================================================

This script loads the trained models and performs validation, testing, and visualization.
It does not retrain the models, only evaluates and visualizes the results.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.enhanced_models import EnhancedDualBranchModel, AttentionBasedModel

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
    Load and preprocess the data for evaluation.
    
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

def evaluate_model(model, test_data, model_name):
    """Evaluate a model and return metrics."""
    # Get predictions
    y_pred = model.predict([test_data[0], test_data[1]])
    
    # Calculate metrics
    mae = mean_absolute_error(test_data[2], y_pred)
    rmse = np.sqrt(mean_squared_error(test_data[2], y_pred))
    mae_x = mean_absolute_error(test_data[2][:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(test_data[2][:, 1], y_pred[:, 1])
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAE_X', 'MAE_Y'],
        'Value': [mae, rmse, mae_x, mae_y]
    })
    
    logger.info(f"\n{model_name} Evaluation Metrics:")
    logger.info(f"Overall MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE X: {mae_x:.4f}")
    logger.info(f"MAE Y: {mae_y:.4f}")
    
    return metrics_df, y_pred

def main():
    """Main function to evaluate and visualize both models."""
    # Set paths
    data_dir = Path('Indoor-Localization-main/data')
    model_dir = Path('Indoor-Localization-main/models/enhanced')
    vis_dir = model_dir / 'visualizations'
    
    # Create visualization directory
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    train_data, val_data, test_data = load_and_preprocess_data(data_dir)
    
    # Load trained models
    logger.info("Loading trained models...")
    
    # Load Enhanced Dual Branch Model
    dual_branch_model_path = model_dir / 'enhanced_dual_branch_best.keras'
    if dual_branch_model_path.exists():
        dual_branch_model = tf.keras.models.load_model(dual_branch_model_path)
        logger.info("Enhanced Dual Branch Model loaded successfully")
    else:
        logger.error(f"Model not found: {dual_branch_model_path}")
        return
    
    # Load Attention-Based Model
    attention_model_path = model_dir / 'attention_based_best.keras'
    if attention_model_path.exists():
        attention_model = tf.keras.models.load_model(attention_model_path)
        logger.info("Attention-Based Model loaded successfully")
    else:
        logger.error(f"Model not found: {attention_model_path}")
        return
    
    # Evaluate models
    logger.info("Evaluating models...")
    
    # Evaluate Enhanced Dual Branch Model
    dual_branch_metrics, dual_branch_pred = evaluate_model(
        dual_branch_model, test_data, 'Enhanced Dual Branch'
    )
    
    # Evaluate Attention-Based Model
    attention_metrics, attention_pred = evaluate_model(
        attention_model, test_data, 'Attention-Based'
    )
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
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
    
    # Save metrics to CSV files
    dual_branch_metrics.to_csv(vis_dir / 'enhanced_dual_branch_metrics.csv', index=False)
    attention_metrics.to_csv(vis_dir / 'attention_based_metrics.csv', index=False)
    
    # Save predictions for further analysis
    np.save(vis_dir / 'enhanced_dual_branch_predictions.npy', dual_branch_pred)
    np.save(vis_dir / 'attention_based_predictions.npy', attention_pred)
    np.save(vis_dir / 'test_true_values.npy', test_data[2])
    
    # Compare models
    logger.info("\nModel Comparison:")
    logger.info("\nEnhanced Dual Branch Model Metrics:")
    logger.info(dual_branch_metrics)
    logger.info("\nAttention-Based Model Metrics:")
    logger.info(attention_metrics)
    
    logger.info(f"\nResults and visualizations saved to {vis_dir}")

if __name__ == "__main__":
    main() 