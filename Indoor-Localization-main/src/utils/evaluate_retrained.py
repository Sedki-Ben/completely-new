"""
Evaluate Retrained Model
=====================

This script evaluates the retrained model and generates visualizations
to analyze its performance.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir: str):
    """Load test data for evaluation."""
    data_dir = Path(data_dir)
    
    # Load CSI data
    csi_features = np.load(data_dir / 'csi_features.npy')
    csi_phase = np.load(data_dir / 'csi_phase.npy')
    
    # Reshape CSI phase data to match features
    if len(csi_phase.shape) == 2:
        csi_phase = csi_phase[..., np.newaxis]  # (samples, time, 1)
    X_csi = np.concatenate([csi_features, csi_phase], axis=-1)
    
    # Load RSSI data
    rssi = np.load(data_dir / 'rssi.npy')
    X_rssi = rssi.reshape(-1, 1)
    
    # Load locations
    locations = np.load(data_dir / 'locations.npy')
    
    # Split data (use the same split as in training)
    _, X_csi_test, _, X_rssi_test, _, y_test = train_test_split(
        X_csi, X_rssi, locations, test_size=0.2, random_state=42
    )
    
    return X_csi_test, X_rssi_test, y_test

def evaluate_model(model_path: str, test_data: tuple, save_dir: str):
    """Evaluate the model and generate visualizations."""
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Get predictions
    y_pred = model.predict([test_data[0], test_data[1]])
    
    # Calculate metrics
    mae = mean_absolute_error(test_data[2], y_pred)
    rmse = np.sqrt(mean_squared_error(test_data[2], y_pred))
    mae_x = mean_absolute_error(test_data[2][:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(test_data[2][:, 1], y_pred[:, 1])
    
    # Log metrics
    logger.info("\nEvaluation Metrics:")
    logger.info(f"Overall MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE X: {mae_x:.4f}")
    logger.info(f"MAE Y: {mae_y:.4f}")
    
    # Create visualizations
    plot_predictions(test_data[2], y_pred, save_path / 'predictions.png')
    plot_error_distribution(test_data[2], y_pred, save_path / 'error_distribution.png')
    plot_spatial_errors(test_data[2], y_pred, save_path / 'spatial_errors.png')
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAE_X', 'MAE_Y'],
        'Value': [mae, rmse, mae_x, mae_y]
    })
    metrics_df.to_csv(save_path / 'metrics.csv', index=False)
    
    # Save predictions and true values for comparison
    np.save(save_path / 'predictions.npy', y_pred)
    np.save(save_path / 'y_test.npy', test_data[2])

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
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
    
    plt.title('True vs Predicted Locations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot error distribution."""
    errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title('Distribution of Localization Errors')
    plt.xlabel('Error Distance (units)')
    plt.ylabel('Count')
    
    # Add mean and median lines
    plt.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
    plt.axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.2f}')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

def plot_spatial_errors(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot spatial distribution of errors."""
    errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(y_true[:, 0], y_true[:, 1], c=errors, cmap='viridis')
    plt.colorbar(scatter, label='Error Distance')
    plt.title('Spatial Distribution of Localization Errors')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    """Main function to run the evaluation pipeline."""
    # Set paths
    data_dir = Path('Indoor-Localization-main/data')
    model_path = Path('Indoor-Localization-main/models/retrained/best_model.keras')
    save_dir = Path('Indoor-Localization-main/models/retrained/evaluation')
    
    # Load test data
    logger.info("Loading test data...")
    test_data = load_data(data_dir)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluate_model(model_path, test_data, save_dir)
    
    logger.info(f"Evaluation results saved to {save_dir}")

if __name__ == "__main__":
    main() 