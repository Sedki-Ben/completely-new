"""
Model Evaluation and Visualization Module
=======================================

This module handles the evaluation of the trained model and visualization of results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir: str) -> tuple:
    """Load test data and model."""
    # Load test data
    X_test = np.load(Path(data_dir) / 'X_test.npy')
    y_test = np.load(Path(data_dir) / 'loc_test.npy')
    rssi_test = np.load(Path(data_dir) / 'rssi_test.npy')
    
    return X_test, y_test, rssi_test

def evaluate_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray, 
                  rssi_test: np.ndarray = None) -> dict:
    """Evaluate model performance."""
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Prepare test data
    test_data = [X_test, rssi_test] if rssi_test is not None else X_test
    
    # Get predictions
    y_pred = model.predict(test_data)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAE_X': mae_x,
        'MAE_Y': mae_y
    }
    
    return metrics, y_pred

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
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
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
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
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    """Main function for evaluation and visualization."""
    # Paths
    data_dir = Path('Indoor-Localization-main/data')
    model_path = Path('Indoor-Localization-main/Indoor Localization Project/models/saved_models/best_model.keras')
    output_dir = Path('Indoor-Localization-main/visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading test data...")
    X_test, y_test, rssi_test = load_data(data_dir)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics, y_pred = evaluate_model(model_path, X_test, y_test, rssi_test)
    
    # Print metrics
    logger.info("\nEvaluation Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    plot_predictions(y_test, y_pred, 
                    save_path=output_dir / 'predictions.png')
    plot_error_distribution(y_test, y_pred, 
                          save_path=output_dir / 'error_distribution.png')

if __name__ == "__main__":
    main() 