"""
Visualize Retrained Model Results
==============================

This script generates visualizations for the retrained model's performance,
focusing on error distribution and predictions plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    plt.title('True vs Predicted Locations (Retrained Model)')
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
    plt.title('Distribution of Localization Errors (Retrained Model)')
    plt.xlabel('Error Distance (units)')
    plt.ylabel('Count')
    
    # Add mean and median lines
    plt.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
    plt.axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.2f}')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

def main():
    """Main function to generate visualizations."""
    # Set paths
    eval_dir = Path('Indoor-Localization-main/models/retrained/evaluation')
    save_dir = Path('Indoor-Localization-main/models/retrained/visualizations')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions and true values
    y_pred = np.load(eval_dir / 'predictions.npy')
    y_true = np.load(eval_dir / 'y_test.npy')
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_predictions(y_true, y_pred, save_dir / 'predictions.png')
    plot_error_distribution(y_true, y_pred, save_dir / 'error_distribution.png')
    
    logger.info(f"Visualizations saved to {save_dir}")

if __name__ == "__main__":
    main() 