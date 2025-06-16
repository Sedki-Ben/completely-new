"""
Save Original Model Results
=========================

This script saves the original model's predictions and metrics for comparison
with the retrained model.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_original_results(data_dir: str, save_dir: str):
    """Save original model results for comparison."""
    data_dir = Path(data_dir)
    save_dir = Path(save_dir) / 'evaluation'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    X_test = np.load(data_dir / 'X_test.npy')
    rssi_test = np.load(data_dir / 'rssi_test.npy')
    y_test = np.load(data_dir / 'loc_test.npy')
    
    # Load original model predictions
    model_path = Path('Indoor-Localization-main/Indoor Localization Project/models/saved_models/best_model.keras')
    model = tf.keras.models.load_model(model_path)
    
    # Get predictions
    y_pred = model.predict([X_test, rssi_test])
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAE_X', 'MAE_Y'],
        'Value': [mae, rmse, mae_x, mae_y]
    })
    metrics_df.to_csv(save_dir / 'metrics.csv', index=False)
    
    # Save predictions and true values
    np.save(save_dir / 'predictions.npy', y_pred)
    np.save(save_dir / 'y_test.npy', y_test)
    
    logger.info(f"Original model results saved to {save_dir}")

def main():
    """Main function to save original model results."""
    data_dir = Path('Indoor-Localization-main/data')
    save_dir = Path('Indoor-Localization-main/models/original')
    
    save_original_results(data_dir, save_dir)

if __name__ == "__main__":
    main() 