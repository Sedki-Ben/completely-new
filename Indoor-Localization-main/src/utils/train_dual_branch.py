"""
Training Script for Dual Branch Indoor Localization Model
======================================================

This script handles the training pipeline for the dual branch model, including:
1. Data loading and preprocessing
2. Feature extraction for CSI and RSSI
3. Model training and validation
4. Results visualization and saving
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.dual_branch_model import DualBranchModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_dir: str):
    """
    Load and preprocess the data for the dual branch model.
    
    Args:
        data_dir: Path to the data directory
    
    Returns:
        Tuple of (X_csi, X_rssi, y) for training and testing
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
        csi_phase = csi_phase[..., np.newaxis]  # (samples, time, 1)
    
    # Combine CSI features
    X_csi = np.concatenate([csi_features, csi_phase], axis=-1)
    
    # Reshape RSSI data
    X_rssi = rssi.reshape(-1, 1)
    
    # Split data
    X_csi_train, X_csi_test, X_rssi_train, X_rssi_test, y_train, y_test = train_test_split(
        X_csi, X_rssi, locations, test_size=0.2, random_state=42
    )
    
    # Scale the data
    csi_scaler = StandardScaler()
    rssi_scaler = StandardScaler()
    
    # Reshape CSI data for scaling
    X_csi_train_reshaped = X_csi_train.reshape(-1, X_csi_train.shape[-1])
    X_csi_test_reshaped = X_csi_test.reshape(-1, X_csi_test.shape[-1])
    
    # Scale CSI data
    X_csi_train_scaled = csi_scaler.fit_transform(X_csi_train_reshaped)
    X_csi_test_scaled = csi_scaler.transform(X_csi_test_reshaped)
    
    # Reshape back to original shape
    X_csi_train = X_csi_train_scaled.reshape(X_csi_train.shape)
    X_csi_test = X_csi_test_scaled.reshape(X_csi_test.shape)
    
    # Scale RSSI data
    X_rssi_train = rssi_scaler.fit_transform(X_rssi_train)
    X_rssi_test = rssi_scaler.transform(X_rssi_test)
    
    return (X_csi_train, X_rssi_train, y_train), (X_csi_test, X_rssi_test, y_test)

def train_model(train_data, test_data, save_dir: str):
    """
    Train the dual branch model and save results.
    
    Args:
        train_data: Tuple of (X_csi_train, X_rssi_train, y_train)
        test_data: Tuple of (X_csi_test, X_rssi_test, y_test)
        save_dir: Directory to save model and results
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = DualBranchModel(
        input_shape_csi=train_data[0].shape[1:],
        input_shape_rssi=train_data[1].shape[1:]
    )
    
    # Train model
    history = model.train(
        train_data=train_data,
        validation_data=test_data,
        save_dir=save_path
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(test_data)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png')
    
    return model, history

def main():
    """Main function to run the training pipeline."""
    # Set paths
    data_dir = Path('Indoor-Localization-main/data')
    save_dir = Path('Indoor-Localization-main/models/dual_branch')
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    train_data, test_data = load_and_preprocess_data(data_dir)
    
    # Train model
    logger.info("Training model...")
    model, history = train_model(train_data, test_data, save_dir)
    
    # Save final model
    model.save(save_dir / 'final_model.keras')
    logger.info(f"Model saved to {save_dir}")

if __name__ == "__main__":
    main() 