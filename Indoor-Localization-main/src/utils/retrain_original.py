"""
Retrain Original Model
====================

This script retrains the original model structure with the same architecture
but as a fresh training run.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_dir: str):
    """Load and preprocess the data."""
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
    
    # Split data
    X_csi_train, X_csi_test, X_rssi_train, X_rssi_test, y_train, y_test = train_test_split(
        X_csi, X_rssi, locations, test_size=0.2, random_state=42
    )
    
    return (X_csi_train, X_rssi_train, y_train), (X_csi_test, X_rssi_test, y_test)

def create_model(input_shape_csi, input_shape_rssi):
    """Create the original model architecture."""
    # CSI branch
    csi_input = tf.keras.layers.Input(shape=input_shape_csi, name='csi_input')
    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(csi_input)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    csi_branch = tf.keras.layers.Dense(128, activation='relu')(x)
    
    # RSSI branch
    rssi_input = tf.keras.layers.Input(shape=input_shape_rssi, name='rssi_input')
    y = tf.keras.layers.Dense(32, activation='relu')(rssi_input)
    y = tf.keras.layers.Dense(64, activation='relu')(y)
    rssi_branch = tf.keras.layers.Dense(32, activation='relu')(y)
    
    # Combine branches
    combined = tf.keras.layers.Concatenate()([csi_branch, rssi_branch])
    z = tf.keras.layers.Dense(128, activation='relu')(combined)
    z = tf.keras.layers.Dropout(0.3)(z)
    z = tf.keras.layers.Dense(64, activation='relu')(z)
    z = tf.keras.layers.Dropout(0.2)(z)
    output = tf.keras.layers.Dense(2)(z)
    
    model = tf.keras.Model(inputs=[csi_input, rssi_input], outputs=output)
    return model

def train_model(train_data, test_data, save_dir: str):
    """Train the model and save results."""
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = create_model(
        input_shape_csi=train_data[0].shape[1:],
        input_shape_rssi=train_data[1].shape[1:]
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path / 'best_model.keras',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        [train_data[0], train_data[1]],
        train_data[2],
        validation_data=([test_data[0], test_data[1]], test_data[2]),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
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
    plt.close()
    
    return model, history

def main():
    """Main function to run the training pipeline."""
    # Set paths
    data_dir = Path('Indoor-Localization-main/data')
    save_dir = Path('Indoor-Localization-main/models/retrained')
    
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