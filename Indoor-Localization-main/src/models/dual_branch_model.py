"""
Dual Branch Indoor Localization Model
====================================

This module implements a dual-branch neural network architecture for indoor localization.
The model uses two separate branches to process CSI and RSSI data independently before
combining them for final position prediction.

Architecture:
- Branch 1 (CSI): Processes CSI data through convolutional layers
- Branch 2 (RSSI): Processes RSSI data through dense layers
- Combined: Merges features from both branches for final prediction

The model is designed to:
1. Reduce overfitting through separate feature extraction
2. Better handle the different characteristics of CSI and RSSI data
3. Provide more interpretable results through branch-specific analysis
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DualBranchModel:
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2):
        """
        Initialize the dual branch model.
        
        Args:
            input_shape_csi: Shape of CSI input data
            input_shape_rssi: Shape of RSSI input data
            output_shape: Number of output coordinates (default: 2 for x,y)
        """
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        self.output_shape = output_shape
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the dual branch model architecture."""
        # CSI Branch
        csi_input = layers.Input(shape=self.input_shape_csi, name='csi_input')
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(csi_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        csi_output = layers.Dense(128, activation='relu')(x)
        
        # RSSI Branch
        rssi_input = layers.Input(shape=self.input_shape_rssi, name='rssi_input')
        y = layers.Dense(64, activation='relu')(rssi_input)
        y = layers.BatchNormalization()(y)
        y = layers.Dense(128, activation='relu')(y)
        y = layers.BatchNormalization()(y)
        rssi_output = layers.Dense(64, activation='relu')(y)
        
        # Combine branches
        combined = layers.Concatenate()([csi_output, rssi_output])
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(self.output_shape, name='output')(x)
        
        # Create model
        model = Model(inputs=[csi_input, rssi_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, train_data, validation_data, epochs=100, batch_size=32, 
              save_dir='Indoor-Localization-main/models/dual_branch'):
        """
        Train the model with early stopping and model checkpointing.
        
        Args:
            train_data: Tuple of (X_train_csi, X_train_rssi, y_train)
            validation_data: Tuple of (X_val_csi, X_val_rssi, y_val)
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            save_dir: Directory to save model checkpoints
        """
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Setup callbacks
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
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Train model
        history = self.model.fit(
            [train_data[0], train_data[1]],
            train_data[2],
            validation_data=([validation_data[0], validation_data[1]], validation_data[2]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Tuple of (X_test_csi, X_test_rssi, y_test)
        """
        return self.model.evaluate(
            [test_data[0], test_data[1]],
            test_data[2]
        )
    
    def predict(self, data):
        """
        Make predictions using the model.
        
        Args:
            data: Tuple of (X_csi, X_rssi)
        """
        return self.model.predict([data[0], data[1]])
    
    def save(self, path):
        """Save the model to disk."""
        self.model.save(path)
    
    @classmethod
    def load(cls, path):
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(path)
        return model 