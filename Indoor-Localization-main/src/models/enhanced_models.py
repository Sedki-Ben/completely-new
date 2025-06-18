"""
Enhanced Indoor Localization Models
=================================

This module implements two enhanced deep neural network architectures for indoor localization:
1. Enhanced Dual Branch Model with residual connections and attention
2. Attention-Based Model with cross-modal learning

Both models are designed to improve upon the original and dual branch architectures.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedDualBranchModel:
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2):
        """Initialize the enhanced dual branch model."""
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        self.output_shape = output_shape
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the enhanced dual branch model architecture."""
        # CSI Branch
        csi_input = layers.Input(shape=self.input_shape_csi, name='csi_input')
        
        # Progressive feature extraction with residual connections
        x = layers.Conv1D(64, 3, padding='same')(csi_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Residual block 1
        residual = x
        x = layers.Conv1D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.MaxPooling1D(2)(x)
        
        # RSSI Branch with attention
        rssi_input = layers.Input(shape=self.input_shape_rssi, name='rssi_input')
        y = layers.Dense(32)(rssi_input)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        
        # Feature fusion with attention
        csi_features = layers.GlobalAveragePooling1D()(x)
        attention = layers.Dense(1, activation='sigmoid')(csi_features)
        csi_features = layers.Multiply()([csi_features, attention])
        
        # Combine with attention
        combined = layers.Concatenate()([csi_features, y])
        
        # Final layers with skip connections
        z = layers.Dense(256)(combined)
        z = layers.BatchNormalization()(z)
        z = layers.Activation('relu')(z)
        z = layers.Dropout(0.3)(z)
        
        z = layers.Dense(128)(z)
        z = layers.BatchNormalization()(z)
        z = layers.Activation('relu')(z)
        z = layers.Dropout(0.2)(z)
        
        output = layers.Dense(self.output_shape)(z)
        
        # Create and compile model
        model = Model(inputs=[csi_input, rssi_input], outputs=output)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model

class AttentionBasedModel:
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2):
        """Initialize the attention-based model."""
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        self.output_shape = output_shape
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the attention-based model architecture."""
        # CSI input with attention
        csi_input = layers.Input(shape=self.input_shape_csi, name='csi_input')
        
        # Spatial attention
        x = layers.Conv1D(64, 3, padding='same')(csi_input)
        attention = layers.Conv1D(1, 1, activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        
        # Feature extraction
        x = layers.Conv1D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)
        
        # RSSI with attention
        rssi_input = layers.Input(shape=self.input_shape_rssi, name='rssi_input')
        y = layers.Dense(32)(rssi_input)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        
        # Cross-attention between CSI and RSSI
        csi_features = layers.GlobalAveragePooling1D()(x)
        attention_weights = layers.Dense(32, activation='softmax')(csi_features)
        y = layers.Multiply()([y, attention_weights])
        
        # Final processing
        combined = layers.Concatenate()([csi_features, y])
        z = layers.Dense(256)(combined)
        z = layers.BatchNormalization()(z)
        z = layers.Activation('relu')(z)
        z = layers.Dropout(0.3)(z)
        
        output = layers.Dense(self.output_shape)(z)
        
        # Create and compile model
        model = Model(inputs=[csi_input, rssi_input], outputs=output)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model

def train_and_evaluate_model(model, train_data, val_data, test_data, save_dir, model_name):
    """
    Train and evaluate a model with visualization of results.
    
    Args:
        model: The model to train
        train_data: Tuple of (X_train_csi, X_train_rssi, y_train)
        val_data: Tuple of (X_val_csi, X_val_rssi, y_val)
        test_data: Tuple of (X_test_csi, X_test_rssi, y_test)
        save_dir: Directory to save results
        model_name: Name of the model for saving files
    """
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
            filepath=save_path / f'{model_name}_best.keras',
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
    history = model.fit(
        [train_data[0], train_data[1]],
        train_data[2],
        validation_data=([val_data[0], val_data[1]], val_data[2]),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(
        [test_data[0], test_data[1]],
        test_data[2]
    )
    
    # Get predictions
    y_pred = model.predict([test_data[0], test_data[1]])
    
    # Calculate metrics
    mae = mean_absolute_error(test_data[2], y_pred)
    rmse = np.sqrt(mean_squared_error(test_data[2], y_pred))
    mae_x = mean_absolute_error(test_data[2][:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(test_data[2][:, 1], y_pred[:, 1])
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAE_X', 'MAE_Y'],
        'Value': [mae, rmse, mae_x, mae_y]
    })
    metrics_df.to_csv(save_path / f'{model_name}_metrics.csv', index=False)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
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
    
    # Plot predictions
    plt.figure(figsize=(10, 8))
    plt.scatter(test_data[2][:, 0], test_data[2][:, 1], c='blue', label='True Locations', alpha=0.6)
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', label='Predicted Locations', alpha=0.6)
    
    for i in range(len(test_data[2])):
        plt.plot([test_data[2][i, 0], y_pred[i, 0]], 
                [test_data[2][i, 1], y_pred[i, 1]], 
                'k--', alpha=0.2)
    
    plt.title(f'{model_name} - True vs Predicted Locations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / f'{model_name}_predictions.png')
    plt.close()
    
    # Plot error distribution
    errors = np.sqrt(np.sum((test_data[2] - y_pred) ** 2, axis=1))
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(f'{model_name} - Error Distribution')
    plt.xlabel('Error Distance')
    plt.ylabel('Count')
    plt.savefig(save_path / f'{model_name}_error_distribution.png')
    plt.close()
    
    return history, metrics_df 