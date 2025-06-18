"""
Advanced Indoor Localization Architectures
=========================================

This module implements three advanced deep neural network architectures for indoor localization,
designed based on analysis of existing models and their performance characteristics.

Architectures:
1. Multi-Scale Inception Network (MSIN) - Addresses frequency domain multi-scale patterns
2. Graph Neural Network with CSI-RSSI Fusion (GNN-Fusion) - Models spatial relationships
3. Temporal Convolutional Network with Attention (TCN-Attention) - Captures temporal dynamics
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

class MultiScaleInceptionNetwork:
    """
    Multi-Scale Inception Network for Indoor Localization
    
    Expected Performance: ~2.8-3.0cm MAE with <150K parameters
    """
    
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2):
        """Initialize the Multi-Scale Inception Network."""
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        self.output_shape = output_shape
        self.model = self._build_model()
        
    def _inception_block(self, x, filters_1x1, filters_3x1, filters_5x1, filters_7x1, filters_pool):
        """Create an inception block with multiple parallel paths."""
        # 1x1 convolution path
        conv1x1 = layers.Conv1D(filters_1x1, 1, padding='same', activation='relu')(x)
        conv1x1 = layers.BatchNormalization()(conv1x1)
        
        # 3x1 convolution path
        conv3x1 = layers.Conv1D(filters_3x1, 1, padding='same', activation='relu')(x)
        conv3x1 = layers.BatchNormalization()(conv3x1)
        conv3x1 = layers.Conv1D(filters_3x1, 3, padding='same', activation='relu')(conv3x1)
        conv3x1 = layers.BatchNormalization()(conv3x1)
        
        # 5x1 convolution path
        conv5x1 = layers.Conv1D(filters_5x1, 1, padding='same', activation='relu')(x)
        conv5x1 = layers.BatchNormalization()(conv5x1)
        conv5x1 = layers.Conv1D(filters_5x1, 5, padding='same', activation='relu')(conv5x1)
        conv5x1 = layers.BatchNormalization()(conv5x1)
        
        # 7x1 convolution path
        conv7x1 = layers.Conv1D(filters_7x1, 1, padding='same', activation='relu')(x)
        conv7x1 = layers.BatchNormalization()(conv7x1)
        conv7x1 = layers.Conv1D(filters_7x1, 7, padding='same', activation='relu')(conv7x1)
        conv7x1 = layers.BatchNormalization()(conv7x1)
        
        # Pooling path
        pool = layers.MaxPooling1D(3, strides=1, padding='same')(x)
        pool = layers.Conv1D(filters_pool, 1, padding='same', activation='relu')(pool)
        pool = layers.BatchNormalization()(pool)
        
        # Concatenate all paths
        output = layers.Concatenate()([conv1x1, conv3x1, conv5x1, conv7x1, pool])
        return output
    
    def _build_model(self):
        """Build the Multi-Scale Inception Network architecture."""
        # CSI Input Branch
        csi_input = layers.Input(shape=self.input_shape_csi, name='csi_input')
        
        # Initial feature extraction
        x = layers.Conv1D(32, 3, padding='same', activation='relu')(csi_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Inception Block 1
        x = self._inception_block(x, 16, 24, 24, 16, 16)
        x = layers.MaxPooling1D(2)(x)
        
        # Inception Block 2
        x = self._inception_block(x, 32, 48, 48, 32, 32)
        x = layers.MaxPooling1D(2)(x)
        
        # Residual connection
        residual = layers.Conv1D(176, 1, padding='same')(x)  # 176 = sum of all filters
        
        # Inception Block 3
        x = self._inception_block(x, 32, 48, 48, 32, 16)
        x = layers.Add()([x, residual])
        
        # Global feature aggregation
        x = layers.GlobalAveragePooling1D()(x)
        
        # RSSI Input Branch (Simplified)
        rssi_input = layers.Input(shape=self.input_shape_rssi, name='rssi_input')
        y = layers.Dense(16, activation='relu')(rssi_input)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.2)(y)
        
        # Feature Fusion
        combined = layers.Concatenate()([x, y])
        
        # Final classification layers
        z = layers.Dense(128, activation='relu')(combined)
        z = layers.BatchNormalization()(z)
        z = layers.Dropout(0.3)(z)
        
        z = layers.Dense(64, activation='relu')(z)
        z = layers.BatchNormalization()(z)
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

class GraphNeuralNetworkFusion:
    """
    Graph Neural Network with CSI-RSSI Fusion
    
    Expected Performance: ~2.5-2.8cm MAE with <200K parameters
    """
    
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2, num_heads=4):
        """Initialize the Graph Neural Network Fusion model."""
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        self.output_shape = output_shape
        self.num_heads = num_heads
        self.model = self._build_model()
        
    def _graph_conv_layer(self, x, filters, activation='relu'):
        """Graph convolution layer for subcarrier relationship modeling."""
        # Learnable adjacency matrix
        adj_matrix = layers.Dense(x.shape[1], activation='softmax')(x)
        adj_matrix = layers.Lambda(lambda x: tf.matmul(x, tf.transpose(x, [0, 2, 1])))(adj_matrix)
        
        # Graph convolution: H' = σ(AHW)
        graph_conv = layers.Dense(filters)(x)
        graph_conv = layers.Lambda(lambda x: tf.matmul(adj_matrix, x))(graph_conv)
        graph_conv = layers.BatchNormalization()(graph_conv)
        graph_conv = layers.Activation(activation)(graph_conv)
        
        return graph_conv, adj_matrix
    
    def _multi_head_attention(self, csi_features, rssi_features):
        """Multi-head attention mechanism for CSI-RSSI fusion."""
        # Project CSI features to query space
        query = layers.Dense(64)(csi_features)
        
        # Project RSSI features to key and value spaces
        key = layers.Dense(64)(rssi_features)
        value = layers.Dense(64)(rssi_features)
        
        # Multi-head attention
        attention_outputs = []
        for head in range(self.num_heads):
            # Scaled dot-product attention
            attention_scores = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
            attention_scores = attention_scores / tf.math.sqrt(64.0)
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            
            # Apply attention weights
            head_output = tf.matmul(attention_weights, value)
            attention_outputs.append(head_output)
        
        # Concatenate all heads
        multi_head_output = tf.concat(attention_outputs, axis=-1)
        return multi_head_output
    
    def _build_model(self):
        """Build the Graph Neural Network Fusion architecture."""
        # CSI Input Branch
        csi_input = layers.Input(shape=self.input_shape_csi, name='csi_input')
        
        # Initial feature extraction
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(csi_input)
        x = layers.BatchNormalization()(x)
        
        # Graph convolution layers
        x, adj1 = self._graph_conv_layer(x, 64)
        x = layers.Dropout(0.2)(x)
        
        x, adj2 = self._graph_conv_layer(x, 128)
        x = layers.Dropout(0.2)(x)
        
        x, adj3 = self._graph_conv_layer(x, 64)
        x = layers.Dropout(0.2)(x)
        
        # Global feature aggregation
        csi_features = layers.GlobalAveragePooling1D()(x)
        
        # RSSI Input Branch
        rssi_input = layers.Input(shape=self.input_shape_rssi, name='rssi_input')
        y = layers.Dense(32, activation='relu')(rssi_input)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.2)(y)
        
        # Multi-head attention fusion
        attended_features = self._multi_head_attention(
            tf.expand_dims(csi_features, axis=1),
            tf.expand_dims(y, axis=1)
        )
        attended_features = layers.Flatten()(attended_features)
        
        # Feature combination
        combined = layers.Concatenate()([csi_features, y, attended_features])
        
        # Final layers with uncertainty estimation
        z = layers.Dense(128, activation='relu')(combined)
        z = layers.BatchNormalization()(z)
        z = layers.Dropout(0.3)(z)
        
        z = layers.Dense(64, activation='relu')(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dropout(0.2)(z)
        
        # Coordinate prediction
        coordinates = layers.Dense(self.output_shape, name='coordinates')(z)
        
        # Uncertainty estimation
        uncertainty = layers.Dense(self.output_shape, activation='exponential', name='uncertainty')(z)
        
        # Create and compile model
        model = Model(inputs=[csi_input, rssi_input], outputs=[coordinates, uncertainty])
        model.compile(
            optimizer='adam',
            loss={
                'coordinates': 'mse',
                'uncertainty': self._uncertainty_loss
            },
            metrics={'coordinates': 'mae'}
        )
        
        return model
    
    def _uncertainty_loss(self, y_true, y_pred):
        """Custom loss function for uncertainty estimation."""
        return tf.reduce_mean(tf.square(y_pred))

class TemporalConvolutionalAttention:
    """
    Temporal Convolutional Network with Attention
    
    Expected Performance: ~2.9-3.1cm MAE with <120K parameters
    """
    
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2, num_blocks=3):
        """Initialize the Temporal Convolutional Attention model."""
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        self.output_shape = output_shape
        self.num_blocks = num_blocks
        self.model = self._build_model()
        
    def _temporal_block(self, x, filters, dilation_rate, kernel_size=3):
        """Temporal convolution block with residual connection."""
        # Causal convolution
        conv = layers.Conv1D(
            filters, kernel_size, 
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Dropout(0.1)(conv)
        
        # Second convolution
        conv = layers.Conv1D(
            filters, kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )(conv)
        conv = layers.BatchNormalization()(conv)
        
        # Residual connection
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same')(x)
        
        output = layers.Add()([x, conv])
        return output
    
    def _temporal_attention(self, x):
        """Temporal attention mechanism."""
        # Self-attention over time dimension
        attention_weights = layers.Dense(1, activation='sigmoid')(x)
        attended = layers.Multiply()([x, attention_weights])
        return attended
    
    def _build_model(self):
        """Build the Temporal Convolutional Attention architecture."""
        # CSI Input Branch
        csi_input = layers.Input(shape=self.input_shape_csi, name='csi_input')
        
        # Initial feature extraction
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(csi_input)
        x = layers.BatchNormalization()(x)
        
        # Temporal convolution blocks with increasing dilation
        for i in range(self.num_blocks):
            dilation_rate = 2 ** i
            x = self._temporal_block(x, 64, dilation_rate)
        
        # Temporal attention
        x = self._temporal_attention(x)
        
        # Global feature aggregation
        csi_features = layers.GlobalAveragePooling1D()(x)
        
        # RSSI Input Branch
        rssi_input = layers.Input(shape=self.input_shape_rssi, name='rssi_input')
        y = layers.Dense(16, activation='relu')(rssi_input)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.2)(y)
        
        # Feature fusion
        combined = layers.Concatenate()([csi_features, y])
        
        # Final layers
        z = layers.Dense(96, activation='relu')(combined)
        z = layers.BatchNormalization()(z)
        z = layers.Dropout(0.3)(z)
        
        z = layers.Dense(48, activation='relu')(z)
        z = layers.BatchNormalization()(z)
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

def train_and_evaluate_advanced_model(model_class, train_data, val_data, test_data, save_dir, model_name):
    """
    Train and evaluate an advanced model with comprehensive analysis.
    
    Args:
        model_class: The model class to instantiate
        train_data: Tuple of (X_train_csi, X_train_rssi, y_train)
        val_data: Tuple of (X_val_csi, X_val_rssi, y_val)
        test_data: Tuple of (X_test_csi, X_test_rssi, y_test)
        save_dir: Directory to save results
        model_name: Name of the model for saving files
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    if model_class == GraphNeuralNetworkFusion:
        model = model_class(
            input_shape_csi=train_data[0].shape[1:],
            input_shape_rssi=train_data[1].shape[1:],
            output_shape=2
        )
    else:
        model = model_class(
            input_shape_csi=train_data[0].shape[1:],
            input_shape_rssi=train_data[1].shape[1:],
            output_shape=2
        )
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
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
            patience=8
        )
    ]
    
    # Train model
    if model_class == GraphNeuralNetworkFusion:
        # Handle uncertainty model
        history = model.model.fit(
            [train_data[0], train_data[1]],
            {'coordinates': train_data[2], 'uncertainty': np.ones_like(train_data[2])},
            validation_data=([val_data[0], val_data[1]], 
                           {'coordinates': val_data[2], 'uncertainty': np.ones_like(val_data[2])}),
            epochs=150,
            batch_size=32,
            callbacks=callbacks
        )
        
        # Evaluate
        test_loss = model.model.evaluate(
            [test_data[0], test_data[1]],
            {'coordinates': test_data[2], 'uncertainty': np.ones_like(test_data[2])}
        )
        
        # Get predictions
        y_pred, uncertainty = model.model.predict([test_data[0], test_data[1]])
    else:
        # Standard model
        history = model.model.fit(
            [train_data[0], train_data[1]],
            train_data[2],
            validation_data=([val_data[0], val_data[1]], val_data[2]),
            epochs=150,
            batch_size=32,
            callbacks=callbacks
        )
        
        # Evaluate
        test_loss, test_mae = model.model.evaluate([test_data[0], test_data[1]], test_data[2])
        
        # Get predictions
        y_pred = model.model.predict([test_data[0], test_data[1]])
        uncertainty = None
    
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
    
    # Save model summary
    with open(save_path / f'{model_name}_summary.txt', 'w') as f:
        model.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    if 'coordinates_mae' in history.history:
        plt.plot(history.history['coordinates_mae'], label='Training MAE')
        plt.plot(history.history['val_coordinates_mae'], label='Validation MAE')
    else:
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.scatter(test_data[2][:, 0], test_data[2][:, 1], c='blue', label='True Locations', alpha=0.6)
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', label='Predicted Locations', alpha=0.6)
    
    for i in range(min(50, len(test_data[2]))):  # Plot first 50 connections
        plt.plot([test_data[2][i, 0], y_pred[i, 0]], 
                [test_data[2][i, 1], y_pred[i, 1]], 
                'k--', alpha=0.2)
    
    plt.title(f'{model_name} - True vs Predicted Locations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / f'{model_name}_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error distribution
    errors = np.sqrt(np.sum((test_data[2] - y_pred) ** 2, axis=1))
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(f'{model_name} - Error Distribution')
    plt.xlabel('Error Distance (cm)')
    plt.ylabel('Count')
    plt.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}cm')
    plt.legend()
    plt.savefig(save_path / f'{model_name}_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\n{model_name} Training Complete:")
    logger.info(f"  Final MAE: {mae:.2f}cm")
    logger.info(f"  Final RMSE: {rmse:.2f}cm")
    logger.info(f"  MAE_X: {mae_x:.2f}cm, MAE_Y: {mae_y:.2f}cm")
    logger.info(f"  Model saved to: {save_path}")
    
    return history, metrics_df, model 