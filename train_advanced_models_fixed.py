"""
Fixed Advanced Indoor Localization Model Training
================================================

This script contains all three advanced DNN architectures and training logic in one file.
Fixed for compatibility with TensorFlow 2.13.0 and Python 3.10.

Architectures:
1. Multi-Scale Inception Network (MSIN)
2. Graph Neural Network with CSI-RSSI Fusion (GNN-Fusion)
3. Temporal Convolutional Network with Attention (TCN-Attention)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

# Import TensorFlow and other dependencies
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model Architecture Definitions
class MultiScaleInceptionNetwork:
    """Multi-Scale Inception Network for Indoor Localization"""
    
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2):
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
        residual = layers.Conv1D(176, 1, padding='same')(x)
        
        # Inception Block 3
        x = self._inception_block(x, 32, 48, 48, 32, 16)
        x = layers.Add()([x, residual])
        
        # Global feature aggregation
        x = layers.GlobalAveragePooling1D()(x)
        
        # RSSI Input Branch
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
    """Graph Neural Network with CSI-RSSI Fusion - Fixed Version"""
    
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2, num_heads=4):
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        self.output_shape = output_shape
        self.num_heads = num_heads
        self.model = self._build_model()
        
    def _graph_conv_layer(self, x, filters, activation='relu'):
        """Graph convolution layer for subcarrier relationship modeling."""
        # Learnable adjacency matrix using Keras layers only
        adj_matrix = layers.Dense(x.shape[1], activation='softmax')(x)
        
        # Use Lambda layer for matrix multiplication
        def matmul_transpose(inputs):
            x, adj = inputs
            return tf.matmul(adj, tf.transpose(adj, [0, 2, 1]))
        
        adj_matrix = layers.Lambda(matmul_transpose)([x, adj_matrix])
        
        # Graph convolution: H' = σ(AHW)
        graph_conv = layers.Dense(filters)(x)
        
        def apply_adjacency(inputs):
            adj, features = inputs
            return tf.matmul(adj, features)
        
        graph_conv = layers.Lambda(apply_adjacency)([adj_matrix, graph_conv])
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
        
        # Multi-head attention using Lambda layers
        def attention_mechanism(inputs):
            q, k, v = inputs
            # Scaled dot-product attention
            attention_scores = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
            attention_scores = attention_scores / tf.math.sqrt(64.0)
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            # Apply attention weights
            return tf.matmul(attention_weights, v)
        
        attention_outputs = []
        for head in range(self.num_heads):
            head_output = layers.Lambda(attention_mechanism)([query, key, value])
            attention_outputs.append(head_output)
        
        # Concatenate all heads
        multi_head_output = layers.Concatenate()(attention_outputs)
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
    """Temporal Convolutional Network with Attention"""
    
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2, num_blocks=3):
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

def load_and_preprocess_data(data_path='demo_data.csv'):
    """Load and preprocess the data for training."""
    logger.info("Loading and preprocessing data...")
    
    # Load data
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
    else:
        # Create synthetic data for demonstration
        logger.info("Creating synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic CSI data (52 subcarriers)
        csi_data = np.random.normal(-42, 2, (n_samples, 52))
        
        # Generate synthetic RSSI data
        rssi_data = np.random.normal(-65, 5, (n_samples, 1))
        
        # Generate synthetic coordinates (x, y)
        coordinates = np.random.uniform(0, 10, (n_samples, 2))
        
        # Combine into DataFrame
        df = pd.DataFrame(csi_data, columns=[f'csi_{i+1}' for i in range(52)])
        df['rssi'] = rssi_data
        df['x'] = coordinates[:, 0]
        df['y'] = coordinates[:, 1]
        
        # Save synthetic data
        df.to_csv(data_path, index=False)
        logger.info(f"Created synthetic data with shape: {df.shape}")
    
    # Extract features and labels
    csi_features = df.iloc[:, :52].values  # First 52 columns are CSI
    rssi_features = df.iloc[:, 52:53].values  # Column 53 is RSSI
    labels = df.iloc[:, 53:].values  # Last 2 columns are x, y coordinates
    
    # Normalize CSI data
    csi_scaler = StandardScaler()
    csi_features_normalized = csi_scaler.fit_transform(csi_features)
    
    # Normalize RSSI data
    rssi_scaler = MinMaxScaler(feature_range=(0, 1))
    rssi_features_normalized = rssi_scaler.fit_transform(rssi_features)
    
    # Reshape CSI data for 1D convolution (samples, timesteps, features)
    csi_features_reshaped = csi_features_normalized.reshape(csi_features_normalized.shape[0], -1, 1)
    
    # Split data
    X_csi_temp, X_csi_test, X_rssi_temp, X_rssi_test, y_temp, y_test = train_test_split(
        csi_features_reshaped, rssi_features_normalized, labels, 
        test_size=0.2, random_state=42
    )
    
    X_csi_train, X_csi_val, X_rssi_train, X_rssi_val, y_train, y_val = train_test_split(
        X_csi_temp, X_rssi_temp, y_temp, 
        test_size=0.2, random_state=42
    )
    
    # Create data tuples
    train_data = (X_csi_train, X_rssi_train, y_train)
    val_data = (X_csi_val, X_rssi_val, y_val)
    test_data = (X_csi_test, X_rssi_test, y_test)
    
    logger.info(f"Data split complete:")
    logger.info(f"  Training: {X_csi_train.shape[0]} samples")
    logger.info(f"  Validation: {X_csi_val.shape[0]} samples")
    logger.info(f"  Test: {X_csi_test.shape[0]} samples")
    
    return train_data, val_data, test_data

def train_and_evaluate_model(model_class, train_data, val_data, test_data, save_dir, model_name):
    """Train and evaluate a model with comprehensive analysis."""
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
    
    # Setup callbacks - FIXED: Use .h5 format instead of .keras
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path / f'{model_name}_best.h5'),  # FIXED: Use .h5 format
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]
    
    # Train model - FIXED: Reduce epochs for faster training
    if model_class == GraphNeuralNetworkFusion:
        # Handle uncertainty model
        history = model.model.fit(
            [train_data[0], train_data[1]],
            {'coordinates': train_data[2], 'uncertainty': np.ones_like(train_data[2])},
            validation_data=([val_data[0], val_data[1]], 
                           {'coordinates': val_data[2], 'uncertainty': np.ones_like(val_data[2])}),
            epochs=50,  # FIXED: Reduced from 150 to 50
            batch_size=16,  # FIXED: Reduced from 32 to 16
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
            epochs=50,  # FIXED: Reduced from 150 to 50
            batch_size=16,  # FIXED: Reduced from 32 to 16
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
    
    # Save metrics - FIXED: Convert Path to string
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAE_X', 'MAE_Y'],
        'Value': [mae, rmse, mae_x, mae_y]
    })
    metrics_df.to_csv(str(save_path / f'{model_name}_metrics.csv'), index=False)  # FIXED: Convert to string
    
    # Save model summary - FIXED: Convert Path to string
    with open(str(save_path / f'{model_name}_summary.txt'), 'w') as f:  # FIXED: Convert to string
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
    plt.savefig(str(save_path / f'{model_name}_results.png'), dpi=300, bbox_inches='tight')  # FIXED: Convert to string
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
    plt.savefig(str(save_path / f'{model_name}_error_distribution.png'), dpi=300, bbox_inches='tight')  # FIXED: Convert to string
    plt.close()
    
    logger.info(f"\n{model_name} Training Complete:")
    logger.info(f"  Final MAE: {mae:.2f}cm")
    logger.info(f"  Final RMSE: {rmse:.2f}cm")
    logger.info(f"  MAE_X: {mae_x:.2f}cm, MAE_Y: {mae_y:.2f}cm")
    logger.info(f"  Model saved to: {save_path}")
    
    return history, metrics_df, model

def main():
    """Main training function."""
    logger.info("Starting Advanced Indoor Localization Model Training")
    logger.info("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and preprocess data
    train_data, val_data, test_data = load_and_preprocess_data()
    
    # Define models to train
    models_to_train = [
        (MultiScaleInceptionNetwork, 'MSIN'),
        (GraphNeuralNetworkFusion, 'GNN_Fusion'),
        (TemporalConvolutionalAttention, 'TCN_Attention')
    ]
    
    # Store results
    all_results = {}
    save_dir = 'models/advanced'
    
    for model_class, model_name in models_to_train:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'='*50}")
        
        try:
            # Create model-specific save directory
            model_save_dir = Path(save_dir) / model_name
            model_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Train and evaluate model
            history, metrics_df, model = train_and_evaluate_model(
                model_class, train_data, val_data, test_data, 
                model_save_dir, model_name
            )
            
            # Store results
            all_results[model_name] = {
                'history': history,
                'metrics': metrics_df,
                'model': model,
                'save_dir': model_save_dir
            }
            
            logger.info(f"{model_name} training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    # Create comparison report
    if all_results:
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        # Print summary
        for model_name, results in all_results.items():
            mae = results['metrics'][results['metrics']['Metric'] == 'MAE']['Value'].iloc[0]
            logger.info(f"{model_name}: MAE = {mae:.2f}cm")
        
        # Find best model
        best_model = min(all_results.items(), 
                        key=lambda x: x[1]['metrics'][x[1]['metrics']['Metric'] == 'MAE']['Value'].iloc[0])
        logger.info(f"\n🏆 BEST MODEL: {best_model[0]} with MAE: {best_model[1]['metrics'][best_model[1]['metrics']['Metric'] == 'MAE']['Value'].iloc[0]:.2f}cm")
        
        logger.info(f"\nAll models saved to: {save_dir}/")
        logger.info("You can now use these trained models for validation, testing, and evaluation!")
        
    else:
        logger.error("No models were trained successfully!")

if __name__ == "__main__":
    main() 