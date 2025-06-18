#!/usr/bin/env python3
"""
Real Data Training for Advanced Indoor Localization Models
=========================================================

This script trains all three advanced DNN architectures on the REAL project data
from Indoor-Localization-main/Data/Amp+Phase_Data, ensuring fair comparison with MSIN.

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
import re
import ast

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
        # CSI Input Branch (104 features: 52 amplitudes + 52 phases)
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
    """Graph Neural Network with CSI-RSSI Fusion - Real Data Version"""
    
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
        """Build the Graph Neural Network with CSI-RSSI Fusion architecture."""
        # CSI Input Branch (104 features)
        csi_input = layers.Input(shape=self.input_shape_csi, name='csi_input')
        
        # Graph convolution layers
        x, adj1 = self._graph_conv_layer(csi_input, 64)
        x = layers.Dropout(0.2)(x)
        
        x, adj2 = self._graph_conv_layer(x, 128)
        x = layers.Dropout(0.2)(x)
        
        x, adj3 = self._graph_conv_layer(x, 64)
        x = layers.Dropout(0.2)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # RSSI Input Branch
        rssi_input = layers.Input(shape=self.input_shape_rssi, name='rssi_input')
        y = layers.Dense(32, activation='relu')(rssi_input)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.2)(y)
        
        y = layers.Dense(16, activation='relu')(y)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.2)(y)
        
        # Multi-head attention fusion
        attention_output = self._multi_head_attention(x, y)
        
        # Feature fusion
        combined = layers.Concatenate()([x, y, attention_output])
        
        # Final layers
        z = layers.Dense(128, activation='relu')(combined)
        z = layers.BatchNormalization()(z)
        z = layers.Dropout(0.3)(z)
        
        z = layers.Dense(64, activation='relu')(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dropout(0.2)(z)
        
        # Output with uncertainty estimation
        coordinates = layers.Dense(self.output_shape, name='coordinates')(z)
        uncertainty = layers.Dense(self.output_shape, activation='exponential', name='uncertainty')(z)
        
        # Create and compile model
        model = Model(inputs=[csi_input, rssi_input], outputs=[coordinates, uncertainty])
        model.compile(
            optimizer='adam',
            loss={
                'coordinates': 'mse',
                'uncertainty': self._uncertainty_loss
            },
            metrics={
                'coordinates': 'mae'
            }
        )
        
        return model
    
    def _uncertainty_loss(self, y_true, y_pred):
        """Custom loss function for uncertainty estimation."""
        return tf.reduce_mean(tf.exp(-y_pred) * tf.square(y_true - y_pred) + y_pred)

class TemporalConvolutionalAttention:
    """Temporal Convolutional Network with Attention - Real Data Version"""
    
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2, num_blocks=3):
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        self.output_shape = output_shape
        self.num_blocks = num_blocks
        self.model = self._build_model()
        
    def _temporal_block(self, x, filters, dilation_rate, kernel_size=3):
        """Temporal convolution block with residual connection."""
        # First convolution
        conv1 = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
                             padding='same', activation='relu')(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Dropout(0.2)(conv1)
        
        # Second convolution
        conv2 = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
                             padding='same', activation='relu')(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Dropout(0.2)(conv2)
        
        # Residual connection
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same')(x)
        
        return layers.Add()([x, conv2])
    
    def _temporal_attention(self, x):
        """Temporal attention mechanism."""
        # Self-attention
        attention = layers.Dense(x.shape[1], activation='softmax')(x)
        attended = layers.Multiply()([x, attention])
        return attended
    
    def _build_model(self):
        """Build the Temporal Convolutional Network with Attention architecture."""
        # CSI Input Branch (104 features)
        csi_input = layers.Input(shape=self.input_shape_csi, name='csi_input')
        
        # Initial convolution
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(csi_input)
        x = layers.BatchNormalization()(x)
        
        # Temporal blocks with increasing dilation rates
        for i in range(self.num_blocks):
            dilation_rate = 2 ** i
            x = self._temporal_block(x, 64, dilation_rate)
        
        # Temporal attention
        x = self._temporal_attention(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # RSSI Input Branch
        rssi_input = layers.Input(shape=self.input_shape_rssi, name='rssi_input')
        y = layers.Dense(32, activation='relu')(rssi_input)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.2)(y)
        
        # Feature fusion
        combined = layers.Concatenate()([x, y])
        
        # Final layers
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

def load_real_data(data_dir='Indoor-Localization-main/Data/Amp+Phase_Data'):
    """Load and preprocess the REAL project data."""
    logger.info("Loading REAL project data...")
    
    data_path = Path(data_dir)
    all_data = []
    
    # Process each file in the data directory
    for file_path in data_path.glob("*.csv"):
        match = re.match(r'([\d.]+),([\d.]+)', file_path.name)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                try:
                    amplitudes = np.array(ast.literal_eval(row['amplitudes'].replace('np.float64(', '').replace(')', '')), dtype=np.float32)
                    phases = np.array(ast.literal_eval(row['phases'].replace('np.float64(', '').replace(')', '')), dtype=np.float32)
                    rssi = float(row['rssi'])
                    
                    if len(amplitudes) == 52 and len(phases) == 52:
                        csi_features = np.concatenate([amplitudes, phases])  # 104 features
                        all_data.append({
                            'x': x, 'y': y, 'rssi': rssi, 'csi_features': csi_features
                        })
                except:
                    continue
    
    logger.info(f"Loaded {len(all_data)} samples from real data")
    
    # Extract features and labels
    csi_features = np.array([sample['csi_features'] for sample in all_data])
    rssi_features = np.array([[sample['rssi'] for sample in all_data]]).T
    coordinates = np.array([[sample['x'], sample['y']] for sample in all_data])
    
    # Normalize features
    csi_scaler = StandardScaler()
    rssi_scaler = StandardScaler()
    coords_scaler = MinMaxScaler()
    
    csi_normalized = csi_scaler.fit_transform(csi_features)
    rssi_normalized = rssi_scaler.fit_transform(rssi_features)
    coords_normalized = coords_scaler.fit_transform(coordinates)
    
    # Reshape CSI data for 1D convolution (samples, timesteps, features)
    csi_reshaped = csi_normalized.reshape(csi_normalized.shape[0], -1, 1)
    
    # Split data
    X_csi_temp, X_csi_test, X_rssi_temp, X_rssi_test, y_temp, y_test = train_test_split(
        csi_reshaped, rssi_normalized, coords_normalized, 
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
            filepath=str(save_path / f'{model_name}_best.h5'),
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
            epochs=100,
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
            epochs=100,
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
    metrics_df.to_csv(str(save_path / f'{model_name}_metrics.csv'), index=False)
    
    # Save model summary
    with open(str(save_path / f'{model_name}_summary.txt'), 'w') as f:
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
    
    for i in range(min(50, len(test_data[2]))):
        plt.plot([test_data[2][i, 0], y_pred[i, 0]], 
                [test_data[2][i, 1], y_pred[i, 1]], 
                'k--', alpha=0.2)
    
    plt.title(f'{model_name} - True vs Predicted Locations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(str(save_path / f'{model_name}_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\n{model_name} Training Complete:")
    logger.info(f"  Final MAE: {mae:.4f} meters ({mae*100:.2f} cm)")
    logger.info(f"  Final RMSE: {rmse:.4f} meters ({rmse*100:.2f} cm)")
    logger.info(f"  MAE_X: {mae_x:.4f} meters ({mae_x*100:.2f} cm)")
    logger.info(f"  MAE_Y: {mae_y:.4f} meters ({mae_y*100:.2f} cm)")
    logger.info(f"  Model saved to: {save_path}")
    
    return history, metrics_df, model

def main():
    """Main training function."""
    logger.info("Starting Advanced Indoor Localization Model Training on REAL DATA")
    logger.info("="*70)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and preprocess REAL data
    train_data, val_data, test_data = load_real_data()
    
    # Define models to train
    models_to_train = [
        (MultiScaleInceptionNetwork, 'MSIN_Real'),
        (GraphNeuralNetworkFusion, 'GNN_Fusion_Real'),
        (TemporalConvolutionalAttention, 'TCN_Attention_Real')
    ]
    
    # Store results
    all_results = {}
    save_dir = 'models/real_data'
    
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
        logger.info("\n" + "="*70)
        logger.info("REAL DATA TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        
        # Print summary
        for model_name, results in all_results.items():
            mae = results['metrics'][results['metrics']['Metric'] == 'MAE']['Value'].iloc[0]
            logger.info(f"{model_name}: MAE = {mae:.4f} meters ({mae*100:.2f} cm)")
        
        # Find best model
        best_model = min(all_results.items(), 
                        key=lambda x: x[1]['metrics'][x[1]['metrics']['Metric'] == 'MAE']['Value'].iloc[0])
        best_mae = best_model[1]['metrics'][best_model[1]['metrics']['Metric'] == 'MAE']['Value'].iloc[0]
        logger.info(f"\n🏆 BEST MODEL: {best_model[0]} with MAE: {best_mae:.4f} meters ({best_mae*100:.2f} cm)")
        
        logger.info(f"\nAll models saved to: {save_dir}/")
        logger.info("You can now use these trained models for validation, testing, and evaluation!")
        
    else:
        logger.error("No models were trained successfully!")

if __name__ == "__main__":
    main() 