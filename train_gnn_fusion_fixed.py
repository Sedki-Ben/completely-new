#!/usr/bin/env python3
"""
Fixed GNN_Fusion Training for Real Data
=======================================

This script trains the GNN_Fusion model on the REAL project data
with proper input shape handling for 104 features (52 amplitudes + 52 phases).
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

class GraphNeuralNetworkFusionFixed:
    """Graph Neural Network with CSI-RSSI Fusion - Fixed for Real Data"""
    
    def __init__(self, input_shape_csi, input_shape_rssi, output_shape=2, num_heads=4):
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        self.output_shape = output_shape
        self.num_heads = num_heads
        self.model = self._build_model()
        
    def _graph_conv_layer(self, x, filters, activation='relu'):
        """Graph convolution layer for subcarrier relationship modeling."""
        # Learnable adjacency matrix - simplified for real data
        adj_matrix = layers.Dense(x.shape[1], activation='softmax')(x)
        
        # Graph convolution: H' = σ(AHW)
        graph_conv = layers.Dense(filters)(x)
        
        # Apply adjacency matrix
        def apply_adjacency(inputs):
            adj, features = inputs
            return tf.matmul(adj, features)
        
        graph_conv = layers.Lambda(apply_adjacency)([adj_matrix, graph_conv])
        graph_conv = layers.BatchNormalization()(graph_conv)
        graph_conv = layers.Activation(activation)(graph_conv)
        
        return graph_conv
    
    def _multi_head_attention(self, csi_features, rssi_features):
        """Multi-head attention mechanism for CSI-RSSI fusion."""
        # Project CSI features to query space
        query = layers.Dense(64)(csi_features)
        
        # Project RSSI features to key and value spaces
        key = layers.Dense(64)(rssi_features)
        value = layers.Dense(64)(rssi_features)
        
        # Multi-head attention
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
        # CSI Input Branch (104 features: 52 amplitudes + 52 phases)
        csi_input = layers.Input(shape=self.input_shape_csi, name='csi_input')
        
        # Graph convolution layers
        x = self._graph_conv_layer(csi_input, 64)
        x = layers.Dropout(0.2)(x)
        
        x = self._graph_conv_layer(x, 128)
        x = layers.Dropout(0.2)(x)
        
        x = self._graph_conv_layer(x, 64)
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

def load_real_data(data_dir='Indoor-Localization-main/Data/Amp+Phase_Data'):
    """Load and preprocess the REAL project data."""
    logger.info("Loading REAL project data for GNN_Fusion...")
    
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
    logger.info(f"  CSI shape: {X_csi_train.shape}")
    logger.info(f"  RSSI shape: {X_rssi_train.shape}")
    
    return train_data, val_data, test_data

def train_gnn_fusion(train_data, val_data, test_data, save_dir='models/real_data/GNN_Fusion_Fixed'):
    """Train and evaluate the GNN_Fusion model."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing GNN_Fusion model...")
    
    # Initialize model
    model = GraphNeuralNetworkFusionFixed(
        input_shape_csi=train_data[0].shape[1:],
        input_shape_rssi=train_data[1].shape[1:],
        output_shape=2
    )
    
    logger.info(f"Model created with {model.model.count_params():,} parameters")
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path / 'GNN_Fusion_Fixed_best.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8
        )
    ]
    
    logger.info("Starting GNN_Fusion training...")
    
    # Train model
    history = model.model.fit(
        [train_data[0], train_data[1]],
        {'coordinates': train_data[2], 'uncertainty': np.ones_like(train_data[2])},
        validation_data=([val_data[0], val_data[1]], 
                       {'coordinates': val_data[2], 'uncertainty': np.ones_like(val_data[2])}),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training completed. Evaluating model...")
    
    # Evaluate
    test_loss = model.model.evaluate(
        [test_data[0], test_data[1]],
        {'coordinates': test_data[2], 'uncertainty': np.ones_like(test_data[2])}
    )
    
    # Get predictions
    y_pred, uncertainty = model.model.predict([test_data[0], test_data[1]])
    
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
    metrics_df.to_csv(str(save_path / 'GNN_Fusion_Fixed_metrics.csv'), index=False)
    
    # Save model summary
    with open(str(save_path / 'GNN_Fusion_Fixed_summary.txt'), 'w') as f:
        model.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('GNN_Fusion_Fixed - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['coordinates_mae'], label='Training MAE')
    plt.plot(history.history['val_coordinates_mae'], label='Validation MAE')
    plt.title('GNN_Fusion_Fixed - Model MAE')
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
    
    plt.title('GNN_Fusion_Fixed - True vs Predicted Locations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(str(save_path / 'GNN_Fusion_Fixed_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nGNN_Fusion_Fixed Training Complete:")
    logger.info(f"  Final MAE: {mae:.4f} meters ({mae*100:.2f} cm)")
    logger.info(f"  Final RMSE: {rmse:.4f} meters ({rmse*100:.2f} cm)")
    logger.info(f"  MAE_X: {mae_x:.4f} meters ({mae_x*100:.2f} cm)")
    logger.info(f"  MAE_Y: {mae_y:.4f} meters ({mae_y*100:.2f} cm)")
    logger.info(f"  Model saved to: {save_path}")
    
    return history, metrics_df, model

def main():
    """Main training function."""
    logger.info("Starting GNN_Fusion Training on REAL DATA")
    logger.info("="*50)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and preprocess REAL data
    train_data, val_data, test_data = load_real_data()
    
    # Train GNN_Fusion model
    try:
        history, metrics_df, model = train_gnn_fusion(train_data, val_data, test_data)
        logger.info("GNN_Fusion training completed successfully!")
        
        # Print final results
        mae = metrics_df[metrics_df['Metric'] == 'MAE']['Value'].iloc[0]
        logger.info(f"\n🏆 GNN_Fusion_Fixed Final Result: MAE = {mae:.4f} meters ({mae*100:.2f} cm)")
        
    except Exception as e:
        logger.error(f"Error during GNN_Fusion training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 