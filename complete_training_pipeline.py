#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete WiFi Indoor Localization Training Pipeline
Processes real data from Amp+Phase_Data directory and trains three new architectures:
1. Multi-Scale Inception Network (MSIN)
2. Graph Neural Network with CSI-RSSI Fusion (GNN-Fusion)  
3. Temporal Convolutional Network with Attention (TCN-Attention)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import plot_model
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DataProcessor:
    """Processes WiFi localization data from Amp+Phase_Data directory"""
    
    def __init__(self, data_dir="Indoor-Localization-main/Data/Amp+Phase_Data"):
        self.data_dir = Path(data_dir)
        self.scaler_csi = StandardScaler()
        self.scaler_rssi = StandardScaler()
        self.scaler_coords = MinMaxScaler()
        
    def parse_coordinates_from_filename(self, filename):
        """Extract x,y coordinates from filename like '0.0,0.0 amplitudes_phases.csv'"""
        match = re.match(r'([\d.]+),([\d.]+)', filename)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None
    
    def parse_array_string(self, array_str):
        """Parse numpy array string to actual array"""
        try:
            # Remove np.float64() wrappers and convert to list
            cleaned = array_str.replace('np.float64(', '').replace(')', '')
            return np.array(ast.literal_eval(cleaned), dtype=np.float32)
        except:
            return np.array([], dtype=np.float32)
    
    def load_and_process_data(self):
        """Load all data files and process into training format"""
        print("Loading and processing WiFi localization data...")
        
        all_data = []
        
        # Process each file in the data directory
        for file_path in self.data_dir.glob("*.csv"):
            x, y = self.parse_coordinates_from_filename(file_path.name)
            if x is None or y is None:
                continue
                
            # Read CSV file
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                # Parse amplitudes and phases
                amplitudes = self.parse_array_string(row['amplitudes'])
                phases = self.parse_array_string(row['phases'])
                rssi = float(row['rssi'])
                
                if len(amplitudes) == 52 and len(phases) == 52:
                    # Combine amplitude and phase into complex CSI
                    csi_complex = amplitudes * np.exp(1j * phases)
                    csi_features = np.concatenate([amplitudes, phases])  # 104 features
                    
                    all_data.append({
                        'x': x,
                        'y': y,
                        'rssi': rssi,
                        'csi_features': csi_features,
                        'amplitudes': amplitudes,
                        'phases': phases
                    })
        
        print(f"Loaded {len(all_data)} samples from {len(list(self.data_dir.glob('*.csv')))} files")
        return all_data
    
    def prepare_training_data(self, data):
        """Prepare data for training with proper scaling"""
        print("Preparing training data...")
        
        # Extract features
        csi_features = np.array([sample['csi_features'] for sample in data])
        rssi_features = np.array([[sample['rssi']] for sample in data])
        coordinates = np.array([[sample['x'], sample['y']] for sample in data])
        
        # Scale features
        csi_scaled = self.scaler_csi.fit_transform(csi_features)
        rssi_scaled = self.scaler_rssi.fit_transform(rssi_features)
        coords_scaled = self.scaler_coords.fit_transform(coordinates)
        
        # Split data
        X_csi_train, X_csi_test, X_rssi_train, X_rssi_test, y_train, y_test = train_test_split(
            csi_scaled, rssi_scaled, coords_scaled, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_csi_train)}")
        print(f"Test samples: {len(X_csi_test)}")
        print(f"CSI features: {X_csi_train.shape[1]}")
        print(f"RSSI features: {X_rssi_train.shape[1]}")
        
        return {
            'X_csi_train': X_csi_train,
            'X_csi_test': X_csi_test,
            'X_rssi_train': X_rssi_train,
            'X_rssi_test': X_rssi_test,
            'y_train': y_train,
            'y_test': y_test,
            'scalers': {
                'csi': self.scaler_csi,
                'rssi': self.scaler_rssi,
                'coords': self.scaler_coords
            }
        }

class MultiScaleInceptionNetwork:
    """Multi-Scale Inception Network for WiFi localization"""
    
    def __init__(self, input_shape_csi=(104,), input_shape_rssi=(1,)):
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        
    def inception_block(self, x, filters_1x1, filters_3x3, filters_5x5, filters_pool):
        """Inception block with multiple kernel sizes"""
        # 1x1 branch
        branch1x1 = layers.Conv1D(filters_1x1, 1, padding='same', activation='relu')(x)
        
        # 3x3 branch
        branch3x3 = layers.Conv1D(filters_3x3, 1, padding='same', activation='relu')(x)
        branch3x3 = layers.Conv1D(filters_3x3, 3, padding='same', activation='relu')(branch3x3)
        
        # 5x5 branch
        branch5x5 = layers.Conv1D(filters_5x5, 1, padding='same', activation='relu')(x)
        branch5x5 = layers.Conv1D(filters_5x5, 5, padding='same', activation='relu')(branch5x5)
        
        # Pooling branch
        branch_pool = layers.MaxPooling1D(3, strides=1, padding='same')(x)
        branch_pool = layers.Conv1D(filters_pool, 1, padding='same', activation='relu')(branch_pool)
        
        # Concatenate all branches
        return layers.Concatenate()([branch1x1, branch3x3, branch5x5, branch_pool])
    
    def build_model(self):
        """Build the Multi-Scale Inception Network"""
        # CSI input branch
        csi_input = layers.Input(shape=self.input_shape_csi)
        csi_reshaped = layers.Reshape((self.input_shape_csi[0], 1))(csi_input)
        
        # Initial convolution
        x = layers.Conv1D(64, 7, padding='same', activation='relu')(csi_reshaped)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
        
        # Inception blocks
        x = self.inception_block(x, 64, 96, 128, 32)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
        
        x = self.inception_block(x, 128, 128, 192, 96)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
        
        x = self.inception_block(x, 192, 96, 208, 16)
        x = layers.BatchNormalization()(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # RSSI input branch
        rssi_input = layers.Input(shape=self.input_shape_rssi)
        rssi_dense = layers.Dense(32, activation='relu')(rssi_input)
        rssi_dense = layers.Dropout(0.3)(rssi_dense)
        
        # Fusion
        fused = layers.Concatenate()([x, rssi_dense])
        
        # Final layers
        dense1 = layers.Dense(256, activation='relu')(fused)
        dense1 = layers.Dropout(0.4)(dense1)
        dense2 = layers.Dense(128, activation='relu')(dense1)
        dense2 = layers.Dropout(0.3)(dense2)
        output = layers.Dense(2, activation='linear')(dense2)
        
        model = models.Model(inputs=[csi_input, rssi_input], outputs=output)
        return model

class GNNCSIFusion:
    """Graph Neural Network with CSI-RSSI Fusion"""
    
    def __init__(self, input_shape_csi=(104,), input_shape_rssi=(1,)):
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        
    def graph_conv_layer(self, x, filters, activation='relu'):
        """Graph convolution layer using 1D convolution"""
        # Self-connection
        self_conv = layers.Conv1D(filters, 1, padding='same', activation=activation)(x)
        
        # Neighbor connections (using 3x3 kernel)
        neighbor_conv = layers.Conv1D(filters, 3, padding='same', activation=activation)(x)
        
        # Combine
        combined = layers.Add()([self_conv, neighbor_conv])
        return layers.BatchNormalization()(combined)
    
    def build_model(self):
        """Build the GNN with CSI-RSSI Fusion"""
        # CSI input branch
        csi_input = layers.Input(shape=self.input_shape_csi)
        csi_reshaped = layers.Reshape((self.input_shape_csi[0], 1))(csi_input)
        
        # Graph convolution layers
        x = self.graph_conv_layer(csi_reshaped, 64)
        x = layers.Dropout(0.2)(x)
        
        x = self.graph_conv_layer(x, 128)
        x = layers.Dropout(0.2)(x)
        
        x = self.graph_conv_layer(x, 256)
        x = layers.Dropout(0.2)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # RSSI processing
        rssi_input = layers.Input(shape=self.input_shape_rssi)
        rssi_dense = layers.Dense(64, activation='relu')(rssi_input)
        rssi_dense = layers.Dropout(0.3)(rssi_dense)
        
        # Attention-based fusion
        attention_weights = layers.Dense(1, activation='sigmoid')(x)
        attended_csi = layers.Multiply()([x, attention_weights])
        
        # Concatenate and process
        fused = layers.Concatenate()([attended_csi, rssi_dense])
        
        # Final layers
        dense1 = layers.Dense(256, activation='relu')(fused)
        dense1 = layers.Dropout(0.4)(dense1)
        dense2 = layers.Dense(128, activation='relu')(dense1)
        dense2 = layers.Dropout(0.3)(dense2)
        output = layers.Dense(2, activation='linear')(dense2)
        
        model = models.Model(inputs=[csi_input, rssi_input], outputs=output)
        return model

class TCNWithAttention:
    """Temporal Convolutional Network with Attention"""
    
    def __init__(self, input_shape_csi=(104,), input_shape_rssi=(1,)):
        self.input_shape_csi = input_shape_csi
        self.input_shape_rssi = input_shape_rssi
        
    def temporal_conv_block(self, x, filters, kernel_size, dilation_rate):
        """Temporal convolution block with residual connection"""
        # Main path
        conv1 = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
                             padding='same', activation='relu')(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Dropout(0.2)(conv1)
        
        conv2 = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
                             padding='same', activation='relu')(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Dropout(0.2)(conv2)
        
        # Residual connection
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same')(x)
        
        return layers.Add()([x, conv2])
    
    def attention_layer(self, x):
        """Self-attention layer"""
        # Query, Key, Value
        query = layers.Dense(64, activation='relu')(x)
        key = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(64, activation='relu')(x)
        
        # Attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention
        attended = tf.matmul(attention_weights, value)
        return attended
    
    def build_model(self):
        """Build the TCN with Attention"""
        # CSI input branch
        csi_input = layers.Input(shape=self.input_shape_csi)
        csi_reshaped = layers.Reshape((self.input_shape_csi[0], 1))(csi_input)
        
        # Initial convolution
        x = layers.Conv1D(64, 7, padding='same', activation='relu')(csi_reshaped)
        x = layers.BatchNormalization()(x)
        
        # TCN blocks with increasing dilation rates
        x = self.temporal_conv_block(x, 64, 3, 1)
        x = self.temporal_conv_block(x, 64, 3, 2)
        x = self.temporal_conv_block(x, 64, 3, 4)
        x = self.temporal_conv_block(x, 64, 3, 8)
        
        # Attention layer
        x = self.attention_layer(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # RSSI processing
        rssi_input = layers.Input(shape=self.input_shape_rssi)
        rssi_dense = layers.Dense(32, activation='relu')(rssi_input)
        rssi_dense = layers.Dropout(0.3)(rssi_dense)
        
        # Fusion
        fused = layers.Concatenate()([x, rssi_dense])
        
        # Final layers
        dense1 = layers.Dense(256, activation='relu')(fused)
        dense1 = layers.Dropout(0.4)(dense1)
        dense2 = layers.Dense(128, activation='relu')(dense1)
        dense2 = layers.Dropout(0.3)(dense2)
        output = layers.Dense(2, activation='linear')(dense2)
        
        model = models.Model(inputs=[csi_input, rssi_input], outputs=output)
        return model

class ModelTrainer:
    """Handles model training, evaluation, and visualization"""
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {}
        self.histories = {}
        self.results = {}
        
    def train_model(self, model_name, model, train_data, epochs=50, batch_size=32):
        """Train a single model"""
        print(f"\nTraining {model_name}...")
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            callbacks.ModelCheckpoint(
                f'models/{model_name}_best.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Train model
        history = model.fit(
            [train_data['X_csi_train'], train_data['X_rssi_train']],
            train_data['y_train'],
            validation_data=([train_data['X_csi_test'], train_data['X_rssi_test']], 
                           train_data['y_test']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.models[model_name] = model
        self.histories[model_name] = history
        
        return model, history
    
    def evaluate_model(self, model_name, model, test_data):
        """Evaluate a trained model"""
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict([test_data['X_csi_test'], test_data['X_rssi_test']])
        
        # Inverse transform predictions
        y_pred_original = self.data_processor.scaler_coords.inverse_transform(y_pred)
        y_test_original = self.data_processor.scaler_coords.inverse_transform(test_data['y_test'])
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_original, y_pred_original)
        mse = mean_squared_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        
        # Calculate MAE for x and y separately
        mae_x = mean_absolute_error(y_test_original[:, 0], y_pred_original[:, 0])
        mae_y = mean_absolute_error(y_test_original[:, 1], y_pred_original[:, 1])
        
        results = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'mae_x': mae_x,
            'mae_y': mae_y,
            'y_pred': y_pred_original,
            'y_true': y_test_original
        }
        
        self.results[model_name] = results
        
        print(f"{model_name} Results:")
        print(f"  MAE: {mae:.4f} meters")
        print(f"  MAE_X: {mae_x:.4f} meters")
        print(f"  MAE_Y: {mae_y:.4f} meters")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return results
    
    def plot_training_history(self, model_name):
        """Plot training history for a model"""
        history = self.histories[model_name]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title(f'{model_name} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title(f'{model_name} - MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, model_name):
        """Plot prediction vs actual for a model"""
        results = self.results[model_name]
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # X coordinate predictions
        ax1.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6)
        ax1.plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                [y_true[:, 0].min(), y_true[:, 0].max()], 'r--', lw=2)
        ax1.set_xlabel('Actual X (meters)')
        ax1.set_ylabel('Predicted X (meters)')
        ax1.set_title(f'{model_name} - X Coordinate Predictions')
        ax1.grid(True)
        
        # Y coordinate predictions
        ax2.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.6)
        ax2.plot([y_true[:, 1].min(), y_true[:, 1].max()], 
                [y_true[:, 1].min(), y_true[:, 1].max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Y (meters)')
        ax2.set_ylabel('Predicted Y (meters)')
        ax2.set_title(f'{model_name} - Y Coordinate Predictions')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comparison(self):
        """Plot comparison of all models"""
        model_names = list(self.results.keys())
        metrics = ['mae', 'mae_x', 'mae_y', 'r2']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in model_names]
            
            bars = axes[i].bar(model_names, values)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save all trained models"""
        os.makedirs('models', exist_ok=True)
        
        for model_name, model in self.models.items():
            model.save(f'models/{model_name}_final.h5')
            print(f"Saved {model_name} model")
    
    def generate_report(self):
        """Generate comprehensive training report"""
        report = []
        report.append("=" * 60)
        report.append("WiFi Indoor Localization Training Report")
        report.append("=" * 60)
        report.append("")
        
        # Model comparison table
        report.append("Model Performance Comparison:")
        report.append("-" * 60)
        report.append(f"{'Model':<25} {'MAE':<10} {'MAE_X':<10} {'MAE_Y':<10} {'R²':<10}")
        report.append("-" * 60)
        
        for model_name in self.results.keys():
            results = self.results[model_name]
            report.append(f"{model_name:<25} {results['mae']:<10.4f} {results['mae_x']:<10.4f} "
                         f"{results['mae_y']:<10.4f} {results['r2']:<10.4f}")
        
        report.append("")
        report.append("Best performing model:")
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        report.append(f"  {best_model} with MAE: {self.results[best_model]['mae']:.4f} meters")
        
        # Save report
        with open('training_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))

def main():
    """Main training pipeline"""
    print("WiFi Indoor Localization Training Pipeline")
    print("=" * 50)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load and process data
    raw_data = data_processor.load_and_process_data()
    train_data = data_processor.prepare_training_data(raw_data)
    
    # Initialize trainer
    trainer = ModelTrainer(data_processor)
    
    # Define models
    models_config = {
        'MSIN': MultiScaleInceptionNetwork(),
        'GNN-Fusion': GNNCSIFusion(),
        'TCN-Attention': TCNWithAttention()
    }
    
    # Train and evaluate each model
    for model_name, model_class in models_config.items():
        print(f"\n{'='*20} {model_name} {'='*20}")
        
        # Build model
        model = model_class.build_model()
        print(f"Model parameters: {model.count_params():,}")
        
        # Train model
        model, history = trainer.train_model(model_name, model, train_data, epochs=30, batch_size=32)
        
        # Evaluate model
        results = trainer.evaluate_model(model_name, model, train_data)
        
        # Plot training history
        trainer.plot_training_history(model_name)
        
        # Plot predictions
        trainer.plot_predictions(model_name)
    
    # Generate comparison plots and report
    trainer.plot_comparison()
    trainer.save_models()
    trainer.generate_report()
    
    print("\nTraining completed successfully!")
    print("Check the 'models/', 'plots/', and 'training_report.txt' files for results.")

if __name__ == "__main__":
    main()
