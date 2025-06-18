#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified TCN-Attention Model for WiFi Indoor Localization
Fixed version that works with real data format
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplifiedTCN_Attention:
    """
    Simplified Temporal Convolutional Network with Attention
    Removes complex temporal convolutions for compatibility
    """
    
    def __init__(self, input_shape=(104,), rssi_shape=(1,), learning_rate=0.001):
        self.input_shape = input_shape
        self.rssi_shape = rssi_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """Build simplified TCN-Attention model"""
        
        # Combined Input (CSI + RSSI concatenated)
        combined_input = layers.Input(shape=(self.input_shape[0] + self.rssi_shape[0],), name='combined_input')
        
        # Reshape for processing
        x = layers.Reshape((self.input_shape[0] + self.rssi_shape[0], 1))(combined_input)
        
        # Simplified Temporal Processing (1D convolutions)
        # First temporal block
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Second temporal block
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Third temporal block
        x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Simplified Self-Attention Mechanism
        # Reshape for attention
        x_reshaped = layers.Reshape((x.shape[1], x.shape[2]))(x)
        
        # Self-attention using Dense layers (simplified)
        attention_weights = layers.Dense(x.shape[1], activation='softmax')(x_reshaped)
        attended_features = layers.Multiply()([x_reshaped, attention_weights])
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(attended_features)
        
        # Dense layers for feature processing
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        # Create model
        model = models.Model(inputs=combined_input, outputs=output)
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()

def load_real_data():
    """Load real project data from Amp+Phase_Data directory"""
    logger.info("Loading REAL project data for TCN_Attention...")
    
    data_dir = Path("Data/Amp+Phase_Data")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    all_data = []
    file_count = 0
    
    # Load all CSV files
    for csv_file in data_dir.glob("*.csv"):
        try:
            # Extract coordinates from filename
            filename = csv_file.stem
            if '_' in filename:
                coords = filename.split('_')
                if len(coords) >= 2:
                    x = float(coords[0])
                    y = float(coords[1])
                    
                    # Load CSV data
                    df = pd.read_csv(csv_file)
                    
                    # Extract features (assuming standard format)
                    if 'RSSI' in df.columns and len(df.columns) > 1:
                        # Get RSSI values (first 4 APs)
                        rssi_values = df['RSSI'].values[:4]  # First 4 RSSI values
                        
                        # Get CSI features (amplitude and phase)
                        csi_features = []
                        for col in df.columns:
                            if col != 'RSSI' and col != 'filename':
                                csi_features.extend(df[col].values)
                        
                        # Ensure we have enough features
                        if len(csi_features) >= 104 and len(rssi_values) >= 4:
                            # Pad or truncate to exactly 104 CSI features
                            if len(csi_features) > 104:
                                csi_features = csi_features[:104]
                            else:
                                csi_features.extend([0] * (104 - len(csi_features)))
                            
                            # Pad RSSI to exactly 4 values
                            if len(rssi_values) > 4:
                                rssi_values = rssi_values[:4]
                            else:
                                rssi_values = np.pad(rssi_values, (0, 4 - len(rssi_values)), 'constant')
                            
                            all_data.append({
                                'x': x,
                                'y': y,
                                'csi': csi_features,
                                'rssi': rssi_values
                            })
                            file_count += 1
                            
        except Exception as e:
            logger.warning(f"Error processing {csv_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(all_data)} samples from real data")
    
    if len(all_data) == 0:
        raise ValueError("No valid data found!")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Prepare features and targets
    X_csi = np.array([row for row in df['csi'].values])
    X_rssi = np.array([row for row in df['rssi'].values])
    y = df[['x', 'y']].values
    
    # Concatenate CSI and RSSI features
    X_combined = np.concatenate([X_csi, X_rssi], axis=1)
    
    # Normalize features
    feature_scaler = StandardScaler()
    coord_scaler = StandardScaler()
    
    X_combined_scaled = feature_scaler.fit_transform(X_combined)
    y_scaled = coord_scaler.fit_transform(y)
    
    return X_combined_scaled, y_scaled, coord_scaler

def train_tcn_attention(train_data, val_data, test_data):
    """Train the simplified TCN-Attention model"""
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    logger.info("Initializing simplified TCN_Attention model...")
    
    # Create model
    model = SimplifiedTCN_Attention(
        input_shape=(108,),  # 104 CSI + 4 RSSI
        rssi_shape=(4,),
        learning_rate=0.001
    )
    
    # Model summary
    model.summary()
    
    # Callbacks
    checkpoint_path = "models/real_data/TCN_Attention_Simple"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_mae',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_mae',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Training
    logger.info("Starting training...")
    history = model.model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluation
    logger.info("Evaluating model...")
    test_loss, test_mae = model.model.evaluate(X_test, y_test, verbose=1)
    
    # Predictions
    y_pred = model.model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAE_X', 'MAE_Y'],
        'Value': [mae, rmse, mae_x, mae_y]
    })
    
    return history, metrics_df, model

def main():
    """Main training function"""
    logger.info("Starting Simplified TCN_Attention Training on REAL DATA")
    logger.info("=" * 50)
    
    try:
        # Load data
        X_combined, y, coord_scaler = load_real_data()
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )
        
        logger.info("Data split complete:")
        logger.info(f"  Training: {len(X_train)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
        logger.info(f"  Combined features shape: {X_train.shape}")
        
        # Prepare data tuples
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
        
        # Train model
        history, metrics_df, model = train_tcn_attention(train_data, val_data, test_data)
        
        # Print results
        logger.info("\nSimplified TCN_Attention Training Complete:")
        logger.info(f"  Final MAE: {metrics_df[metrics_df['Metric']=='MAE']['Value'].iloc[0]:.4f} meters ({metrics_df[metrics_df['Metric']=='MAE']['Value'].iloc[0]*100:.2f} cm)")
        logger.info(f"  Final RMSE: {metrics_df[metrics_df['Metric']=='RMSE']['Value'].iloc[0]:.4f} meters ({metrics_df[metrics_df['Metric']=='RMSE']['Value'].iloc[0]*100:.2f} cm)")
        logger.info(f"  MAE_X: {metrics_df[metrics_df['Metric']=='MAE_X']['Value'].iloc[0]:.4f} meters ({metrics_df[metrics_df['Metric']=='MAE_X']['Value'].iloc[0]*100:.2f} cm)")
        logger.info(f"  MAE_Y: {metrics_df[metrics_df['Metric']=='MAE_Y']['Value'].iloc[0]:.4f} meters ({metrics_df[metrics_df['Metric']=='MAE_Y']['Value'].iloc[0]*100:.2f} cm)")
        logger.info(f"  Model saved to: models\\real_data\\TCN_Attention_Simple")
        logger.info("Simplified TCN_Attention training completed successfully!")
        
        # Save metrics
        metrics_df.to_csv("models/real_data/TCN_Attention_Simple_metrics.csv", index=False)
        
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
        plt.savefig("models/real_data/TCN_Attention_Simple_training.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logger.error(f"Error during simplified TCN_Attention training: {e}")
        raise

if __name__ == "__main__":
    main() 