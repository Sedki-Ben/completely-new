"""
Model Architecture Module
========================

This module defines the deep learning model architecture for indoor localization
using CSI and RSSI data. The model uses a CNN-based architecture with dual input
branches for processing CSI and RSSI data.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout,
    Flatten, Concatenate, BatchNormalization
)
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndoorLocalizationModel:
    """
    A class implementing the indoor localization model architecture.
    
    This model uses a CNN-based architecture with dual input branches:
    1. CSI Branch: Processes CSI data (amplitude and phase)
    2. RSSI Branch: Processes RSSI data (optional)
    
    The model outputs (x, y) coordinates for indoor localization.
    
    Attributes:
        num_subcarriers (int): Number of subcarriers in the CSI data
        use_rssi (bool): Whether to include RSSI data in the model
        model (tf.keras.Model): The compiled Keras model
    """
    
    def __init__(self, num_subcarriers: int = 52, use_rssi: bool = True):
        """
        Initialize the model.
        
        Args:
            num_subcarriers: Number of subcarriers in the CSI data
            use_rssi: Whether to include RSSI data in the model
        """
        self.num_subcarriers = num_subcarriers
        self.use_rssi = use_rssi
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build the model architecture.
        
        Returns:
            Compiled Keras model
        """
        # CSI input branch
        csi_input = Input(shape=(self.num_subcarriers, 2), name='csi_input')
        
        # CSI processing branch
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(csi_input)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        csi_features = Flatten()(x)
        
        if self.use_rssi:
            # RSSI input branch
            rssi_input = Input(shape=(1,), name='rssi_input')
            
            # Combine CSI and RSSI branches
            combined = Concatenate()([csi_features, rssi_input])
            
            x = Dense(256, activation='relu')(combined)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            inputs = [csi_input, rssi_input]
        else:
            x = Dense(256, activation='relu')(csi_features)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            inputs = csi_input
        
        # Additional dense layers
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer for (x, y) coordinates
        outputs = Dense(2, activation=None)(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def get_model_summary(self) -> str:
        """
        Get a string representation of the model architecture.
        
        Returns:
            Model summary as a string
        """
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return '\n'.join(string_list)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str) -> 'IndoorLocalizationModel':
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded IndoorLocalizationModel instance
        """
        try:
            model = tf.keras.models.load_model(filepath)
            instance = cls()
            instance.model = model
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def create_model(num_subcarriers: int = 52, use_rssi: bool = True) -> IndoorLocalizationModel:
    """
    Factory function to create a new model instance.
    
    Args:
        num_subcarriers: Number of subcarriers in the CSI data
        use_rssi: Whether to include RSSI data in the model
        
    Returns:
        New IndoorLocalizationModel instance
    """
    return IndoorLocalizationModel(num_subcarriers=num_subcarriers, use_rssi=use_rssi)

if __name__ == "__main__":
    # Example usage
    model = create_model()
    print(model.get_model_summary()) 