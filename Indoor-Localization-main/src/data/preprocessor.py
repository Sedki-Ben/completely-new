"""
Data Preprocessing Module
========================

This module handles the preprocessing of CSI and RSSI data for model training.
It includes normalization, feature extraction, and data splitting functionality.
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class for preprocessing CSI and RSSI data for model training.
    
    This class handles:
    1. Loading raw CSI and RSSI data
    2. Normalizing the data
    3. Splitting into training and testing sets
    4. Saving processed data and scalers
    
    Attributes:
        num_subcarriers (int): Number of subcarriers in the CSI data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    """
    
    def __init__(self, num_subcarriers: int = 52, test_size: float = 0.2, 
                 random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            num_subcarriers: Number of subcarriers in the CSI data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.num_subcarriers = num_subcarriers
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}
        
    def preprocess_data(self, csi_amplitude: np.ndarray, csi_phase: np.ndarray, 
                       rssi_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Preprocess CSI and RSSI data.
        
        Args:
            csi_amplitude: Raw CSI amplitude data
            csi_phase: Raw CSI phase data
            rssi_values: Raw RSSI values
            
        Returns:
            Tuple containing:
            - Processed CSI features
            - Normalized RSSI values
            - Dictionary of fitted scalers
        """
        # Normalize CSI amplitude
        scaler_amp = StandardScaler()
        csi_amplitude_norm = scaler_amp.fit_transform(csi_amplitude)
        
        # Normalize CSI phase
        scaler_phase = MinMaxScaler(feature_range=(-1, 1))
        csi_phase_norm = scaler_phase.fit_transform(csi_phase)
        
        # Normalize RSSI values
        rssi_values = rssi_values.reshape(-1, 1)
        scaler_rssi = MinMaxScaler(feature_range=(0, 1))
        rssi_norm = scaler_rssi.fit_transform(rssi_values)
        
        # Reshape CSI data to (samples, time_steps, features)
        if len(csi_amplitude_norm.shape) == 2:
            csi_amplitude_norm = csi_amplitude_norm.reshape(csi_amplitude_norm.shape[0], -1, 1)
        if len(csi_phase_norm.shape) == 2:
            csi_phase_norm = csi_phase_norm.reshape(csi_phase_norm.shape[0], -1, 1)
        
        # Stack amplitude and phase as channels
        csi_features = np.concatenate([csi_amplitude_norm, csi_phase_norm], axis=-1)
        
        # Store scalers
        self.scalers = {
            'amplitude': scaler_amp,
            'phase': scaler_phase,
            'rssi': scaler_rssi
        }
        
        return csi_features, rssi_norm, self.scalers
    
    def load_and_prepare_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, np.ndarray, 
                                                           Dict[str, Any]]:
        """
        Load and prepare data for training.
        
        Args:
            data_dir: Directory containing the data files
            
        Returns:
            Tuple containing:
            - Training CSI features
            - Testing CSI features
            - Training RSSI values
            - Testing RSSI values
            - Training locations
            - Testing locations
            - Dictionary of scalers
        """
        try:
            # Load data
            csi_amp = np.load(os.path.join(data_dir, 'csi_amplitude.npy'))
            csi_phase = np.load(os.path.join(data_dir, 'csi_phase.npy'))
            rssi = np.load(os.path.join(data_dir, 'rssi.npy'))
            locations = np.load(os.path.join(data_dir, 'locations.npy'))
            
            # Preprocess data
            csi_features, rssi_norm, scalers = self.preprocess_data(csi_amp, csi_phase, rssi)
            
            # Split data
            X_train, X_test, rssi_train, rssi_test, loc_train, loc_test = train_test_split(
                csi_features, rssi_norm, locations,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            return X_train, X_test, rssi_train, rssi_test, loc_train, loc_test, scalers
            
        except Exception as e:
            logger.error(f"Error loading and preparing data: {e}")
            raise
    
    def save_processed_data(self, output_dir: str, data: Dict[str, np.ndarray]) -> None:
        """
        Save processed data and scalers.
        
        Args:
            output_dir: Directory to save the processed data
            data: Dictionary containing the processed data arrays
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save processed data
            for name, array in data.items():
                np.save(os.path.join(output_dir, f'{name}.npy'), array)
            
            # Save scalers
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, os.path.join(output_dir, f'scaler_{name}.pkl'))
                
            logger.info(f"Processed data saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CSI and RSSI data')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input data files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory for processed output files')
    parser.add_argument('--num_subcarriers', type=int, default=52,
                      help='Number of subcarriers in the CSI data')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        num_subcarriers=args.num_subcarriers,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Load and prepare data
    X_train, X_test, rssi_train, rssi_test, loc_train, loc_test, scalers = \
        preprocessor.load_and_prepare_data(args.input_dir)
    
    # Save processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'rssi_train': rssi_train,
        'rssi_test': rssi_test,
        'loc_train': loc_train,
        'loc_test': loc_test
    }
    preprocessor.save_processed_data(args.output_dir, processed_data)

if __name__ == "__main__":
    main() 