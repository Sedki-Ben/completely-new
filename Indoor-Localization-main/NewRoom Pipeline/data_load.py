# data_load.py
import numpy as np
import os
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data

def load_data(data_path):
    """
    Load raw CSI and RSSI data from specified path
    
    Parameters:
    data_path: Path to the data files
    
    Returns:
    Raw data arrays
    """
    csi_amplitude = np.load(os.path.join(data_path, "csi_amplitude.npy"))
    csi_phase = np.load(os.path.join(data_path, "csi_phase.npy"))
    rssi_values = np.load(os.path.join(data_path, "rssi.npy"))
    locations = np.load(os.path.join(data_path, "locations.npy"))
    
    return csi_amplitude, csi_phase, rssi_values, locations

def prepare_data_for_training(data_path, save_preprocessed=False):
    """
    Load, preprocess, and split data for training
    
    Parameters:
    data_path: Path to the data files
    save_preprocessed: Whether to save the preprocessed data
    
    Returns:
    Training and validation data splits
    """
    # Load raw data
    csi_amplitude, csi_phase, rssi_values, locations = load_data(data_path)
    
    # Preprocess data
    csi_features, rssi_norm, scalers = preprocess_data(csi_amplitude, csi_phase, rssi_values)
    
    # Optionally save preprocessed data
    if save_preprocessed:
        np.save(os.path.join(data_path, "csi_features.npy"), csi_features)
        np.save(os.path.join(data_path, "rssi_norm.npy"), rssi_norm)
    
    # Split data
    X_csi_train, X_csi_val, X_rssi_train, X_rssi_val, y_train, y_val = train_test_split(
        csi_features, rssi_norm, locations, test_size=0.2, random_state=42
    )
    
    return X_csi_train, X_csi_val, X_rssi_train, X_rssi_val, y_train, y_val, scalers
