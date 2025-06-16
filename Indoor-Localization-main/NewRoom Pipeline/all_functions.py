import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(data_path):
    """
    Load raw CSI and RSSI data from specified path.
    
    Parameters:
    data_path (str): Path to the data files.
    
    Returns:
    tuple: Raw data arrays (csi_amplitude, csi_phase, rssi_values, locations).
    """
    csi_amplitude = np.load(os.path.join(data_path, "csi_amplitude.npy"))
    csi_phase = np.load(os.path.join(data_path, "csi_phase.npy"))
    rssi_values = np.load(os.path.join(data_path, "rssi.npy"))
    locations = np.load(os.path.join(data_path, "locations.npy"))
    
    return csi_amplitude, csi_phase, rssi_values, locations

def preprocess_data(csi_amplitude, csi_phase, rssi_values):
    """
    Preprocess CSI and RSSI data
    
    Parameters:
    - csi_amplitude, csi_phase, rssi_values: raw data arrays
    
    Returns:
    - csi_features: Processed CSI features
    - rssi_norm: Normalized RSSI values
    - scalers: Tuple of fitted scalers (amplitude, phase, rssi)
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
    
    # Stack amplitude and phase as channels
    csi_features = np.stack([csi_amplitude_norm, csi_phase_norm], axis=2)
    
    return csi_features, rssi_norm, (scaler_amp, scaler_phase, scaler_rssi)

def split_data(csi_features, rssi_norm, locations, random_state=42):
    """
    Split data into training, validation, and testing sets with exact sizes:
    5600 training, 1200 validation, and 1200 testing samples.
    
    Parameters:
    csi_features (np.array): Preprocessed CSI features.
    rssi_norm (np.array): Preprocessed RSSI values.
    locations (np.array): Location labels.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    tuple: Split data (X_csi_train, X_csi_val, X_csi_test, X_rssi_train, X_rssi_val, X_rssi_test, y_train, y_val, y_test).
    """
    # Get total number of samples
    total_samples = len(csi_features)
    
    # Ensure we have enough samples
    if total_samples < 8000:  # 5600 + 1200 + 1200
        raise ValueError(f"Not enough samples in dataset. Need 8000, but got {total_samples}")
    
    # Create indices for all samples and shuffle them
    indices = np.arange(total_samples)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    # Split indices into train, validation, and test sets
    test_indices = indices[:1200]
    val_indices = indices[1200:2400]
    train_indices = indices[2400:8000]  # Take up to 8000th sample if available
    
    # If we have more than 8000 samples, limit train set to exactly 5600 samples
    if len(train_indices) > 5600:
        train_indices = train_indices[:5600]
    
    # Split the data using the indices
    X_csi_train, X_csi_val, X_csi_test = csi_features[train_indices], csi_features[val_indices], csi_features[test_indices]
    X_rssi_train, X_rssi_val, X_rssi_test = rssi_norm[train_indices], rssi_norm[val_indices], rssi_norm[test_indices]
    y_train, y_val, y_test = locations[train_indices], locations[val_indices], locations[test_indices]
    
    return X_csi_train, X_csi_val, X_csi_test, X_rssi_train, X_rssi_val, X_rssi_test, y_train, y_val, y_test

def prepare_data_for_training(data_path, save_preprocessed=False):
    """
    Load, preprocess, and split data for training.
    
    Parameters:
    data_path (str): Path to the data files.
    save_preprocessed (bool): Whether to save the preprocessed data.
    
    Returns:
    tuple: Training, validation, and testing data splits, along with scalers.
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
    X_csi_train, X_csi_val, X_csi_test, X_rssi_train, X_rssi_val, X_rssi_test, y_train, y_val, y_test = split_data(
        csi_features, rssi_norm, locations
    )
    
    return X_csi_train, X_csi_val, X_csi_test, X_rssi_train, X_rssi_val, X_rssi_test, y_train, y_val, y_test, scalers