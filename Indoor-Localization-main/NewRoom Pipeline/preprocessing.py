# preprocessing.py
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

# data_loader.py
def load_and_prepare_data(data_dir):
    """Load and preprocess CSI and RSSI data."""
    csi_amp = np.load(os.path.join(data_dir, 'csi_amplitude.npy'))
    csi_phase = np.load(os.path.join(data_dir, 'csi_phase.npy'))
    rssi = np.load(os.path.join(data_dir, 'rssi.npy'))
    locations = np.load(os.path.join(data_dir, 'locations.npy'))
    
    csi_features, rssi_norm, scalers = preprocess_data(csi_amp, csi_phase, rssi)
    # train_test_split returns a list, so we need to convert it to a tuple before adding scalers
    split_results = train_test_split(csi_features, rssi_norm, locations, test_size=0.2, random_state=42)
    # Return all results as a tuple
    return tuple(split_results) + (scalers,)

# Add this at the end of your file or in a separate script:

if __name__ == "__main__":
    # Specify the directory where your data files are located
    data_directory = r"C:\MasterArbeit\NewRoom Pipeline\Data"
    
    # Specify where to save the results
    output_directory = r"C:\MasterArbeit\NewRoom Pipeline\Data"
    os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist
    
    # First, apply preprocess_data directly to save intermediate results
    csi_amp = np.load(os.path.join(data_directory, 'csi_amplitude.npy'))
    csi_phase = np.load(os.path.join(data_directory, 'csi_phase.npy'))
    rssi = np.load(os.path.join(data_directory, 'rssi.npy'))
    
    # Process the data
    csi_features, rssi_norm, scalers_tuple = preprocess_data(csi_amp, csi_phase, rssi)
    scaler_amp, scaler_phase, scaler_rssi = scalers_tuple
    
    # Save intermediate results
    np.save(os.path.join(output_directory, 'csi_features.npy'), csi_features)
    np.save(os.path.join(output_directory, 'rssi_norm.npy'), rssi_norm)
    
    # Save scalers using joblib (better for sklearn objects)
    import joblib
    joblib.dump(scaler_amp, os.path.join(output_directory, 'scaler_amp.pkl'))
    joblib.dump(scaler_phase, os.path.join(output_directory, 'scaler_phase.pkl'))
    joblib.dump(scaler_rssi, os.path.join(output_directory, 'scaler_rssi.pkl'))
    
    # Now load and prepare the data to get train/test splits
    X_train, X_test, rssi_train, rssi_test, loc_train, loc_test, scalers = load_and_prepare_data(data_directory)
    
    # Save train/test splits
    np.save(os.path.join(output_directory, 'X_train.npy'), X_train)
    np.save(os.path.join(output_directory, 'X_test.npy'), X_test)
    np.save(os.path.join(output_directory, 'rssi_train.npy'), rssi_train)
    np.save(os.path.join(output_directory, 'rssi_test.npy'), rssi_test)
    np.save(os.path.join(output_directory, 'loc_train.npy'), loc_train)
    np.save(os.path.join(output_directory, 'loc_test.npy'), loc_test)
    
    # Save the combined scalers object
    joblib.dump(scalers, os.path.join(output_directory, 'scalers.pkl'))
    
    print("All data processed and saved successfully!")