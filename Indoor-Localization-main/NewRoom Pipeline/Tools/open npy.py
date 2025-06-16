import numpy as np
import os
import joblib

# Directory where data is saved
data_directory = r"C:\MasterArbeit\NewRoom Pipeline\Data"

# Load both original and processed data
print("Loading original data...")
csi_amplitudes = np.load(os.path.join(data_directory, "csi_amplitude.npy"), allow_pickle=True)
csi_phases = np.load(os.path.join(data_directory, "csi_phase.npy"), allow_pickle=True)
rssi_values = np.load(os.path.join(data_directory, "rssi.npy"), allow_pickle=True)
locations = np.load(os.path.join(data_directory, "locations.npy"), allow_pickle=True)

# Print information about original data
print("\n=== ORIGINAL DATA ===")
print("CSI Amplitudes Shape:", csi_amplitudes.shape)
print("CSI Phases Shape:", csi_phases.shape)
print("RSSI Values Shape:", rssi_values.shape)
print("Locations Shape:", locations.shape)

# Print sample data
print("\nSample Locations:\n", locations[:3])
print("Sample RSSI Values:\n", rssi_values[:3])

# Load processed data
print("\n\n=== PROCESSED DATA ===")
try:
    # Load intermediate results
    csi_features = np.load(os.path.join(data_directory, "csi_features.npy"), allow_pickle=True)
    rssi_norm = np.load(os.path.join(data_directory, "rssi_norm.npy"), allow_pickle=True)
    
    print("CSI Features Shape:", csi_features.shape)
    print("RSSI Normalized Shape:", rssi_norm.shape)
    print("\nSample CSI Features (first entry, first few values):\n", csi_features[0, 0:3, :])
    print("Sample Normalized RSSI Values:\n", rssi_norm[:3])
    
    # Load train/test splits
    X_train = np.load(os.path.join(data_directory, "X_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(data_directory, "X_test.npy"), allow_pickle=True)
    rssi_train = np.load(os.path.join(data_directory, "rssi_train.npy"), allow_pickle=True)
    rssi_test = np.load(os.path.join(data_directory, "rssi_test.npy"), allow_pickle=True)
    loc_train = np.load(os.path.join(data_directory, "loc_train.npy"), allow_pickle=True)
    loc_test = np.load(os.path.join(data_directory, "loc_test.npy"), allow_pickle=True)
    
    # Print train/test information
    print("\n=== TRAIN/TEST SPLITS ===")
    print(f"X_train shape: {X_train.shape} ({X_train.shape[0]} samples)")
    print(f"X_test shape: {X_test.shape} ({X_test.shape[0]} samples)")
    print(f"RSSI train shape: {rssi_train.shape}")
    print(f"RSSI test shape: {rssi_test.shape}")
    print(f"Location train shape: {loc_train.shape}")
    print(f"Location test shape: {loc_test.shape}")
    
    # Verify train/test ratio
    total_samples = X_train.shape[0] + X_test.shape[0]
    test_ratio = X_test.shape[0] / total_samples
    print(f"\nTest set ratio: {test_ratio:.2f} (expected 0.20)")
    
    # Load scalers
    print("\n=== SCALERS ===")
    try:
        scaler_amp = joblib.load(os.path.join(data_directory, "scaler_amp.pkl"))
        scaler_phase = joblib.load(os.path.join(data_directory, "scaler_phase.pkl"))
        scaler_rssi = joblib.load(os.path.join(data_directory, "scaler_rssi.pkl"))
        
        print("Amplitude Scaler mean:", scaler_amp.mean_[:5], "...")
        print("Amplitude Scaler scale:", scaler_amp.scale_[:5], "...")
        print("Phase Scaler min:", scaler_phase.min_[:5], "...")
        print("Phase Scaler scale:", scaler_phase.scale_[:5], "...")
        print("RSSI Scaler data min:", scaler_rssi.data_min_)
        print("RSSI Scaler data max:", scaler_rssi.data_max_)
    except Exception as e:
        print(f"Error loading individual scalers: {e}")
    
    # Try to load combined scalers object
    try:
        scalers = joblib.load(os.path.join(data_directory, "scalers.pkl"))
        print("\nCombined scalers object loaded successfully")
    except Exception as e:
        print(f"Error loading combined scalers: {e}")

except Exception as e:
    print(f"Error loading processed data: {e}")