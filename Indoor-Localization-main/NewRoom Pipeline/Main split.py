import os
import numpy as np
from all_functions import load_data, preprocess_data, split_data

def main():
    # Define the path to your data
    data_path = r"C:\MasterArbeit\NewRoom Pipeline\Data\Stacked_Data\sub 52 to 72"
    
    # Create output directory for saving intermediary values
    output_dir = os.path.join(data_path, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data directly
    print("Loading data...")
    csi_amplitude, csi_phase, rssi_values, locations = load_data(data_path)
    
    # Save loaded data
    print("Saving loaded data...")
    np.save(os.path.join(output_dir, "csi_amplitude_loaded.npy"), csi_amplitude)
    np.save(os.path.join(output_dir, "csi_phase_loaded.npy"), csi_phase)
    np.save(os.path.join(output_dir, "rssi_values_loaded.npy"), rssi_values)
    np.save(os.path.join(output_dir, "locations_loaded.npy"), locations)
    
    # Preprocess the data
    print("Preprocessing data...")
    csi_features, rssi_norm, scalers = preprocess_data(csi_amplitude, csi_phase, rssi_values)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    np.save(os.path.join(output_dir, "csi_features.npy"), csi_features)
    np.save(os.path.join(output_dir, "rssi_norm.npy"), rssi_norm)
    
    # Save scalers using pickle for later use
    import pickle
    with open(os.path.join(output_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    
    # Split the data
    print("Splitting data...")
    X_csi_train, X_csi_val, X_csi_test, X_rssi_train, X_rssi_val, X_rssi_test, y_train, y_val, y_test = split_data(
        csi_features, rssi_norm, locations
    )
    
    # Save split data
    print("Saving split data...")
    np.save(os.path.join(output_dir, "X_csi_train.npy"), X_csi_train)
    np.save(os.path.join(output_dir, "X_csi_val.npy"), X_csi_val)
    np.save(os.path.join(output_dir, "X_csi_test.npy"), X_csi_test)
    np.save(os.path.join(output_dir, "X_rssi_train.npy"), X_rssi_train)
    np.save(os.path.join(output_dir, "X_rssi_val.npy"), X_rssi_val)
    np.save(os.path.join(output_dir, "X_rssi_test.npy"), X_rssi_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    # Print shapes to verify
    print("\nData Processing Summary:")
    print("-----------------------")
    print("Raw Data:")
    print(f"CSI Amplitude: {csi_amplitude.shape}")
    print(f"CSI Phase: {csi_phase.shape}")
    print(f"RSSI Values: {rssi_values.shape}")
    print(f"Locations: {locations.shape}")
    
    print("\nPreprocessed Data:")
    print(f"CSI Features: {csi_features.shape}")
    print(f"RSSI Normalized: {rssi_norm.shape}")
    
    print("\nSplit Data:")
    print(f"Training CSI shape: {X_csi_train.shape}")
    print(f"Validation CSI shape: {X_csi_val.shape}")
    print(f"Testing CSI shape: {X_csi_test.shape}")
    print(f"Training RSSI shape: {X_rssi_train.shape}")
    print(f"Validation RSSI shape: {X_rssi_val.shape}")
    print(f"Testing RSSI shape: {X_rssi_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Testing labels shape: {y_test.shape}")
    
    print("\nAll data has been processed and saved to:", output_dir)

if __name__ == "__main__":
    main()