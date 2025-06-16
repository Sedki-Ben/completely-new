from data_load import prepare_data_for_training

# Call the function to preprocess and split data
X_csi_train, X_csi_val, X_rssi_train, X_rssi_val, y_train, y_val, scalers = prepare_data_for_training("data")

# Print shapes to verify
print("Training CSI shape:", X_csi_train.shape)
print("Validation CSI shape:", X_csi_val.shape)
print("Training RSSI shape:", X_rssi_train.shape)
print("Validation RSSI shape:", X_rssi_val.shape)
print("Training labels shape:", y_train.shape)
print("Validation labels shape:", y_val.shape)