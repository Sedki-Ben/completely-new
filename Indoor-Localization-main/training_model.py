import numpy as np
import os
import joblib

# Specify the directory where your preprocessed data is saved
output_directory = r"C:\MasterArbeit\NewRoom Pipeline\Data"

# Load the preprocessed data
X_train = np.load(os.path.join(output_directory, 'X_train.npy'))
X_test = np.load(os.path.join(output_directory, 'X_test.npy'))
rssi_train = np.load(os.path.join(output_directory, 'rssi_train.npy'))
rssi_test = np.load(os.path.join(output_directory, 'rssi_test.npy'))
loc_train = np.load(os.path.join(output_directory, 'loc_train.npy'))
loc_test = np.load(os.path.join(output_directory, 'loc_test.npy'))

# Load the scalers (optional, if needed for later use)
scaler_amp = joblib.load(os.path.join(output_directory, 'scaler_amp.pkl'))
scaler_phase = joblib.load(os.path.join(output_directory, 'scaler_phase.pkl'))
scaler_rssi = joblib.load(os.path.join(output_directory, 'scaler_rssi.pkl'))


from model import create_single_antenna_model

# Create the model
model = create_single_antenna_model(num_subcarriers=52, use_rssi=True)

# Print the model summary
model.summary()     

from model import train_model

# Train the model
model, history = train_model(
    model, 
    X_train, rssi_train, loc_train,  # Training data
    X_test, rssi_test, loc_test,     # Validation data
    use_rssi=True                    # Whether to use RSSI
)