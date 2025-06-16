import numpy as np
import os
import pickle
import tensorflow as tf
from model import create_single_antenna_model, train_model

# Specify the directory where your preprocessed data is saved
data_path = r"C:\MasterArbeit\NewRoom Pipeline\Data\Stacked_Data\sub 50 to 70"
output_directory = os.path.join(data_path, "processed")

# Load the preprocessed data with correct filenames
print("Loading preprocessed data...")
X_csi_train = np.load(os.path.join(output_directory, 'X_csi_train.npy'))
X_csi_val = np.load(os.path.join(output_directory, 'X_csi_val.npy'))
X_csi_test = np.load(os.path.join(output_directory, 'X_csi_test.npy'))
X_rssi_train = np.load(os.path.join(output_directory, 'X_rssi_train.npy'))
X_rssi_val = np.load(os.path.join(output_directory, 'X_rssi_val.npy'))
X_rssi_test = np.load(os.path.join(output_directory, 'X_rssi_test.npy'))
y_train = np.load(os.path.join(output_directory, 'y_train.npy'))
y_val = np.load(os.path.join(output_directory, 'y_val.npy'))
y_test = np.load(os.path.join(output_directory, 'y_test.npy'))

# Load the scalers
print("Loading scalers...")
with open(os.path.join(output_directory, 'scalers.pkl'), 'rb') as f:
    scalers = pickle.load(f)
    scaler_amp, scaler_phase, scaler_rssi = scalers

# Create the model
print("Creating model...")
num_subcarriers = X_csi_train.shape[1]
model = create_single_antenna_model(num_subcarriers=num_subcarriers, use_rssi=True)

# Print the model summary
model.summary()

# Train the model
print("Training model...")
model, history = train_model(
    model, 
    X_csi_train, X_rssi_train, y_train,  # Training data
    X_csi_val, X_rssi_val, y_val,        # Validation data
    use_rssi=True                        # Whether to use RSSI
)

# Evaluate on test set after training
print("Evaluating model on test set...")
if True:  # use_rssi is True
    test_inputs = [X_csi_test, X_rssi_test]
else:
    test_inputs = X_csi_test

test_loss, test_mae = model.evaluate(test_inputs, y_test, verbose=1)
print(f"Test MAE: {test_mae:.4f}")

# Plot training history
print("Plotting training history...")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'training_history.png'))
plt.show()

print(f"Training history saved to {os.path.join(output_directory, 'training_history.png')}")
print("Training complete!")