import numpy as np

# Load the .npy file
data = np.load("rssi.npy")

# Print the dimensions of the array
print("Dimensions of the file:", data.shape)