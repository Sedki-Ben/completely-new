import numpy as np
import pandas as pd
import os

# Define x, y coordinate range and step
x_values = np.arange(0.0, 2.0, 0.5)  # 0.0 to 1.5 with step 0.5
y_values = np.arange(0.0, 2.0, 0.5)

# Data directory
data_dir = r'C:\MasterArbeit\NewRoom Pipeline\Data\Amp+Phase_Data\sub 52 to 72\updated\Raw Phases'

# Initialize lists to store data
csi_amplitudes = []
csi_phases = []
rssi_values = []
locations = []

# Load data from each file
for x in x_values:
    for y in y_values:
        file_path = os.path.join(data_dir, f"Amp+Phase_{x},{y} processed.csv")  # Updated file naming
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue
        
        data = pd.read_csv(file_path)
        
        # Extract amplitudes and phases
        amplitudes = np.array([eval(amp) for amp in data['amplitudes']])
        phases = np.array([eval(phase) for phase in data['phases']])
        
        # Append to lists
        csi_amplitudes.append(amplitudes)
        csi_phases.append(phases)
        rssi_values.append(data['rssi'].values)
        
        # Create location labels (x, y coordinates)
        locations.append(np.tile([x, y], (len(data), 1)))

# Convert lists to numpy arrays
csi_amplitudes = np.vstack(csi_amplitudes)
csi_phases = np.vstack(csi_phases)
rssi_values = np.hstack(rssi_values)
locations = np.vstack(locations)

# Save arrays in the same directory
np.save(os.path.join(data_dir, "csi_amplitude.npy"), csi_amplitudes)
np.save(os.path.join(data_dir, "csi_phase.npy"), csi_phases)
np.save(os.path.join(data_dir, "rssi.npy"), rssi_values)
np.save(os.path.join(data_dir, "locations.npy"), locations)

print("Data accumulation completed successfully!")