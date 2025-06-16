import numpy as np
import matplotlib.pyplot as plt

# Load the NPY file - use raw string (r prefix) or double backslashes or forward slashes
data = np.load(r"C:\MasterArbeit\NewRoom Pipeline\Data\Stacked_Data\sub 52 to 72\csi_amplitude.npy")  # Using raw string

# Verify the shape
print(f"Data shape: {data.shape}")

# Create subcarrier indices from -26 to 26, skipping 0
subcarrier_indices = np.array(list(range(-26, 0)) + list(range(1, 27)))

# Create a 4x4 grid of plots (16 plots)
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

# Create coordinate pairs for naming plots
x_coords = [0.0, 0.5, 1.0, 1.5]
y_coords = [0.0, 0.5, 1.0, 1.5]

# Generate all coordinate pairs
coord_pairs = []
for y in y_coords:
    for x in x_coords:
        coord_pairs.append((x, y))

# Assume there are 8000 samples (500 samples for each of the 16 points)
samples_per_point = 500

for plot_idx in range(16):
    ax = axes[plot_idx]
    
    # Get the 500 samples for this point
    start_idx = plot_idx * samples_per_point
    end_idx = start_idx + samples_per_point
    point_data = data[start_idx:end_idx]
    
    # Get coordinates for this plot
    x_coord, y_coord = coord_pairs[plot_idx]
    
    # Create scatter plot for all 500 samples
    for sample_idx in range(point_data.shape[0]):
        sample = point_data[sample_idx]
        ax.scatter(subcarrier_indices, sample, s=1, alpha=0.5)
    
    # Set title with coordinates
    ax.set_title(f"Position ({x_coord}, {y_coord})")
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-27, 27)  # A bit wider to see all points
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a vertical line at x=0 to separate negative and positive indices
    ax.axvline(x=0, color='r', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('subcarrier_amplitude_plots_52.png', dpi=300)
plt.show()