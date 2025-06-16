import numpy as np
import matplotlib.pyplot as plt
import os

# Load the NPY file for phase data
phase_data = np.load(r"C:\MasterArbeit\NewRoom Pipeline\Data\Stacked_Data\sub 50 to 70\csi_phase50.npy")  # Using raw string

# Verify the shape
print(f"Phase data shape: {phase_data.shape}")

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

# Normalize phases to the range [-π, π] for better visualization
phase_data = np.arctan2(np.sin(phase_data), np.cos(phase_data))

# Plotting
for plot_idx in range(16):
    ax = axes[plot_idx]
    
    # Get the 500 samples for this point
    start_idx = plot_idx * samples_per_point
    end_idx = start_idx + samples_per_point
    point_data = phase_data[start_idx:end_idx]
    
    # Get coordinates for this plot
    x_coord, y_coord = coord_pairs[plot_idx]
    
    # Create scatter plot for all 500 samples
    for sample_idx in range(point_data.shape[0]):
        sample = point_data[sample_idx]
        ax.scatter(subcarrier_indices, sample, s=1, alpha=0.5)
    
    # Set title with coordinates
    ax.set_title(f"Position ({x_coord}, {y_coord})")
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Phase (radians)")
    ax.set_xlim(-27, 27)  # A bit wider to see all points
    ax.set_ylim(-np.pi, np.pi)  # Set y-axis limits for phase data
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a vertical line at x=0 to separate negative and positive indices
    ax.axvline(x=0, color='r', linestyle='-', alpha=0.3)

# Save the plot in the specified directory
output_dir = r"C:\MasterArbeit\NewRoom Pipeline\Plots"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
output_path = os.path.join(output_dir, 'subcarrier_phase_plots_50.png')
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Phase plots saved to: {output_path}")