import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re

# File path
file_path = r"C:\MasterArbeit\NewRoom Pipeline\Data\Amp+Phase_Data\sub 50 to 70\updated\sanitized051 phases\sanitized051_Amp+Phase_0.0,1.5 processednew.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Function to safely convert string representation of lists to actual lists
def parse_list_string(list_string):
    """
    Parse a string representation of a list, handling cases like np.float64 and space-separated values.
    """
    try:
        # Remove np.float64 and other unwanted syntax
        cleaned_string = re.sub(r"np\.float64\(([-+]?\d*\.\d+|\d+)\)", r"\1", list_string)
        # Remove square brackets and split by spaces or commas
        cleaned_string = cleaned_string.strip("[]")
        # Split into individual elements (handling spaces, commas, and multiple spaces)
        elements = re.split(r"[,\s]+", cleaned_string)
        # Filter out empty strings and convert to floats
        return [float(x) for x in elements if x]
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing list string: {list_string}")
        print(f"Error: {e}")
        return []

# Apply the function to the relevant columns
df['amplitudes'] = df['amplitudes'].apply(parse_list_string)
df['phases'] = df['phases'].apply(parse_list_string)
df['sanitized_phases'] = df['sanitized_phases'].apply(parse_list_string)

# Randomly select 10 rows
random_rows = df.sample(n=10, random_state=42)

# Subcarrier indices (from -26 to -1 and 1 to 26, excluding 0)
subcarrier_indices = list(range(-26, -1)) + list(range(1, 26))

# Function to plot phases on a disk
def plot_phases_on_disk(ax, raw_phases, sanitized_phases, amplitudes, sample_index):
    """
    Plot raw and sanitized phases on a disk normalized to the highest amplitude.
    Points are indexed by their subcarrier indices (-26 to -1 and 1 to 26).
    
    Parameters:
    - ax: The subplot axis to plot on.
    - raw_phases: List of raw phase values.
    - sanitized_phases: List of sanitized phase values.
    - amplitudes: List of amplitude values.
    - sample_index: Index of the sample for the plot title.
    """
    # Skip if any of the lists are empty
    if not amplitudes or not raw_phases or not sanitized_phases:
        print(f"Skipping sample {sample_index} due to empty data.")
        return

    # Normalize amplitudes to the highest amplitude
    max_amplitude = max(amplitudes)
    normalized_amplitudes = np.array(amplitudes) / max_amplitude

    # Convert phases to Cartesian coordinates on the disk, scaled by normalized amplitudes
    def polar_to_cartesian(phases, normalized_amplitudes):
        x = normalized_amplitudes * np.cos(phases)
        y = normalized_amplitudes * np.sin(phases)
        return x, y

    # Convert raw and sanitized phases to Cartesian coordinates
    raw_x, raw_y = polar_to_cartesian(raw_phases, normalized_amplitudes)
    sanitized_x, sanitized_y = polar_to_cartesian(sanitized_phases, normalized_amplitudes)

    # Plot raw phases
    ax.scatter(raw_x, raw_y, color='red', label='Raw Phases', alpha=0.6)
    # Plot sanitized phases
    ax.scatter(sanitized_x, sanitized_y, color='blue', label='Sanitized Phases', alpha=0.6)

    # Draw the unit circle (disk)
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
    ax.add_patch(circle)

    # Label points with subcarrier indices
    for i, idx in enumerate(subcarrier_indices):
        ax.text(raw_x[i], raw_y[i], str(idx), fontsize=6, color='red', ha='center', va='center')
        ax.text(sanitized_x[i], sanitized_y[i], str(idx), fontsize=6, color='blue', ha='center', va='center')

    # Set plot limits and labels
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(f"Sample {sample_index}")
    ax.set_xlabel("Real (Normalized Amplitude * cos(θ))")
    ax.set_ylabel("Imaginary (Normalized Amplitude * sin(θ))")
    ax.legend()
    ax.grid(True)

# Create a figure with 2 rows and 5 columns of subplots
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot the 10 random samples
for i, (index, row) in enumerate(random_rows.iterrows()):
    plot_phases_on_disk(axes[i], row['phases'], row['sanitized_phases'], row['amplitudes'], index)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()