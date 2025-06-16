import numpy as np
import pandas as pd
import os
import glob
import re
import ast

def phase_sanitization(raw_phases, subcarrier_indices):
    """
    Perform phase sanitization to remove CFO and SFO effects.
    
    Parameters:
    - raw_phases: A list of raw phase values for each subcarrier.
    - subcarrier_indices: A list of subcarrier indices (e.g., [0, 1, ..., 51]).
    
    Returns:
    - calibrated_phases: A list of calibrated phase values.
    """
    # Step 1: Calculate the slope (a)
    n1 = subcarrier_indices[0]  # First subcarrier index (e.g., 0)
    nN = subcarrier_indices[-1]  # Last subcarrier index (e.g., 51)
    phase1 = raw_phases[0]       # Phase of first subcarrier
    phaseN = raw_phases[-1]      # Phase of last subcarrier
    a = (phaseN - phase1) / (nN - n1)
    
    # Step 2: Calculate the offset (b)
    b = np.mean(raw_phases)
    
    # Step 3: Calibrate the phases
    calibrated_phases = raw_phases - a * np.array(subcarrier_indices) - b
    
    return calibrated_phases

def extract_phases_from_string(phase_string):
    """
    Extract phase values from a string representation of a list.
    
    Parameters:
    - phase_string: A string representation of a list of phase values.
    
    Returns:
    - A list of phase values as floats.
    """
    # Safely evaluate the string to convert it into a list
    return ast.literal_eval(phase_string)

# Subcarrier indices (from 0 to 51)
subcarrier_indices = list(range(0, 52))

# Directory containing the CSV files
directory = r"C:\MasterArbeit\NewRoom Pipeline\Data\Amp+Phase_Data\sub 50 to 70\updated\Raw Phases"

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(directory, "*.csv"))

# Process each CSV file
for file_path in csv_files:
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Initialize a list to store sanitized phases
    sanitized_phases_list = []
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract phase values from the string representation
        raw_phases = extract_phases_from_string(row['phases'])
        
        # Ensure raw_phases has exactly 52 elements
        if len(raw_phases) != 52:
            raise ValueError(f"Expected 52 phase values, but found {len(raw_phases)} in row {index}.")
        
        # Perform phase sanitization
        sanitized_phases = phase_sanitization(raw_phases, subcarrier_indices)
        
        # Append the sanitized phases to the list
        sanitized_phases_list.append(sanitized_phases)
    
    # Add the sanitized phases as a new column in the DataFrame
    df['sanitized_phases'] = sanitized_phases_list
    
    # Create the new filename with 'sanitized_' prefix
    file_name = os.path.basename(file_path)
    new_file_name = f"sanitized051_{file_name}"
    new_file_path = os.path.join(directory, new_file_name)
    
    # Save the updated DataFrame to the new CSV file
    df.to_csv(new_file_path, index=False)
    
    print(f"Processed and saved: {new_file_path}")

print("All files have been processed and saved with sanitized phases.")