# extract phases and amplitudes  

'''
A Python script that processes WiFi Channel State Information (CSI) data from 52 subcarriers to extract amplitudes, phases, and RSSI values. 
The script can process single files or entire directories making it effecient to process large datasets.
'''

import pandas as pd
import numpy as np
import csv
import ast
import os
import glob

def process_csi_file(input_file, output_file=None):
    """
    Process a single CSI data file to extract amplitudes, phases, and RSSI values.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to the output CSV file. If None, generates based on input name.
    
    Returns:
        list: Processed data rows
    """
    # If output_file is not specified, generate it based on the input_file name
    if output_file is None:
        base_name = os.path.basename(input_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(os.path.dirname(input_file), f"{file_name_without_ext}_amplitudes_phases.csv")
    
    print(f"Processing {input_file}...")
    
    # Read the CSV into a pandas DataFrame
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return []
    
    # Initialize the list to store new rows for the output CSV
    output_data = []

    # Loop through each row in the DataFrame
    for idx, row in df.iterrows():
        # Extract csi_data from the current row and safely convert it to a list
        try:
            csi_data = ast.literal_eval(row['csi_data'])
        except (SyntaxError, ValueError) as e:
            print(f"Error processing row {idx} in {input_file}: {e}")
            continue

        # Extract the RSSI value
        rssi = row['rssi']

        # Initialize lists to store amplitudes and phases
        amplitudes = []
        phases = []

        # Loop through the CSI data in pairs (imaginary, real)
        for i in range(0, len(csi_data) - 1, 2):
            imaginary = csi_data[i]
            real = csi_data[i + 1]
            
            # Calculate the amplitude (magnitude)
            amplitude = np.sqrt(imaginary**2 + real**2)
            
            # Calculate the phase (angle) in radians
            phase = np.arctan2(imaginary, real)
            
            # Append to the lists
            amplitudes.append(amplitude)
            phases.append(phase)

        # Add the processed row to the output list
        output_data.append([idx, rssi, amplitudes, phases])

    # Write the output data to a new CSV
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write the header
            writer.writerow(['index', 'rssi', 'amplitudes', 'phases'])
            
            # Write the rows
            writer.writerows(output_data)
        
        print(f"Processed CSV file has been saved as '{output_file}'")
    except Exception as e:
        print(f"Error writing to output file {output_file}: {e}")
    
    return output_data

def process_all_csv_files_in_directory(directory_path):
    """
    Process all CSV files in the specified directory to extract amplitudes, phases, and RSSI values.
    
    Args:
        directory_path (str): Path to the directory containing CSV files
    """
    # Ensure the directory path exists
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    # Process each CSV file
    for file_path in csv_files:
        process_csi_file(file_path)
    
    print("All CSV files have been processed.")

if __name__ == "__main__":
    # Path to the directory containing the CSV files with raw data
    data_directory = r'C:\MasterArbeit\NewRoom Pipeline\Data\Processed_Data\sub 50 to 70'
    
    # Process a single file
    # single_file_path = os.path.join(data_directory, '1,5-0.csv')
    # process_csi_file(single_file_path)
    
    # Or process all CSV files in the directory
    process_all_csv_files_in_directory(data_directory)
