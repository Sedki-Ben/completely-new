#CSI processing 50 to 70

''' 
A simple Python script that processes WiFi Channel State Information (CSI) by removing non-informative subcarriers. 
It reduces the original 64 subcarriers to 52 by eliminating guard bands, DC subcarriers, and pilot tones that don't carry useful signal information.
The script can process all CSV files in a directory at once, making it efficient for processing large datasets.
'''

import csv
import ast
import json
import os
import glob

def process_csi_data(input_file, output_file=None):
    """
    Process CSI data from a CSV file by removing specific indices.
    
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
        output_file = os.path.join(os.path.dirname(input_file), f"{file_name_without_ext}_processed.csv")
    
    print(f"Processing {input_file}...")
    
    # Read the input CSV file
    rows = []
    try:
        with open(input_file, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Get the header row
            
            for row in csv_reader:
                rows.append(row)
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return []
    
    # Create a new header with the count column
    new_header = header + ['csi_data_count']
    
    # Process each row
    processed_rows = []
    for row_idx, row in enumerate(rows):
        # Skip empty rows
        if not row:
            print(f"Skipping empty row at index {row_idx}")
            continue
            
        try:
            # Get the CSI data which is the last column
            csi_data = row[-1]
            
            # Convert string representation of list to actual list
            csi_list = ast.literal_eval(csi_data)
            
            # Remove indices 0 to 3
            csi_list = csi_list[4:]
            
            # Remove indices 56 to 75 (which are now 52 to 71 after removing the first 4 elements)
            csi_list = csi_list[:50] + csi_list[70:]
            
            # Count the remaining elements
            count = len(csi_list)
            
            # Convert back to string representation
            new_csi_data = json.dumps(csi_list)
            
            # Replace the CSI data in the row
            new_row = row[:-1] + [new_csi_data, str(count)]
            processed_rows.append(new_row)
            
        except (SyntaxError, ValueError, IndexError) as e:
            # If the CSI data is incomplete or malformed
            print(f"Error processing row {row_idx} in {input_file}: {e}")
            # Keep the original row and set count to error
            if row:  # Only add if row is not empty
                processed_rows.append(row + ["ERROR"])
    
    # Write to the output file
    try:
        with open(output_file, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(new_header)
            csv_writer.writerows(processed_rows)
        
        print(f"Processed CSV file has been saved as '{output_file}'")
        print(f"Removed indices 0-3 and 56-75 from the original CSI data")
        print(f"Added a new column 'csi_data_count' showing the number of elements in the truncated CSI data")
        
        # Display the first few rows of processed data
        print("\nFirst 5 rows of processed data (or fewer if less available):")
        print(",".join(new_header))
        for i, row in enumerate(processed_rows):
            if i < 5:  # Display only first 5 rows for readability
                print(",".join(str(cell) for cell in row))
            else:
                break
    except Exception as e:
        print(f"Error writing to output file {output_file}: {e}")
    
    return processed_rows

def process_all_csv_files_in_directory(directory_path):
    """
    Process all CSV files in the specified directory.
    
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
        process_csi_data(file_path)
    
    print("All CSV files have been processed.")

if __name__ == "__main__":
    # Path to the directory containing the CSV files with raw data_data
    data_directory = r'C:\MasterArbeit\NewRoom Pipeline\Data\raw_data'
    
    # Process a single file
    # single_file_path = os.path.join(data_directory, '1,5-0.csv')
    # process_csi_data(single_file_path)
    
    # Or process all CSV files in the directory
    process_all_csv_files_in_directory(data_directory)
