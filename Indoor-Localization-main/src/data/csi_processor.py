"""
CSI Data Processing Module
=========================

This module handles the processing of raw Channel State Information (CSI) data
from WiFi signals. It processes the raw CSI data by removing non-informative
subcarriers and preparing it for further analysis and model training.
"""

import csv
import ast
import json
import os
import glob
from typing import List, Tuple, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSIProcessor:
    """
    A class for processing CSI data from WiFi signals.
    
    This class handles the processing of raw CSI data by:
    1. Removing non-informative subcarriers (guard bands, DC subcarriers, pilot tones)
    2. Normalizing the data
    3. Saving the processed data in a standardized format
    
    Attributes:
        num_subcarriers (int): Number of subcarriers in the processed data
        bandwidth (float): Bandwidth of the WiFi signal in MHz
    """
    
    def __init__(self, num_subcarriers: int = 52, bandwidth: float = 20.0):
        """
        Initialize the CSI processor.
        
        Args:
            num_subcarriers: Number of subcarriers to keep in the processed data
            bandwidth: Bandwidth of the WiFi signal in MHz
        """
        self.num_subcarriers = num_subcarriers
        self.bandwidth = bandwidth
        
    def process_csi_data(self, input_file: str, output_file: Optional[str] = None) -> List[List]:
        """
        Process CSI data from a CSV file.
        
        Args:
            input_file: Path to the input CSV file
            output_file: Path to the output CSV file. If None, generates based on input name.
            
        Returns:
            List of processed data rows
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input data is malformed
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        if output_file is None:
            base_name = os.path.basename(input_file)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_file = os.path.join(os.path.dirname(input_file), 
                                     f"{file_name_without_ext}_processed.csv")
        
        logger.info(f"Processing {input_file}...")
        
        # Read and process the data
        processed_rows = self._read_and_process_file(input_file)
        
        # Write the processed data
        self._write_processed_data(processed_rows, output_file)
        
        return processed_rows
    
    def _read_and_process_file(self, input_file: str) -> List[List]:
        """
        Read and process the input file.
        
        Args:
            input_file: Path to the input file
            
        Returns:
            List of processed data rows
        """
        rows = []
        try:
            with open(input_file, 'r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)
                
                for row_idx, row in enumerate(csv_reader):
                    if not row:
                        logger.warning(f"Skipping empty row at index {row_idx}")
                        continue
                        
                    try:
                        processed_row = self._process_row(row, row_idx)
                        rows.append(processed_row)
                    except (SyntaxError, ValueError, IndexError) as e:
                        logger.error(f"Error processing row {row_idx}: {e}")
                        if row:
                            rows.append(row + ["ERROR"])
                            
        except Exception as e:
            logger.error(f"Error reading file {input_file}: {e}")
            raise
            
        return rows
    
    def _process_row(self, row: List[str], row_idx: int) -> List[str]:
        """
        Process a single row of CSI data.
        
        Args:
            row: List of values in the row
            row_idx: Index of the row being processed
            
        Returns:
            Processed row as a list of strings
        """
        csi_data = row[-1]
        csi_list = ast.literal_eval(csi_data)
        
        # Remove non-informative subcarriers
        csi_list = csi_list[4:]  # Remove first 4 elements
        csi_list = csi_list[:50] + csi_list[70:]  # Remove middle elements
        
        # Count remaining elements
        count = len(csi_list)
        
        # Convert back to string
        new_csi_data = json.dumps(csi_list)
        
        return row[:-1] + [new_csi_data, str(count)]
    
    def _write_processed_data(self, processed_rows: List[List], output_file: str) -> None:
        """
        Write processed data to output file.
        
        Args:
            processed_rows: List of processed data rows
            output_file: Path to the output file
        """
        try:
            with open(output_file, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['timestamp', 'rssi', 'csi_data', 'csi_data_count'])
                csv_writer.writerows(processed_rows)
                
            logger.info(f"Processed data saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error writing to output file {output_file}: {e}")
            raise
    
    def process_directory(self, directory_path: str) -> None:
        """
        Process all CSV files in a directory.
        
        Args:
            directory_path: Path to the directory containing CSV files
        """
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return
            
        csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {directory_path}")
            return
            
        logger.info(f"Found {len(csv_files)} CSV files in {directory_path}")
        
        for file_path in csv_files:
            try:
                self.process_csi_data(file_path)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process CSI data from CSV files')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory for processed output files')
    parser.add_argument('--num_subcarriers', type=int, default=52,
                      help='Number of subcarriers to keep')
    parser.add_argument('--bandwidth', type=float, default=20.0,
                      help='Bandwidth in MHz')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor and process files
    processor = CSIProcessor(
        num_subcarriers=args.num_subcarriers,
        bandwidth=args.bandwidth
    )
    processor.process_directory(args.input_dir)

if __name__ == "__main__":
    main() 