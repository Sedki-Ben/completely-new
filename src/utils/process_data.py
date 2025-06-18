"""
Data Processing Script
====================

This script processes raw CSI and RSSI data into the format needed for training.
It includes:
1. Loading raw data
2. Extracting CSI features and phase
3. Processing RSSI data
4. Saving processed data in numpy format
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_raw_data(data_dir: Path):
    """
    Load raw CSI and RSSI data from CSV files.
    
    Args:
        data_dir: Path to the raw data directory
    
    Returns:
        Tuple of (csi_data, rssi_data, locations)
    """
    csi_data = []
    rssi_data = []
    locations = []
    
    # Process each CSV file in the raw data directory
    for csv_file in data_dir.glob('*.csv'):
        try:
            # Extract location from filename (assuming format: "x,y.csv")
            location = csv_file.stem.split(',')
            x, y = float(location[0]), float(location[1])
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Convert CSI data to complex numbers
            csi_complex = df.iloc[:, :-1].apply(lambda x: x.str.replace('i', 'j')).astype(complex).values
            rssi = df.iloc[:, -1].values.astype(float)
            
            csi_data.append(csi_complex)
            rssi_data.append(rssi)
            locations.append([x, y])
            
        except Exception as e:
            logger.warning(f"Error processing {csv_file}: {str(e)}")
    
    return np.array(csi_data), np.array(rssi_data), np.array(locations)

def process_csi_data(csi_data: np.ndarray):
    """
    Process CSI data to extract features and phase.
    
    Args:
        csi_data: Raw CSI data as complex numbers
    
    Returns:
        Tuple of (csi_features, csi_phase)
    """
    # Calculate amplitude and phase
    amplitude = np.abs(csi_data)
    phase = np.angle(csi_data)
    
    # Normalize amplitude
    scaler = StandardScaler()
    amplitude_normalized = scaler.fit_transform(amplitude.reshape(-1, amplitude.shape[-1])).reshape(amplitude.shape)
    
    return amplitude_normalized, phase

def process_rssi_data(rssi_data: np.ndarray):
    """
    Process RSSI data.
    
    Args:
        rssi_data: Raw RSSI data
    
    Returns:
        Processed RSSI data
    """
    # Normalize RSSI
    scaler = StandardScaler()
    rssi_normalized = scaler.fit_transform(rssi_data.reshape(-1, 1)).reshape(rssi_data.shape)
    
    return rssi_normalized

def main():
    """Main function to process data."""
    # Set paths
    raw_data_dir = Path('Indoor-Localization-main/Data/Raw_Data')
    processed_data_dir = Path('data/processed')
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    logger.info("Loading raw data...")
    csi_data, rssi_data, locations = load_raw_data(raw_data_dir)
    
    # Process CSI data
    logger.info("Processing CSI data...")
    csi_features, csi_phase = process_csi_data(csi_data)
    
    # Process RSSI data
    logger.info("Processing RSSI data...")
    rssi_processed = process_rssi_data(rssi_data)
    
    # Save processed data
    logger.info("Saving processed data...")
    np.save(processed_data_dir / 'processed_csi.npy', csi_features)
    np.save(processed_data_dir / 'processed_rssi.npy', rssi_processed)
    np.save(processed_data_dir / 'processed_labels.npy', locations)
    
    logger.info(f"Processed data saved to {processed_data_dir}")

if __name__ == "__main__":
    main() 