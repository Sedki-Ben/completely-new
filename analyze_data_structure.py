#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Structure Analysis Script
Analyzes the feature shapes and structure of the WiFi localization data
"""

import pandas as pd
import numpy as np
import ast
import os
from pathlib import Path

def analyze_data_structure():
    """Analyze the complete data structure and feature shapes"""
    
    print("=" * 60)
    print("WiFi Indoor Localization - Data Structure Analysis")
    print("=" * 60)
    
    # 1. Analyze Amp+Phase_Data structure
    print("\n1. AMPLITUDE + PHASE DATA STRUCTURE")
    print("-" * 40)
    
    data_file = "Indoor-Localization-main/Data/Amp+Phase_Data/0.0,0.0 amplitudes_phases.csv"
    
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        print(f"File shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Analyze first row
        first_row = df.iloc[0]
        print(f"\nFirst row analysis:")
        print(f"  Index: {first_row['index']}")
        print(f"  RSSI: {first_row['rssi']}")
        
        # Parse amplitudes and phases
        try:
            amplitudes = ast.literal_eval(first_row['amplitudes'])
            phases = ast.literal_eval(first_row['phases'])
            
            print(f"  Amplitudes length: {len(amplitudes)}")
            print(f"  Phases length: {len(phases)}")
            print(f"  Sample amplitudes (first 5): {amplitudes[:5]}")
            print(f"  Sample phases (first 5): {phases[:5]}")
            
            # Check if they're equal length
            if len(amplitudes) == len(phases):
                print(f"  ✓ Amplitudes and phases have same length: {len(amplitudes)}")
            else:
                print(f"  ✗ Amplitudes and phases have different lengths!")
                
        except Exception as e:
            print(f"  Error parsing amplitudes/phases: {e}")
    
    # 2. Analyze processed data files
    print("\n2. PROCESSED DATA FILES")
    print("-" * 40)
    
    processed_files = [
        "Indoor-Localization-main/Data/X_train.npy",
        "Indoor-Localization-main/Data/X_test.npy", 
        "Indoor-Localization-main/Data/csi_phase.npy",
        "Indoor-Localization-main/Data/rssi_train.npy"
    ]
    
    for file_path in processed_files:
        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                print(f"{os.path.basename(file_path)}:")
                print(f"  Shape: {data.shape}")
                print(f"  Data type: {data.dtype}")
                print(f"  Min value: {data.min():.4f}")
                print(f"  Max value: {data.max():.4f}")
                print(f"  Mean value: {data.mean():.4f}")
                print()
            except Exception as e:
                print(f"{os.path.basename(file_path)}: Error loading - {e}")
    
    # 3. Analyze all coordinate files
    print("\n3. COORDINATE FILES ANALYSIS")
    print("-" * 40)
    
    coord_dir = "Indoor-Localization-main/Data/Amp+Phase_Data"
    if os.path.exists(coord_dir):
        coord_files = [f for f in os.listdir(coord_dir) if f.endswith('.csv')]
        print(f"Total coordinate files: {len(coord_files)}")
        
        # Extract coordinates from filenames
        coordinates = []
        for filename in coord_files:
            coord_str = filename.replace(' amplitudes_phases.csv', '')
            try:
                x, y = map(float, coord_str.split(','))
                coordinates.append((x, y))
            except:
                pass
        
        coordinates = sorted(set(coordinates))
        print(f"Unique coordinates: {len(coordinates)}")
        print(f"Coordinate range: X: {min(c[0] for c in coordinates)} to {max(c[0] for c in coordinates)}")
        print(f"Coordinate range: Y: {min(c[1] for c in coordinates)} to {max(c[1] for c in coordinates)}")
        print(f"Sample coordinates: {coordinates[:5]}")
    
    # 4. Feature structure summary
    print("\n4. FEATURE STRUCTURE SUMMARY")
    print("-" * 40)
    
    print("Raw Data Structure:")
    print("  - Each coordinate has 500 samples")
    print("  - Each sample contains:")
    print("    * RSSI: 1 value (Received Signal Strength)")
    print("    * Amplitudes: 52 values (CSI amplitude per subcarrier)")
    print("    * Phases: 52 values (CSI phase per subcarrier)")
    print("  - Total features per sample: 1 + 52 + 52 = 105 features")
    
    print("\nProcessed Data Structure:")
    print("  - X_train.npy: Training features")
    print("  - X_test.npy: Testing features") 
    print("  - csi_phase.npy: CSI phase information")
    print("  - rssi_train.npy: RSSI training data")
    
    print("\nModel Input Structure:")
    print("  - CSI Input: 104 features (52 amplitudes + 52 phases)")
    print("  - RSSI Input: 1 feature")
    print("  - Target: 2 features (x, y coordinates)")
    
    print("\nData Flow:")
    print("  1. Raw CSI packets → CSV files with amplitudes/phases")
    print("  2. CSV parsing → NumPy arrays")
    print("  3. Feature engineering → Model-ready format")
    print("  4. Model training → Coordinate predictions")

if __name__ == "__main__":
    analyze_data_structure() 