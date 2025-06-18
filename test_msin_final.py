#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import re
import ast
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import os

print('MSIN Model Testing...')

# Load model
model = keras.models.load_model('models/MSIN_best.h5')
print(f'Model loaded with {model.count_params():,} parameters')

# Load test data
data_dir = Path('Indoor-Localization-main/Data/Amp+Phase_Data')
all_data = []

for file_path in data_dir.glob('*.csv'):
    match = re.match(r'([\d.]+),([\d.]+)', file_path.name)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        df = pd.read_csv(file_path)
        
        for _, row in df.iterrows():
            try:
                amplitudes = np.array(ast.literal_eval(row['amplitudes'].replace('np.float64(', '').replace(')', '')), dtype=np.float32)
                phases = np.array(ast.literal_eval(row['phases'].replace('np.float64(', '').replace(')', '')), dtype=np.float32)
                rssi = float(row['rssi'])
                
                if len(amplitudes) == 52 and len(phases) == 52:
                    csi_features = np.concatenate([amplitudes, phases])
                    all_data.append({
                        'x': x, 'y': y, 'rssi': rssi, 'csi_features': csi_features
                    })
            except:
                continue

print(f'Loaded {len(all_data)} test samples')

# Prepare data
csi_features = np.array([sample['csi_features'] for sample in all_data])
rssi_features = np.array([[sample['rssi'] for sample in all_data]]).T
coordinates = np.array([[sample['x'], sample['y']] for sample in all_data])

# Scale features
scaler_csi = StandardScaler()
scaler_rssi = StandardScaler()
scaler_coords = MinMaxScaler()

csi_scaled = scaler_csi.fit_transform(csi_features)
rssi_scaled = scaler_rssi.fit_transform(rssi_features)
coords_scaled = scaler_coords.fit_transform(coordinates)

# Test model
print('Making predictions...')
y_pred_scaled = model.predict([csi_scaled, rssi_scaled])
y_pred_original = scaler_coords.inverse_transform(y_pred_scaled)
y_true_original = scaler_coords.inverse_transform(coords_scaled)

# Calculate metrics
mae = mean_absolute_error(y_true_original, y_pred_original)
mae_x = mean_absolute_error(y_true_original[:, 0], y_pred_original[:, 0])
mae_y = mean_absolute_error(y_true_original[:, 1], y_pred_original[:, 1])
r2 = r2_score(y_true_original, y_pred_original)
point_errors = np.sqrt(np.sum((y_true_original - y_pred_original)**2, axis=1))

print(f'MSIN Test Results:')
print(f'  MAE: {mae:.4f} meters')
print(f'  MAE_X: {mae_x:.4f} meters')
print(f'  MAE_Y: {mae_y:.4f} meters')
print(f'  R²: {r2:.4f}')
print(f'  Average Point Error: {np.mean(point_errors):.4f} meters')
print(f'  Max Point Error: {np.max(point_errors):.4f} meters')
print(f'  Min Point Error: {np.min(point_errors):.4f} meters')

# Save results
with open('MSIN_test_results.txt', 'w') as f:
    f.write(f'MSIN Model Test Results\n')
    f.write(f'MAE: {mae:.4f} meters\n')
    f.write(f'MAE_X: {mae_x:.4f} meters\n')
    f.write(f'MAE_Y: {mae_y:.4f} meters\n')
    f.write(f'R²: {r2:.4f}\n')
    f.write(f'Average Point Error: {np.mean(point_errors):.4f} meters\n')
    f.write(f'Max Point Error: {np.max(point_errors):.4f} meters\n')
    f.write(f'Min Point Error: {np.min(point_errors):.4f} meters\n')

print('MSIN testing completed! Check MSIN_test_results.txt') 