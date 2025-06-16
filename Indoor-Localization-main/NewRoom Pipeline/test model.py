import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Specify paths
data_path = r"C:\MasterArbeit\NewRoom Pipeline\Data\Stacked_Data\sub 50 to 70"
processed_dir = os.path.join(data_path, "processed")
model_path = os.path.join(r"C:\MasterArbeit\NewRoom Pipeline\models", "saved_models", "best_model.keras")

# Create output directory for results
results_dir = os.path.join(data_path, "results")
os.makedirs(results_dir, exist_ok=True)

# Load test data
print("Loading test data...")
X_csi_test = np.load(os.path.join(processed_dir, 'X_csi_test.npy'))
X_rssi_test = np.load(os.path.join(processed_dir, 'X_rssi_test.npy'))
y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))

# Load model
print(f"Loading model from {model_path}...")
model = load_model(model_path)

# Make predictions
print("Making predictions...")
test_inputs = [X_csi_test, X_rssi_test]  # Assuming use_rssi=True
predictions = model.predict(test_inputs)

# Calculate error metrics
mae = np.mean(np.abs(predictions - y_test), axis=0)
rmse = np.sqrt(np.mean(np.square(predictions - y_test), axis=0))
euclidean_distance = np.sqrt(np.sum(np.square(predictions - y_test), axis=1))
median_error = np.median(euclidean_distance)
mean_error = np.mean(euclidean_distance)

print(f"Mean Absolute Error (X, Y): {mae}")
print(f"RMSE (X, Y): {rmse}")
print(f"Median Euclidean Distance Error: {median_error:.4f} meters")
print(f"Mean Euclidean Distance Error: {mean_error:.4f} meters")

# Calculate CDF (Cumulative Distribution Function) of errors
sorted_errors = np.sort(euclidean_distance)
p = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

# Plot CDF of positioning errors
plt.figure(figsize=(10, 6))
plt.plot(sorted_errors, p, 'b-', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Error (meters)', fontsize=12)
plt.ylabel('CDF', fontsize=12)
plt.title('Cumulative Distribution Function of Positioning Errors', fontsize=14)

# Add annotation for median and 90th percentile errors
percentile_90 = np.percentile(euclidean_distance, 90)
plt.axvline(x=median_error, color='r', linestyle='--')
plt.axvline(x=percentile_90, color='g', linestyle='--')
plt.text(median_error + 0.1, 0.5, f'Median: {median_error:.2f}m', color='r')
plt.text(percentile_90 + 0.1, 0.9, f'90th percentile: {percentile_90:.2f}m', color='g')

plt.savefig(os.path.join(results_dir, 'error_cdf.png'), dpi=300, bbox_inches='tight')
plt.show()

# Visualize actual vs predicted positions (scatter plot)
plt.figure(figsize=(10, 10))
plt.scatter(y_test[:, 0], y_test[:, 1], c='blue', label='Actual', alpha=0.5, s=20)
plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='Predicted', alpha=0.5, s=20)

# Connect actual and predicted points with lines for a sample of points (for clarity)
sample_size = min(100, len(y_test))
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
for i in sample_indices:
    plt.plot([y_test[i, 0], predictions[i, 0]], [y_test[i, 1], predictions[i, 1]], 'k-', alpha=0.2)

plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('X coordinate (meters)', fontsize=12)
plt.ylabel('Y coordinate (meters)', fontsize=12)
plt.title('Actual vs Predicted Positions', fontsize=14)
plt.legend()
plt.axis('equal')  # Equal scaling for both axes

plt.savefig(os.path.join(results_dir, 'position_predictions.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save error metrics to a text file
with open(os.path.join(results_dir, 'error_metrics.txt'), 'w') as f:
    f.write(f"Mean Absolute Error (X, Y): {mae}\n")
    f.write(f"RMSE (X, Y): {rmse}\n")
    f.write(f"Median Euclidean Distance Error: {median_error:.4f} meters\n")
    f.write(f"Mean Euclidean Distance Error: {mean_error:.4f} meters\n")
    f.write(f"90th percentile Error: {percentile_90:.4f} meters\n")

print(f"Results saved to {results_dir}")