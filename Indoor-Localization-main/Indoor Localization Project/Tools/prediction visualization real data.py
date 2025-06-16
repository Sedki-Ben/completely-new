import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import random

def visualize_random_samples(y_true, y_pred, num_samples=10, grid_size=(10, 10), step_size=0.5, room_name="Test Room"):
    """
    Visualize a random subset of localization results
    
    Parameters:
    y_true: True coordinates
    y_pred: Predicted coordinates
    num_samples: Number of random samples to visualize
    grid_size: Size of the grid (rows, columns)
    step_size: Step size in meters
    room_name: Name of the room/environment for the title
    """
    # Select random indices
    if len(y_true) <= num_samples:
        indices = list(range(len(y_true)))
    else:
        indices = random.sample(range(len(y_true)), num_samples)
    
    # Get the selected samples
    selected_true = y_true[indices]
    selected_pred = y_pred[indices]
    
    plt.figure(figsize=(12, 10))
    
    # Plot the grid
    for i in range(int(grid_size[0]/step_size) + 1):
        plt.axhline(y=i * step_size, color='gray', linestyle='-', alpha=0.3)
    for i in range(int(grid_size[1]/step_size) + 1):
        plt.axvline(x=i * step_size, color='gray', linestyle='-', alpha=0.3)
    
    # Plot true positions
    plt.scatter(selected_true[:, 0], selected_true[:, 1], c='blue', marker='o', label='True Position', s=100, alpha=0.7)
    
    # Plot predicted positions
    plt.scatter(selected_pred[:, 0], selected_pred[:, 1], c='red', marker='x', label='Predicted Position', s=100)
    
    # Connect true and predicted positions with lines and add point numbers
    for i, idx in enumerate(indices):
        plt.plot([selected_true[i, 0], selected_pred[i, 0]], 
                 [selected_true[i, 1], selected_pred[i, 1]], 'g-', alpha=0.5)
        
        # Add point number labels
        plt.text(selected_true[i, 0] + 0.05, selected_true[i, 1] + 0.05, 
                 f'{i+1}', color='blue', fontweight='bold')
        plt.text(selected_pred[i, 0] + 0.05, selected_pred[i, 1] + 0.05, 
                 f'{i+1}', color='red', fontweight='bold')
    
    plt.title(f'Indoor Localization Results - {room_name} ({num_samples} Random Samples)')
    plt.xlabel('X coordinate (m)')
    plt.ylabel('Y coordinate (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set axis limits to match the room dimensions
    plt.xlim(-0.5, grid_size[1] + 0.5)
    plt.ylim(-0.5, grid_size[0] + 0.5)
    
    plt.tight_layout()
    
    # Calculate and display metrics for the selected samples
    errors = np.sqrt(np.sum((selected_true - selected_pred)**2, axis=1))
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    
    # Add a table with errors for each point
    error_text = "Point   Error (m)\n" + "-"*18 + "\n"
    for i in range(len(errors)):
        error_text += f"{i+1:5d}   {errors[i]:.3f}\n"
    
    error_text += "-"*18 + f"\nMean:   {mean_error:.3f} m\nMedian: {median_error:.3f} m\nMax:    {max_error:.3f} m"
    
    plt.figtext(0.02, 0.02, error_text, 
                bbox=dict(facecolor='white', alpha=0.8),
                fontfamily='monospace')
    
    return plt.gcf()

# Main execution script
if __name__ == "__main__":
    # Specify paths
    data_path = r"C:\MasterArbeit\NewRoom Pipeline\Data\Stacked_Data\sub 50 to 70"
    processed_dir = os.path.join(data_path, "processed")
    model_path = os.path.join(r"C:\MasterArbeit\NewRoom Pipeline\models", "saved_models", "best_model.keras")
    results_dir = os.path.join(data_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    #np.random.seed(42)
    #random.seed(42)
    
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
    y_pred = model.predict(test_inputs)
    
    # Create visualization
    print("Creating visualization...")
    room_dimensions = (1.5, 1.5)  # Adjust based on your actual room dimensions
    fig = visualize_random_samples(y_test, y_pred, num_samples=10, grid_size=room_dimensions, 
                                  step_size=0.5, room_name="Test Environment")
    
    # Save the figure
    fig.savefig(os.path.join(results_dir, 'prediction_vis.png'), dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {os.path.join(results_dir, 'prediction_vis.png')}")
    
    # Show the plot
    plt.show()
    
    # Also generate a version with more samples (20) for comparison
    fig2 = visualize_random_samples(y_test, y_pred, num_samples=20, grid_size=room_dimensions, 
                                  step_size=0.5, room_name="Test Environment")
    fig2.savefig(os.path.join(results_dir, 'prediction_vis_20_samples.png'), dpi=300, bbox_inches='tight')