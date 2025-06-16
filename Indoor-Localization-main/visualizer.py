# visualization.py

"""
This script visualizes indoor localization results by plotting true and predicted positions.  
- True positions are shown as blue circles.  
- Predicted positions are shown as red crosses.  
- Green lines connect them to show errors.  
It also calculates and displays error metrics (mean, median, max).
"""


import numpy as np
import matplotlib.pyplot as plt

def visualize_results(y_true, y_pred, grid_size=(4, 4), step_size=0.5):
    """
    Visualize the localization results
    
    Parameters:
    y_true: True coordinates
    y_pred: Predicted coordinates
    grid_size: Size of the grid (rows, columns)
    step_size: Step size in meters
    """
    plt.figure(figsize=(10, 8))
    
    # Plot the grid
    for i in range(grid_size[0] + 1):
        plt.axhline(y=i * step_size, color='gray', linestyle='-', alpha=0.3)
    for i in range(grid_size[1] + 1):
        plt.axvline(x=i * step_size, color='gray', linestyle='-', alpha=0.3)
    
    # Plot true positions
    plt.scatter(y_true[:, 0], y_true[:, 1], c='blue', marker='o', label='True Position', s=100, alpha=0.7)
    
    # Plot predicted positions
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', marker='x', label='Predicted Position', s=100)
    
    # Connect true and predicted positions with lines
    for i in range(len(y_true)):
        plt.plot([y_true[i, 0], y_pred[i, 0]], [y_true[i, 1], y_pred[i, 1]], 'g-', alpha=0.5)
    
    plt.title('Indoor Localization Results')
    plt.xlabel('X coordinate (m)')
    plt.ylabel('Y coordinate (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Calculate and display metrics
    errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    
    plt.figtext(0.02, 0.02, f'Mean Error: {mean_error:.3f} m\nMedian Error: {median_error:.3f} m\nMax Error: {max_error:.3f} m', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig('localization_results.png', dpi=300)
    plt.show()
