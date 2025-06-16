"""
Pipeline Runner
==============

This script runs the complete indoor localization pipeline including:
1. Data loading and preprocessing
2. Model training and validation
3. Model testing
4. Results visualization and evaluation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Tuple, Any

from utils.config import get_config
from data.preprocessor import DataPreprocessor
from models.model import IndoorLocalizationModel
from models.trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """
    A class to run the complete indoor localization pipeline.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the pipeline runner.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = get_config(config_path)
        self.preprocessor = None
        self.model = None
        self.trainer = None
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns:
            Dictionary containing results and metrics
        """
        logger.info("Starting pipeline execution...")
        
        # Create necessary directories
        self.config.create_directories()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        self._load_and_preprocess_data()
        
        # Train model
        logger.info("Training model...")
        self._train_model()
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = self._evaluate_model()
        
        # Visualize results
        logger.info("Visualizing results...")
        self._visualize_results(results)
        
        logger.info("Pipeline execution completed")
        return results
    
    def _load_and_preprocess_data(self) -> None:
        """Load and preprocess the data."""
        self.preprocessor = DataPreprocessor(
            num_subcarriers=self.config.get_model_param('num_subcarriers'),
            test_size=self.config.get_data_param('test_size'),
            random_state=self.config.get_data_param('random_state')
        )
        
        # Load and prepare data
        data_dir = self.config.get_path('processed_data_dir')
        (self.X_train, self.X_test, self.rssi_train, self.rssi_test,
         self.y_train, self.y_test, self.scalers) = self.preprocessor.load_and_prepare_data(data_dir)
    
    def _train_model(self) -> None:
        """Train the model."""
        # Create model
        self.model = IndoorLocalizationModel(
            num_subcarriers=self.config.get_model_param('num_subcarriers'),
            use_rssi=self.config.get_model_param('use_rssi')
        )
        
        # Create trainer
        self.trainer = ModelTrainer(
            model=self.model,
            batch_size=self.config.get_model_param('batch_size'),
            epochs=self.config.get_model_param('epochs'),
            patience=self.config.get_model_param('patience'),
            model_dir=self.config.get_path('model_dir')
        )
        
        # Train model
        self.history = self.trainer.train(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_test,  # Using test set as validation set for simplicity
            y_val=self.y_test,
            rssi_train=self.rssi_train,
            rssi_val=self.rssi_test
        )
    
    def _evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the model and calculate metrics.
        
        Returns:
            Dictionary containing evaluation results
        """
        # Get predictions
        test_data = self.trainer._prepare_data(self.X_test, self.rssi_test)
        predictions = self.model.model.predict(test_data)
        
        # Calculate metrics
        errors = np.sqrt(np.sum(np.square(predictions - self.y_test), axis=1))
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.square(errors)))
        error_95th = np.percentile(errors, 95)
        
        return {
            'predictions': predictions,
            'true_values': self.y_test,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                '95th_percentile_error': error_95th
            },
            'errors': errors
        }
    
    def _visualize_results(self, results: Dict[str, Any]) -> None:
        """
        Visualize the results.
        
        Args:
            results: Dictionary containing evaluation results
        """
        # Create visualization directory
        vis_dir = self.config.get_path('visualization_dir')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot training history
        self._plot_training_history(vis_dir)
        
        # Plot predictions vs true values
        self._plot_predictions(results, vis_dir)
        
        # Plot error distribution
        self._plot_error_distribution(results, vis_dir)
    
    def _plot_training_history(self, vis_dir: str) -> None:
        """
        Plot training history.
        
        Args:
            vis_dir: Directory to save visualizations
        """
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history['mae'], label='Training MAE')
        plt.plot(self.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'training_history.png'))
        plt.close()
    
    def _plot_predictions(self, results: Dict[str, Any], vis_dir: str) -> None:
        """
        Plot predictions vs true values.
        
        Args:
            results: Dictionary containing evaluation results
            vis_dir: Directory to save visualizations
        """
        predictions = results['predictions']
        true_values = results['true_values']
        
        plt.figure(figsize=(10, 10))
        plt.scatter(true_values[:, 0], true_values[:, 1], 
                   label='True Positions', alpha=0.5)
        plt.scatter(predictions[:, 0], predictions[:, 1], 
                   label='Predicted Positions', alpha=0.5)
        
        # Draw lines between true and predicted positions
        for true, pred in zip(true_values, predictions):
            plt.plot([true[0], pred[0]], [true[1], pred[1]], 
                    'r-', alpha=0.1)
        
        plt.title('True vs Predicted Positions')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(vis_dir, 'predictions.png'))
        plt.close()
    
    def _plot_error_distribution(self, results: Dict[str, Any], vis_dir: str) -> None:
        """
        Plot error distribution.
        
        Args:
            results: Dictionary containing evaluation results
            vis_dir: Directory to save visualizations
        """
        errors = results['errors']
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, density=True)
        plt.title('Error Distribution')
        plt.xlabel('Euclidean Distance Error')
        plt.ylabel('Density')
        plt.grid(True)
        
        # Add vertical lines for mean and 95th percentile
        mean_error = np.mean(errors)
        error_95th = np.percentile(errors, 95)
        plt.axvline(mean_error, color='r', linestyle='--', 
                   label=f'Mean Error: {mean_error:.2f}m')
        plt.axvline(error_95th, color='g', linestyle='--', 
                   label=f'95th Percentile: {error_95th:.2f}m')
        plt.legend()
        
        plt.savefig(os.path.join(vis_dir, 'error_distribution.png'))
        plt.close()

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run indoor localization pipeline')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run pipeline
    runner = PipelineRunner(args.config)
    results = runner.run()
    
    # Print results
    print("\nEvaluation Results:")
    print("------------------")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 