"""
Model Training Module
===================

This module handles the training process for the indoor localization model.
It includes functionality for training, validation, and model checkpointing.
"""

import os
import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from pathlib import Path
from .model import IndoorLocalizationModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A class for training the indoor localization model.
    
    This class handles:
    1. Model training and validation
    2. Learning rate scheduling
    3. Early stopping
    4. Model checkpointing
    5. Training history tracking
    
    Attributes:
        model (IndoorLocalizationModel): The model to train
        batch_size (int): Batch size for training
        epochs (int): Maximum number of training epochs
        patience (int): Patience for early stopping
        model_dir (str): Directory to save model checkpoints
    """
    
    def __init__(self, model: IndoorLocalizationModel, batch_size: int = 32,
                 epochs: int = 100, patience: int = 10, model_dir: str = 'models'):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            patience: Patience for early stopping
            model_dir: Directory to save model checkpoints
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model_dir = model_dir
        self.history = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              rssi_train: Optional[np.ndarray] = None,
              rssi_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training CSI features
            y_train: Training target coordinates
            X_val: Validation CSI features
            y_val: Validation target coordinates
            rssi_train: Training RSSI values
            rssi_val: Validation RSSI values
            
        Returns:
            Training history dictionary
        """
        # Prepare training data
        train_data = self._prepare_data(X_train, rssi_train)
        val_data = self._prepare_data(X_val, rssi_val) if X_val is not None else None
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Train model
        logger.info("Starting model training...")
        self.history = self.model.model.fit(
            train_data, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_data, y_val) if val_data is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        return self.history.history
    
    def _prepare_data(self, X: np.ndarray, rssi: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Prepare input data for training.
        
        Args:
            X: CSI features
            rssi: RSSI values
            
        Returns:
            Prepared input data
        """
        if rssi is not None and self.model.use_rssi:
            return [X, rssi]
        return X
    
    def _create_callbacks(self) -> list:
        """
        Create training callbacks.
        
        Returns:
            List of callbacks
        """
        callbacks = [
            # Model checkpointing
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2,
                verbose=1
            ),
            
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.model_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def save_training_history(self, filepath: str) -> None:
        """
        Save training history to disk.
        
        Args:
            filepath: Path to save the history
        """
        if self.history is None:
            logger.warning("No training history available to save")
            return
            
        try:
            np.save(filepath, self.history.history)
            logger.info(f"Training history saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                rssi_test: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test CSI features
            y_test: Test target coordinates
            rssi_test: Test RSSI values
            
        Returns:
            Dictionary of evaluation metrics
        """
        test_data = self._prepare_data(X_test, rssi_test)
        
        try:
            metrics = self.model.model.evaluate(test_data, y_test, verbose=1)
            return dict(zip(self.model.model.metrics_names, metrics))
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train indoor localization model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing processed data')
    parser.add_argument('--model_dir', type=str, default='models',
                      help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                      help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # Load data
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))
    
    try:
        rssi_train = np.load(os.path.join(args.data_dir, 'rssi_train.npy'))
        rssi_val = np.load(os.path.join(args.data_dir, 'rssi_val.npy'))
    except:
        rssi_train = rssi_val = None
    
    # Create and train model
    model = IndoorLocalizationModel(use_rssi=(rssi_train is not None))
    trainer = ModelTrainer(
        model=model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        model_dir=args.model_dir
    )
    
    # Train model
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        rssi_train=rssi_train,
        rssi_val=rssi_val
    )
    
    # Save training history
    trainer.save_training_history(
        os.path.join(args.model_dir, 'training_history.npy')
    )

if __name__ == "__main__":
    main() 