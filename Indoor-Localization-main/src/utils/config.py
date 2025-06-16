"""
Configuration Module
==================

This module handles configuration settings and paths for the indoor localization system.
It provides a centralized way to manage settings and paths across the project.
"""

import os
from pathlib import Path
from typing import Dict, Any
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """
    Configuration class for the indoor localization system.
    
    This class manages:
    1. Project paths and directories
    2. Model parameters
    3. Training settings
    4. Data processing settings
    
    Attributes:
        base_dir (Path): Base directory of the project
        config (Dict): Configuration dictionary
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.base_dir = Path(__file__).parent.parent.parent
        self.config = self._load_default_config()
        
        if config_file:
            self.load_config(config_file)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration.
        
        Returns:
            Dictionary containing default configuration
        """
        return {
            # Project paths
            'paths': {
                'data_dir': str(self.base_dir / 'data'),
                'raw_data_dir': str(self.base_dir / 'data' / 'raw'),
                'processed_data_dir': str(self.base_dir / 'data' / 'processed'),
                'model_dir': str(self.base_dir / 'data' / 'models'),
                'log_dir': str(self.base_dir / 'logs'),
                'visualization_dir': str(self.base_dir / 'visualizations')
            },
            
            # Model parameters
            'model': {
                'num_subcarriers': 52,
                'use_rssi': True,
                'batch_size': 32,
                'epochs': 100,
                'patience': 10,
                'learning_rate': 0.001
            },
            
            # Data processing
            'data': {
                'test_size': 0.2,
                'random_state': 42,
                'bandwidth': 20.0,  # MHz
                'num_antennas': 4
            },
            
            # Hardware settings
            'hardware': {
                'router_model': 'TP-Link Archer A5',
                'receiver_model': 'ESP32-WROOM-32',
                'num_antennas': 4
            }
        }
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                
            # Update configuration
            self._update_config(loaded_config)
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            new_config: New configuration dictionary
        """
        def update_dict(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_dict(self.config, new_config)
    
    def save_config(self, config_file: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to save configuration
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_path(self, key: str) -> str:
        """
        Get path from configuration.
        
        Args:
            key: Path key (e.g., 'data_dir', 'model_dir')
            
        Returns:
            Path string
        """
        try:
            return self.config['paths'][key]
        except KeyError:
            logger.error(f"Path key not found: {key}")
            raise
    
    def get_model_param(self, key: str) -> Any:
        """
        Get model parameter from configuration.
        
        Args:
            key: Parameter key
            
        Returns:
            Parameter value
        """
        try:
            return self.config['model'][key]
        except KeyError:
            logger.error(f"Model parameter not found: {key}")
            raise
    
    def get_data_param(self, key: str) -> Any:
        """
        Get data parameter from configuration.
        
        Args:
            key: Parameter key
            
        Returns:
            Parameter value
        """
        try:
            return self.config['data'][key]
        except KeyError:
            logger.error(f"Data parameter not found: {key}")
            raise
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        for path in self.config['paths'].values():
            os.makedirs(path, exist_ok=True)
        logger.info("All directories created successfully")

def get_config(config_file: str = None) -> Config:
    """
    Factory function to get configuration instance.
    
    Args:
        config_file: Path to configuration file (optional)
        
    Returns:
        Config instance
    """
    return Config(config_file)

if __name__ == "__main__":
    # Example usage
    config = get_config()
    config.create_directories()
    
    # Save default configuration
    config.save_config('config.json') 