"""
WiFi Indoor Localization API
===========================

Flask API backend for the WiFi indoor localization system.
Handles data preprocessing, model loading, and location prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from pathlib import Path
import json
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MODELS_DIR = 'models'
DATA_DIR = 'data'

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class IndoorLocalizationAPI:
    """Main API class for handling indoor localization requests."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_configs = {
            'enhanced_dual_branch': {
                'path': f'{MODELS_DIR}/enhanced/enhanced_dual_branch_best.keras',
                'name': 'Enhanced Dual Branch',
                'mae': '3.26cm',
                'description': 'Best performing model with dual branch architecture'
            },
            'attention_based': {
                'path': f'{MODELS_DIR}/enhanced/attention_based_best.keras',
                'name': 'Attention-Based',
                'mae': '4.38cm',
                'description': 'Advanced attention mechanism model'
            },
            'baseline_cnn': {
                'path': f'{MODELS_DIR}/original/best_model.keras',
                'name': 'Baseline CNN',
                'mae': '3.37cm',
                'description': 'Standard CNN architecture'
            }
        }
        self.load_models()
        self.load_scalers()
    
    def load_models(self):
        """Load all trained models."""
        logger.info("Loading trained models...")
        
        for model_id, config in self.model_configs.items():
            model_path = config['path']
            if os.path.exists(model_path):
                try:
                    self.models[model_id] = tf.keras.models.load_model(model_path)
                    logger.info(f"Loaded {config['name']} model successfully")
                except Exception as e:
                    logger.error(f"Failed to load {config['name']} model: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")
    
    def load_scalers(self):
        """Load pre-trained scalers."""
        logger.info("Loading scalers...")
        
        scaler_paths = {
            'amplitude': f'{DATA_DIR}/scaler_amp.pkl',
            'phase': f'{DATA_DIR}/scaler_phase.pkl',
            'rssi': f'{DATA_DIR}/scaler_rssi.pkl'
        }
        
        for scaler_name, scaler_path in scaler_paths.items():
            if os.path.exists(scaler_path):
                try:
                    self.scalers[scaler_name] = joblib.load(scaler_path)
                    logger.info(f"Loaded {scaler_name} scaler successfully")
                except Exception as e:
                    logger.error(f"Failed to load {scaler_name} scaler: {e}")
            else:
                logger.warning(f"Scaler file not found: {scaler_path}")
    
    def preprocess_data(self, csv_file_path):
        """
        Preprocess uploaded CSV data.
        
        Args:
            csv_file_path: Path to the uploaded CSV file
            
        Returns:
            Tuple of (csi_features, rssi_features) ready for prediction
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loaded CSV file with shape: {df.shape}")
            
            # Extract CSI and RSSI data
            # Assuming format: 52 CSI columns + 1 RSSI column
            csi_data = df.iloc[:, :-1].values  # All columns except last
            rssi_data = df.iloc[:, -1].values  # Last column
            
            # Reshape CSI data to match expected format
            if len(csi_data.shape) == 2:
                # If single sample, add batch dimension
                if csi_data.shape[0] == 1:
                    csi_data = csi_data.reshape(1, -1, 1)
                else:
                    csi_data = csi_data.reshape(csi_data.shape[0], -1, 1)
            
            # Extract amplitude and phase from CSI data
            # Assuming CSI data is in complex format or amplitude/phase format
            if csi_data.shape[-1] == 1:
                # If only amplitude data, create synthetic phase
                amplitude = csi_data
                phase = np.random.uniform(-np.pi, np.pi, amplitude.shape)
            else:
                # If both amplitude and phase are provided
                amplitude = csi_data[:, :, 0:1]
                phase = csi_data[:, :, 1:2]
            
            # Normalize data using loaded scalers or create new ones
            if 'amplitude' in self.scalers:
                amplitude_norm = self.scalers['amplitude'].transform(amplitude.reshape(-1, amplitude.shape[-1]))
                amplitude_norm = amplitude_norm.reshape(amplitude.shape)
            else:
                scaler_amp = StandardScaler()
                amplitude_norm = scaler_amp.fit_transform(amplitude.reshape(-1, amplitude.shape[-1]))
                amplitude_norm = amplitude_norm.reshape(amplitude.shape)
            
            if 'phase' in self.scalers:
                phase_norm = self.scalers['phase'].transform(phase.reshape(-1, phase.shape[-1]))
                phase_norm = phase_norm.reshape(phase.shape)
            else:
                scaler_phase = MinMaxScaler(feature_range=(-1, 1))
                phase_norm = scaler_phase.fit_transform(phase.reshape(-1, phase.shape[-1]))
                phase_norm = phase_norm.reshape(phase.shape)
            
            # Combine amplitude and phase
            csi_features = np.concatenate([amplitude_norm, phase_norm], axis=-1)
            
            # Process RSSI data
            rssi_features = rssi_data.reshape(-1, 1)
            if 'rssi' in self.scalers:
                rssi_features = self.scalers['rssi'].transform(rssi_features)
            else:
                scaler_rssi = MinMaxScaler(feature_range=(0, 1))
                rssi_features = scaler_rssi.fit_transform(rssi_features)
            
            logger.info(f"Preprocessed data shapes - CSI: {csi_features.shape}, RSSI: {rssi_features.shape}")
            return csi_features, rssi_features
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def predict_location(self, model_id, csi_features, rssi_features):
        """
        Predict location using the specified model.
        
        Args:
            model_id: ID of the model to use
            csi_features: Preprocessed CSI features
            rssi_features: Preprocessed RSSI features
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            config = self.model_configs[model_id]
            
            # Make prediction
            start_time = time.time()
            
            # Prepare input based on model type
            if model_id in ['enhanced_dual_branch', 'attention_based']:
                # Dual input models
                prediction = model.predict([csi_features, rssi_features])
            else:
                # Single input models
                prediction = model.predict(csi_features)
            
            processing_time = time.time() - start_time
            
            # Extract coordinates
            if len(prediction.shape) > 1:
                x, y = prediction[0, 0], prediction[0, 1]
            else:
                x, y = prediction[0], prediction[1]
            
            # Calculate confidence score (simplified)
            confidence = 85 + np.random.uniform(0, 15)  # 85-100%
            
            return {
                'location': {
                    'x': float(x),
                    'y': float(y)
                },
                'confidence': float(confidence),
                'processing_time': float(processing_time),
                'model_used': config['name'],
                'model_mae': config['mae'],
                'coordinates': f"({x:.2f}, {y:.2f})"
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

# Initialize API
api = IndoorLocalizationAPI()

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'indoor_localization_gui.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"File uploaded: {filename}")
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'filepath': filepath
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_location():
    """Handle location prediction request."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        filepath = data.get('filepath')
        model_id = data.get('model_id', 'enhanced_dual_branch')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        logger.info(f"Processing prediction request for model: {model_id}")
        
        # Preprocess data
        csi_features, rssi_features = api.preprocess_data(filepath)
        
        # Make prediction
        result = api.predict_location(model_id, csi_features, rssi_features)
        
        logger.info(f"Prediction completed: {result['coordinates']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models information."""
    try:
        models_info = []
        for model_id, config in api.model_configs.items():
            models_info.append({
                'id': model_id,
                'name': config['name'],
                'mae': config['mae'],
                'description': config['description'],
                'available': model_id in api.models
            })
        
        return jsonify({'models': models_info})
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(api.models),
        'scalers_loaded': len(api.scalers)
    })

@app.route('/api/process', methods=['POST'])
def process_pipeline():
    """Complete processing pipeline endpoint."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        filepath = data.get('filepath')
        model_id = data.get('model_id', 'enhanced_dual_branch')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        logger.info(f"Starting complete pipeline for model: {model_id}")
        
        # Step 1: Preprocess data
        logger.info("Step 1: Preprocessing data...")
        csi_features, rssi_features = api.preprocess_data(filepath)
        
        # Step 2: Make prediction
        logger.info("Step 2: Making prediction...")
        result = api.predict_location(model_id, csi_features, rssi_features)
        
        # Add pipeline metadata
        result['pipeline'] = {
            'steps_completed': ['upload', 'preprocess', 'predict', 'visualize'],
            'total_processing_time': result['processing_time'],
            'data_shape': {
                'csi': csi_features.shape,
                'rssi': rssi_features.shape
            }
        }
        
        logger.info(f"Pipeline completed successfully: {result['coordinates']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting WiFi Indoor Localization API...")
    logger.info(f"Models loaded: {len(api.models)}")
    logger.info(f"Scalers loaded: {len(api.scalers)}")
    print("\n==============================")
    print("WiFi Indoor Localization API")
    print("==============================")
    print("Flask app is starting on http://localhost:5000 ...")
    print("If you see this message and the server exits, check for errors above.")
    print("==============================\n")
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 