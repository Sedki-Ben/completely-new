# Indoor Localization System

## Overview
This project implements an indoor localization system using WiFi Channel State Information (CSI) and Received Signal Strength Indicator (RSSI) measurements. The system uses a TP-Link Archer A5 router (4 antennas) and an ESP32-WROOM-32 board as the receiver to capture and process wireless signals for accurate indoor positioning.

## Hardware Requirements
- TP-Link Archer A5 router (4 antennas)
- ESP32-WROOM-32 board
- Computer with Python 3.8+ installed

## Software Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn
- pandas
- matplotlib (for visualization)

## Project Structure
```
Indoor-Localization-main/
├── src/                    # Source code directory
│   ├── data/              # Data processing modules
│   │   ├── __init__.py
│   │   ├── csi_processor.py
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── models/            # Model-related modules
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── trainer.py
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   ├── visualizer.py
│   │   └── config.py
│   └── __init__.py
├── data/                  # Data directory
│   ├── raw/              # Raw CSI and RSSI data
│   ├── processed/        # Processed data files
│   └── models/           # Saved model checkpoints
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
├── setup.py            # Package setup file
└── README.md           # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Indoor-Localization.git
cd Indoor-Localization
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Collection

### Hardware Setup
1. Connect the ESP32-WROOM-32 board to your computer
2. Ensure the TP-Link Archer A5 router is powered and connected to the network
3. Place the router and receiver in the desired locations for data collection

### Data Collection Process
1. Run the data collection script:
```bash
python src/data/collect_data.py --router_ip <ROUTER_IP> --output_dir data/raw
```

2. The script will collect:
   - CSI measurements (amplitude and phase)
   - RSSI values
   - Location coordinates

## Data Processing

1. Process raw CSI data:
```bash
python src/data/csi_processor.py --input_dir data/raw --output_dir data/processed
```

2. Preprocess the data for training:
```bash
python src/data/preprocessor.py --input_dir data/processed --output_dir data/processed
```

## Model Training

1. Train the model:
```bash
python src/models/trainer.py --data_dir data/processed --model_dir data/models
```

2. Monitor training progress:
```bash
tensorboard --logdir data/models/logs
```

## Model Evaluation

1. Evaluate the model:
```bash
python src/models/evaluate.py --model_path data/models/best_model.keras --test_data data/processed/test_data.npy
```

## Visualization

1. Generate visualizations:
```bash
python src/utils/visualizer.py --data_dir data/processed --output_dir visualizations
```

## Project Details

### CSI Processing
- Bandwidth: 20MHz (configurable)
- Subcarriers: 64 total
  - 52 data-carrying subcarriers
  - 12 non-data subcarriers (guard bands, DC subcarriers, pilot tones)

### Model Architecture
- Input: CSI (amplitude and phase) + RSSI
- Architecture: CNN-based with dual input branches
- Output: (x, y) coordinates
- Training: Adam optimizer with MSE loss

### Performance Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- 95th percentile error

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Master thesis project
- Special thanks to [Advisor Name] for guidance and support

## Contact
For questions and support, please open an issue in the repository.
