# model.py
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Concatenate

def create_single_antenna_model(num_subcarriers=52, use_rssi=True):
    """
    Create a deep learning model for single-antenna CSI-based localization
    with optional RSSI integration
    
    Parameters:
    num_subcarriers: Number of subcarriers in the CSI data
    use_rssi: Whether to include RSSI data in the model
    
    Returns:
    Keras model
    """
    # Input for CSI data (amplitude and phase as channels)
    csi_input = Input(shape=(num_subcarriers, 2), name='csi_input')
    
    # CSI processing branch
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(csi_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    csi_features = Flatten()(x)
    
    if use_rssi:
        # RSSI input branch
        rssi_input = Input(shape=(1,), name='rssi_input')
        
        # Combine CSI and RSSI branches
        combined = Concatenate()([csi_features, rssi_input])
        
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.3)(x)
        
        inputs = [csi_input, rssi_input]
    else:
        x = Dense(256, activation='relu')(csi_features)
        x = Dropout(0.3)(x)
        
        inputs = csi_input
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer for (x, y) coordinates
    outputs = Dense(2, activation=None)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(model, X_csi_train, X_rssi_train, y_train, X_csi_val, X_rssi_val, y_val, use_rssi=True):
    """
    Train the model with the provided data
    """

     # Define the output directory
    output_directory = r"C:\MasterArbeit\NewRoom Pipeline\models"
    os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist


    # Prepare inputs based on whether RSSI is used
    if use_rssi:
        train_inputs = [X_csi_train, X_rssi_train]
        val_inputs = [X_csi_val, X_rssi_val]
    else:
        train_inputs = X_csi_train
        val_inputs = X_csi_val
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Define the checkpoint callback to save the best model
    checkpoint_path = os.path.join(output_directory, 'saved_models', 'best_model.keras')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # Create directory if it doesn't exist
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Train model
    history = model.fit(
        train_inputs, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(val_inputs, y_val),
        callbacks=[
            checkpoint,
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
    )
    
    return model, history

    