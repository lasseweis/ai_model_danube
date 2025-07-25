# train.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config_ai import Config as cfg
from model import build_lstm_model

def create_sequences(data, look_back):
    """Converts a time-series DataFrame into sequences for the LSTM."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :-1]) # Features
        y.append(data[i + look_back, -1])      # Target
    return np.array(X), np.array(y)

def run_training():
    """Executes the entire training workflow."""
    # --- 1. Load and prepare data ---
    df = pd.read_csv(cfg.PROCESSED_DATA_FILE, index_col='time', parse_dates=True)
    
    # Ensure the column order is correct
    df = df[cfg.FEATURES + [cfg.TARGET]]
    
    # Split data (chronologically!)
    df_train = df[df.index.year <= cfg.TRAIN_SPLIT_YEAR]
    df_val = df[(df.index.year > cfg.TRAIN_SPLIT_YEAR) & (df.index.year <= cfg.VALIDATION_SPLIT_YEAR)]
    df_test = df[df.index.year > cfg.VALIDATION_SPLIT_YEAR]

    # --- 2. Scale data ---
    # IMPORTANT: Fit the scaler only on training data!
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(df_train)
    scaled_val = scaler.transform(df_val)
    scaled_test = scaler.transform(df_test)
    
    # --- 3. Create sequences ---
    X_train, y_train = create_sequences(scaled_train, cfg.LOOK_BACK_WINDOW)
    X_val, y_val = create_sequences(scaled_val, cfg.LOOK_BACK_WINDOW)
    X_test, y_test = create_sequences(scaled_test, cfg.LOOK_BACK_WINDOW)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    # --- 4. Build and train model ---
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=cfg.LSTM_UNITS,
        dropout_rate=cfg.DROPOUT_RATE,
        learning_rate=cfg.LEARNING_RATE
    )
    model.summary()
    
    # Callbacks for training
    os.makedirs(cfg.MODEL_OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(cfg.MODEL_OUTPUT_DIR, 'best_model.h5')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # --- 5. Evaluate model ---
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss (MSE): {test_loss}")
    
    # --- 6. Save scaler and training history ---
    with open(os.path.join(cfg.MODEL_OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(cfg.MODEL_OUTPUT_DIR, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
        
    print("\nTraining complete. Model, scaler, and history saved in:", cfg.MODEL_OUTPUT_DIR)

if __name__ == '__main__':
    run_training()