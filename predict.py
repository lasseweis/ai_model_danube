# predict.py
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from config_ai import Config as cfg
from train import create_sequences # Reuse helper function

def run_prediction():
    """Loads the trained model and visualizes predictions on the test data."""
    
    # --- 1. Load model and scaler ---
    model_path = os.path.join(cfg.MODEL_OUTPUT_DIR, 'best_model.h5')
    scaler_path = os.path.join(cfg.MODEL_OUTPUT_DIR, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Model or scaler not found. Please run 'train.py' first.")
        return

    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # --- 2. Prepare test data ---
    df = pd.read_csv(cfg.PROCESSED_DATA_FILE, index_col='time', parse_dates=True)
    df = df[cfg.FEATURES + [cfg.TARGET]]
    df_test = df[df.index.year > cfg.VALIDATION_SPLIT_YEAR]
    
    scaled_test = scaler.transform(df_test)
    X_test, y_test_scaled = create_sequences(scaled_test, cfg.LOOK_BACK_WINDOW)
    
    # --- 3. Make predictions ---
    predictions_scaled = model.predict(X_test)
    
    # --- 4. Inverse transform predictions and actual values ---
    # Important: The scaler expects the same number of features as during training.
    # We must pad the predictions (1 column) with dummy values for the other features.
    dummy_features = np.zeros((len(predictions_scaled), len(cfg.FEATURES)))
    
    # Inverse transform predictions
    predictions_padded = np.hstack([dummy_features, predictions_scaled])
    predictions = scaler.inverse_transform(predictions_padded)[:, -1] # Only the last column (target)
    
    # Inverse transform actual test values
    y_test_padded = np.hstack([dummy_features, y_test_scaled.reshape(-1, 1)])
    y_test_actual = scaler.inverse_transform(y_test_padded)[:, -1]
    
    # --- 5. Visualize results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Create the time axis for the plot
    plot_index = df_test.index[cfg.LOOK_BACK_WINDOW:]
    
    ax.plot(plot_index, y_test_actual, label='Actual Discharge (Test Data)', color='royalblue', lw=2)
    ax.plot(plot_index, predictions, label='Predicted Discharge (LSTM)', color='crimson', linestyle='--', alpha=0.8)
    
    ax.set_title('Danube River Discharge Prediction with LSTM', fontsize=16, weight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(f'Discharge [{cfg.TARGET}]', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    
    # Save the plot
    plot_filename = os.path.join(cfg.MODEL_OUTPUT_DIR, 'prediction_vs_actual.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Prediction plot saved to: {plot_filename}")
    #plt.show()

if __name__ == '__main__':
    run_prediction()