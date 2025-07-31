# predict.py

import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

import config_ai as config
from model import LSTMForecastModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_predictions():
    device = config.DEVICE
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    logging.info("Loading test data and scalers...")
    try:
        test_data = torch.load(config.TEST_DATA_PATH)
        target_scaler = joblib.load(config.TARGET_SCALER_PATH)
    except FileNotFoundError:
        logging.error("Test data or scaler not found. Run 'prepare_dataset.py' and 'train.py' first.")
        return
        
    X_test, y_test = test_data['X'], test_data['y']

    logging.info(f"Loading model from {config.MODEL_PATH}...")
    config.INPUT_SIZE = X_test.shape[2] 
    model = LSTMForecastModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    except FileNotFoundError:
        logging.error(f"Model file not found at {config.MODEL_PATH}. Train the model first.")
        return

    model.eval()

    logging.info("Generating predictions on the test set...")
    with torch.no_grad():
        predictions_scaled = model(X_test.to(device)).cpu().numpy()

    actuals_scaled = y_test.numpy()

    predictions_unscaled = np.zeros_like(predictions_scaled)
    actuals_unscaled = np.zeros_like(actuals_scaled)

    for i in range(config.OUTPUT_SIZE):
      pred_horizon = predictions_scaled[:, i].reshape(-1, 1)
      act_horizon = actuals_scaled[:, i].reshape(-1, 1)
      predictions_unscaled[:, i] = target_scaler.inverse_transform(pred_horizon).flatten()
      actuals_unscaled[:, i] = target_scaler.inverse_transform(act_horizon).flatten()

    logging.info(f"Saving prediction plots to '{config.PLOTS_DIR}'...")
    for i, horizon in enumerate(config.PREDICTION_HORIZONS):
        plt.figure(figsize=(15, 7))
        plt.plot(actuals_unscaled[:, i], label='Actual Water Level', color='blue', alpha=0.8)
        plt.plot(predictions_unscaled[:, i], label='Predicted Water Level', color='red', linestyle='--')
        plt.title(f'Water Level Prediction vs Actual ({horizon} Days Ahead)')
        plt.xlabel('Time Steps (Test Set)')
        plt.ylabel('Water Level / Discharge')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(config.PLOTS_DIR, f'prediction_horizon_{horizon}days.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Plot saved: {plot_path}")

    logging.info("Prediction and plotting finished.")

if __name__ == '__main__':
    run_predictions()