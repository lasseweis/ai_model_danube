# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import numpy as np

import config_ai as config
from model import LSTMForecastModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_training():
    # *** HIER DIE ÄNDERUNG EINFÜGEN ***
    # Limitiere die Anzahl der genutzten CPU-Kerne auf 48
    torch.set_num_threads(48)
    # *** ENDE DER ÄNDERUNG ***

    device = config.DEVICE
    logging.info(f"Using device: {device}")

    # Dieser Log wird jetzt auch die Anzahl der genutzten Threads anzeigen
    logging.info(f"PyTorch is configured to use a maximum of {torch.get_num_threads()} threads.")

    logging.info("Loading preprocessed data...")
    try:
        train_data = torch.load(config.TRAIN_DATA_PATH)
        valid_data = torch.load(config.VALID_DATA_PATH)
    except FileNotFoundError:
        logging.error("Processed data not found. Please run 'prepare_dataset.py' first.")
        return

    X_train, y_train = train_data['X'], train_data['y']
    X_valid, y_valid = valid_data['X'], valid_data['y']

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=config.BATCH_SIZE, shuffle=False)

    config.INPUT_SIZE = X_train.shape[2]
    model = LSTMForecastModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    logging.info("Model architecture:\n" + str(model))

    best_val_loss = float('inf')
    logging.info("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        logging.info(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], Validation Loss: {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_PATH)
            logging.info(f"Model saved with improved validation loss: {best_val_loss:.6f}")

    logging.info("Training finished.")

if __name__ == '__main__':
    run_training()