# prepare_dataset.py

import os
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import joblib
import logging

import config_ai as config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_process_era5(file_path, var_name, desired_lat, desired_lon):
    """Loads and processes a single ERA5 NetCDF file."""
    logging.info(f"Processing ERA5 data from {file_path} for variable '{var_name}'...")
    try:
        ds = xr.open_dataset(file_path)
        data_point = ds[var_name].sel(lat=desired_lat, lon=desired_lon, method='nearest')
        df = data_point.to_dataframe(name=var_name).drop(columns=['lat', 'lon'], errors='ignore')
        
        # Rename the column to be more generic for concatenation
        column_rename_map = {
            'tp': 'precipitation',
            't2m': 'temperature',
            'u': 'wind_u_component'
        }
        df.rename(columns={var_name: column_rename_map.get(var_name, var_name)}, inplace=True)

        if var_name == 'tp': # Special handling for total precipitation
             logging.info("Converting precipitation from 'm' to 'mm/day'.")
             df['precipitation'] *= 1000
             
        return df
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return pd.DataFrame()

def create_cyclical_features(df):
    """Creates sine and cosine features for the day of the year."""
    logging.info("Creating cyclical features for seasonality...")
    df['dayofyear'] = df.index.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df = df.drop(columns=['dayofyear'])
    return df

def create_sequences(features, targets, seq_length, horizons):
    """Creates input sequences and corresponding future targets."""
    logging.info(f"Creating sequences with length {seq_length}...")
    X, y = [], []
    max_horizon = max(horizons)
    
    for i in range(len(features) - seq_length - max_horizon + 1):
        X.append(features[i : i + seq_length])
        current_targets = []
        for h in horizons:
            current_targets.append(targets[i + seq_length + h - 1])
        y.append(current_targets)
    return np.array(X), np.array(y)

def main():
    """Main function to run the data preparation pipeline."""
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    # --- 1. Load Data ---
    danube_lat, danube_lon = 48.2082, 16.3738 # Example: Vienna (please adjust if needed)

    # *** FIXED: Using correct variable names from config ***
    df_pr = load_and_process_era5(config.ERA5_PRECIPITATION_NC_PATH, config.ERA5_VARS["precipitation"], danube_lat, danube_lon)
    df_tas = load_and_process_era5(config.ERA5_TEMPERATURE_NC_PATH, config.ERA5_VARS["temperature"], danube_lat, danube_lon)
    df_ua = load_and_process_era5(config.ERA5_WIND_NC_PATH, config.ERA5_VARS["wind"], danube_lat, danube_lon)
    
    logging.info(f"Loading discharge data from {config.DISCHARGE_XLSX_PATH}...")
    # *** FIXED: Using correct column names from original repository ***
    date_column_name = 'Date (YYYY-MM-DD)'
    discharge_column_name = 'Discharge (m3/s)'
    target_rename = 'discharge'
    
    try:
        df_discharge = pd.read_excel(config.DISCHARGE_XLSX_PATH)
        df_discharge = df_discharge.rename(columns={
            date_column_name: 'date',
            discharge_column_name: target_rename
        })
        df_discharge['date'] = pd.to_datetime(df_discharge['date'])
        df_discharge = df_discharge.set_index('date')
    except FileNotFoundError:
        logging.error(f"Discharge file not found at {config.DISCHARGE_XLSX_PATH}.")
        return
    except KeyError:
        logging.error(f"Could not find expected columns '{date_column_name}' or '{discharge_column_name}'. Please check the Excel file.")
        return

    logging.info("Upsampling monthly discharge data to daily frequency using forward fill.")
    df_discharge_daily = df_discharge.resample('D').ffill()

    # --- 2. Combine and Preprocess ---
    logging.info("Combining all data sources...")
    df_combined = pd.concat([df_discharge_daily, df_pr, df_tas, df_ua], axis=1)
    df_combined = df_combined.dropna()
    df_combined.index = pd.to_datetime(df_combined.index)
    df_combined = df_combined.sort_index()

    df_processed = create_cyclical_features(df_combined.copy())
    
    # --- 3. Split Features and Targets ---
    features_df = df_processed.drop(columns=[target_rename])
    targets_df = df_processed[[target_rename]]

    # --- 4. Scale Data ---
    logging.info("Scaling features and targets...")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    features_scaled = feature_scaler.fit_transform(features_df)
    targets_scaled = target_scaler.fit_transform(targets_df)
    
    joblib.dump(feature_scaler, config.SCALER_PATH)
    joblib.dump(target_scaler, config.TARGET_SCALER_PATH)
    logging.info(f"Scalers saved to {config.PROCESSED_DATA_DIR}")

    # --- 5. Create Sequences ---
    X, y = create_sequences(features_scaled, targets_scaled.flatten(), config.SEQUENCE_LENGTH, config.PREDICTION_HORIZONS)

    if len(X) == 0:
        logging.error("Not enough data to create sequences. Check data length and parameters.")
        return

    # --- 6. Split into Train, Validation, Test Sets ---
    logging.info("Splitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.15, shuffle=False)
    logging.info(f"Training set: {X_train.shape[0]}, Validation set: {X_valid.shape[0]}, Test set: {X_test.shape[0]}")

    # --- 7. Save as PyTorch Tensors ---
    torch.save({'X': torch.FloatTensor(X_train), 'y': torch.FloatTensor(y_train)}, config.TRAIN_DATA_PATH)
    torch.save({'X': torch.FloatTensor(X_valid), 'y': torch.FloatTensor(y_valid)}, config.VALID_DATA_PATH)
    torch.save({'X': torch.FloatTensor(X_test), 'y': torch.FloatTensor(y_test)}, config.TEST_DATA_PATH)
    logging.info(f"Processed data saved to {config.PROCESSED_DATA_DIR}")

if __name__ == '__main__':
    main()