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

def load_and_process_era5_grid(file_path, var_name, lat_min, lat_max, lon_min, lon_max, pressure_level=None):
    """
    Loads ERA5 data, slices a bounding box, selects a pressure level if specified,
    and flattens the grid cells into individual feature columns.
    """
    logging.info(f"Processing ERA5 data from {file_path} for var '{var_name}' as a spatial grid...")
    try:
        ds = xr.open_dataset(file_path)

        if 'longitude' in ds.coords and ds['longitude'].max() > 180:
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')

        if pressure_level:
            if 'level' in ds.coords:
                logging.info(f"Selecting pressure level {pressure_level} hPa/millibars from 'level' coordinate...")
                ds = ds.sel(level=pressure_level, method='nearest')
            elif 'plev' in ds.coords:
                logging.info(f"Selecting pressure level {pressure_level} hPa/millibars from 'plev' coordinate...")
                ds = ds.sel(plev=pressure_level, method='nearest')
            else:
                 logging.warning(f"Pressure level coordinate not found in {file_path}. Cannot select level {pressure_level}.")

        data_box = ds[var_name].sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )

        df_flat = data_box.to_dataframe()

        df_pivot = df_flat.reset_index().pivot_table(
            index='time',
            columns=['latitude', 'longitude'],
            values=var_name
        )
        
        # *** FIXED: Normalize the index to remove the time component ***
        df_pivot.index = pd.to_datetime(df_pivot.index).normalize()
        df_pivot.index.name = 'date'


        base_name_map = {
            'tp': 'precipitation',
            't2m': 'temperature',
            'u': 'wind_u_component'
        }
        base_name = base_name_map.get(var_name, var_name)

        df_pivot.columns = [f"{base_name}_lat{lat:.2f}_lon{lon:.2f}" for lat, lon in df_pivot.columns]

        if var_name == 'tp':
             logging.info("Converting precipitation from 'm' to 'mm/day'.")
             df_pivot *= 1000

        return df_pivot
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
    HYDROLOGICAL_BOX_LAT_MIN, HYDROLOGICAL_BOX_LAT_MAX = 46.0, 51.0
    HYDROLOGICAL_BOX_LON_MIN, HYDROLOGICAL_BOX_LON_MAX = 8.0, 18.0

    JET_STREAM_BOX_LAT_MIN, JET_STREAM_BOX_LAT_MAX = 40.0, 60.0
    JET_STREAM_BOX_LON_MIN, JET_STREAM_BOX_LON_MAX = -20.0, 20.0

    df_pr = load_and_process_era5_grid(config.ERA5_PRECIPITATION_NC_PATH, config.ERA5_VARS["precipitation"], HYDROLOGICAL_BOX_LAT_MIN, HYDROLOGICAL_BOX_LAT_MAX, HYDROLOGICAL_BOX_LON_MIN, HYDROLOGICAL_BOX_LON_MAX)
    df_tas = load_and_process_era5_grid(config.ERA5_TEMPERATURE_NC_PATH, config.ERA5_VARS["temperature"], HYDROLOGICAL_BOX_LAT_MIN, HYDROLOGICAL_BOX_LAT_MAX, HYDROLOGICAL_BOX_LON_MIN, HYDROLOGICAL_BOX_LON_MAX)
    df_ua = load_and_process_era5_grid(
        config.ERA5_WIND_NC_PATH,
        config.ERA5_VARS["wind"],
        JET_STREAM_BOX_LAT_MIN,
        JET_STREAM_BOX_LAT_MAX,
        JET_STREAM_BOX_LON_MIN,
        JET_STREAM_BOX_LON_MAX,
        pressure_level=850
    )

    logging.info(f"Loading discharge data from {config.DISCHARGE_XLSX_PATH}...")

    target_rename = 'discharge'

    try:
        df_discharge = pd.read_excel(config.DISCHARGE_XLSX_PATH)
        df_discharge['date'] = pd.to_datetime(df_discharge['year'].astype(str) + '-' + df_discharge['month'].astype(str) + '-01')
        df_discharge = df_discharge.set_index('date')
        df_discharge = df_discharge[['Wien']].rename(columns={'Wien': target_rename})

    except FileNotFoundError:
        logging.error(f"Discharge file not found at {config.DISCHARGE_XLSX_PATH}.")
        return
    except KeyError:
        logging.error(f"Could not find expected columns 'year', 'month' or 'Wien'. Please check the Excel file.")
        return

    logging.info("Upsampling monthly discharge data to daily frequency using forward fill.")
    df_discharge_daily = df_discharge.resample('D').ffill()

    # --- 2. Combine and Preprocess ---
    logging.info("Combining all data sources...")
    
    all_dfs = [df_discharge_daily, df_pr, df_tas, df_ua]
    all_dfs = [df for df in all_dfs if not df.empty]

    if len(all_dfs) < 4:
        logging.error("One or more data sources could not be loaded or are empty. Aborting.")
        return

    df_combined = pd.concat(all_dfs, axis=1, join='inner')

    if df_combined.empty:
        logging.error("The combined dataframe is empty after inner join. Check for time overlap in data sources.")
        return

    df_combined.index = pd.to_datetime(df_combined.index)
    df_combined = df_combined.sort_index()

    df_processed = create_cyclical_features(df_combined.copy())

    # --- 3. Split Features and Targets ---
    features_df = df_processed.drop(columns=[target_rename])
    targets_df = df_processed[[target_rename]]

    logging.info(f"Total number of features created: {len(features_df.columns)}")

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