# config_ai.py

import torch
import os

# --- Data Paths ---
DISCHARGE_BASE_PATH = '/nas/home/vlw/Desktop/STREAM/STREAM/'
DISCHARGE_XLSX_PATH = os.path.join(DISCHARGE_BASE_PATH, 'danube_discharge_monthly_1893-2021.xlsx')

ERA5_PRECIPITATION_NC_PATH = '/data/reloclim/normal/ERA5_daily/ERA5_0p25_day_PR_19500101-20221231.nc'
ERA5_TEMPERATURE_NC_PATH = '/data/reloclim/normal/ERA5_daily/ERA5_0p25_day_TAS_19500101-20221231.nc'
ERA5_WIND_NC_PATH = '/data/reloclim/normal/ERA5_daily/ERA5_025_day_ua850_19500101-20211231.nc'

# --- Correct ERA5 Variable Names ---
# These are the actual variable names found in the NetCDF files.
ERA5_VARS = {
    "precipitation": "tp",
    "temperature": "t2m",
    "wind": "u"
}

# --- Processed Data Paths ---
PROCESSED_DATA_DIR = './processed_data'
TRAIN_DATA_PATH = f'{PROCESSED_DATA_DIR}/train.pt'
VALID_DATA_PATH = f'{PROCESSED_DATA_DIR}/valid.pt'
TEST_DATA_PATH = f'{PROCESSED_DATA_DIR}/test.pt'
SCALER_PATH = f'{PROCESSED_DATA_DIR}/scaler.pkl'
TARGET_SCALER_PATH = f'{PROCESSED_DATA_DIR}/target_scaler.pkl'

# --- Model & Training Parameters ---
SEQUENCE_LENGTH = 120
PREDICTION_HORIZONS = [10, 20, 30]
INPUT_SIZE = -1
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = len(PREDICTION_HORIZONS)
DROPOUT = 0.2
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Prediction & Plotting ---
MODEL_PATH = './danube_predictor.pth'
PLOTS_DIR = './plots'