# config_ai.py
import os

class Config:
    """Configuration for the AI Danube discharge prediction project."""
    
    # --- DATA PATHS (adopted from the original project) ---
    # Base path for reanalysis data
    DATA_BASE_PATH = '/nas/home/vlw/Desktop/STREAM/STREAM/'

    # Use the higher-resolution ERA5 data as the primary source
    ERA5_BASE_PATH = '/data/reloclim/normal/ERA5_daily/'
    ERA5_UA_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_025_day_ua850_19500101-20211231.nc')
    ERA5_PR_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_0p25_day_PR_19500101-20221231.nc')
    ERA5_TAS_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_0p25_day_TAS_19500101-20221231.nc')
    
    # Discharge data
    DISCHARGE_FILE = os.path.join(DATA_BASE_PATH, 'danube_discharge_monthly_1893-2021.xlsx')

    # --- REGION DEFINITIONS (from the original project) ---
    # Analysis box for temperature and precipitation (Danube catchment area)
    BOX_LAT_MIN, BOX_LAT_MAX = 46.0, 51.0
    BOX_LON_MIN, BOX_LON_MAX = 8.0, 18.0

    # Jet index boxes
    JET_SPEED_BOX_LAT_MIN, JET_SPEED_BOX_LAT_MAX = 40.0, 60.0
    JET_SPEED_BOX_LON_MIN, JET_SPEED_BOX_LON_MAX = -20.0, 20.0
    JET_LAT_BOX_LAT_MIN, JET_LAT_BOX_LAT_MAX = 30.0, 70.0
    JET_LAT_BOX_LON_MIN, JET_LAT_BOX_LON_MAX = -20.0, 0.0
    
    # Wind level
    WIND_LEVEL = 850  # 850 hPa

    # --- AI MODEL & TRAINING PARAMETERS ---
    # Name of the final CSV file that serves as input for the model
    PROCESSED_DATA_FILE = 'danube_monthly_features_for_ai.csv'
    
    # Model output directory
    MODEL_OUTPUT_DIR = 'ai_model_output'
    
    # Definition of features (predictors) and the target (variable to predict)
    FEATURES = [
        'tas_box', 'pr_box', 'spei_4_box', 
        'jet_speed_djf', 'jet_lat_djf',
        'jet_speed_jja', 'jet_lat_jja'
    ]
    TARGET = 'discharge'
    
    # Look-back window for the LSTM: How many past months do we use 
    # to predict the next month?
    LOOK_BACK_WINDOW = 12  # 12 months
    
    # Data split for training, validation, and testing
    TRAIN_SPLIT_YEAR = 1990
    VALIDATION_SPLIT_YEAR = 2010
    
    # LSTM model hyperparameters
    LSTM_UNITS = 100
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32