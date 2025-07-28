# prepare_dataset.py (with Dask and error correction)
import sys
import os

# This block allows Python to import modules from the sibling project directory.
# Your setup is assumed to be:
# /nas/home/vlw/Desktop/STREAM/Code/
#  |-- paper1-project/
#  |   |-- data_processing.py
#  |   +-- ...
#  +-- ai_model_danube/
#      +-- prepare_dataset.py  (this file)

# 1. Get the absolute path to the directory containing this script (danube-ai-prediction)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory (which is /nas/home/vlw/Desktop/STREAM/Code/)
parent_dir = os.path.dirname(current_dir)

# 3. Construct the path to your analysis project directory
analysis_project_path = os.path.join(parent_dir, 'paper1-project')

# 4. Add this path to Python's list of search paths
if analysis_project_path not in sys.path:
    sys.path.append(analysis_project_path)

import pandas as pd
import xarray as xr
import numpy as np
import logging
import dask
from dask.distributed import Client

# Import the original classes from your previous project
from data_processing import DataProcessor
from jet_analyzer import JetStreamAnalyzer
from stats_analyzer import StatsAnalyzer
from config_ai import Config as cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_monthly_indices(da_monthly):
    """
    Calculates monthly jet indices from monthly U850 data.
    Unlike the seasonal calculation, this produces a value for each month.
    """
    if da_monthly is None:
        return None, None
        
    # Jet speed index for each month
    jet_speed_monthly = JetStreamAnalyzer.calculate_jet_speed_index(da_monthly)
    
    # Jet latitude index for each month
    jet_lat_monthly = JetStreamAnalyzer.calculate_jet_lat_index(da_monthly)
    
    return jet_speed_monthly, jet_lat_monthly


def run_data_preparation():
    """
    Executes the entire data preparation process, parallelized with Dask.
    1. Loads ERA5 data in parallel.
    2. Calculates monthly box averages and indices.
    3. Loads discharge data.
    4. Combines everything into a single DataFrame.
    5. Saves the final dataset.
    """
    # Start a local Dask client to utilize all available CPU cores.
    client = Client()
    logging.info(f"Dask client started, dashboard available at: {client.dashboard_link}")
    
    logging.info("Starting data preparation for the AI model...")

    # --- 1. Load and process climate data in parallel ---
    logging.info("Lazily scheduling ERA5 data loading tasks...")
    
    lazy_pr = dask.delayed(DataProcessor.process_era5_file)(cfg.ERA5_PR_FILE, 'pr')
    lazy_tas = dask.delayed(DataProcessor.process_era5_file)(cfg.ERA5_TAS_FILE, 'tas')
    lazy_ua850 = dask.delayed(DataProcessor.process_era5_file)(cfg.ERA5_UA_FILE, 'u', 'ua', level_val=cfg.WIND_LEVEL)

    logging.info("Executing data loading tasks in parallel...")
    pr_monthly, tas_monthly, ua850_monthly = dask.compute(lazy_pr, lazy_tas, lazy_ua850)
    logging.info("Finished parallel data loading.")

    # --- 2. Calculate spatial means for the box (monthly) ---
    logging.info("Calculating monthly spatial means (box)...")
    pr_box_monthly = DataProcessor.calculate_spatial_mean(pr_monthly, cfg.BOX_LAT_MIN, cfg.BOX_LAT_MAX, cfg.BOX_LON_MIN, cfg.BOX_LON_MAX)
    tas_box_monthly = DataProcessor.calculate_spatial_mean(tas_monthly, cfg.BOX_LAT_MIN, cfg.BOX_LAT_MAX, cfg.BOX_LON_MIN, cfg.BOX_LON_MAX)

    # --- 3. Calculate SPEI for the box (monthly) ---
    logging.info("Calculating monthly SPEI...")
    lat_center_of_box = (cfg.BOX_LAT_MIN + cfg.BOX_LAT_MAX) / 2
    spei_4_box_monthly = DataProcessor.calculate_spei(pr_box_monthly, tas_box_monthly, lat=lat_center_of_box, scale=4)

    # --- 4. Calculate seasonal jet indices (to be mapped to months) ---
    logging.info("Calculating seasonal jet indices...")
    ua850_seasonal = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(ua850_monthly))
    
    jet_data = {}
    for season, s_label in [('Winter', 'djf'), ('Summer', 'jja')]:
        ua_season = DataProcessor.filter_by_season(ua850_seasonal, season)
        jet_data[f'jet_speed_{s_label}'] = JetStreamAnalyzer.calculate_jet_speed_index(ua_season)
        jet_data[f'jet_lat_{s_label}'] = JetStreamAnalyzer.calculate_jet_lat_index(ua_season)

    # --- 5. Load discharge data ---
    logging.info("Loading Danube discharge data...")
    discharge_df = pd.read_excel(cfg.DISCHARGE_FILE, usecols='A,H', names=['time', 'discharge'])
    discharge_df['time'] = pd.to_datetime(discharge_df['time'], dayfirst=True)
    discharge_df = discharge_df.set_index('time').dropna()

    # --- 6. Combine all data into one DataFrame ---
    logging.info("Combining all time series into a final DataFrame...")
    
    # Create the base dataframe with the monthly data
    df = pd.DataFrame({
        'tas_box': tas_box_monthly.to_series(),
        'pr_box': pr_box_monthly.to_series(),
        'spei_4_box': spei_4_box_monthly.to_series()
    })
    
    # Add the seasonal jet indices
    for key, da in jet_data.items():
        if da is not None:
            # Convert the DataArray (indexed by 'year') to a pandas Series
            jet_series = da.to_series()
            jet_series.name = key
            
            # Create a DatetimeIndex to align the yearly data correctly
            # DJF (Winter) is assigned to December of the respective year
            # JJA (Summer) is assigned to June of the respective year
            if 'djf' in key:
                month = 12
            elif 'jja' in key:
                month = 6
            else:
                continue # Skip if other seasons are present

            new_index = [pd.to_datetime(f'{year}-{month}-01') for year in jet_series.index]
            jet_series.index = new_index
            
            # Join with the main dataframe
            df = df.join(jet_series, how='left')
    
    # Forward-fill the seasonal values to propagate them to subsequent months
    for col in jet_data.keys():
        if col in df.columns:
            df[col] = df[col].ffill()

    # Add the target variable (discharge) and clean up NaN values
    final_df = df.join(discharge_df, how='inner') 
    final_df = final_df.dropna() 

    # --- 7. Save the final dataset ---
    final_df.to_csv(cfg.PROCESSED_DATA_FILE)
    logging.info(f"Data preparation complete. Final dataset saved to: {cfg.PROCESSED_DATA_FILE}")
    logging.info(f"Shape of the dataset: {final_df.shape}")
    logging.info(f"Time range: {final_df.index.min()} to {final_df.index.max()}")
    logging.info(f"Available columns: {final_df.columns.tolist()}")
    
    # --- 8. Shutdown Dask client ---
    client.close()


if __name__ == '__main__':
    run_data_preparation()