"""
Configuration file for Flood Prediction System
Contains all constants and settings for GIS data sources and model parameters
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Google Earth Engine Authentication (user needs to authenticate)
# Run: earthengine authenticate (first time)
GEE_SERVICE_ACCOUNT = os.getenv('GEE_SERVICE_ACCOUNT', None)
GEE_PRIVATE_KEY = os.getenv('GEE_PRIVATE_KEY', None)
GEE_PROJECT = os.getenv('GEE_PROJECT', 'flood-482213')  # Your GEE project name

# NASA POWER API - No API key required (public API)
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Target Region: Assam - Brahmaputra River Basin
# Approximate bounding box for Assam
ASSAM_BOUNDS = {
    'min_lat': 24.0,
    'max_lat': 28.0,
    'min_lon': 89.0,
    'max_lon': 96.0
}

# GIS Dataset Parameters
# SRTM DEM (30m resolution)
SRTM_DATASET = "USGS/SRTMGL1_003"

# NASA SMAP Soil Moisture (Surface, 0-5cm)
SMAP_DATASET ="NASA/SMAP/SPL4SMGP/008"


# Sentinel-2 for NDWI (Normalized Difference Water Index)
SENTINEL2_DATASET = "COPERNICUS/S2_SR_HARMONIZED"

# Date range for training data collection
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2023-12-31"

# Machine Learning Parameters
ML_MODEL_TYPE = "lightgbm"  # Options: "lightgbm" or "randomforest"
RANDOM_STATE = 42

# Flood Risk Thresholds (for classification)
FLOOD_RISK_THRESHOLDS = {
    'low': 0.33,
    'medium': 0.66,
    'high': 1.0
}

# Model file paths (Flood Risk Classification)
MODEL_PATH = "models/flood_prediction_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"

# New model file paths (Affected Population Regression)
# These are used by the population impact module and are completely
# separate from the flood risk classification model above.
AFFECTED_POP_MODEL_PATH = "models/affected_population_model.pkl"
AFFECTED_POP_PREPROCESSOR_PATH = "models/affected_population_preprocessor.pkl"
AFFECTED_POP_META_PATH = "models/affected_population_meta.json"

# Flask API Configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
DEBUG_MODE = True



