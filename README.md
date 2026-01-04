# Flood Prediction System for India - Assam Brahmaputra River Basin

A comprehensive flood risk prediction system using **Python, Google Earth Engine, and NASA free GIS APIs** for real-time flood risk assessment.

## 🎯 Project Overview

This system predicts flood risk levels (Low/Medium/High) and flood probability (%) for any location in the Assam - Brahmaputra River Basin using:

- **NASA POWER API** - Daily and cumulative rainfall data
- **Google Earth Engine** - For accessing:
  - NASA SMAP (Soil Moisture)
  - SRTM (30m DEM for elevation)
  - Sentinel-2 (NDWI calculation)
- **Machine Learning** - LightGBM/RandomForest models
- **Flask API** - RESTful backend for predictions

## 📋 Features

### GIS & Remote Sensing Features Extracted:

1. **Rainfall Data** (NASA POWER API)
   - Daily rainfall (mm)
   - Cumulative 30-day rainfall (mm)
   - Average daily rainfall

2. **Soil Moisture** (NASA SMAP via Google Earth Engine)
   - Surface soil moisture at 0-5cm depth

3. **Elevation & Topography** (SRTM 30m DEM)
   - Elevation in meters
   - Slope calculation (degrees)

4. **Hydrological Features**
   - Flow accumulation (water collection zones)

5. **Surface Water Detection** (Sentinel-2)
   - NDWI (Normalized Difference Water Index)

### Flood Risk Prediction Logic:

```
High Flood Risk = High Rainfall + Low Slope + High Soil Moisture + High Flow Accumulation + High NDWI
```

The system uses a trained machine learning model to combine these features and predict:
- **Flood Risk Level**: Low / Medium / High
- **Flood Probability**: 0-100%

## 🏗️ Project Structure

```
major project/
│
├── app.py                    # Flask API server
├── feature_extraction.py     # GIS feature extraction module
├── train_model.py           # ML model training pipeline
├── config.py                # Configuration file
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── models/                 # Trained models (created after training)
│   ├── flood_prediction_model.pkl
│   ├── feature_scaler.pkl
│   └── feature_scaler_features.pkl
│
└── .env                    # Environment variables (create if needed)
```

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Google Earth Engine Authentication

First-time setup requires authentication:

```bash
earthengine authenticate
```

This will open a browser window for authentication. Follow the instructions to authenticate your Google account.

**Alternative:** If you have service account credentials, add them to `.env`:
```
GEE_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com
GEE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n..."
```

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Generate synthetic training data (or you can use real historical data)
- Train a LightGBM model
- Save the model to `models/` directory

**Note:** For production, replace synthetic data with real historical flood data from your study area.

### 4. Run the API Server

```bash
python app.py
```

The API will start at: `http://localhost:5000`

## 📡 API Usage

### Health Check

```bash
curl http://localhost:5000/
```

### Flood Prediction

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 26.1865,
    "longitude": 91.7439,
    "date": "2023-07-15"
  }'
```

**Response:**
```json
{
  "success": true,
  "input": {
    "latitude": 26.1865,
    "longitude": 91.7439,
    "date": "2023-07-15"
  },
  "features": {
    "rainfall_mm": 15.5,
    "cumulative_rainfall_30d_mm": 450.2,
    "avg_daily_rainfall_mm": 12.3,
    "soil_moisture_mm": 85.4,
    "elevation_m": 120.5,
    "slope_degree": 5.2,
    "slope_percentage": 9.1,
    "flow_accumulation": 75.3,
    "ndwi": 0.35
  },
  "prediction": {
    "flood_risk": "High",
    "flood_risk_level": 2,
    "flood_probability": 78.5,
    "interpretation": "High flood risk (78.5%). Take immediate precautions. Evacuate if necessary."
  }
}
```

### Get Features Only

```bash
curl -X POST http://localhost:5000/features \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 26.1865,
    "longitude": 91.7439,
    "date": "2023-07-15"
  }'
```

## 🔬 Technical Details

### Feature Extraction Pipeline

1. **Rainfall (NASA POWER API)**
   - Retrieves daily precipitation data
   - Calculates 30-day cumulative rainfall
   - API: `https://power.larc.nasa.gov/api/temporal/daily/point`

2. **Elevation & Slope (SRTM)**
   - Extracts elevation from SRTMGL1_003 dataset (30m resolution)
   - Calculates slope using terrain analysis tools

3. **Flow Accumulation**
   - Analyzes local topography to estimate water collection potential
   - Higher values indicate flood-prone areas

4. **Soil Moisture (SMAP)**
   - Surface soil moisture from NASA SMAP (11km resolution)
   - Indicates saturation level

5. **NDWI (Sentinel-2)**
   - Formula: `NDWI = (Green - NIR) / (Green + NIR)`
   - Values range from -1 to 1
   - Positive values indicate surface water presence

### Machine Learning Model

- **Algorithm**: LightGBM (Gradient Boosting) or RandomForest
- **Features**: 8 input features (rainfall, soil moisture, elevation, slope, flow accumulation, NDWI)
- **Output**: 
  - Classification: Low (0), Medium (1), High (2)
  - Probability: 0-100%

### Flood Risk Logic

The model considers:
- **Rainfall** (30% weight) - Recent precipitation
- **Slope** (20% weight) - Lower slopes = higher risk
- **Soil Moisture** (20% weight) - Saturated soil = higher risk
- **Flow Accumulation** (15% weight) - Water collection zones
- **NDWI** (15% weight) - Existing surface water

## 🎓 Viva/Explanation Guide

### System Architecture

1. **Data Collection Layer**
   - Multiple free/open GIS data sources
   - Real-time and historical data access
   - Automated feature extraction

2. **Feature Engineering**
   - Normalization and scaling
   - Derived features (slope, NDWI)
   - Temporal aggregations (30-day cumulative)

3. **Machine Learning Layer**
   - Trained on historical patterns
   - Handles non-linear relationships
   - Provides probability estimates

4. **API Layer**
   - RESTful interface
   - Input validation
   - Error handling

### Key Advantages

- ✅ **Free & Open Data Sources** - No licensing costs
- ✅ **Scalable** - Can process multiple locations
- ✅ **Real-time** - Live data from satellite sources
- ✅ **Interpretable** - Feature-based predictions
- ✅ **Production-ready** - Modular, well-documented code

### Potential Improvements

- Historical flood event data for training
- Multi-temporal analysis (trends over time)
- Integration with weather forecasts
- Regional model calibration for Assam
- Frontend visualization dashboard

## 📚 Data Sources Reference

1. **NASA POWER**: https://power.larc.nasa.gov/
2. **Google Earth Engine**: https://earthengine.google.com/
3. **SRTM**: https://earthengine.google.com/datasets/catalog/USGS_SRTMGL1_003
4. **SMAP**: https://earthengine.google.com/datasets/catalog/NASA_USDA_HSL_SMAP_soil_moisture
5. **Sentinel-2**: https://earthengine.google.com/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

## 🔧 Configuration

Edit `config.py` to customize:
- Model type (LightGBM/RandomForest)
- Assam bounding box
- API host/port
- Model file paths

## 🐛 Troubleshooting

### GEE Authentication Error
```bash
earthengine authenticate
```

### Model Not Found
```bash
python train_model.py
```

### NASA API Timeout
- Check internet connection
- NASA POWER API may be slow at times
- Increase timeout in `feature_extraction.py`

### Memory Issues
- Reduce batch size in training
- Process locations sequentially

## 📝 License

This project is for educational/academic purposes.

## 👨‍💻 Author

Major College Project - Flood Prediction System

---

**Note**: This system uses synthetic training data for demonstration. For production use, train the model with real historical flood event data from Assam region.

