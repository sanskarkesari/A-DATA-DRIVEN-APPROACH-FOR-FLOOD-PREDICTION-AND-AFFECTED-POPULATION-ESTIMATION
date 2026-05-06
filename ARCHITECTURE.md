# Flood Prediction System - Architecture Documentation

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FLOOD PREDICTION SYSTEM                          │
│                    Assam - Brahmaputra River Basin                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            PRESENTATION LAYER                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Frontend (HTML/CSS/JS)                      │    │
│  │  • Input: Latitude, Longitude, Date (Assam bounds)             │    │
│  │  • Optional: District, River level, Population density         │    │
│  │  • POST /predict → flood risk + features                       │    │
│  │  • POST /predict-affected-population (when optional set)       │    │
│  │  • Risk card + Feature cards + Population card                 │    │
│  │  • Error Handling & Validation                                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↕ HTTP/JSON                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LAYER                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Flask REST API Server                        │    │
│  │  • POST /predict              – Flood risk prediction           │    │
│  │  • POST /features             – Features only (no prediction)   │    │
│  │  • POST /predict-affected-population – Affected population est.  │    │
│  │  • GET  /api/health           – Health + model load status       │    │
│  │  • POST /api/test-feature/<name> – Test single feature          │    │
│  │  • POST /api/test-extraction  – Test extraction (progressive)   │    │
│  │  • POST /api/test-full-extraction – Full extraction, no model   │    │
│  │  • Request Validation • Error Handling • CORS Enabled           │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↕                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           BUSINESS LOGIC LAYER                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              Feature Extraction Module                         │    │
│  │  • Coordinates Feature Extraction                              │    │
│  │  • Multi-source Data Integration                               │    │
│  │  • Error Handling & Fallbacks                                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↕                                           │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              Flood Risk ML Module                               │    │
│  │  • Model Loading (LightGBM or RandomForest per config)         │    │
│  │  • Feature Scaling (StandardScaler)                            │    │
│  │  • Multi-class Risk Prediction (Low/Medium/High)               │    │
│  │  • Probability + hybrid rule-based Low risk when 2-class       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              Affected Population ML Module                      │    │
│  │  • Regression model (LightGBMRegressor)                         │    │
│  │  • Preprocessor (district one-hot + numeric passthrough)       │    │
│  │  • Predict affected_population + confidence interval           │    │
│  │  • Optional: used after flood prediction when inputs provided  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↕                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES LAYER                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│
│  │ NASA POWER  │  │ Google Earth │  │ Google Earth │  │ Google Earth ││
│  │    API      │  │  Engine -    │  │  Engine -    │  │  Engine -    ││
│  │             │  │    SRTM      │  │    SMAP      │  │  Sentinel-2  ││
│  │ • Rainfall  │  │ • Elevation  │  │ • Soil       │  │ • NDWI       ││
│  │ • Daily     │  │ • Slope      │  │   Moisture   │  │ • Water      ││
│  │ • Cumulative│  │ • Flow Acc.  │  │              │  │   Detection  ││
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            PERSISTENCE LAYER                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Flood Risk Model Storage                     │    │
│  │  • flood_prediction_model.pkl (LightGBM or RandomForest)       │    │
│  │  • feature_scaler.pkl (StandardScaler)                         │    │
│  │  • feature_scaler_features.pkl (Feature column metadata)        │    │
│  └────────────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Affected Population Model Storage            │    │
│  │  • affected_population_model.pkl (LightGBMRegressor)           │    │
│  │  • affected_population_preprocessor.pkl (ColumnTransformer)     │    │
│  │  • affected_population_meta.json (metrics/metadata)             │    │
│  └────────────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Training Data Storage                        │    │
│  │  • assam_flood_dataset.csv (flood events + 8 features)          │    │
│  │  • assam_population_impact_dataset.csv (population impact data) │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Detailed Component Architecture

### 1. Frontend Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Components                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  index.html                                                  │
│  ├── Input Form                                              │
│  │   ├── Latitude (24°-28°), Longitude (89°-96°), Date       │
│  │   └── Optional: District, River level (m), Pop. density   │
│  │                                                           │
│  ├── Results Display                                         │
│  │   ├── Risk Level (Low/Medium/High) + Probability %       │
│  │   ├── Feature Cards (8 flood features)                    │
│  │   ├── Interpretation Text                                 │
│  │   └── Population Card (affected people + range, optional)│
│  │                                                           │
│  └── Error Handling                                          │
│      ├── Validation Messages                                 │
│      └── API Error Display                                   │
│                                                              │
│  script.js                                                   │
│  ├── API Integration                                         │
│  ├── Form Validation                                         │
│  ├── Result Rendering                                        │
│  └── Error Handling                                          │
│                                                              │
│  style.css                                                   │
│  ├── Responsive Design                                       │
│  ├── Color-coded Risk Levels                                 │
│  └── Modern UI Components                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Technologies:**
- HTML5, CSS3, JavaScript (ES6+)
- Font Awesome Icons
- Responsive Grid Layout

---

### 2. Flask API Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    Flask Application                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  app.py                                                      │
│  ├── POST /predict                                            │
│  │   ├── Validate lat/lon/date (Assam bounds)                │
│  │   ├── Feature extraction → Flood model → risk + probability│
│  │   └── JSON: features, prediction (flood_risk, level, %)   │
│  ├── POST /features         – Features only, no prediction   │
│  ├── POST /predict-affected-population                        │
│  │   ├── Requires: district, lat, lon, year, month, rainfall, │
│  │   │   soil_moisture_mm, river_water_level_m, elevation_m,  │
│  │   │   flood_risk_level, population_density_per_sqkm        │
│  │   └── JSON: prediction.affected_population, confidence_range│
│  ├── GET /api/health        – status, model_loaded, population_model_loaded │
│  ├── POST /api/test-feature/<name>, /api/test-extraction,     │
│  │      /api/test-full-extraction (diagnostics)               │
│  ├── Error Handling (try/catch, HTTP codes, JSON errors)     │
│  └── CORS enabled                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Technologies:**
- Flask (Python Web Framework)
- Flask-CORS (Cross-Origin Support)
- JSON API Responses

---

### 3. Feature Extraction Module

```
┌─────────────────────────────────────────────────────────────┐
│              Feature Extraction Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: (latitude, longitude, date)                          │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  1. Rainfall Data Extraction                 │           │
│  │     • NASA POWER API (HTTP REST)             │           │
│  │     • Daily & Cumulative (30-day)           │           │
│  │     • Output: 3 features                    │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  2. Elevation & Slope Extraction             │           │
│  │     • Google Earth Engine - SRTM             │           │
│  │     • Elevation sampling                     │           │
│  │     • Slope calculation                      │           │
│  │     • Output: 3 features                    │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  3. Flow Accumulation Calculation             │           │
│  │     • Derived from SRTM elevation             │           │
│  │     • Hydrological analysis                  │           │
│  │     • Output: 1 feature                      │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  4. Soil Moisture Extraction                  │           │
│  │     • Google Earth Engine - SMAP             │           │
│  │     • Surface soil moisture (0-5cm)          │           │
│  │     • Output: 1 feature                      │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  5. NDWI Calculation                         │           │
│  │     • Google Earth Engine - Sentinel-2       │           │
│  │     • Water index from satellite images      │           │
│  │     • Output: 1 feature                      │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  Output: 8 Features + Metadata                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Total Features Extracted: 8**
1. `daily_rainfall_mm`
2. `cumulative_rainfall_30d_mm`
3. `avg_daily_rainfall_mm`
4. `elevation_m`
5. `slope_degree`
6. `flow_accumulation`
7. `soil_moisture_mm`
8. `ndwi`

---

### 4. Machine Learning Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              ML Model Architecture                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Training Phase:                                             │
│  ┌──────────────────────────────────────────────┐           │
│  │  Training Data (synthetic or assam_flood_dataset.csv)    │           │
│  │  ├── Features (8 features per sample)         │           │
│  │  └── Labels (0=Low, 1=Medium, 2=High)       │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  Feature Preparation                         │           │
│  │  ├── Feature Selection                       │           │
│  │  ├── Missing Value Handling                  │           │
│  │  └── Standardization                         │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  Model Training (LightGBM or RandomForest)    │           │
│  │  ├── train_test_split (configurable)         │           │
│  │  └── LightGBM: n_estimators 200, max_depth 10, lr 0.05   │           │
│  │      RandomForest: n_estimators 200, max_depth 15        │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  Model Evaluation                            │           │
│  │  ├── Accuracy, Classification Report         │           │
│  │  └── Confusion Matrix                        │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  Model Persistence                           │           │
│  │  ├── flood_prediction_model.pkl              │           │
│  │  ├── feature_scaler.pkl                     │           │
│  │  └── feature_scaler_features.pkl             │           │
│  └──────────────────────────────────────────────┘           │
│                                                              │
│  Prediction Phase:                                           │
│  ┌──────────────────────────────────────────────┐           │
│  │  New Features (8 features)                    │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  Feature Scaling                              │           │
│  │  └── StandardScaler transformation            │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  Model Prediction                             │           │
│  │  ├── Risk Level (0/1/2)                      │           │
│  │  └── Probabilities [Low, Medium, High]       │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                          │
│  Output: Flood Risk Level + Probability                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Flood Model Details:**
- **Algorithm:** LightGBM or RandomForest (config: `ML_MODEL_TYPE`)
- **Type:** Multi-class Classification
- **Classes:** Low (0), Medium (1), High (2); model may be trained with 2 or 3 classes
- **Inference:** If 2-class model, hybrid rule-based logic can override to Low when conditions (low rainfall, high slope, low soil moisture, low flow accumulation) indicate low risk
- **Output:** `flood_risk_level` (0/1/2), `flood_probability` (%), interpretation text
- **Features:** 8 (daily_rainfall_mm, cumulative_rainfall_30d_mm, avg_daily_rainfall_mm, soil_moisture_mm, elevation_m, slope_degree, flow_accumulation, ndwi)
- **Training:** From `assam_flood_dataset.csv` or synthetic data; StandardScaler fitted on same features

---

### 5. Affected Population Prediction Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              Affected Population Regression                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Inputs (from /predict response + user):                    │
│  • district (categorical)                                    │
│  • latitude, longitude, year, month                          │
│  • daily_rainfall_mm, cumulative_rainfall_30d_mm              │
│  • soil_moisture_mm, river_water_level_m, elevation_m         │
│  • flood_risk_level (from flood model)                       │
│  • population_density_per_sqkm                               │
│    ↓                                                          │
│  Preprocessor (ColumnTransformer)                             │
│  • OneHotEncoder(district, handle_unknown='ignore')          │
│  • Passthrough numeric features                              │
│    ↓                                                          │
│  LightGBMRegressor (log1p target optional)                   │
│    ↓                                                          │
│  Output: affected_population (count) + confidence_range     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Population Model Details:**
- **Algorithm:** LightGBMRegressor (regression)
- **Target:** `affected_population` (people count); log1p transform for skew
- **Training Data:** `assam_population_impact_dataset.csv`
- **Artifacts:** `affected_population_model.pkl`, `affected_population_preprocessor.pkl`, `affected_population_meta.json`
- **Usage:** Optional; frontend calls `/predict-affected-population` after `/predict` when user provides district, river level, and population density

---

## 🔄 Data Flow Diagram

```
┌─────────────┐
│   User      │
│  (Browser)  │
└──────┬──────┘
       │
       │ 1. Input: (lat, lon, date)
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    Flask API Server                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  POST /predict                                       │   │
│  │  • Validate Input                                    │   │
│  │  • Check Bounds (Assam region)                       │   │
│  └──────────────────────────────────────────────────────┘   │
└──────┬───────────────────────────────────────────────────────┘
       │
       │ 2. Extract Features
       ↓
┌─────────────────────────────────────────────────────────────┐
│            Feature Extraction Module                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ NASA POWER  │  │ Google Earth │  │ Google Earth │     │
│  │    API      │  │  Engine      │  │  Engine      │     │
│  │             │  │              │  │              │     │
│  │ Rainfall    │  │ Elevation    │  │ Soil         │     │
│  │ (3 features)│  │ Slope        │  │ Moisture     │     │
│  │             │  │ Flow Acc.    │  │ NDWI         │     │
│  │             │  │ (4 features) │  │ (2 features) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  Output: 8 Features + Metadata                              │
└──────┬───────────────────────────────────────────────────────┘
       │
       │ 3. Prepare Features
       ↓
┌─────────────────────────────────────────────────────────────┐
│            Machine Learning Module                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Load Model & Scaler                                 │   │
│  │  • flood_prediction_model.pkl                         │   │
│  │  • feature_scaler.pkl                                │   │
│  └──────────────────────────────────────────────────────┘   │
│       ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Feature Scaling                                     │   │
│  │  • StandardScaler transformation                     │   │
│  └──────────────────────────────────────────────────────┘   │
│       ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Model Prediction                                    │   │
│  │  • Risk Level: 0/1/2                                 │   │
│  │  • Probabilities: [Low, Medium, High]                │   │
│  └──────────────────────────────────────────────────────┘   │
└──────┬───────────────────────────────────────────────────────┘
       │
       │ 4. Format Response
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    JSON Response (/predict)                  │
│  { success, input, features, prediction: { flood_risk,       │
│    flood_risk_level, flood_probability, interpretation } }   │
└──────┬───────────────────────────────────────────────────────┘
       │
       │ 5a. If district, river level, population density provided:
       │     POST /predict-affected-population (payload from /predict + user)
       ↓
┌─────────────────────────────────────────────────────────────┐
│            Affected Population Model (optional)              │
│  → prediction.affected_population, confidence_range          │
└──────┬───────────────────────────────────────────────────────┘
       │
       │ 5b. Display Results (risk + features + population card if available)
       ↓
┌─────────────┐
│   User      │
│  (Browser)  │
└─────────────┘
```

---

## 🗄️ Data Storage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    File System Storage                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  models/                                                     │
│  ├── flood_prediction_model.pkl   (LightGBM or RandomForest) │
│  ├── feature_scaler.pkl           (StandardScaler)           │
│  ├── feature_scaler_features.pkl  (feature column list)      │
│  ├── affected_population_model.pkl (LightGBMRegressor)        │
│  ├── affected_population_preprocessor.pkl (ColumnTransformer)│
│  └── affected_population_meta.json (metrics/metadata)        │
│                                                              │
│  data/                                                       │
│  ├── assam_flood_dataset.csv       (flood events + 8 features, labels) │
│  └── assam_population_impact_dataset.csv (population impact training)│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Storage Details:**
- **Flood model:** Pickle (classifier + scaler + feature list)
- **Population model:** Pickle (regressor + preprocessor), JSON (meta)
- **Training data:** CSV (flood dataset, population impact dataset)

---

## 🌐 External Services Integration

```
┌─────────────────────────────────────────────────────────────┐
│              External APIs & Services                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  NASA POWER API                                      │   │
│  │  • Endpoint: power.larc.nasa.gov/api/...            │   │
│  │  • Protocol: HTTP REST                               │   │
│  │  • Authentication: None (Public API)                 │   │
│  │  • Data: Daily rainfall (1981-present)              │   │
│  │  • Response Time: ~2-3 seconds                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Google Earth Engine                                 │   │
│  │  • Platform: earthengine.google.com                 │   │
│  │  • Authentication: OAuth2 (earthengine authenticate) │   │
│  │  • Datasets:                                        │   │
│  │    - USGS/SRTMGL1_003 (Elevation)                   │   │
│  │    - NASA/SMAP/SPL4SMGP/008 (Soil Moisture)         │   │
│  │    - COPERNICUS/S2_SR_HARMONIZED (Sentinel-2)       │   │
│  │  • Response Time: ~5-10 seconds per dataset          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔐 Security & Authentication

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Google Earth Engine:                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • OAuth2 Authentication                             │   │
│  │  • Project: flood-482213                             │   │
│  │  • Credentials: ~/.config/earthengine/              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Flask API:                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • CORS Enabled (for frontend)                       │   │
│  │  • Input Validation (lat/lon bounds)                 │   │
│  │  • Error Handling (no sensitive data exposure)       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  NASA POWER:                                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Public API (no authentication)                   │   │
│  │  • Rate limiting handled by delays                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Performance Characteristics

```
┌─────────────────────────────────────────────────────────────┐
│                    Performance Metrics                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Feature Extraction:                                         │
│  ├── NASA POWER API:        ~2-3 seconds                    │
│  ├── SRTM (Elevation/Slope): ~3-4 seconds                   │
│  ├── Flow Accumulation:     ~2-3 seconds                   │
│  ├── SMAP (Soil Moisture):   ~5-7 seconds                   │
│  └── Sentinel-2 (NDWI):      ~3-5 seconds                   │
│  Total: ~15-22 seconds                                       │
│                                                              │
│  Model Prediction:                                           │
│  ├── Feature Scaling:       <0.1 seconds                    │
│  ├── Model Inference:        <0.1 seconds                    │
│  └── Total:                 <0.2 seconds                    │
│                                                              │
│  Total API Response Time: ~15-25 seconds                     │
│                                                              │
│  Concurrent Requests:                                        │
│  ├── Flask: Single-threaded (development)                   │
│  ├── GEE: Rate-limited by API                               │
│  └── Recommendation: Use async/queue for production         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Technology Stack                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Backend:                                                    │
│  ├── Python 3.10+                                            │
│  ├── Flask 2.3+ (Web Framework)                             │
│  ├── Flask-CORS (Cross-Origin Support)                      │
│  ├── LightGBM (Machine Learning)                            │
│  ├── scikit-learn (ML Utilities)                            │
│  ├── pandas (Data Processing)                               │
│  ├── numpy (Numerical Computing)                             │
│  └── earthengine-api (Google Earth Engine)                  │
│                                                              │
│  Frontend:                                                   │
│  ├── HTML5                                                   │
│  ├── CSS3 (Modern Styling)                                   │
│  ├── JavaScript (ES6+)                                       │
│  └── Font Awesome (Icons)                                    │
│                                                              │
│  External Services:                                          │
│  ├── NASA POWER API (HTTP REST)                              │
│  └── Google Earth Engine (Python API)                        │
│                                                              │
│  Data Storage:                                               │
│  ├── Pickle (Model Serialization)                           │
│  └── CSV (Training Data)                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📡 API Endpoints

```
┌─────────────────────────────────────────────────────────────┐
│                    API Endpoints                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. POST /predict                                            │
│     Request: { "latitude", "longitude", "date" }             │
│     Response: success, input, features (8), prediction:      │
│       flood_risk, flood_risk_level (0/1/2), flood_probability│
│       (%), interpretation                                    │
│                                                              │
│  2. POST /features                                           │
│     Same input as /predict; returns features only (no model) │
│                                                              │
│  3. POST /predict-affected-population                        │
│     Request: district, latitude, longitude, year, month,     │
│       daily_rainfall_mm, cumulative_rainfall_30d_mm,          │
│       soil_moisture_mm, river_water_level_m, elevation_m,     │
│       flood_risk_level, population_density_per_sqkm          │
│     Response: success, prediction: affected_population,       │
│       confidence_range, unit; flood_risk_level                │
│     Note: 503 if population model not loaded.                 │
│                                                              │
│  4. GET /api/health                                          │
│     Returns: status, service, version, model_loaded,         │
│       feature_extractor_loaded, population_model_loaded       │
│                                                              │
│  5. POST /api/test-feature/<name>  (elevation|flow|rainfall| │
│       smap|ndwi) – Test single feature extraction            │
│  6. POST /api/test-extraction      – Progressive extraction  │
│  7. POST /api/test-full-extraction – Full extraction, no ML   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 System Workflow

### Complete Request-Response Cycle

```
1. User Input
   └─> Frontend validates (lat: 24-28°, lon: 89-96°, date format)

2. API Request
   └─> Flask receives POST /predict
   └─> Validates JSON payload
   └─> Checks Assam region bounds

3. Feature Extraction (Parallel where possible)
   ├─> NASA POWER: Fetch rainfall (2-3s)
   ├─> GEE SRTM: Extract elevation & slope (3-4s)
   ├─> GEE SRTM: Calculate flow accumulation (2-3s)
   ├─> GEE SMAP: Extract soil moisture (5-7s)
   └─> GEE Sentinel-2: Calculate NDWI (3-5s)
   Total: ~15-22 seconds

4. Feature Preparation
   └─> Create DataFrame
   └─> Align columns with training data
   └─> Handle missing values

5. Model Prediction (flood)
   ├─> Prepare features (align columns, fillna)
   ├─> Scale with StandardScaler
   ├─> Predict risk level (2 or 3 classes)
   ├─> If 2-class and conditions indicate low risk → hybrid Low
   └─> Compute flood_probability (%)
   Total: <0.2 seconds

6. Response Formatting
   └─> Return JSON: success, input, features, prediction

7. (Optional) Affected Population
   └─> Frontend sends /predict-affected-population with /predict output + district, river_water_level_m, population_density_per_sqkm
   └─> Backend: preprocess → regression → affected_population + confidence_range

8. Frontend Display
   └─> Risk card (level + probability + interpretation)
   └─> Feature cards (8 features)
   └─> Population card if population prediction succeeded
```

---

## 🎯 Key Design Decisions

### 1. **Modular Architecture**
- Separate feature extraction from prediction
- Independent modules for each data source
- Easy to add new features or data sources

### 2. **Error Handling**
- Graceful degradation (default values on failure)
- Individual try-catch for each feature
- Never crashes Flask server

### 3. **Hybrid Risk Detection (Flood)**
- ML model may be 2-class (Medium/High) or 3-class (Low/Medium/High)
- When 2-class: rule-based override to Low risk when conditions (low cumulative rainfall, high slope, low soil moisture, low flow accumulation) indicate low risk
- Ensures all three risk levels can be returned to the user

### 4. **Real-time Feature Extraction**
- No pre-computation required
- Always uses latest available data
- Handles future dates (uses current data)

### 5. **Free & Open Datasets**
- No paid APIs required
- All datasets are publicly available
- Sustainable for long-term use

### 6. **Two-Stage Prediction**
- Flood risk is always computed from location + date (feature extraction + classifier).
- Affected population is optional: regression model uses flood_risk_level plus district, river level, population density, and shared features; frontend only calls it when user provides the extra inputs.

---

## 📊 System Scalability

```
Current Capacity:
├── Single-threaded Flask (development)
├── Sequential feature extraction
└── ~15-25 seconds per request

Production Recommendations:
├── Use Gunicorn/uWSGI (multi-worker)
├── Implement async feature extraction
├── Add Redis caching for common locations
├── Use Celery for background processing
└── Deploy on cloud (AWS/GCP/Azure)
```

---

## 🔍 Monitoring & Logging

```
Logging Points:
├── Feature extraction progress
├── API request/response
├── Model prediction results
├── Error messages with tracebacks
└── Performance metrics

Log Output:
├── Console (development)
└── File logging (production ready)
```

---

## 📝 File Structure

```
major project/
├── app.py                         # Flask API (predict, features, predict-affected-population, health, test endpoints)
├── config.py                      # Bounds, paths, ML_MODEL_TYPE, GEE/NASA config
├── feature_extraction.py          # FeatureExtractor: NASA POWER, GEE SRTM/SMAP/S2, 8 features
├── train_model.py                # FloodPredictionModel (LightGBM/RandomForest), train_flood_model()
├── population_model.py            # AffectedPopulationModel (LightGBMRegressor), load/save, predict_with_interval
├── population_preprocessing.py    # ColumnTransformer (district one-hot, numeric passthrough)
├── population_predictor.py        # Batch prediction script for population model
├── requirements.txt               # Dependencies
│
├── models/
│   ├── flood_prediction_model.pkl
│   ├── feature_scaler.pkl
│   ├── feature_scaler_features.pkl
│   ├── affected_population_model.pkl
│   ├── affected_population_preprocessor.pkl
│   └── affected_population_meta.json
│
├── data/
│   ├── assam_flood_dataset.csv
│   └── assam_population_impact_dataset.csv
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── README.md
│
└── tests/
    ├── test_api.py
    ├── test_feature_extraction.py
    └── test_flask_endpoints.py
```

---

## 🎓 For Viva Presentation

### Architecture Highlights:

1. **3-Tier Architecture:**
   - Presentation (Frontend)
   - Application (Flask API)
   - Data Sources (External APIs)

2. **Dual ML Models:**
   - Flood risk: classification (LightGBM/RandomForest), 8 features, optional hybrid Low-risk rules
   - Affected population: regression (LightGBMRegressor), optional second step using flood_risk_level + district, river level, population density

3. **Real-time Processing:**
   - No database required
   - Direct API integration
   - Always uses latest data

4. **Production-Ready:**
   - Error handling
   - Input validation
   - CORS support
   - Comprehensive logging

---

**Last Updated:** 2026-02-09  
**Version:** 1.1.0  
**Status:** Production Ready ✅  

*Architecture aligned with flood risk prediction (LightGBM/RandomForest, 2/3-class, hybrid Low) and affected population regression (LightGBMRegressor, optional frontend integration).*

