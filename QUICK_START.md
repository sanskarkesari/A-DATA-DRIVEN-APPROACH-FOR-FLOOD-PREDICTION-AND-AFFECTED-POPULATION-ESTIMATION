# Quick Start Guide

## Step-by-Step Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Authenticate Google Earth Engine
```bash
earthengine authenticate
```
Follow the browser prompts to complete authentication.

### 3. Train the Model
```bash
python train_model.py
```
This will:
- Generate synthetic training data
- Train the LightGBM model
- Save models to `models/` directory

**Expected output:**
```
Generating synthetic training data...
Training lightgbm model...
Model Accuracy: 0.95xx
Model saved to models/flood_prediction_model.pkl
```

### 4. Start the API Server
```bash
python app.py
```

**Expected output:**
```
============================================================
Flood Prediction System API
============================================================
✓ Model loaded successfully
Starting Flask server...
API will be available at: http://0.0.0.0:5000
```

### 5. Test the API
Open a new terminal and run:
```bash
python test_api.py
```

Or manually test with curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 26.1865, "longitude": 91.7439, "date": "2023-07-15"}'
```

## Example Usage in Python

```python
import requests

# Make prediction request
response = requests.post(
    "http://localhost:5000/predict",
    json={
        "latitude": 26.1865,
        "longitude": 91.7439,
        "date": "2023-07-15"
    }
)

result = response.json()
print(f"Flood Risk: {result['prediction']['flood_risk']}")
print(f"Probability: {result['prediction']['flood_probability']}%")
```

## Common Issues

### Issue: `earthengine` command not found
**Solution:** Make sure Google Earth Engine is installed:
```bash
pip install earthengine-api
```

### Issue: Model files not found
**Solution:** Train the model first:
```bash
python train_model.py
```

### Issue: NASA API timeout
**Solution:** 
- Check internet connection
- API may be slow, wait and retry
- Check if date range is valid

### Issue: GEE authentication error
**Solution:** Re-authenticate:
```bash
earthengine authenticate --force
```

## Next Steps

1. Replace synthetic training data with real historical flood data
2. Fine-tune model parameters for Assam region
3. Add frontend dashboard
4. Deploy to cloud (Heroku, AWS, etc.)

