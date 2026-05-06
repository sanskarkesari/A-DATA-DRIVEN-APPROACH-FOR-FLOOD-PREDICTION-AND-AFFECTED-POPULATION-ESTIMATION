
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import signal
import config
from feature_extraction import FeatureExtractor

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Prevent signals from killing Flask during request processing
def signal_handler(signum, frame):
    """Handle signals gracefully instead of crashing"""
    print(f"\n  Received signal {signum}, but continuing...")
    # Don't exit - just log and continue

# Register signal handlers (but don't override SIGKILL which can't be caught)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Global variables for models and feature extractor
feature_extractor = None
model = None  # Flood risk classification model
population_model = None  # Affected population regression model


def initialize_model():
    """Initialize the trained models and feature extractor"""
    global feature_extractor, model, population_model
    
    # Initialize feature extractor (GIS + remote sensing features)
    try:
        print("  Initializing feature extractor...")
        feature_extractor = FeatureExtractor()
        print(" Feature extractor initialized")
    except Exception as e:
        print(f" Error initializing feature extractor: {e}")
        import traceback
        traceback.print_exc()
        feature_extractor = None
    
    # Load flood risk classification model
    try:
        print("  Loading flood risk classification model...")
        # Lazy import so the server can still start if ML libs are restricted
        from train_model import FloodPredictionModel
        model = FloodPredictionModel(model_type=config.ML_MODEL_TYPE)
        
        if os.path.exists(config.MODEL_PATH) and os.path.exists(config.SCALER_PATH):
            model.load_model(config.MODEL_PATH, config.SCALER_PATH)
            print(" Flood risk model loaded successfully")
        else:
            print("⚠ Warning: Flood risk model files not found. Please train the model first.")
            print("  Run: python train_model.py")
            model = None
    except Exception as e:
        print(f" Error initializing flood risk model: {e}")
        import traceback
        traceback.print_exc()
        model = None
    
    # Load affected population regression model (optional extension)
    try:
        print(" Loading affected population regression model...")
        # Lazy import so the server can still start if ML libs are restricted
        from population_model import AffectedPopulationModel
        pop_model = AffectedPopulationModel()
        if os.path.exists(config.AFFECTED_POP_MODEL_PATH) and os.path.exists(config.AFFECTED_POP_PREPROCESSOR_PATH):
            pop_model.load(
                model_path=config.AFFECTED_POP_MODEL_PATH,
                preprocessor_path=config.AFFECTED_POP_PREPROCESSOR_PATH,
                meta_path=config.AFFECTED_POP_META_PATH,
            )
            population_model = pop_model
            print(" Affected population model loaded successfully")
        else:
            print("ℹ Affected population model not found. Population impact predictions will be disabled.")
            population_model = None
    except Exception as e:
        print(f" Error initializing affected population model: {e}")
        import traceback
        traceback.print_exc()
        population_model = None


@app.route('/risk-map', methods=['GET'])
def district_risk_map():
    """
    Generate and serve an interactive district-wise flood risk map (Assam).

    This endpoint does NOT modify any existing prediction endpoints.
    It generates an HTML file (risk_map.html) and serves it.

    Query params (optional):
      - live=1 : attempt live feature extraction (slower; requires GEE/NASA access)
      - date=YYYY-MM-DD : date for live extraction (default: today)
      - district=Morigaon : if provided, generate map for only that district
    """
    try:
        live = request.args.get('live', '0').strip() in ('1', 'true', 'True', 'yes', 'YES')
        date_str = request.args.get('date', None)
        district = request.args.get('district', None)

        from risk_map import generate_risk_map

        out_path = generate_risk_map(
            output_path="risk_map.html",
            date_str=date_str,
            use_live_features=live,
            district=district,
        )
        return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to generate risk map',
            'details': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'Flood Prediction System API',
        'version': '1.0.0',
        'model_loaded': model is not None,
        'feature_extractor_loaded': feature_extractor is not None,
        'population_model_loaded': population_model is not None
    })


@app.route('/api/test-feature/<feature_name>', methods=['POST'])
def test_individual_feature(feature_name):
    """
    Test individual features one at a time to isolate issues
    Available features: elevation, flow, rainfall, smap, ndwi
    """
    try:
        data = request.get_json(silent=True) or {}
        latitude = float(data.get('latitude', 26.2006))
        longitude = float(data.get('longitude', 92.9376))
        date = data.get('date', '2023-07-15')
        
        print(f"\n Testing individual feature: {feature_name}")
        print(f"   Location: ({latitude}, {longitude})")
        
        if feature_extractor is None:
            return jsonify({'error': 'Feature extractor not initialized'}), 503
        
        result = None
        error = None
        
        try:
            if feature_name == 'elevation':
                result = feature_extractor.get_elevation_and_slope(latitude, longitude)
            elif feature_name == 'flow':
                result = feature_extractor.get_flow_accumulation(latitude, longitude)
            elif feature_name == 'rainfall':
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                result = feature_extractor.get_rainfall_data(latitude, longitude, start_date, end_date)
            elif feature_name == 'smap':
                result = feature_extractor.get_soil_moisture(latitude, longitude, date)
            elif feature_name == 'ndwi':
                result = feature_extractor.get_ndwi(latitude, longitude, date)
            else:
                return jsonify({'error': f'Unknown feature: {feature_name}'}), 400
            
            print(f" {feature_name} test successful")
            return jsonify({
                'success': True,
                'feature': feature_name,
                'result': result
            }), 200
            
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            print(f" {feature_name} test failed: {error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'feature': feature_name,
                'error': error
            }), 500
            
    except Exception as e:
        print(f" Test endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test-full-extraction', methods=['POST'])
def test_full_extraction():
    """
    Test extract_all_features() method directly (without model prediction)
    This helps isolate if the issue is in feature extraction or model prediction
    """
    try:
        data = request.get_json(silent=True) or {}
        latitude = float(data.get('latitude', 26.2006))
        longitude = float(data.get('longitude', 92.9376))
        date = data.get('date', '2023-07-15')
        
        print(f"\n Testing full feature extraction (no model prediction)")
        print(f"   Location: ({latitude}, {longitude}), Date: {date}")
        
        if feature_extractor is None:
            return jsonify({'error': 'Feature extractor not initialized'}), 503
        
        try:
            features = feature_extractor.extract_all_features(latitude, longitude, date)
            print(f" Full feature extraction successful")
            return jsonify({
                'success': True,
                'features': features
            }), 200
            
        except BaseException as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f" Full feature extraction failed: {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    except Exception as e:
        print(f" Test endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict-affected-population', methods=['POST'])
def predict_affected_population():
    """
    Predict affected_population using the regression model.

    This endpoint expects the same features used during training, including:
      - district (categorical)
      - latitude, longitude
      - year, month
      - daily_rainfall_mm
      - cumulative_rainfall_30d_mm
      - soil_moisture_mm
      - river_water_level_m
      - elevation_m
      - flood_risk_level (output of existing flood model)
      - population_density_per_sqkm
    """
    try:
        if population_model is None:
            return jsonify({
                'success': False,
                'error': 'Affected population model not loaded. Train the model and restart the server.'
            }), 503
        
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        # Required fields for regression model
        required_fields = [
            'district',
            'latitude',
            'longitude',
            'year',
            'month',
            'daily_rainfall_mm',
            'cumulative_rainfall_30d_mm',
            'soil_moisture_mm',
            'river_water_level_m',
            'elevation_m',
            'flood_risk_level',
            'population_density_per_sqkm',
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing)}'
            }), 400
        
        # Build single-row DataFrame
        try:
            row = {
                'district': str(data['district']),
                'latitude': float(data['latitude']),
                'longitude': float(data['longitude']),
                'year': int(data['year']),
                'month': int(data['month']),
                'daily_rainfall_mm': float(data['daily_rainfall_mm']),
                'cumulative_rainfall_30d_mm': float(data['cumulative_rainfall_30d_mm']),
                'soil_moisture_mm': float(data['soil_moisture_mm']),
                'river_water_level_m': float(data['river_water_level_m']),
                'elevation_m': float(data['elevation_m']),
                'flood_risk_level': int(data['flood_risk_level']),
                'population_density_per_sqkm': float(data['population_density_per_sqkm']),
            }
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': f'Invalid input format: {e}'
            }), 400
        
        features_df = pd.DataFrame([row])
        
        # Predict affected population with simple confidence interval
        preds, low, high = population_model.predict_with_interval(features_df, interval=1.0)
        pred_value = float(preds[0])
        low_value = max(0.0, float(low[0]))
        high_value = float(high[0])
        
        # Build human-readable confidence range string
        confidence_range = f"{int(round(low_value))}–{int(round(high_value))}"
        
        return jsonify({
            'success': True,
            'input': data,
            'prediction': {
                'affected_population': round(pred_value, 0),
                'confidence_range': confidence_range,
                'unit': 'people'
            },
            'flood_risk_level': int(row['flood_risk_level'])
        }), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to predict affected population',
            'details': str(e)
        }), 500


@app.route('/api/test-extraction', methods=['POST'])
def test_extraction_simple():
    """
    Simple test endpoint to isolate feature extraction issues
    Tests only elevation (fastest feature) to see if GEE works in Flask context
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'No JSON data'}), 400
        
        latitude = float(data.get('latitude', 26.2006))
        longitude = float(data.get('longitude', 92.9376))
        test_type = data.get('test_type', 'elevation')  # elevation, progressive, or all
        
        print(f"\n Testing feature extraction: ({latitude}, {longitude}), type: {test_type}")
        
        if feature_extractor is None:
            return jsonify({'error': 'Feature extractor not initialized'}), 503
        
        results = {}
        errors = {}
        
        # Test 1: Elevation (fastest)
        if test_type in ['elevation', 'progressive', 'all']:
            try:
                print("  → Testing elevation...")
                elev_slope = feature_extractor.get_elevation_and_slope(latitude, longitude)
                results['elevation'] = elev_slope
                print("   Elevation OK")
            except Exception as e:
                errors['elevation'] = str(e)
                print(f"   Elevation failed: {e}")
        
        # Test 2: Flow accumulation
        if test_type in ['progressive', 'all']:
            try:
                print("  → Testing flow accumulation...")
                # Add timeout protection
                import threading
                flow_result = [None]
                flow_error = [None]
                
                def run_flow():
                    try:
                        flow_result[0] = feature_extractor.get_flow_accumulation(latitude, longitude)
                    except Exception as e:
                        flow_error[0] = str(e)
                
                thread = threading.Thread(target=run_flow)
                thread.daemon = True
                thread.start()
                thread.join(timeout=60)  # 60 second timeout
                
                if thread.is_alive():
                    errors['flow_accumulation'] = "Timeout after 60 seconds"
                    print("   Flow accumulation timed out")
                elif flow_error[0]:
                    errors['flow_accumulation'] = flow_error[0]
                    print(f"   Flow accumulation failed: {flow_error[0]}")
                else:
                    results['flow_accumulation'] = flow_result[0]
                    print("  Flow accumulation OK")
            except BaseException as e:
                errors['flow_accumulation'] = f"{type(e).__name__}: {str(e)}"
                print(f"   Flow accumulation failed: {e}")
        
        # Test 3: Rainfall (NASA POWER - not GEE)
        if test_type in ['progressive', 'all']:
            try:
                print("  → Testing rainfall...")
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                rainfall = feature_extractor.get_rainfall_data(latitude, longitude, start_date, end_date)
                results['rainfall'] = rainfall
                print("   Rainfall OK")
            except Exception as e:
                errors['rainfall'] = str(e)
                print(f"   Rainfall failed: {e}")
        
        # Test 4: SMAP soil moisture (often problematic)
        if test_type in ['progressive', 'all']:
            try:
                print("  → Testing SMAP soil moisture...")
                date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
                # Add timeout protection for SMAP
                import threading
                soil_result = [None]
                soil_error = [None]
                
                def run_smap():
                    try:
                        soil_result[0] = feature_extractor.get_soil_moisture(latitude, longitude, date)
                    except Exception as e:
                        soil_error[0] = str(e)
                
                thread = threading.Thread(target=run_smap)
                thread.daemon = True
                thread.start()
                thread.join(timeout=90)  # 90 second timeout for SMAP
                
                if thread.is_alive():
                    errors['soil_moisture'] = "Timeout after 90 seconds"
                    print("   SMAP soil moisture timed out")
                elif soil_error[0]:
                    errors['soil_moisture'] = soil_error[0]
                    print(f"   SMAP soil moisture failed: {soil_error[0]}")
                else:
                    results['soil_moisture'] = soil_result[0]
                    print("   SMAP soil moisture OK")
            except BaseException as e:
                errors['soil_moisture'] = f"{type(e).__name__}: {str(e)}"
                print(f"   SMAP soil moisture failed: {e}")
        
        # Test 5: Sentinel-2 NDWI (often slowest)
        if test_type == 'all':
            try:
                print("  → Testing Sentinel-2 NDWI...")
                date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
                ndwi = feature_extractor.get_ndwi(latitude, longitude, date)
                results['ndwi'] = ndwi
                print("   NDWI OK")
            except Exception as e:
                errors['ndwi'] = str(e)
                print(f"   NDWI failed: {e}")
        
        return jsonify({
            'success': len(errors) == 0,
            'test_type': test_type,
            'results': results,
            'errors': errors if errors else None
        }), 200
            
    except Exception as e:
        print(f" Test endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict_flood():
    """
    Main prediction endpoint
    
    Expected JSON input:
    {
        "latitude": 26.1865,
        "longitude": 91.7439,
        "date": "2023-07-15"
    }
    
    Returns:
    JSON with all features and flood prediction
    """
    # TOP-LEVEL ERROR HANDLER: Ensures Flask never crashes silently
    try:
        # Get and validate input data
        data = request.get_json(silent=True)
        if not data:
            print(" Error: No JSON data provided or invalid JSON")
            return jsonify({
                'success': False,
                'error': 'Invalid or missing JSON body'
            }), 400
        
        # Log incoming request
        print(f"\n  Received prediction request: {data}")
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'date']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            error_msg = f'Missing required fields: {", ".join(missing_fields)}'
            print(f"  {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        try:
            latitude = float(data['latitude'])
            longitude = float(data['longitude'])
            date = data['date'].strip()
        except (ValueError, KeyError, AttributeError) as e:
            error_msg = f'Invalid input format: {str(e)}'
            print(f"  {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Validate latitude/longitude (Assam region approximately)
        if not (config.ASSAM_BOUNDS['min_lat'] <= latitude <= config.ASSAM_BOUNDS['max_lat']):
            error_msg = f'Latitude out of range for Assam region ({latitude})'
            print(f"  {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        if not (config.ASSAM_BOUNDS['min_lon'] <= longitude <= config.ASSAM_BOUNDS['max_lon']):
            error_msg = f'Longitude out of range for Assam region ({longitude})'
            print(f"  {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Validate date format
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            error_msg = f'Invalid date format. Expected YYYY-MM-DD, got: {date}'
            print(f"  {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Check if feature extractor is initialized
        if feature_extractor is None:
            error_msg = 'Feature extractor not initialized. Server may be starting up.'
            print(f"  {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 503  # Service Unavailable
        
        print(f"🔍 Processing prediction request:")
        print(f"   📍 Location: ({latitude}, {longitude})")
        print(f"   📅 Date: {date}")
        
        # Extract features with error handling
        # Use a more defensive approach to catch ALL possible exceptions
        features = None
        extraction_error = None
        
        try:
            print("🔍 Starting feature extraction (this may take 60-120 seconds)...")
            print("   (If this hangs, check Flask console for last progress message)")
            
            # Ensure feature extractor is still valid
            if feature_extractor is None:
                raise ValueError("Feature extractor is None - initialization failed")
            
            features = feature_extractor.extract_all_features(latitude, longitude, date)
            
            if not features:
                extraction_error = "Feature extraction returned no data"
                print(f"  {extraction_error}")
            else:
                print("  Successfully extracted features")
                
        except KeyboardInterrupt:
            extraction_error = "Feature extraction interrupted"
            print(f"  {extraction_error}")
        except SystemExit as e:
            extraction_error = f"System exit during feature extraction: {e}"
            print(f"  {extraction_error}")
            # Don't let SystemExit kill the server
            import sys
            sys.exit = lambda *args: None
        except BaseException as e:
            # Catch ALL exceptions including SystemExit, KeyboardInterrupt, etc.
            extraction_error = f'Feature extraction failed: {type(e).__name__}: {str(e)}'
            print(f"  {extraction_error}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            # Log to file for debugging
            try:
                with open('flask_errors.log', 'a') as f:
                    f.write(f"\n{datetime.now()}: {extraction_error}\n")
                    traceback.print_exc(file=f)
            except:
                pass
        
        if extraction_error or not features:
            return jsonify({
                'success': False,
                'error': 'Failed to extract location features',
                'details': extraction_error or 'Unknown error during feature extraction',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        # Prepare features for model prediction
        try:
            if model is not None:
                # Create DataFrame with features
                features_df = pd.DataFrame([features])
                X = model.prepare_features(features_df)
                
                # Predict
                risk_levels, probabilities = model.predict(X)
                
                # Get prediction results
                risk_level = int(risk_levels[0])
                risk_probs = probabilities[0]
                
                # Handle different number of classes (model might have 2 or 3 classes)
                num_classes = len(risk_probs)
                risk_names = ['Low', 'Medium', 'High']
                
                # Check if conditions suggest Low risk (hybrid approach)
                # Since model was trained on mostly Medium/High, use rule-based for Low
                rainfall = features.get('cumulative_rainfall_30d_mm', 0)
                slope = features.get('slope_degree', 0)
                soil_moisture = features.get('soil_moisture_mm', 0)
                flow_accum = features.get('flow_accumulation', 50)
                
                # Low risk conditions: low rainfall, high slope, low soil moisture
                is_low_risk_conditions = (
                    rainfall < 200 and  # Low cumulative rainfall
                    slope > 15 and      # High slope (water drains)
                    soil_moisture < 50 and  # Low soil moisture
                    flow_accum < 40     # Low flow accumulation
                )
                
                # Map probabilities to risk levels
                if num_classes == 2:
                    # Model trained on 2 classes: map 0->Medium, 1->High
                    if is_low_risk_conditions:
                        # Override with Low risk if conditions clearly indicate it
                        risk_name = 'Low'
                        risk_level = 0
                        # Calculate Low risk probability (inverse of High risk)
                        flood_probability = float((1 - risk_probs[1]) * 100) if len(risk_probs) > 1 else 10.0
                        print("     Using hybrid approach: Conditions indicate Low risk")
                    elif risk_level == 0:
                        risk_name = 'Medium'
                        flood_probability = float(risk_probs[1] * 100)  # Probability of High
                    elif risk_level == 1:
                        risk_name = 'High'
                        flood_probability = float(risk_probs[1] * 100)  # Probability of High
                    else:
                        risk_name = 'High'  # Default
                        flood_probability = float(risk_probs[-1] * 100)
                elif num_classes == 3:
                    # Model has 3 classes: Low (0), Medium (1), High (2)
                    risk_name = risk_names[risk_level] if risk_level < len(risk_names) else f'Level {risk_level}'
                    flood_probability = float(risk_probs[2] * 100)  # Probability of High risk
                else:
                    # Fallback for unexpected number of classes
                    risk_name = risk_names[risk_level] if risk_level < len(risk_names) else f'Level {risk_level}'
                    flood_probability = float(risk_probs[-1] * 100)  # Use last probability
                
                print(f"  Prediction complete - Risk: {risk_name} ({flood_probability:.2f}%)")
                
            else:
                # Fallback: calculate risk manually if model not loaded
                print("⚠ Model not loaded, using rule-based fallback")
                risk_level, risk_name, flood_probability = calculate_fallback_risk(features)
                print(f"  Fallback prediction - Risk: {risk_name} ({flood_probability:.2f}%)")
            
            # Prepare response
            response = {
                'success': True,
                'input': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'date': date
                },
                'features': {
                    'rainfall_mm': round(float(features.get('daily_rainfall_mm', 0.0)), 2),
                    'cumulative_rainfall_30d_mm': round(float(features.get('cumulative_rainfall_30d_mm', 0.0)), 2),
                    'avg_daily_rainfall_mm': round(float(features.get('avg_daily_rainfall_mm', 0.0)), 2),
                    'soil_moisture_mm': round(float(features.get('soil_moisture_mm', 0.0)), 2),
                    'elevation_m': round(float(features.get('elevation_m', 0.0)), 2),
                    'slope_degree': round(float(features.get('slope_degree', 0.0)), 2),
                    'slope_percentage': round(float(features.get('slope_percentage', 0.0)), 2),
                    'flow_accumulation': round(float(features.get('flow_accumulation', 0.0)), 2),
                    'ndwi': round(float(features.get('ndwi', 0.0)), 4)
                },
                'prediction': {
                    'flood_risk': risk_name,
                    'flood_risk_level': risk_level,  # 0=Low, 1=Medium, 2=High
                    'flood_probability': round(flood_probability, 2),
                    'interpretation': get_risk_interpretation(risk_name, flood_probability)
                }
            }
            
            print("  Request processed successfully")
            return jsonify(response), 200
            
        except Exception as e:
            error_msg = f'Prediction failed: {str(e)}'
            print(f"  {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Failed to generate prediction',
                'details': str(e)
            }), 500
            
    except Exception as e:
        # Catch any unexpected errors
        error_msg = f'Unexpected error: {str(e)}'
        print(f"  {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500


def calculate_fallback_risk(features: dict) -> tuple:
    """
    Fallback risk calculation using rule-based approach
    Used when ML model is not available
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        Tuple of (risk_level, risk_name, flood_probability)
    """
    # Extract features
    rainfall = features.get('cumulative_rainfall_30d_mm', 0)
    slope = features.get('slope_degree', 45)
    soil_moisture = features.get('soil_moisture_mm', 0)
    flow_accum = features.get('flow_accumulation', 50)
    ndwi = features.get('ndwi', 0)
    
    # Score components (0-1)
    rainfall_score = min(rainfall / 500.0, 1.0)
    slope_score = 1.0 - min(slope / 45.0, 1.0)  # Lower slope = higher risk
    soil_score = min(soil_moisture / 100.0, 1.0)
    flow_score = flow_accum / 100.0
    ndwi_score = max(0, min((ndwi + 0.3) / 0.6, 1.0))
    
    # Weighted flood score
    flood_score = (
        0.30 * rainfall_score +
        0.20 * slope_score +
        0.20 * soil_score +
        0.15 * flow_score +
        0.15 * ndwi_score
    )
    
    # Determine risk level
    if flood_score < 0.33:
        risk_level = 0
        risk_name = 'Low'
    elif flood_score < 0.66:
        risk_level = 1
        risk_name = 'Medium'
    else:
        risk_level = 2
        risk_name = 'High'
    
    flood_probability = flood_score * 100
    
    return risk_level, risk_name, flood_probability


def get_risk_interpretation(risk_name: str, probability: float) -> str:
    """
    Get human-readable interpretation of flood risk
    
    Args:
        risk_name: Risk level name
        probability: Flood probability percentage
        
    Returns:
        Interpretation string
    """
    if risk_name == 'Low':
        return f"Low flood risk ({probability:.1f}%). Conditions are relatively safe, but monitor weather updates."
    elif risk_name == 'Medium':
        return f"Medium flood risk ({probability:.1f}%). Exercise caution and stay alert to weather warnings."
    else:
        return f"High flood risk ({probability:.1f}%). Take immediate precautions. Evacuate if necessary."


@app.route('/features', methods=['POST'])
def get_features_only():
    """
    Endpoint to get only features without prediction
    Useful for debugging or data collection
    
    Expected JSON input:
    {
        "latitude": 26.1865,
        "longitude": 91.7439,
        "date": "2023-07-15"
    }
    """
    # TOP-LEVEL ERROR HANDLER: Ensures Flask never crashes silently
    try:
        data = request.get_json(silent=True)
        
        if not data:
            print("  Error: No JSON data provided to /features endpoint")
            return jsonify({
                'success': False,
                'error': 'Invalid or missing JSON body'
            }), 400
        
        print(f"\n  Received features request: {data}")
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'date']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            error_msg = f'Missing required fields: {", ".join(missing_fields)}'
            print(f"  {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        try:
            latitude = float(data['latitude'])
            longitude = float(data['longitude'])
            date = data['date'].strip()
        except (ValueError, KeyError, AttributeError) as e:
            error_msg = f'Invalid input format: {str(e)}'
            print(f"  {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Check if feature extractor is initialized
        if feature_extractor is None:
            error_msg = 'Feature extractor not initialized. Server may be starting up.'
            print(f"  {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 503
        
        print(f"🔍 Extracting features:")
        print(f"    Location: ({latitude}, {longitude})")
        print(f"    Date: {date}")
        
        # Extract features with error handling
        try:
            features = feature_extractor.extract_all_features(latitude, longitude, date)
            if not features:
                raise ValueError("Feature extraction returned no data")
            print("  Successfully extracted features")
            
            return jsonify({
                'success': True,
                'features': features
            }), 200
            
        except Exception as e:
            error_msg = f'Feature extraction failed: {str(e)}'
            print(f"  {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Failed to extract location features',
                'details': str(e)
            }), 500
        
    except Exception as e:
        # Catch any unexpected errors
        error_msg = f'Unexpected error: {str(e)}'
        print(f"  {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500


# Optional: Serve frontend from Flask (placed last to not interfere with API routes)
# Frontend can also be opened directly in browser or served separately
@app.route('/', methods=['GET'])
def serve_frontend():
    """Serve the frontend HTML file"""
    return send_from_directory('frontend', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from frontend directory"""
    # Don't serve API routes as static files
    if path.startswith('api/') or path in ['predict', 'features']:
        return jsonify({'error': 'Not found'}), 404
    # Don't try to serve non-existent files - return 404 instead of crashing
    try:
        return send_from_directory('frontend', path)
    except:
        return jsonify({'error': 'Not found'}), 404


if __name__ == '__main__':
    print("="*60)
    print("Flood Prediction System API")
    print("="*60)
    
    # Initialize model
    initialize_model()
    
    print(f"\nStarting Flask server...")
    print(f"API will be available at: http://{config.API_HOST}:{config.API_PORT}")
    print(f"\nAPI Endpoints:")
    print(f"  GET  /api/health    - Health check")
    print(f"  POST /predict       - Get flood prediction")
    print(f"  POST /features      - Get features only (no prediction)")
    print(f"\nFrontend:")
    print(f"  GET  /              - Frontend UI (if enabled)")
    print("\n" + "="*60 + "\n")
    
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG_MODE,
        use_reloader=False  # Disable auto-reload to prevent interrupting requests
    )
