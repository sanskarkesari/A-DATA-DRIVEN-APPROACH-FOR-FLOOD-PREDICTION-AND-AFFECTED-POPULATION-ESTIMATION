"""
Feature Extraction Module for Flood Prediction System
Extracts GIS and remote sensing features using Google Earth Engine and NASA POWER API

Features extracted:
1. Rainfall data (NASA POWER API)
2. Soil moisture (NASA SMAP via GEE)
3. Elevation/DEM (SRTM)
4. Slope (derived from DEM)
5. Flow accumulation (hydrological analysis)
6. NDWI (Sentinel-2)
"""

import ee
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import time
import json
import warnings
import config

# Suppress Earth Engine deprecation warnings to prevent Flask auto-reload issues
warnings.filterwarnings('ignore', category=DeprecationWarning, module='ee')

# Initialize Google Earth Engine with safe fallback
try:
    ee.Initialize(project=config.GEE_PROJECT)
    print("✅ Earth Engine initialized successfully")
except Exception:
    try:
        ee.Authenticate()
        ee.Initialize(project=config.GEE_PROJECT)
        print("✅ Earth Engine initialized after authentication")
    except Exception as e:
        print(f"❌ GEE initialization error: {e}")
        print("Please run: earthengine authenticate")


class FeatureExtractor:
    """
    Main class for extracting flood prediction features from various data sources
    """
    
    def __init__(self):
        """Initialize the feature extractor"""
        self.srtm = ee.Image("USGS/SRTMGL1_003")
        
    def get_rainfall_data(self, latitude: float, longitude: float, 
                         start_date: str, end_date: str) -> Dict:
        """
        Fetch daily and cumulative rainfall data from NASA POWER API
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            
        Returns:
            Dictionary with daily rainfall and cumulative rainfall
        """
        try:
            # Convert dates to NASA POWER format (YYYYMMDD)
            if isinstance(start_date, str) and '-' in start_date:
                start_date = start_date.replace('-', '')
            if isinstance(end_date, str) and '-' in end_date:
                end_date = end_date.replace('-', '')
            
            # NASA POWER API endpoint
            url = (
                "https://power.larc.nasa.gov/api/temporal/daily/point"
                f"?parameters=PRECTOTCORR"
                f"&community=AG"
                f"&longitude={longitude}"
                f"&latitude={latitude}"
                f"&start={start_date}"
                f"&end={end_date}"
                f"&format=JSON"
            )
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract rainfall values
            if 'properties' in data and 'parameter' in data['properties']:
                rainfall_data = data['properties']['parameter']['PRECTOTCORR']
                
                # Convert to DataFrame for easier processing
                dates = []
                rainfall_values = []
                
                for date_str, value in rainfall_data.items():
                    dates.append(pd.to_datetime(date_str))
                    rainfall_values.append(float(value))
                
                df = pd.DataFrame({
                    'date': dates,
                    'rainfall_mm': rainfall_values
                })
                
                # Calculate cumulative rainfall (30-day window)
                df['cumulative_rainfall_30d'] = df['rainfall_mm'].rolling(window=30, min_periods=1).sum()
                
                # Get latest values
                latest_rainfall = df['rainfall_mm'].iloc[-1] if len(df) > 0 else 0.0
                cumulative_rainfall = df['cumulative_rainfall_30d'].iloc[-1] if len(df) > 0 else 0.0
                
                return {
                    'daily_rainfall_mm': round(latest_rainfall, 2),
                    'cumulative_rainfall_30d_mm': round(cumulative_rainfall, 2),
                    'avg_daily_rainfall_mm': round(df['rainfall_mm'].mean(), 2)
                }
            
            return {
                'daily_rainfall_mm': 0.0,
                'cumulative_rainfall_30d_mm': 0.0,
                'avg_daily_rainfall_mm': 0.0
            }
            
        except Exception as e:
            print(f"Error fetching rainfall data: {e}")
            return {
                'daily_rainfall_mm': 0.0,
                'cumulative_rainfall_30d_mm': 0.0,
                'avg_daily_rainfall_mm': 0.0
            }
    
    def get_elevation_and_slope(self, latitude: float, longitude: float) -> Dict:
        """
        Extract elevation and calculate slope from SRTM DEM (30m resolution)
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            
        Returns:
            Dictionary with elevation and slope values
        """
        try:
            point = ee.Geometry.Point([longitude, latitude])
            
            # Extract elevation from SRTM
            elevation = self.srtm.select('elevation')
            print("  → Sampling elevation from SRTM...")
            try:
                elevation_value = elevation.sample(
                    region=point,
                    scale=30,
                    numPixels=1
                ).first().get('elevation').getInfo()
                print(f"  ✓ Elevation: {elevation_value}m")
            except Exception as e:
                print(f"  ⚠ Elevation sampling failed: {e}")
                elevation_value = 0.0
            
            # Calculate slope from DEM
            # Slope is calculated as rise/run * 100 to get percentage
            # Then converted to degrees using arctan
            slope_percentage = ee.Terrain.slope(elevation)
            print("  → Sampling slope from DEM...")
            try:
                slope_value = slope_percentage.sample(
                    region=point,
                    scale=30,
                    numPixels=1
                ).first().get('slope').getInfo()
                print(f"  ✓ Slope: {slope_value}%")
            except Exception as e:
                print(f"  ⚠ Slope sampling failed: {e}")
                slope_value = 0.0
            
            # Convert slope from percentage to degrees
            slope_degrees = np.arctan(slope_value / 100) * (180 / np.pi)
            
            return {
                'elevation_m': round(float(elevation_value), 2),
                'slope_degree': round(float(slope_degrees), 2),
                'slope_percentage': round(float(slope_value), 2)
            }
            
        except Exception as e:
            print(f"Error fetching elevation/slope: {e}")
            return {
                'elevation_m': 0.0,
                'slope_degree': 0.0,
                'slope_percentage': 0.0
            }
    
    def get_flow_accumulation(self, latitude: float, longitude: float) -> Dict:
        """
        Calculate flow accumulation to identify water collection zones
        High flow accumulation = areas where water collects (flood-prone)
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            
        Returns:
            Dictionary with flow accumulation value
        """
        try:
            point = ee.Geometry.Point([longitude, latitude])
            
            # Get elevation at the point
            elevation = self.srtm.select('elevation')
            
            # Create a small region around the point (1km buffer)
            region = point.buffer(1000)  # 1km buffer
            
            # Calculate flow accumulation using hydrological tools
            # Flow accumulation represents the number of cells that drain into each cell
            flow_dir = ee.Terrain.products(elevation).select('flowdir')
            
            # Get flow accumulation (simplified - using contributing area concept)
            # For a more accurate calculation, we'd use flow accumulation algorithm
            # Here we approximate using elevation and local topography
            
            # Sample elevation in surrounding area to estimate local drainage
            elevation_sample = elevation.sample(
                region=region,
                scale=90,  # 90m for faster processing
                numPixels=100
            )
            
            # Get elevation at point
            print("  → Sampling point elevation for flow accumulation...")
            try:
                point_elev = elevation.sample(
                    region=point,
                    scale=30,
                    numPixels=1
                ).first().get('elevation').getInfo()
                print(f"  ✓ Point elevation: {point_elev}m")
            except Exception as e:
                print(f"  ⚠ Point elevation sampling failed: {e}")
                point_elev = 0.0
            
            # Estimate flow accumulation based on local elevation gradient
            # Lower areas (relative to surroundings) = higher flow accumulation
            print("  → Aggregating elevation array for flow accumulation...")
            try:
                elevation_list = elevation_sample.aggregate_array('elevation').getInfo()
                print(f"  ✓ Got {len(elevation_list) if elevation_list else 0} elevation samples")
            except Exception as e:
                print(f"  ⚠ Elevation array aggregation failed: {e}")
                elevation_list = []
            
            if elevation_list and len(elevation_list) > 0:
                mean_elevation = np.mean([float(e) for e in elevation_list if e is not None])
                elevation_diff = mean_elevation - float(point_elev)
                
                # Normalize flow accumulation (0-100 scale)
                # Higher value = more water collection potential
                flow_accumulation = max(0, min(100, elevation_diff * 0.1 + 50))
            else:
                flow_accumulation = 50.0  # Default moderate value
            
            return {
                'flow_accumulation': round(flow_accumulation, 2)
            }
            
        except Exception as e:
            print(f"Error calculating flow accumulation: {e}")
            return {
                'flow_accumulation': 50.0  # Default moderate value
            }
    
    def get_soil_moisture(self, latitude: float, longitude: float, 
                         target_date: str) -> Dict:
        """
        Extract soil moisture from NASA SMAP dataset via Google Earth Engine
        SMAP provides soil moisture at 0-5cm depth (surface soil moisture)
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            target_date: Date to query (YYYY-MM-DD format)
            
        Returns:
            Dictionary with soil moisture values
        """
        try:
            point = ee.Geometry.Point([longitude, latitude])
            
            # NASA SMAP soil moisture dataset (using config)
            smap_collection = ee.ImageCollection(config.SMAP_DATASET)
            
            # Filter by date (get closest image to target date)
            start_date = pd.to_datetime(target_date) - timedelta(days=7)
            end_date = pd.to_datetime(target_date) + timedelta(days=7)
            
            filtered = smap_collection.filterDate(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # SAFE: Check if collection is empty before processing
            print("  → Checking SMAP collection size...")
            try:
                collection_size = filtered.size().getInfo()
                print(f"  ✓ SMAP collection size: {collection_size} images")
            except Exception as e:
                print(f"  ⚠ Failed to get SMAP collection size: {e}")
                collection_size = 0
            
            if collection_size == 0:
                print("  ⚠ SMAP collection is empty, returning default value")
                return {'soil_moisture_mm': 0.0}
            
            # Get the most recent image
            image = filtered.sort('system:time_start', False).first()
            
            # Extract soil moisture - For NASA/SMAP/SPL4SMGP/007, use 'sm_surface' band
            # Try different band names and track which one works
            soil_moisture_band = None
            try:
                soil_moisture = image.select('sm_surface')
                soil_moisture_band = 'sm_surface'
                print("  → Using 'sm_surface' band for SMAP")
            except:
                try:
                    soil_moisture = image.select('ssm')
                    soil_moisture_band = 'ssm'
                    print("  → Using 'ssm' band for SMAP")
                except:
                    soil_moisture = image.select(0)  # Select first band as fallback
                    soil_moisture_band = None
                    print("  → Using first band as fallback for SMAP")
            
            # Sample at the point
            print("  → Sampling soil moisture from SMAP...")
            sample = soil_moisture.sample(
                region=point,
                scale=11000,  # SMAP native resolution ~11km
                numPixels=1
            )
            
            # Get the value using the band name that was selected
            try:
                if soil_moisture_band:
                    print(f"  → Attempting to get '{soil_moisture_band}' band...")
                    soil_moisture_value = sample.first().get(soil_moisture_band).getInfo()
                    print(f"  ✓ Soil moisture ({soil_moisture_band}): {soil_moisture_value}mm")
                else:
                    # If no specific band, try to get first property
                    print("  → Getting first property from sample...")
                    props = sample.first().getInfo()
                    if props.get('properties'):
                        # Get the first numeric property value
                        for key, value in props['properties'].items():
                            if isinstance(value, (int, float)) and value is not None:
                                soil_moisture_value = value
                                print(f"  ✓ Soil moisture ({key}): {soil_moisture_value}mm")
                                break
                        else:
                            soil_moisture_value = None
                    else:
                        soil_moisture_value = None
            except Exception as e:
                print(f"  ⚠ Soil moisture extraction failed: {e}")
                soil_moisture_value = None
            
            if soil_moisture_value is None:
                # If no data, use median of collection
                print("  → No soil moisture value, trying median of collection...")
                try:
                    median_image = smap_collection.median()
                    # Try sm_surface first (correct band for this dataset)
                    try:
                        median_soil = median_image.select('sm_surface')
                        median_band = 'sm_surface'
                    except:
                        try:
                            median_soil = median_image.select('ssm')
                            median_band = 'ssm'
                        except:
                            median_soil = median_image.select(0)
                            median_band = None
                    
                    median_sample = median_soil.sample(
                        region=point,
                        scale=11000,
                        numPixels=1
                    )
                    try:
                        if median_band:
                            soil_moisture_value = median_sample.first().get(median_band).getInfo()
                        else:
                            props = median_sample.first().getInfo()
                            if props.get('properties'):
                                soil_moisture_value = list(props['properties'].values())[0]
                            else:
                                soil_moisture_value = 0.0
                        print(f"  ✓ Soil moisture (median): {soil_moisture_value}mm")
                    except Exception as e:
                        print(f"  ⚠ Median soil moisture extraction failed: {e}")
                        soil_moisture_value = 0.0
                except Exception as e:
                    print(f"  ⚠ Median image creation failed: {e}")
                    soil_moisture_value = 0.0
            
            return {
                'soil_moisture_mm': round(float(soil_moisture_value) if soil_moisture_value else 0.0, 2)
            }
            
        except Exception as e:
            print(f"Error fetching soil moisture: {e}")
            return {
                'soil_moisture_mm': 0.0
            }
    
    def get_ndwi(self, latitude: float, longitude: float, 
                 target_date: str) -> Dict:
        """
        Calculate NDWI (Normalized Difference Water Index) from Sentinel-2
        NDWI = (Green - NIR) / (Green + NIR)
        High NDWI = presence of surface water (potential flooding)
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            target_date: Date to query (YYYY-MM-DD format)
            
        Returns:
            Dictionary with NDWI value
        """
        try:
            point = ee.Geometry.Point([longitude, latitude])
            
            # Sentinel-2 Surface Reflectance dataset
            sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            
            # Filter by date and location
            start_date = pd.to_datetime(target_date) - timedelta(days=30)
            end_date = pd.to_datetime(target_date)
            
            filtered = sentinel2.filterDate(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ).filterBounds(point).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            # SAFE: Check if collection is empty before processing
            print("  → Checking Sentinel-2 collection size...")
            try:
                collection_size = filtered.size().getInfo()
                print(f"  ✓ Sentinel-2 collection size: {collection_size} images")
            except Exception as e:
                print(f"  ⚠ Failed to get Sentinel-2 collection size: {e}")
                collection_size = 0
            
            if collection_size == 0:
                print("  ⚠ Sentinel-2 collection is empty, returning default value")
                return {'ndwi': 0.0}
            
            # Get median composite to reduce cloud effects
            composite = filtered.median()
            
            # Calculate NDWI: (Green - NIR) / (Green + NIR)
            # Sentinel-2 bands: B3 = Green, B8 = NIR
            green = composite.select('B3')
            nir = composite.select('B8')
            
            ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
            
            # Sample at the point
            print("  → Sampling NDWI from Sentinel-2...")
            sample = ndwi.sample(
                region=point,
                scale=10,  # Sentinel-2 resolution is 10m
                numPixels=1
            )
            
            try:
                ndwi_value = sample.first().get('NDWI').getInfo()
                print(f"  ✓ NDWI: {ndwi_value}")
            except Exception as e:
                print(f"  ⚠ NDWI sampling failed: {e}")
                ndwi_value = 0.0
            
            # If no valid data, use a default
            if ndwi_value is None:
                ndwi_value = 0.0
            
            return {
                'ndwi': round(float(ndwi_value), 3)
            }
            
        except Exception as e:
            print(f"Error calculating NDWI: {e}")
            return {
                'ndwi': 0.0
            }
    
    def extract_all_features(self, latitude: float, longitude: float, 
                            target_date: str, start_date: Optional[str] = None) -> Dict:
        """
        Extract all features required for flood prediction
        Uses defensive error handling - if one feature fails, others still work
        
        For future dates, uses latest available data (closest to today)
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            target_date: Target date for prediction (YYYY-MM-DD)
            start_date: Start date for rainfall cumulative calculation (default: 30 days before)
            
        Returns:
            Dictionary with all extracted features
        """
        print(f"Extracting features for location ({latitude}, {longitude}) on {target_date}")
        
        # Check if target_date is in the future
        target_dt = pd.to_datetime(target_date)
        today_dt = pd.to_datetime(datetime.now().date())
        is_future = target_dt > today_dt
        
        if is_future:
            days_ahead = (target_dt - today_dt).days
            print(f"⚠️  Warning: Target date is {days_ahead} days in the future")
            print(f"   Using latest available data (today: {today_dt.strftime('%Y-%m-%d')})")
            # Use today's date for data extraction
            effective_date = today_dt.strftime('%Y-%m-%d')
        else:
            effective_date = target_date
        
        if start_date is None:
            # Default to 30 days before effective date for rainfall
            start_date_obj = pd.to_datetime(effective_date) - timedelta(days=30)
            start_date = start_date_obj.strftime('%Y-%m-%d')
        
        # Convert dates for NASA POWER API (use effective_date for future dates)
        start_date_power = start_date.replace('-', '')
        end_date_power = effective_date.replace('-', '')
        
        # Extract all features with individual error handling
        features = {}
        
        # 1. Rainfall data (NASA POWER - not GEE, usually safe)
        try:
            print("  → Fetching rainfall data from NASA POWER...")
            rainfall = self.get_rainfall_data(latitude, longitude, start_date_power, end_date_power)
            features.update(rainfall)
            print("  ✓ Rainfall data extracted")
        except BaseException as e:
            print(f"  ⚠ Rainfall extraction failed: {e}")
            features.update({
                'daily_rainfall_mm': 0.0,
                'cumulative_rainfall_30d_mm': 0.0,
                'avg_daily_rainfall_mm': 0.0
            })
        
        # Small delay to prevent GEE state issues
        time.sleep(1)
        
        # 2. Elevation and Slope
        try:
            print("  → Extracting elevation and calculating slope from SRTM...")
            elev_slope = self.get_elevation_and_slope(latitude, longitude)
            features.update(elev_slope)
            print("  ✓ Elevation and slope extracted")
        except BaseException as e:
            print(f"  ⚠ Elevation/slope extraction failed: {e}")
            features.update({
                'elevation_m': 0.0,
                'slope_degree': 0.0,
                'slope_percentage': 0.0
            })
        
        # Small delay to prevent GEE state issues
        time.sleep(1)
        
        # 3. Flow Accumulation
        try:
            print("  → Calculating flow accumulation...")
            flow_accum = self.get_flow_accumulation(latitude, longitude)
            features.update(flow_accum)
            print("  ✓ Flow accumulation calculated")
        except BaseException as e:
            print(f"  ⚠ Flow accumulation failed: {e}")
            features.update({'flow_accumulation': 50.0})  # Default moderate value
        
        # Small delay to prevent GEE state issues
        time.sleep(1)
        
        # 4. Soil Moisture
        try:
            print("  → Extracting soil moisture from NASA SMAP...")
            soil_moist = self.get_soil_moisture(latitude, longitude, effective_date)
            features.update(soil_moist)
            print("  ✓ Soil moisture extracted")
        except BaseException as e:
            print(f"  ⚠ Soil moisture extraction failed: {e}")
            features.update({'soil_moisture_mm': 0.0})
        
        # Small delay to prevent GEE state issues
        time.sleep(1)
        
        # 5. NDWI
        try:
            print("  → Calculating NDWI from Sentinel-2...")
            ndwi_data = self.get_ndwi(latitude, longitude, effective_date)
            features.update(ndwi_data)
            print("  ✓ NDWI calculated")
        except BaseException as e:
            print(f"  ⚠ NDWI calculation failed: {e}")
            features.update({'ndwi': 0.0})
        
        # Add metadata
        features['latitude'] = latitude
        features['longitude'] = longitude
        features['date'] = target_date  # Keep original requested date
        features['effective_date'] = effective_date  # Date actually used for data extraction
        if is_future:
            features['is_future_prediction'] = True
            features['days_ahead'] = days_ahead
        
        print(f"  ✓ Feature extraction complete! ({len(features)} features)")
        return features


# Utility function for batch feature extraction
def extract_training_data(locations: list, date_range: tuple) -> pd.DataFrame:
    """
    Extract features for multiple locations and dates (for training data)
    
    Args:
        locations: List of (latitude, longitude) tuples
        date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
        
    Returns:
        DataFrame with extracted features
    """
    extractor = FeatureExtractor()
    all_features = []
    
    start_date, end_date = date_range
    date_list = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for lat, lon in locations:
        for date in date_list:
            try:
                features = extractor.extract_all_features(lat, lon, date.strftime('%Y-%m-%d'))
                all_features.append(features)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error extracting features for ({lat}, {lon}) on {date}: {e}")
                continue
    
    return pd.DataFrame(all_features)


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    # Test location in Assam (Brahmaputra Basin)
    test_lat = 26.1865
    test_lon = 91.7439
    test_date = "2023-07-15"
    
    features = extractor.extract_all_features(test_lat, test_lon, test_date)
    print("\nExtracted Features:")
    print(json.dumps(features, indent=2))



