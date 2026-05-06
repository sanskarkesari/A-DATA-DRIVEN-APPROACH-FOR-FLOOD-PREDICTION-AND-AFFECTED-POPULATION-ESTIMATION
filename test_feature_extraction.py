#!/usr/bin/env python3
"""
Standalone test script for feature extraction
Tests feature extraction without Flask to identify issues

Usage:
    python test_feature_extraction.py
"""

import sys
import traceback
from feature_extraction import FeatureExtractor

def test_feature_extraction():
    """Test feature extraction for a sample location in Assam"""
    
    print("="*60)
    print("Testing Feature Extraction (Standalone)")
    print("="*60)
    
    # Test location in Assam (Brahmaputra Basin)
    test_lat = 26.2006
    test_lon = 92.9376
    test_date = "2023-07-15"
    
    print(f"\n📍 Test Location:")
    print(f"   Latitude: {test_lat}")
    print(f"   Longitude: {test_lon}")
    print(f"   Date: {test_date}")
    print("\n" + "-"*60)
    
    try:
        # Initialize extractor
        print("\n🔧 Initializing FeatureExtractor...")
        extractor = FeatureExtractor()
        print("✅ FeatureExtractor initialized")
        
        # Extract all features
        print("\n🔍 Starting feature extraction...")
        print("="*60)
        
        features = extractor.extract_all_features(
            latitude=test_lat,
            longitude=test_lon,
            target_date=test_date
        )
        
        print("\n" + "="*60)
        print("✅ Feature Extraction Complete!")
        print("="*60)
        
        # Display results
        print("\n📊 Extracted Features:")
        print("-"*60)
        print(f"  Rainfall (daily):           {features.get('daily_rainfall_mm', 0.0):.2f} mm")
        print(f"  Rainfall (30-day cum):      {features.get('cumulative_rainfall_30d_mm', 0.0):.2f} mm")
        print(f"  Rainfall (avg daily):       {features.get('avg_daily_rainfall_mm', 0.0):.2f} mm")
        print(f"  Soil Moisture:              {features.get('soil_moisture_mm', 0.0):.2f} mm")
        print(f"  Elevation:                 {features.get('elevation_m', 0.0):.2f} m")
        print(f"  Slope:                      {features.get('slope_degree', 0.0):.2f}°")
        print(f"  Flow Accumulation:          {features.get('flow_accumulation', 0.0):.2f}")
        print(f"  NDWI:                       {features.get('ndwi', 0.0):.4f}")
        print("-"*60)
        
        # Check if all features are valid
        required_features = [
            'daily_rainfall_mm',
            'cumulative_rainfall_30d_mm',
            'soil_moisture_mm',
            'elevation_m',
            'slope_degree',
            'flow_accumulation',
            'ndwi'
        ]
        
        missing = [f for f in required_features if f not in features]
        if missing:
            print(f"\n⚠️  Warning: Missing features: {', '.join(missing)}")
        else:
            print("\n✅ All required features extracted successfully!")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return False
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Feature Extraction Failed!")
        print("="*60)
        print(f"\nError: {str(e)}")
        print("\nFull traceback:")
        print("-"*60)
        traceback.print_exc()
        print("-"*60)
        return False


if __name__ == "__main__":
    print("\n")
    success = test_feature_extraction()
    
    print("\n" + "="*60)
    if success:
        print("✅ Test PASSED - Feature extraction works correctly")
        print("   You can now use this in Flask without issues")
    else:
        print("❌ Test FAILED - Fix feature extraction before using Flask")
        print("   Check the error messages above to identify the issue")
    print("="*60)
    print("\n")
    
    sys.exit(0 if success else 1)

