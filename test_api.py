"""
Test script for Flood Prediction API
Usage: python test_api.py
"""

import requests
import json
from datetime import datetime, timedelta

API_BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check endpoint...")
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_prediction():
    """Test prediction endpoint"""
    print("Testing prediction endpoint...")
    
    # Test location in Assam (near Guwahati)
    test_data = {
        "latitude": 26.1865,
        "longitude": 91.7439,
        "date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    }
    
    print(f"Input: {json.dumps(test_data, indent=2)}")
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction successful!")
        print(f"\nFeatures extracted:")
        for key, value in result['features'].items():
            print(f"  {key}: {value}")
        
        print(f"\nPrediction:")
        print(f"  Flood Risk: {result['prediction']['flood_risk']}")
        print(f"  Flood Probability: {result['prediction']['flood_probability']}%")
        print(f"  Interpretation: {result['prediction']['interpretation']}")
    else:
        print(f"Error: {response.text}")
    
    print()

def test_features_only():
    """Test features-only endpoint"""
    print("Testing features-only endpoint...")
    
    test_data = {
        "latitude": 26.1865,
        "longitude": 91.7439,
        "date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    }
    
    response = requests.post(
        f"{API_BASE_URL}/features",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Features extracted successfully!")
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.text}")
    
    print()

if __name__ == "__main__":
    print("="*60)
    print("Flood Prediction API Test Suite")
    print("="*60)
    print()
    
    try:
        test_health_check()
        test_prediction()
        test_features_only()
        
        print("="*60)
        print("All tests completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("Make sure the Flask server is running:")
        print("  python app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

