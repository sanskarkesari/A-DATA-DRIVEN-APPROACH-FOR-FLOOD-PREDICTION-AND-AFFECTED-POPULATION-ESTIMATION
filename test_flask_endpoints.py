#!/usr/bin/env python3
"""
Test Flask endpoints to diagnose issues
"""

import requests
import json
import sys

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("="*60)
    print("Testing /api/health endpoint")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Flask server is not running")
        print("   Start Flask with: python3 app.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ Timeout: Flask server is not responding")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_progressive_extraction():
    """Test progressive feature extraction to identify problematic features"""
    print("\n" + "="*60)
    print("Testing Progressive Feature Extraction")
    print("="*60)
    
    payload = {
        "latitude": 26.2006,
        "longitude": 92.9376,
        "date": "2023-07-15",
        "test_type": "progressive"  # Test elevation, flow, rainfall, SMAP (not NDWI)
    }
    
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    print("\nSending progressive test (timeout: 180 seconds)...")
    print("This will test: elevation, flow accumulation, rainfall, SMAP")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/test-extraction",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=180  # 3 minute timeout
        )
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Progressive Test Results:")
            print(json.dumps(data, indent=2))
            
            if data.get('errors'):
                print("\n⚠️  Some features failed:")
                for feature, error in data['errors'].items():
                    print(f"   - {feature}: {error}")
                return False
            return True
        else:
            print(f"\n❌ Error Response:")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ TIMEOUT: Request took longer than 180 seconds")
        print("   Check Flask console to see which feature is hanging")
        return False
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Flask server disconnected")
        print("   Flask likely crashed during request processing")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predict():
    """Test predict endpoint"""
    print("\n" + "="*60)
    print("Testing /predict endpoint")
    print("="*60)
    
    payload = {
        "latitude": 26.2006,
        "longitude": 92.9376,
        "date": "2023-07-15"
    }
    
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    print("\nSending request (timeout: 180 seconds)...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=180  # 3 minute timeout for feature extraction
        )
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print(f"\n✅ Success!")
            result = response.json()
            # Print summary
            if 'prediction' in result:
                pred = result['prediction']
                print(f"\n📊 Prediction: {pred.get('flood_risk')} ({pred.get('flood_probability')}%)")
            print(f"\nFull Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"\n❌ Error Response:")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ TIMEOUT: Request took longer than 180 seconds")
        print("   This means feature extraction is hanging on a .getInfo() call")
        print("   Check Flask console logs to see which step is hanging")
        return False
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Flask server disconnected")
        print("   Flask likely crashed during request processing")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔍 Flask Endpoint Diagnostic Tool\n")
    
    # Test 1: Health check
    health_ok = test_health()
    
    if not health_ok:
        print("\n" + "="*60)
        print("❌ Health check failed. Flask server may not be running.")
        print("   Start Flask with: python3 app.py")
        print("="*60)
        sys.exit(1)
    
    # Test 2: Progressive extraction (identify problematic features)
    print("\n⚠️  Running progressive test first to identify any problematic features...")
    progressive_ok = test_progressive_extraction()
    
    if not progressive_ok:
        print("\n" + "="*60)
        print("❌ Progressive test FAILED")
        print("   One or more features are causing issues")
        print("   Check the errors above to see which feature failed")
        print("="*60)
        print("\n")
        sys.exit(1)
    
    # Test 3: Full predict endpoint (only if progressive test passed)
    print("\n✅ Progressive test passed, testing full predict endpoint...")
    predict_ok = test_predict()
    
    print("\n" + "="*60)
    if predict_ok:
        print("✅ All tests PASSED")
    else:
        print("❌ Predict endpoint test FAILED")
        print("\nTroubleshooting:")
        print("1. Check Flask console for error messages")
        print("2. Run: python3 test_feature_extraction.py")
        print("3. Check Earth Engine authentication")
        print("4. Progressive test passed, so issue might be with NDWI or model prediction")
    print("="*60)
    print("\n")
    
    sys.exit(0 if predict_ok else 1)

