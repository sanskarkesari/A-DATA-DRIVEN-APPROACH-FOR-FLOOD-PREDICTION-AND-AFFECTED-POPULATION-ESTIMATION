#!/usr/bin/env python3
"""
Extract features for real historical flood events
Creates a training dataset from your real flood data
"""

import pandas as pd
from feature_extraction import FeatureExtractor
from datetime import datetime
import time
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Initialize feature extractor
extractor = FeatureExtractor()

# ============================================================
# STEP 1: ADD YOUR REAL FLOOD EVENTS HERE
# ============================================================
# Replace this with your actual flood events data
# Format: List of dictionaries with location, date, and flood level

flood_events = [
    {
        "location_name": "Guwahati",
        "latitude": 26.1865,
        "longitude": 91.7439,
        "date": "2023-07-15",
        "flood_risk_level": 2  # 0=Low, 1=Medium, 2=High
    },
    {
        "location_name": "Dibrugarh",
        "latitude": 27.4728,
        "longitude": 94.9120,
        "date": "2022-06-20",
        "flood_risk_level": 1
    },
    # Add more flood events here...
    # Example:
    # {
    #     "location_name": "Jorhat",
    #     "latitude": 26.7500,
    #     "longitude": 94.2167,
    #     "date": "2021-08-10",
    #     "flood_risk_level": 2
    # },
]

# ============================================================
# STEP 2: EXTRACT FEATURES FOR EACH EVENT
# ============================================================

print("="*60)
print("Creating Real Training Dataset from Flood Events")
print("="*60)
print(f"Total events to process: {len(flood_events)}")
print()

all_features = []
failed_events = []

for i, event in enumerate(flood_events, 1):
    print(f"[{i}/{len(flood_events)}] Processing: {event['location_name']} on {event['date']}")
    print(f"   Location: ({event['latitude']}, {event['longitude']})")
    
    try:
        # Extract features for this event
        features = extractor.extract_all_features(
            latitude=event['latitude'],
            longitude=event['longitude'],
            target_date=event['date']
        )
        
        # Add the real flood label
        features['flood_risk_level'] = event['flood_risk_level']
        features['location_name'] = event['location_name']
        
        all_features.append(features)
        print(f"   ✅ Success - Features extracted")
        
        # Rate limiting (don't overload APIs)
        if i < len(flood_events):  # Don't sleep after last event
            time.sleep(2)  # 2 second delay between requests
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed_events.append({
            'event': event,
            'error': str(e)
        })
        continue

# ============================================================
# STEP 3: SAVE DATASET
# ============================================================

if not all_features:
    print("\n❌ Error: No features extracted. Check your flood events data.")
    exit(1)

# Create DataFrame
df = pd.DataFrame(all_features)

# Save to CSV
output_file = "data/assam_flood_dataset.csv"
df.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"✅ Successfully processed {len(all_features)} events")
print(f"📁 Saved to: {output_file}")

if failed_events:
    print(f"\n⚠️  {len(failed_events)} events failed:")
    for failed in failed_events:
        print(f"   - {failed['event']['location_name']} ({failed['event']['date']}): {failed['error']}")

# ============================================================
# STEP 4: DISPLAY SUMMARY
# ============================================================

print("\n" + "="*60)
print("Dataset Summary:")
print("="*60)
print(f"Total samples: {len(df)}")
print(f"\nFlood Risk Distribution:")
print(df['flood_risk_level'].value_counts().sort_index())
print(f"\n  Low (0):    {len(df[df['flood_risk_level'] == 0])}")
print(f"  Medium (1): {len(df[df['flood_risk_level'] == 1])}")
print(f"  High (2):   {len(df[df['flood_risk_level'] == 2])}")

print("\n" + "="*60)
print("✅ Dataset creation complete!")
print("="*60)
print(f"\nNext step: Train the model with:")
print(f"  python3 train_with_real_data.py")
print("="*60)

