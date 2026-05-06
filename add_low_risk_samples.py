#!/usr/bin/env python3
"""
Add Low risk samples to your dataset
Since your dataset has only 1 Low sample, this script helps add more
"""

import pandas as pd
from feature_extraction import FeatureExtractor
import time
import os

# Initialize feature extractor
extractor = FeatureExtractor()

# Load existing dataset
dataset_path = "data/assam_flood_dataset.csv"
df = pd.read_csv(dataset_path)

print("="*60)
print("Adding Low Risk Samples to Dataset")
print("="*60)
print(f"Current dataset: {len(df)} samples")
print(f"Low (0): {len(df[df['flood_risk_level'] == 0])}")
print(f"Medium (1): {len(df[df['flood_risk_level'] == 1])}")
print(f"High (2): {len(df[df['flood_risk_level'] == 2])}")
print()

# ============================================================
# Add Low Risk Locations/Dates
# ============================================================
# Low risk conditions: Dry season, high elevation, high slope
# Add locations and dates that are likely Low risk

low_risk_events = [
    # Dry season dates (November - February) in Assam
    {"location": "Guwahati", "lat": 26.1865, "lon": 91.7439, "date": "2023-01-15", "level": 0},
    {"location": "Dibrugarh", "lat": 27.4728, "lon": 94.9120, "date": "2023-02-10", "level": 0},
    {"location": "Jorhat", "lat": 26.7500, "lon": 94.2167, "date": "2022-12-20", "level": 0},
    {"location": "Tezpur", "lat": 26.6333, "lon": 92.8000, "date": "2023-01-25", "level": 0},
    {"location": "Silchar", "lat": 24.8333, "lon": 92.7833, "date": "2022-11-15", "level": 0},
    # Add more dry season dates...
    # You can add 50-100 more Low risk samples
]

print(f"Extracting features for {len(low_risk_events)} Low risk events...")
print("="*60)

new_samples = []

for i, event in enumerate(low_risk_events, 1):
    print(f"[{i}/{len(low_risk_events)}] {event['location']} on {event['date']}")
    
    try:
        features = extractor.extract_all_features(
            latitude=event['lat'],
            longitude=event['lon'],
            target_date=event['date']
        )
        
        # Add metadata
        features['flood_risk_level'] = event['level']
        features['location_name'] = event['location']
        # Add year if not present
        if 'year' not in features:
            features['year'] = int(event['date'].split('-')[0])
        if 'district' not in features:
            features['district'] = event['location']
        
        new_samples.append(features)
        print(f"  ✅ Success")
        
        time.sleep(2)  # Rate limiting
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        continue

# Combine with existing dataset
if new_samples:
    new_df = pd.DataFrame(new_samples)
    
    # Ensure same columns
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = None
    
    # Reorder columns to match
    new_df = new_df[df.columns]
    
    # Combine
    combined_df = pd.concat([df, new_df], ignore_index=True)
    
    # Save
    backup_path = dataset_path.replace('.csv', '_backup.csv')
    df.to_csv(backup_path, index=False)
    print(f"\n✅ Backup saved to: {backup_path}")
    
    combined_df.to_csv(dataset_path, index=False)
    
    print("\n" + "="*60)
    print("✅ Dataset updated!")
    print("="*60)
    print(f"Total samples: {len(combined_df)}")
    print(f"Low (0): {len(combined_df[combined_df['flood_risk_level'] == 0])}")
    print(f"Medium (1): {len(combined_df[combined_df['flood_risk_level'] == 1])}")
    print(f"High (2): {len(combined_df[combined_df['flood_risk_level'] == 2])}")
    print(f"\n📁 Updated dataset: {dataset_path}")
    print("\nNext step: Retrain the model")
    print("  python3 train_with_real_data.py")
    print("="*60)
else:
    print("\n❌ No new samples added")

