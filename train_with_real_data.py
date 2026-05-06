#!/usr/bin/env python3
"""
Train model with real historical flood data
"""

from train_model import train_flood_model
import os

# Path to your real dataset
DATA_PATH = "data/assam_flood_dataset.csv"

# Check if file exists
if not os.path.exists(DATA_PATH):
    print("="*60)
    print("❌ Error: Dataset file not found!")
    print("="*60)
    print(f"   Expected file: {DATA_PATH}")
    print("\n   Please create the dataset first:")
    print("   1. Edit create_real_dataset.py")
    print("   2. Add your flood events")
    print("   3. Run: python3 create_real_dataset.py")
    print("="*60)
    exit(1)

print("="*60)
print("Training Model with Real Historical Flood Data")
print("="*60)
print(f"Dataset: {DATA_PATH}")
print()

# Train model with real data
try:
    model, metrics = train_flood_model(
        synthetic=False,  # Use real data, not synthetic
        data_path=DATA_PATH
    )
    
    print("\n" + "="*60)
    print("✅ Model training complete!")
    print("="*60)
    print(f"Model saved to: models/flood_prediction_model.pkl")
    print(f"Scaler saved to: models/feature_scaler.pkl")
    print(f"\nModel Accuracy: {metrics['accuracy']:.4f}")
    print("\n✅ Model is now ready to use with real predictions!")
    print("   Restart Flask server to use the new model.")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

