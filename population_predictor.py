#!/usr/bin/env python3
"""
Batch prediction script for affected population regression model.

Usage:
    python population_predictor.py --input data/assam_population_impact_dataset.csv --output data/assam_population_predictions.csv

This script:
    - Loads the trained affected population model and preprocessor
    - Reads a CSV file with the same features used during training
    - Adds a new column `affected_population_pred` (and optional intervals)
    - Saves the result to a new CSV file
"""

import argparse
import os
from typing import Optional

import pandas as pd

from population_model import AffectedPopulationModel


def run_batch_prediction(input_path: str, output_path: Optional[str] = None) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("=" * 60)
    print("Batch Prediction - Affected Population")
    print("=" * 60)
    print(f"Input:  {input_path}")

    df = pd.read_csv(input_path)

    model = AffectedPopulationModel()
    model.load()

    preds, low, high = model.predict_with_interval(df)
    df["affected_population_pred"] = preds
    df["affected_population_low"] = low
    df["affected_population_high"] = high

    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_with_predictions{ext}"

    df.to_csv(output_path, index=False)

    print(f"\n✅ Predictions saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction for affected population.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with features.")
    parser.add_argument("--output", type=str, help="Output CSV file to save predictions.")
    args = parser.parse_args()

    run_batch_prediction(args.input, args.output)

