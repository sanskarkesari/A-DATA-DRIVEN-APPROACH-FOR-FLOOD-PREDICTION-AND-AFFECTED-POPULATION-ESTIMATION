"""
Preprocessing utilities for affected population regression model.

Responsible for:
- Handling categorical features (e.g., district)
- Passing through numeric features
- Building a reusable sklearn ColumnTransformer

The regression model itself is defined in `population_model.py`.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Categorical and numeric feature definitions for the population model
CATEGORICAL_FEATURES: List[str] = ["district"]

# Numeric features used by the regression model
# These should match the columns present in your training CSV
NUMERIC_FEATURES: List[str] = [
    "latitude",
    "longitude",
    "year",
    "month",
    "daily_rainfall_mm",
    "cumulative_rainfall_30d_mm",
    "soil_moisture_mm",
    "river_water_level_m",
    "elevation_m",
    "flood_risk_level",  # Reuse output from flood risk classifier
    "population_density_per_sqkm",
]


def get_feature_names() -> Tuple[List[str], List[str]]:
    """
    Return the lists of categorical and numeric feature names.
    Useful for consistency across training and prediction code.
    """
    return CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_population_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - One-hot encodes the `district` categorical feature
    - Passes numeric features through without scaling (tree models handle scale well)

    Returns
    -------
    preprocessor : ColumnTransformer
        Fitted during training, reused during prediction.
    """
    # One-hot encode categorical features; ignore unseen districts at prediction time
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    # Simple sanity check
    cats, nums = get_feature_names()
    print("Categorical features:", cats)
    print("Numeric features:", nums)

