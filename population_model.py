"""
Affected Population Regression Model
------------------------------------

This module implements a regression model to predict:

    affected_population (numeric)

using tabular features including:
- district (categorical)
- latitude, longitude
- year, month
- daily & cumulative rainfall
- soil moisture, river water level
- elevation
- flood_risk_level (output from existing classifier)
- population_density_per_sqkm

Key design goals:
- Treat as regression (LightGBMRegressor)
- Handle categorical district via one-hot encoding
- Apply log1p transform to skewed target
- Keep this model completely separate from the flood risk model
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import config
from population_preprocessing import (
    build_population_preprocessor,
    get_feature_names,
)


@dataclass
class PopulationModelMetrics:
    """Container for regression evaluation metrics."""

    rmse: float
    mae: float
    r2: float


class AffectedPopulationModel:
    """
    Regression model for predicting affected_population.

    This class wraps:
    - A sklearn ColumnTransformer preprocessor
    - A LightGBMRegressor model
    - Optional log-target transformation
    """

    def __init__(self, log_target: bool = True):
        self.log_target = log_target
        self.preprocessor = None
        self.model: Optional[lgb.LGBMRegressor] = None
        self.metrics: Optional[PopulationModelMetrics] = None

        self.cat_features, self.num_features = get_feature_names()
        self.feature_columns = self.cat_features + self.num_features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, target_col: str = "affected_population") -> PopulationModelMetrics:
        """
        Train the regression model on the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with all required feature columns and target.
        target_col : str
            Name of the target column (default: 'affected_population').
        """
        missing_cols = [c for c in self.feature_columns + [target_col] if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in training data: {missing_cols}")

        # Drop rows with missing target
        df = df.dropna(subset=[target_col])

        # Features and target
        X = df[self.feature_columns].copy()
        y = df[target_col].astype(float).values

        # Ensure target is non-negative for log transform
        if self.log_target:
            y = np.where(y < 0, 0.0, y)

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=config.RANDOM_STATE,
        )

        # Build and fit preprocessor
        self.preprocessor = build_population_preprocessor()
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_val_proc = self.preprocessor.transform(X_val)

        # Optional log transform on target to handle skewness
        if self.log_target:
            y_train_trans = np.log1p(y_train)
        else:
            y_train_trans = y_train

        # Define LightGBM regressor (good for tabular data)
        self.model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )

        print("Training LightGBM regression model for affected_population...")
        self.model.fit(X_train_proc, y_train_trans)

        # Evaluation on validation set
        if self.log_target:
            y_val_pred_log = self.model.predict(X_val_proc)
            y_val_pred = np.expm1(y_val_pred_log)
        else:
            y_val_pred = self.model.predict(X_val_proc)

        rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        mae = float(mean_absolute_error(y_val, y_val_pred))
        r2 = float(r2_score(y_val, y_val_pred))

        self.metrics = PopulationModelMetrics(rmse=rmse, mae=mae, r2=r2)

        print(f"\nAffected Population Regression Metrics:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE : {mae:.2f}")
        print(f"  R²  : {r2:.4f}")

        return self.metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self.model is None or self.preprocessor is None:
            raise ValueError("AffectedPopulationModel not loaded/trained.")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict affected_population for a batch of feature rows.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame with the same feature columns used during training.

        Returns
        -------
        np.ndarray
            Predicted affected_population values (original scale).
        """
        self._ensure_loaded()

        # Align columns
        X = pd.DataFrame(columns=self.feature_columns)
        for col in self.feature_columns:
            if col in features.columns:
                X[col] = features[col]
        # Fill any missing numeric columns with 0 (safe default)
        X = X.fillna(0)

        X_proc = self.preprocessor.transform(X)

        if self.log_target:
            preds_log = self.model.predict(X_proc)
            preds = np.expm1(preds_log)
        else:
            preds = self.model.predict(X_proc)

        # Ensure non-negative predictions
        preds = np.where(preds < 0, 0.0, preds)

        return preds

    def predict_with_interval(self, features: pd.DataFrame, interval: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict affected_population and return a simple confidence interval.

        The interval is based on RMSE from validation:
            [pred - k * RMSE, pred + k * RMSE]

        Parameters
        ----------
        features : pd.DataFrame
            Input feature rows.
        interval : float
            Multiplier for RMSE (1.0 ≈ 68% CI, 2.0 ≈ 95% CI approx).
        """
        preds = self.predict(features)

        if self.metrics is None or self.metrics.rmse is None:
            # Fallback: ±20% of prediction
            low = preds * 0.8
            high = preds * 1.2
        else:
            rmse = self.metrics.rmse
            low = np.maximum(0.0, preds - interval * rmse)
            high = preds + interval * rmse

        return preds, low, high

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        meta_path: Optional[str] = None,
    ) -> None:
        """Save model, preprocessor, and metrics to disk."""
        model_path = model_path or config.AFFECTED_POP_MODEL_PATH
        preprocessor_path = preprocessor_path or config.AFFECTED_POP_PREPROCESSOR_PATH
        meta_path = meta_path or config.AFFECTED_POP_META_PATH

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        joblib.dump(self.model, model_path)
        print(f"✅ Population model saved to {model_path}")

        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"✅ Population preprocessor saved to {preprocessor_path}")

        meta: Dict[str, float] = {}
        if self.metrics is not None:
            meta = asdict(self.metrics)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"✅ Population model metrics saved to {meta_path}")

    def load(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        meta_path: Optional[str] = None,
    ) -> None:
        """Load model, preprocessor, and metrics from disk."""
        model_path = model_path or config.AFFECTED_POP_MODEL_PATH
        preprocessor_path = preprocessor_path or config.AFFECTED_POP_PREPROCESSOR_PATH
        meta_path = meta_path or config.AFFECTED_POP_META_PATH

        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Affected population model or preprocessor not found at {model_path} / {preprocessor_path}"
            )

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        print(f"✅ Population model loaded from {model_path}")
        print(f"✅ Population preprocessor loaded from {preprocessor_path}")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.metrics = PopulationModelMetrics(
                rmse=meta.get("rmse", 0.0),
                mae=meta.get("mae", 0.0),
                r2=meta.get("r2", 0.0),
            )
            print(f"✅ Population metrics loaded from {meta_path}")
        else:
            print("ℹ️ Population metrics file not found; confidence intervals will use default heuristic.")


def train_population_model(
    data_path: str,
    target_col: str = "affected_population",
    log_target: bool = True,
) -> Tuple[AffectedPopulationModel, PopulationModelMetrics]:
    """
    Convenience function to train the affected population model from a CSV file.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    print("=" * 60)
    print("Training Affected Population Regression Model")
    print("=" * 60)
    print(f"Dataset: {data_path}\n")

    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print("\nColumns:", list(df.columns))
    print("\nTarget column:", target_col)
    print("\nTarget summary:")
    print(df[target_col].describe())

    model = AffectedPopulationModel(log_target=log_target)
    metrics = model.fit(df, target_col=target_col)
    model.save()

    print("\n✓ Affected population model training complete!\n")
    return model, metrics


if __name__ == "__main__":
    # Example CLI usage:
    import argparse

    parser = argparse.ArgumentParser(description="Train affected population regression model.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/assam_population_impact_dataset.csv",
        help="Path to CSV file with training data.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log1p transform on target.",
    )
    args = parser.parse_args()

    train_population_model(
        data_path=args.data,
        target_col="affected_population",
        log_target=not args.no_log,
    )

