"""
Standalone script to generate model performance visualizations for papers/reports.
Does not modify any application code or models; only loads saved models and data,
then produces and saves figures.

Outputs:
  - figures/confusion_matrix.png   : Flood risk classification (confusion matrix)
  - figures/regression_performance.png : Affected population (actual vs predicted + metrics)

Usage:
  pip install matplotlib seaborn   # if not already installed
  python plot_model_performance.py

Figures are saved under the 'figures/' directory (created if missing).
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Optional: matplotlib and seaborn for plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print("Please install matplotlib and seaborn: pip install matplotlib seaborn")
    sys.exit(1)

# Project imports (read-only: load model and config)
import config
from train_model import FloodPredictionModel
from population_model import AffectedPopulationModel
from population_preprocessing import get_feature_names


# Output directory for figures
FIGURES_DIR = "figures"
FLOOD_DATA_PATH = "data/assam_flood_dataset.csv"
POPULATION_DATA_PATH = "data/assam_population_impact_dataset.csv"


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_confusion_matrix():
    """
    Load flood model and dataset, compute predictions on a test split,
    then plot and save confusion matrix.
    """
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
        print(f"Skipping confusion matrix: model or scaler not found at {config.MODEL_PATH} / {config.SCALER_PATH}")
        return

    if not os.path.exists(FLOOD_DATA_PATH):
        print(f"Skipping confusion matrix: data not found at {FLOOD_DATA_PATH}")
        return

    print("Loading flood model and data for confusion matrix...")
    model = FloodPredictionModel(model_type=config.ML_MODEL_TYPE)
    model.load_model(config.MODEL_PATH, config.SCALER_PATH)

    df = pd.read_csv(FLOOD_DATA_PATH)
    df = df.dropna(subset=["flood_risk_level"])
    y_true = df["flood_risk_level"].astype(int)

    X = model.prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_true
    )

    # Scale using the loaded scaler (fit on train only for this script's split)
    X_test_scaled = model.scaler.transform(X_test)
    y_pred = model.model.predict(X_test_scaled)

    labels = [0, 1, 2]
    labels = [l for l in labels if l in np.unique(y_test) or l in np.unique(y_pred)]
    class_names = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[class_names[i] for i in labels],
        yticklabels=[class_names[i] for i in labels],
        ax=ax,
        cbar_kws={"label": "Count"},
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Flood Risk Classification — Confusion Matrix", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_regression_performance():
    """
    Load affected population model and dataset, compute predictions on a test split,
    then plot actual vs predicted and optional metrics bar.
    """
    if not os.path.exists(config.AFFECTED_POP_MODEL_PATH) or not os.path.exists(
        config.AFFECTED_POP_PREPROCESSOR_PATH
    ):
        print(
            f"Skipping regression plot: population model not found at {config.AFFECTED_POP_MODEL_PATH}"
        )
        return

    if not os.path.exists(POPULATION_DATA_PATH):
        print(f"Skipping regression plot: data not found at {POPULATION_DATA_PATH}")
        return

    print("Loading population model and data for regression performance...")
    pop_model = AffectedPopulationModel()
    pop_model.load(
        model_path=config.AFFECTED_POP_MODEL_PATH,
        preprocessor_path=config.AFFECTED_POP_PREPROCESSOR_PATH,
        meta_path=config.AFFECTED_POP_META_PATH,
    )

    df = pd.read_csv(POPULATION_DATA_PATH)
    df = df.dropna(subset=["affected_population"])
    cat_features, num_features = get_feature_names()
    feature_cols = cat_features + num_features
    missing = [c for c in feature_cols + ["affected_population"] if c not in df.columns]
    if missing:
        print(f"Skipping regression plot: missing columns {missing}")
        return

    X = df[feature_cols].copy()
    y = df["affected_population"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )

    y_pred = pop_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Actual vs Predicted scatter
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors="k", linewidths=0.3)
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect prediction")
    ax1.set_xlabel("Actual affected population", fontsize=11)
    ax1.set_ylabel("Predicted affected population", fontsize=11)
    ax1.set_title("Actual vs Predicted", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Metrics bar chart
    ax2 = axes[1]
    metrics_names = ["MAE", "RMSE", "R²"]
    metrics_vals = [mae, rmse, r2]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = ax2.bar(metrics_names, metrics_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_title("Regression metrics (test set)", fontsize=12)
    for bar, val in zip(bars, metrics_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(metrics_vals) * 0.02),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "regression_performance.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.4f}")


def main():
    _ensure_figures_dir()
    plot_confusion_matrix()
    plot_regression_performance()
    print("Done. Check the 'figures' folder for PNG files.")


if __name__ == "__main__":
    main()
