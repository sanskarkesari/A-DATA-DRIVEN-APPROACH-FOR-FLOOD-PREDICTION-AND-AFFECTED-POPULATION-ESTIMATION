

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import config
from train_model import FloodPredictionModel

try:
    import folium
except ImportError as e:  # pragma: no cover
    raise ImportError("folium is required. Install with: python -m pip install folium") from e


DISTRICT_COORDS: Dict[str, Tuple[float, float]] = {
    "Morigaon": (26.25, 92.34),
    "Dhubri": (26.02, 89.97),
    "Kamrup": (26.18, 91.73),
    "Sonitpur": (26.62, 92.79),
    "Jorhat": (26.75, 94.22),
    "Lakhimpur": (27.23, 94.10),
    "Darrang": (26.44, 92.03),
    "Barpeta": (26.32, 91.00),
    "Nagaon": (26.35, 92.68),
    "Nalbari": (26.44, 91.44),
}


# Static district populations (approx. census-scale values; update if you have official numbers)
# Used only for affected_population = population * probability
DISTRICT_POPULATION: Dict[str, int] = {
    "Morigaon": 957423,
    "Dhubri": 1948632,
    "Kamrup": 1517542,
    "Sonitpur": 1925975,
    "Jorhat": 1092256,
    "Lakhimpur": 1042137,
    "Darrang": 928500,
    "Barpeta": 1693190,
    "Nagaon": 2823768,
    "Nalbari": 769919,
}


FLOOD_FEATURE_COLUMNS: List[str] = [
    "daily_rainfall_mm",
    "cumulative_rainfall_30d_mm",
    "avg_daily_rainfall_mm",
    "soil_moisture_mm",
    "elevation_m",
    "slope_degree",
    "flow_accumulation",
    "ndwi",
]


@dataclass
class DistrictRiskResult:
    district: str
    latitude: float
    longitude: float
    probability: float  # 0..1 probability of High risk (or highest class)
    risk_level: str  # Low/Medium/High
    color: str  # green/orange/red
    population: int
    affected_population: int
    alert: str


def _risk_bucket(probability: float) -> Tuple[str, str]:
    """Return (risk_level, color) based on probability thresholds."""
    if probability < 0.3:
        return ("Low", "green")
    if probability <= 0.7:
        return ("Medium", "orange")
    return ("High", "red")


def _probability_from_model_output(probabilities: np.ndarray) -> float:
    """
    Convert model predict_proba output to a single flood probability (0..1).
    We use the probability of the 'High' class when present; otherwise the last class.
    """
    probs = np.asarray(probabilities).reshape(-1)
    if len(probs) == 0:
        return 0.0
    if len(probs) >= 3:
        return float(probs[2])
    if len(probs) == 2:
        return float(probs[1])
    return float(probs[-1])


def _load_flood_model() -> FloodPredictionModel:
    """
    Load the trained flood model + scaler from disk (no training).
    """
    model = FloodPredictionModel(model_type=config.ML_MODEL_TYPE)
    model.load_model(config.MODEL_PATH, config.SCALER_PATH)
    return model


def _get_features_from_dataset(district: str, data_path: str = "data/assam_flood_dataset.csv") -> Optional[Dict[str, float]]:
    """
    Fast path: use a representative row from the existing dataset per district.
    Returns a dict of the 8 model features, or None if unavailable.
    """
    if not os.path.exists(data_path):
        return None

    df = pd.read_csv(data_path)
    if "district" not in df.columns:
        return None

    ddf = df[df["district"].astype(str).str.lower() == district.lower()].copy()
    if ddf.empty:
        return None

    # Prefer the most recent year if present; otherwise last row
    if "year" in ddf.columns:
        ddf = ddf.sort_values("year")
    row = ddf.iloc[-1]

    features: Dict[str, float] = {}
    for col in FLOOD_FEATURE_COLUMNS:
        if col in ddf.columns:
            try:
                features[col] = float(row[col])
            except Exception:
                features[col] = 0.0
        else:
            features[col] = 0.0
    return features


def _get_features_live(latitude: float, longitude: float, date_str: str) -> Optional[Dict[str, float]]:
    """
    Slow path: use live feature extraction (GEE + NASA POWER).
    Returns 8-feature dict or None on failure.
    """
    try:
        from feature_extraction import FeatureExtractor

        extractor = FeatureExtractor()
        feats = extractor.extract_all_features(latitude, longitude, date_str)
        return {k: float(feats.get(k, 0.0)) for k in FLOOD_FEATURE_COLUMNS}
    except Exception:
        return None


def compute_district_risks(
    date_str: Optional[str] = None,
    use_live_features: bool = False,
    district: Optional[str] = None,
) -> List[DistrictRiskResult]:
    """
    Compute risk probability and affected population per district.

    - If `use_live_features=False`, features come from `data/assam_flood_dataset.csv` (fast).
    - If `use_live_features=True`, attempts live feature extraction and falls back to dataset.
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    model = _load_flood_model()

    district_filter = district.strip() if isinstance(district, str) and district.strip() else None
    items = list(DISTRICT_COORDS.items())
    if district_filter:
        items = [(d, coords) for (d, coords) in items if d.lower() == district_filter.lower()]

    results: List[DistrictRiskResult] = []
    for district_name, (lat, lon) in items:
        features = None
        if use_live_features:
            features = _get_features_live(lat, lon, date_str)
        if features is None:
            features = _get_features_from_dataset(district_name)
        if features is None:
            # Safe final fallback: zeros (keeps script demo-able)
            features = {k: 0.0 for k in FLOOD_FEATURE_COLUMNS}

        X_df = pd.DataFrame([features])
        X = model.prepare_features(X_df)
        risk_levels, probas = model.predict(X)
        probability = _probability_from_model_output(probas[0] if len(probas) else np.array([0.0]))

        risk_level, color = _risk_bucket(probability)
        population = int(DISTRICT_POPULATION.get(district_name, 0))
        affected = int(round(population * probability))
        alert = " High Alert" if probability > 0.7 else "Normal Monitoring"

        results.append(
            DistrictRiskResult(
                district=district_name,
                latitude=lat,
                longitude=lon,
                probability=float(probability),
                risk_level=risk_level,
                color=color,
                population=population,
                affected_population=affected,
                alert=alert,
            )
        )

    return results


def _add_legend(m: "folium.Map") -> None:
    """Add a simple fixed legend to the Folium map."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        padding: 12px 14px;
        border: 2px solid rgba(0,0,0,0.2);
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-family: Arial, sans-serif;
        font-size: 13px;
        ">
      <div style="font-weight: 700; margin-bottom: 8px;">Flood Risk Legend</div>
      <div style="margin-bottom: 6px;">
        <span style="display:inline-block;width:12px;height:12px;background:#2ecc71;border-radius:50%;margin-right:8px;"></span>
        Low (&lt; 0.3)
      </div>
      <div style="margin-bottom: 6px;">
        <span style="display:inline-block;width:12px;height:12px;background:#f39c12;border-radius:50%;margin-right:8px;"></span>
        Medium (0.3–0.7)
      </div>
      <div>
        <span style="display:inline-block;width:12px;height:12px;background:#e74c3c;border-radius:50%;margin-right:8px;"></span>
        High (&gt; 0.7)
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def generate_risk_map(
    output_path: str = "risk_map.html",
    date_str: Optional[str] = None,
    use_live_features: bool = False,
    district: Optional[str] = None,
) -> str:
    """
    Generate and save the interactive district-wise flood risk map.

    Parameters
    ----------
    output_path : str
        Where to save the HTML (default: risk_map.html).
    date_str : Optional[str]
        Date used for live extraction (YYYY-MM-DD). Dataset mode ignores it.
    use_live_features : bool
        If True, attempts live feature extraction per district (slower).

    Returns
    -------
    str
        Absolute path of the generated HTML file.
    """
    results = compute_district_risks(date_str=date_str, use_live_features=use_live_features, district=district)

    # If a single district is requested, zoom closer and center on it.
    if len(results) == 1:
        center = [results[0].latitude, results[0].longitude]
        zoom = 9
    else:
        center = [26.2, 92.0]
        zoom = 7

    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    for r in results:
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; min-width: 240px;">
          <div style="font-size: 15px; font-weight: 700; margin-bottom: 6px;">{r.district}</div>
          <div><b>Risk level:</b> {r.risk_level}</div>
          <div><b>Flood probability:</b> {r.probability:.2f}</div>
          <div><b>Estimated affected:</b> {r.affected_population:,} people</div>
          <div><b>Alert:</b> {r.alert}</div>
        </div>
        """
        folium.CircleMarker(
            location=[r.latitude, r.longitude],
            radius=10,
            color=r.color,
            fill=True,
            fill_color=r.color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{r.district}: {r.risk_level} ({r.probability:.2f})",
        ).add_to(m)

    _add_legend(m)

    abs_path = os.path.abspath(output_path)
    m.save(abs_path)
    return abs_path


if __name__ == "__main__":  # pragma: no cover
    path = generate_risk_map(output_path="risk_map.html", use_live_features=False)
    print(f" Saved risk map to: {path}")
