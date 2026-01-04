"""
Machine Learning Training Pipeline for Flood Prediction System
Trains LightGBM or RandomForest model to predict flood risk levels

Model Output:
- Flood Risk Level: Low / Medium / High
- Flood Probability: 0-100%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
import joblib
import os
from typing import Tuple
import config

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)


class FloodPredictionModel:
    """
    Machine Learning model for flood risk prediction
    """
    
    def __init__(self, model_type: str = 'lightgbm'):
        """
        Initialize the model
        
        Args:
            model_type: 'lightgbm' or 'randomforest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        if model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                random_state=config.RANDOM_STATE,
                verbose=-1
            )
        elif model_type == 'randomforest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature dataframe for training/prediction
        Selects and prepares the relevant columns
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with prepared features
        """
        # Feature columns used for prediction
        feature_cols = [
            'daily_rainfall_mm',
            'cumulative_rainfall_30d_mm',
            'avg_daily_rainfall_mm',
            'soil_moisture_mm',
            'elevation_m',
            'slope_degree',
            'flow_accumulation',
            'ndwi'
        ]
        
        # Check which columns exist in the dataframe
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Create feature dataframe
        features_df = df[available_cols].copy()
        
        # Fill missing values with median
        features_df = features_df.fillna(features_df.median())
        
        # Store feature columns for later use
        self.feature_columns = available_cols
        
        return features_df
    
    def create_flood_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create flood risk labels based on features (synthetic labels for training)
        
        Flood Logic:
        High rainfall + low slope + high soil moisture + high flow accumulation = High Flood Risk
        
        Args:
            df: DataFrame with features
            
        Returns:
            Series with flood risk labels (0=Low, 1=Medium, 2=High)
        """
        labels = []
        
        for idx, row in df.iterrows():
            # Normalize features for scoring
            rainfall_score = min(row.get('cumulative_rainfall_30d_mm', 0) / 500.0, 1.0)  # Normalize to 0-1
            slope_score = 1.0 - min(row.get('slope_degree', 45) / 45.0, 1.0)  # Low slope = high risk
            soil_moisture_score = min(row.get('soil_moisture_mm', 0) / 100.0, 1.0)
            flow_accum_score = row.get('flow_accumulation', 50) / 100.0
            ndwi_score = max(0, min((row.get('ndwi', 0) + 0.3) / 0.6, 1.0))  # NDWI typically -1 to 1
            
            # Combined flood risk score (weighted)
            flood_score = (
                0.30 * rainfall_score +
                0.20 * slope_score +
                0.20 * soil_moisture_score +
                0.15 * flow_accum_score +
                0.15 * ndwi_score
            )
            
            # Convert to risk level
            if flood_score < 0.33:
                labels.append(0)  # Low
            elif flood_score < 0.66:
                labels.append(1)  # Medium
            else:
                labels.append(2)  # High
        
        return pd.Series(labels, name='flood_risk_level')
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """
        Train the model
        
        Args:
            X: Feature DataFrame
            y: Target labels
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {self.model_type} model...")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        if self.model_type == 'lightgbm':
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=20)]
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        # Get unique classes in test set
        unique_classes = sorted(set(y_test) | set(y_pred))
        class_names = ['Low', 'Medium', 'High']
        # Only include class names for classes that exist
        target_names = [class_names[i] for i in unique_classes if i < len(class_names)]
        print(classification_report(y_test, y_pred, 
                                  labels=unique_classes,
                                  target_names=target_names))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred, labels=unique_classes))
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict flood risk for given features
        
        Args:
            features: DataFrame with features
            
        Returns:
            Tuple of (risk_levels, probabilities)
            risk_levels: Array of predicted risk levels (0=Low, 1=Medium, 2=High)
            probabilities: Array of probability arrays [prob_low, prob_medium, prob_high]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure same columns as training
        features_aligned = pd.DataFrame(0, index=features.index, columns=self.feature_columns)
        for col in self.feature_columns:
            if col in features.columns:
                features_aligned[col] = features[col]
        
        # Scale features
        features_scaled = self.scaler.transform(features_aligned)
        
        # Predict
        risk_levels = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return risk_levels, probabilities
    
    def save_model(self, model_path: str = None, scaler_path: str = None):
        """
        Save trained model and scaler
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        if model_path is None:
            model_path = config.MODEL_PATH
        if scaler_path is None:
            scaler_path = config.SCALER_PATH
        
        # Save model
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        # Save feature columns
        feature_cols_path = scaler_path.replace('.pkl', '_features.pkl')
        joblib.dump(self.feature_columns, feature_cols_path)
        print(f"Feature columns saved to {feature_cols_path}")
    
    def load_model(self, model_path: str = None, scaler_path: str = None):
        """
        Load trained model and scaler
        
        Args:
            model_path: Path to load model from
            scaler_path: Path to load scaler from
        """
        if model_path is None:
            model_path = config.MODEL_PATH
        if scaler_path is None:
            scaler_path = config.SCALER_PATH
        
        # Load model
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        
        # Load feature columns
        feature_cols_path = scaler_path.replace('.pkl', '_features.pkl')
        if os.path.exists(feature_cols_path):
            self.feature_columns = joblib.load(feature_cols_path)
            print(f"Feature columns loaded from {feature_cols_path}")


def generate_synthetic_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic training data for demonstration
    In production, use real historical data from feature extraction
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic features
    """
    np.random.seed(config.RANDOM_STATE)
    
    data = {
        'daily_rainfall_mm': np.random.gamma(2, 10, n_samples),  # Right-skewed (realistic rainfall)
        'cumulative_rainfall_30d_mm': np.random.gamma(5, 50, n_samples),
        'avg_daily_rainfall_mm': np.random.gamma(2, 8, n_samples),
        'soil_moisture_mm': np.random.uniform(0, 150, n_samples),
        'elevation_m': np.random.uniform(50, 300, n_samples),  # Assam elevation range
        'slope_degree': np.random.uniform(0, 45, n_samples),
        'flow_accumulation': np.random.uniform(0, 100, n_samples),
        'ndwi': np.random.uniform(-0.5, 0.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic correlations
    # High rainfall -> high soil moisture
    df['soil_moisture_mm'] = df['soil_moisture_mm'] + df['cumulative_rainfall_30d_mm'] * 0.1
    
    # Low elevation -> higher flow accumulation
    df['flow_accumulation'] = 100 - (df['elevation_m'] / 3) + np.random.normal(0, 10, n_samples)
    df['flow_accumulation'] = df['flow_accumulation'].clip(0, 100)
    
    # High rainfall + low slope -> higher NDWI (water present)
    df['ndwi'] = df['ndwi'] + (df['cumulative_rainfall_30d_mm'] / 1000) - (df['slope_degree'] / 100)
    df['ndwi'] = df['ndwi'].clip(-1, 1)
    
    return df


def train_flood_model(synthetic: bool = True, data_path: str = None):
    """
    Main function to train the flood prediction model
    
    Args:
        synthetic: If True, use synthetic data. If False, load from data_path
        data_path: Path to CSV file with training data
    """
    # Initialize model
    model = FloodPredictionModel(model_type=config.ML_MODEL_TYPE)
    
    # Load or generate data
    if synthetic:
        print("Generating synthetic training data...")
        df = generate_synthetic_training_data(n_samples=2000)
    else:
        if data_path is None:
            raise ValueError("data_path required when synthetic=False")
        print(f"Loading training data from {data_path}...")
        df = pd.read_csv(data_path)
    
    print(f"\nDataset shape: {df.shape}")
    print("\nDataset statistics:")
    print(df.describe())
    
    # Prepare features
    X = model.prepare_features(df)
    print(f"\nFeatures: {list(X.columns)}")
    
    # Create labels
    y = model.create_flood_labels(df)
    print(f"\nLabel distribution:")
    print(y.value_counts().sort_index())
    
    # Train model
    metrics = model.train(X, y)
    
    # Save model
    model.save_model()
    
    print("\n✓ Model training complete!")
    return model, metrics


if __name__ == "__main__":
    # Train model with synthetic data
    model, metrics = train_flood_model(synthetic=True)
    
    # Test prediction
    print("\n" + "="*50)
    print("Testing predictions...")
    
    # Create test sample
    test_features = generate_synthetic_training_data(n_samples=5)
    test_X = model.prepare_features(test_features)
    
    risk_levels, probabilities = model.predict(test_X)
    
    risk_names = ['Low', 'Medium', 'High']
    
    for i in range(len(test_features)):
        print(f"\nSample {i+1}:")
        print(f"  Risk Level: {risk_names[risk_levels[i]]}")
        print(f"  Probabilities - Low: {probabilities[i][0]:.2%}, "
              f"Medium: {probabilities[i][1]:.2%}, "
              f"High: {probabilities[i][2]:.2%}")
        print(f"  Flood Probability: {probabilities[i][2] * 100:.2f}%")



