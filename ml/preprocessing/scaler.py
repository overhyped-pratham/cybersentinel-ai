"""
CyberSentinel AI - Feature Scaler & Normalization Engine.

Strictly adheres to data-leakage prevention:
1. Scaler statistics are calculated ONLY on training splits.
2. Inference and evaluation sets are transformed using frozen training statistics.
3. Feature names and column ordering are validated to ensure deterministic alignment.
"""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from ml.state.state_builder import FEATURE_NAMES


class FeatureScaler:
    """
    Leakage-free feature scaler for network state representations S_t.
    """

    def __init__(
        self,
        scaler_type: str = "robust",
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.scaler_type = scaler_type.lower()
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.is_fitted = False

        if self.scaler_type == "robust":
            self._scaler = RobustScaler()
        elif self.scaler_type == "standard":
            self._scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'robust' or 'standard'.")

    def fit(self, df_train: pd.DataFrame) -> "FeatureScaler":
        """
        Fits scaling parameters strictly on training split data.
        
        Args:
            df_train: DataFrame containing training network states S_t.
        """
        missing = [f for f in self.feature_names if f not in df_train.columns]
        if missing:
            raise ValueError(f"Training data missing required features: {missing}")

        X_train = df_train[self.feature_names].to_numpy(dtype=np.float32)
        self._scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms state features using frozen fitted parameters.
        
        Args:
            df: DataFrame containing network states to scale.
            
        Returns:
            np.ndarray: Scaled feature matrix of shape (N, num_features).
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fit on training data before calling transform.")

        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"Data missing required features for transform: {missing}")

        X = df[self.feature_names].to_numpy(dtype=np.float32)
        return self._scaler.transform(X).astype(np.float32)

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        """Fits on training data and returns transformed matrix."""
        self.fit(df_train)
        return self.transform(df_train)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverses scaled feature matrix back to raw feature scale."""
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fit before calling inverse_transform.")
        return self._scaler.inverse_transform(X_scaled).astype(np.float32)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves fitted scaler and feature schema to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted FeatureScaler.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler_type": self.scaler_type,
            "feature_names": self.feature_names,
            "scaler_obj": self._scaler,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FeatureScaler":
        """Loads a frozen FeatureScaler from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        scaler = cls(
            scaler_type=payload["scaler_type"],
            feature_names=payload["feature_names"],
        )
        scaler._scaler = payload["scaler_obj"]
        scaler.is_fitted = True
        return scaler
