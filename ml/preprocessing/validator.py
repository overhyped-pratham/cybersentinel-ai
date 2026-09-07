"""
CyberSentinel AI - Feature Validation Engine.

Enforces schema contracts, detects missing features, verifies mathematical boundaries,
and ensures numerical integrity (no NaNs, no Infs) across network states S_t.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from ml.state.state_builder import FEATURE_NAMES


class FeatureValidationError(Exception):
    """Raised when state dataframe violates feature schema or bounds."""
    pass


class FeatureValidator:
    """
    Validates network state DataFrames against expected schema,
    value ranges, and numerical validity.
    """

    def __init__(self, expected_features: Optional[List[str]] = None) -> None:
        self.expected_features = expected_features or list(FEATURE_NAMES)

    def validate(self, df: pd.DataFrame, raise_on_error: bool = True) -> bool:
        """
        Validates feature presence, ordering, finiteness, and valid ranges.
        
        Args:
            df: DataFrame of network state features.
            raise_on_error: If True, raises FeatureValidationError on breach.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        # Check missing features
        missing = [f for f in self.expected_features if f not in df.columns]
        if missing:
            msg = f"Network state DataFrame missing expected features: {missing}"
            if raise_on_error:
                raise FeatureValidationError(msg)
            return False

        # Extract only the feature columns in canonical order
        feat_df = df[self.expected_features]

        # Check for NaNs
        nan_cols = feat_df.columns[feat_df.isna().any()].tolist()
        if nan_cols:
            msg = f"Network state features contain NaN values in columns: {nan_cols}"
            if raise_on_error:
                raise FeatureValidationError(msg)
            return False

        # Check for Infs
        inf_mask = np.isinf(feat_df.to_numpy())
        if np.any(inf_mask):
            msg = "Network state features contain Infinite values."
            if raise_on_error:
                raise FeatureValidationError(msg)
            return False

        # Check bounds: ratios in [0.0, 1.0], non-negative counts
        ratio_cols = [c for c in self.expected_features if c.endswith("_ratio") or c.endswith("_share")]
        for col in ratio_cols:
            vals = feat_df[col]
            if (vals < -1e-5).any() or (vals > 1.0 + 1e-5).any():
                msg = f"Ratio feature '{col}' contains values outside [0.0, 1.0]: min={vals.min()}, max={vals.max()}"
                if raise_on_error:
                    raise FeatureValidationError(msg)
                return False

        # Non-negative counts/rates/durations
        non_neg_cols = [c for c in self.expected_features if "count" in c or "rate" in c or "bytes" in c or "packets" in c or "entropy" in c or "duration" in c]
        for col in non_neg_cols:
            vals = feat_df[col]
            if (vals < -1e-5).any():
                msg = f"Feature '{col}' contains negative values: min={vals.min()}"
                if raise_on_error:
                    raise FeatureValidationError(msg)
                return False

        return True

    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces NaNs, clips ratios to [0, 1], and replaces infinities.
        """
        df_clean = df.copy()
        for col in self.expected_features:
            if col not in df_clean.columns:
                df_clean[col] = 0.0

        # Replace infs with nan then fillna
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean.fillna(0.0, inplace=True)

        ratio_cols = [c for c in self.expected_features if c.endswith("_ratio") or c.endswith("_share")]
        for col in ratio_cols:
            df_clean[col] = df_clean[col].clip(0.0, 1.0)

        non_neg_cols = [c for c in self.expected_features if "count" in c or "rate" in c or "bytes" in c or "packets" in c or "entropy" in c or "duration" in c]
        for col in non_neg_cols:
            df_clean[col] = df_clean[col].clip(lower=0.0)

        return df_clean
