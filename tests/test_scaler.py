"""
Unit tests for ml.preprocessing.scaler.FeatureScaler.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from ml.preprocessing.scaler import FeatureScaler
from ml.state.state_builder import FEATURE_NAMES


def test_scaler_fit_transform(tmp_path: Path):
    # Create synthetic states dataframe
    n_samples = 20
    data = {col: np.random.uniform(0.0, 100.0, size=n_samples) for col in FEATURE_NAMES}
    df = pd.DataFrame(data)

    scaler = FeatureScaler(scaler_type="robust")
    assert not scaler.is_fitted

    X_scaled = scaler.fit_transform(df)
    assert scaler.is_fitted
    assert X_scaled.shape == (n_samples, len(FEATURE_NAMES))

    # Test inverse transform
    X_recovered = scaler.inverse_transform(X_scaled)
    assert np.allclose(df[FEATURE_NAMES].to_numpy(), X_recovered, atol=1e-4)

    # Test persistence
    scaler_file = tmp_path / "scaler.pkl"
    scaler.save(scaler_file)
    assert scaler_file.exists()

    loaded_scaler = FeatureScaler.load(scaler_file)
    assert loaded_scaler.is_fitted
    X_loaded_scaled = loaded_scaler.transform(df)
    assert np.allclose(X_scaled, X_loaded_scaled)


def test_scaler_unfitted_error():
    scaler = FeatureScaler()
    df = pd.DataFrame({col: [1.0] for col in FEATURE_NAMES})
    with pytest.raises(RuntimeError):
        scaler.transform(df)
