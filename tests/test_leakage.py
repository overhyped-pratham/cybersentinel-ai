"""
Explicit Leakage Detection Tests for CyberSentinel AI.

Enforces Rule 5: Data leakage is a critical bug.
Verifies:
1. Historical sequence context never contains target S_{t+1}.
2. Scenario splits have zero scenario overlap (Train vs Val vs Test).
3. Window state calculations use only in-window flows (no lookahead).
4. Feature scaler fits strictly on train split without test contamination.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES
from ml.preprocessing.sequence_builder import SequenceBuilder
from ml.preprocessing.splitter import ScenarioBasedSplitter
from ml.preprocessing.scaler import FeatureScaler
from network.flow.flow_record import FlowRecord


def test_no_target_in_sequence_history():
    """Verify target S_{t+1} is strictly future and not included in sequence input X."""
    n_windows = 10
    feature_dim = len(FEATURE_NAMES)
    
    # Each window t has distinct unique recognizable values: all features = t * 10
    rows = []
    for t in range(n_windows):
        r = {f: float(t * 10) for f in FEATURE_NAMES}
        r["scenario_id"] = "trace_clean"
        r["stage_id"] = t
        r["is_attack"] = 0.0
        r["timestamp_start"] = float(100 + t * 30)
        rows.append(r)
    df = pd.DataFrame(rows)

    builder = SequenceBuilder(sequence_length=4, pad_short_sequences=True)
    batch = builder.build_sequences(df)

    # For every sample i:
    for i in range(len(batch.x_seq)):
        # target value
        target_val = batch.y_next_state[i, 0].item()
        
        # input history values at valid mask positions
        mask_i = batch.mask[i]
        hist_vals = batch.x_seq[i, mask_i, 0].tolist()

        # Target value must NEVER appear in the history
        assert target_val not in hist_vals, f"Leakage detected! Target {target_val} found in history {hist_vals}"
        # Target value must be strictly greater than the latest history value
        assert target_val > max(hist_vals), f"Future ordering violated! Target: {target_val}, max history: {max(hist_vals)}"


def test_scenario_splitter_zero_overlap():
    """Verify train, val, and test splits have 0 common scenarios."""
    scenarios = [f"scenario_{i}" for i in range(20)]
    splitter = ScenarioBasedSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    splits = splitter.split_scenarios(scenarios)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set) + len(val_set) + len(test_set) == len(scenarios)


def test_no_future_flow_leakage_in_state_builder():
    """Verify flows occurring at or after window_end are never included in window state."""
    builder = NetworkStateBuilder(window_size_seconds=30.0)

    # Window 0: [0.0, 30.0)
    # Window 1: [30.0, 60.0)
    flows = [
        FlowRecord(timestamp=10.0, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1, dst_port=80, protocol=6, packets=5, bytes=500),
        FlowRecord(timestamp=29.999, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=2, dst_port=80, protocol=6, packets=5, bytes=500),
        # Future flow exactly at boundary 30.0
        FlowRecord(timestamp=30.0, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=3, dst_port=80, protocol=6, packets=100, bytes=50000),
    ]

    df_states = builder.build_states(flows, base_timestamp=0.0)
    assert len(df_states) == 2

    w0 = df_states.iloc[0]
    # Window 0 must only have the 2 flows before 30.0 (total_packets = 10, bytes = 1000)
    assert w0["flow_count"] == 2
    assert w0["total_packets"] == 10
    assert w0["total_bytes"] == 1000

    w1 = df_states.iloc[1]
    # Window 1 has the flow at 30.0
    assert w1["flow_count"] == 1
    assert w1["total_packets"] == 100
    assert w1["total_bytes"] == 50000


def test_scaler_isolated_from_test_distribution():
    """Verify test distribution does not leak into scaler fitted parameters."""
    rng = np.random.default_rng(123)
    
    # Train distribution centered around 10.0
    df_train = pd.DataFrame({col: rng.normal(10.0, 1.0, size=50) for col in FEATURE_NAMES})
    # Test distribution with extreme shift centered around 1000.0
    df_test = pd.DataFrame({col: rng.normal(1000.0, 1.0, size=50) for col in FEATURE_NAMES})

    scaler = FeatureScaler(scaler_type="standard")
    scaler.fit(df_train)

    # Scaler mean must reflect train (~10.0), NOT shifted by test
    fitted_means = scaler._scaler.mean_
    assert np.allclose(fitted_means, 10.0, atol=0.5)

    # When transforming test, output should reflect standard deviations from train mean
    X_test_scaled = scaler.transform(df_test)
    assert np.all(X_test_scaled > 500.0) # Correctly recognized as extreme outlier without leaking
