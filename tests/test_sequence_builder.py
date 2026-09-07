"""
Unit tests for ml.preprocessing.sequence_builder.SequenceBuilder.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from ml.preprocessing.sequence_builder import SequenceBuilder
from ml.state.state_builder import FEATURE_NAMES


def test_sequence_builder_shapes_and_masks():
    # 5 windows in scenario_A
    seq_len = 4
    n_windows = 5
    feature_dim = len(FEATURE_NAMES)

    data = {f: np.ones(n_windows, dtype=np.float32) * i for i, f in enumerate(FEATURE_NAMES)}
    data["scenario_id"] = ["scenario_A"] * n_windows
    data["stage_id"] = [1, 2, 3, 4, 5]
    data["is_attack"] = [0.0, 1.0, 1.0, 1.0, 1.0]
    data["timestamp_start"] = [100.0, 130.0, 160.0, 190.0, 220.0]

    df_states = pd.DataFrame(data)

    builder = SequenceBuilder(sequence_length=seq_len, pad_short_sequences=True)
    batch = builder.build_sequences(df_states)

    # With 5 states (t=0,1,2,3,4), valid transitions (t -> t+1) are 4: t=0,1,2,3
    assert batch.x_seq.shape == (4, seq_len, feature_dim)
    assert batch.mask.shape == (4, seq_len)
    assert batch.y_next_state.shape == (4, feature_dim)
    assert batch.y_current_stage.shape == (4,)
    assert batch.y_next_stage.shape == (4,)

    # Check padding mask for first step (t=0 has 1 real state, 3 padded)
    assert batch.mask[0].tolist() == [False, False, False, True]
    # Check t=3 (has 4 real states, 0 padded)
    assert batch.mask[3].tolist() == [True, True, True, True]

    # Verify stage progression
    assert batch.y_current_stage.tolist() == [1, 2, 3, 4]
    assert batch.y_next_stage.tolist() == [2, 3, 4, 5]


def test_sequence_builder_no_cross_scenario():
    # 2 scenarios: scenario_1 has 3 windows, scenario_2 has 3 windows
    seq_len = 4
    feature_dim = len(FEATURE_NAMES)

    rows = []
    # Scenario 1
    for t in range(3):
        r = {f: float(t) for f in FEATURE_NAMES}
        r["scenario_id"] = "scenario_1"
        r["stage_id"] = t
        r["is_attack"] = 0.0
        r["timestamp_start"] = float(100 + t * 30)
        rows.append(r)

    # Scenario 2
    for t in range(3):
        r = {f: float(100 + t) for f in FEATURE_NAMES}
        r["scenario_id"] = "scenario_2"
        r["stage_id"] = 10 + t
        r["is_attack"] = 1.0
        r["timestamp_start"] = float(500 + t * 30)
        rows.append(r)

    df_states = pd.DataFrame(rows)

    builder = SequenceBuilder(sequence_length=seq_len, pad_short_sequences=True)
    batch = builder.build_sequences(df_states)

    # Scenario 1 produces 2 sequences (t=0->1, t=1->2)
    # Scenario 2 produces 2 sequences (t=0->1, t=1->2)
    # Total = 4 sequences
    assert batch.x_seq.shape[0] == 4

    # Assert sequence IDs and scenario IDs never mix
    assert batch.scenario_ids == ["scenario_1", "scenario_1", "scenario_2", "scenario_2"]

    # Verify that scenario_2 sequences NEVER contain scenario_1 values (which are < 10)
    sc2_seq0 = batch.x_seq[2] # first sequence of scenario_2
    # The valid positions in sc2_seq0 must only have values >= 100
    valid_positions = batch.mask[2]
    sc2_values = sc2_seq0[valid_positions]
    assert (sc2_values >= 100.0).all()
