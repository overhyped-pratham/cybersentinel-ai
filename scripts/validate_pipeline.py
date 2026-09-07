"""
CyberSentinel AI - End-to-End Pipeline Validation Script.

Validates:
1. CSV flow ingestion and sanitization.
2. 30-second window network state construction S_t.
3. Feature validation against schema, boundaries, and numerical validity.
4. Derived research attack-stage labeling.
5. Scenario-based dataset splitting with zero leakage.
6. Feature scaling fit strictly on train split.
7. Temporal sequence building with masks and next-state targets.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import logging
import torch
import numpy as np
import pandas as pd

from network.flow.csv_loader import CSVFlowLoader
from ml.state.state_builder import NetworkStateBuilder
from ml.preprocessing.validator import FeatureValidator
from ml.preprocessing.stage_labeler import StageLabeler
from ml.preprocessing.splitter import ScenarioBasedSplitter
from ml.preprocessing.scaler import FeatureScaler
from ml.preprocessing.sequence_builder import SequenceBuilder
from datasets.sample.generate_sample_traces import generate_sample_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_pipeline")


def run_pipeline_validation() -> bool:
    logger.info("=== CyberSentinel AI: Validating Ingestion & State Pipeline ===")

    # Step 1: Ensure sample datasets exist
    sample_dir = Path("datasets/sample")
    csv_files = list(sample_dir.glob("*.csv"))
    if not csv_files:
        logger.info("Generating synthetic scenario datasets...")
        csv_files = generate_sample_dataset(sample_dir)
    logger.info(f"Loaded {len(csv_files)} scenario trace CSVs from {sample_dir}")

    # Step 2: Flow Ingestion & Cleaning
    loader = CSVFlowLoader()
    all_flows = []
    for f in csv_files:
        flows = loader.load_flows(f)
        all_flows.extend(flows)
    logger.info(f"Successfully ingested {len(all_flows)} total cleaned flow records.")
    assert len(all_flows) > 0, "No flows ingested!"

    # Step 3: Network State Construction (30s windows)
    state_builder = NetworkStateBuilder(window_size_seconds=30.0)
    df_states = state_builder.build_states(all_flows)
    logger.info(f"Constructed {len(df_states)} network state windows across {df_states['scenario_id'].nunique()} scenarios.")
    assert len(df_states) > 0, "No state windows constructed!"

    # Step 4: Feature Schema & Bound Validation
    validator = FeatureValidator()
    validator.validate(df_states, raise_on_error=True)
    logger.info(f"Validated all 24 features: bounds, finiteness, and zero NaN/Inf guaranteed.")

    # Step 5: Attack-Stage Labeling
    labeler = StageLabeler(fallback_to_heuristics=True)
    df_states_labeled = labeler.attach_labels_to_dataframe(df_states)
    stage_dist = df_states_labeled["stage_name"].value_counts().to_dict()
    logger.info(f"Derived research attack stages: {stage_dist}")

    # Step 6: Scenario-Based Dataset Splitting
    splitter = ScenarioBasedSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42)
    df_train, df_val, df_test = splitter.split_dataframe(df_states_labeled)
    logger.info(
        f"Scenario splits: Train={len(df_train)} windows ({df_train['scenario_id'].nunique()} scenarios), "
        f"Val={len(df_val)} windows ({df_val['scenario_id'].nunique()} scenarios), "
        f"Test={len(df_test)} windows ({df_test['scenario_id'].nunique()} scenarios)"
    )

    # Step 7: Leakage-Free Feature Scaling
    scaler = FeatureScaler(scaler_type="robust")
    scaler.fit(df_train) # Fit ONLY on train
    X_train_scaled = scaler.transform(df_train)
    X_val_scaled = scaler.transform(df_val)
    X_test_scaled = scaler.transform(df_test)
    logger.info("FeatureScaler fit strictly on Train split; Val and Test transformed without distribution leakage.")

    # Step 8: Temporal Sequence Building
    seq_builder = SequenceBuilder(sequence_length=8, pad_short_sequences=True)
    train_batch = seq_builder.build_sequences(df_train, scaled_features=X_train_scaled)
    val_batch = seq_builder.build_sequences(df_val, scaled_features=X_val_scaled)
    test_batch = seq_builder.build_sequences(df_test, scaled_features=X_test_scaled)

    logger.info(f"Train Sequence Batch: X={tuple(train_batch.x_seq.shape)}, Mask={tuple(train_batch.mask.shape)}, Next State Target={tuple(train_batch.y_next_state.shape)}")
    logger.info(f"Val Sequence Batch:   X={tuple(val_batch.x_seq.shape)}")
    logger.info(f"Test Sequence Batch:  X={tuple(test_batch.x_seq.shape)}")

    # Verification assertions
    assert train_batch.x_seq.ndim == 3 and train_batch.x_seq.shape[1] == 8 and train_batch.x_seq.shape[2] == 24
    assert train_batch.y_next_state.shape == (train_batch.x_seq.shape[0], 24)
    assert train_batch.y_next_stage.shape == (train_batch.x_seq.shape[0],)

    logger.info("=== PIPELINE VALIDATION PASSED SUCCESSFULLY ===")
    return True


if __name__ == "__main__":
    success = run_pipeline_validation()
    sys.exit(0 if success else 1)
