"""
CyberSentinel AI — Replay Session Service (Phase 10).

Manages in-memory step-through replay sessions for test scenarios.

Sessions are keyed by UUID. Each session holds pre-loaded sequence data
so that stepping forward requires only model inference, no disk I/O.

GUARANTEE: Replay uses the ACTUAL trained CyberWorldModelV2.
           No canned predictions are stored.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ml.defense.risk_engine import STAGE_TAXONOMY
from ml.state.state_builder import FEATURE_NAMES
from ml.world_model.world_model_v2 import INPUT_DIM

logger = logging.getLogger(__name__)

_WORKSPACE = Path(__file__).resolve().parent.parent.parent


def _stage_name(idx: int) -> str:
    if 0 <= idx < len(STAGE_TAXONOMY):
        return STAGE_TAXONOMY[idx]
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class ReplaySession:
    session_id: str
    scenario_id: str
    x_seqs: torch.Tensor           # (N, T, D)
    masks: torch.Tensor             # (N, T)
    true_stages: List[str]          # ground-truth stage per window
    true_next_stages: List[str]
    k_steps: int
    current_window: int = 0
    is_complete: bool = False
    started_at: float = field(default_factory=time.time)
    telemetry_rows: Optional[List[Dict[str, float]]] = None


# ---------------------------------------------------------------------------
# ReplayService
# ---------------------------------------------------------------------------

class ReplayService:
    """
    Creates and manages in-memory replay sessions.

    Usage:
        svc = ReplayService()
        session_id, info = svc.start_session("trace_multistage_03", k_steps=4)
        step_result = svc.step(session_id)     # returns model forecast dict
        status = svc.get_status(session_id)
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ReplaySession] = {}
        self._scaler = None

    def _load_data(self):
        """Load and segment dataset. Cached after first call."""
        if self._scaler is not None:
            return

        try:
            from network.flow.csv_loader import CSVFlowLoader
            from ml.state.state_builder import NetworkStateBuilder
            from ml.preprocessing.scaler import FeatureScaler
            from ml.preprocessing.sequence_builder import SequenceBuilder
            from ml.preprocessing.stage_labeler import StageLabeler

            dataset_dir = _WORKSPACE / "datasets" / "sample"
            loader = CSVFlowLoader()
            all_flows = []
            for p in sorted(dataset_dir.glob("*.csv")):
                all_flows.extend(loader.load_flows(p))

            states_df = NetworkStateBuilder(window_size_seconds=30.0).build_states(all_flows)
            states_df = StageLabeler(fallback_to_heuristics=True).attach_labels_to_dataframe(states_df)

            test_sc = ["trace_multistage_03", "trace_multistage_theta",
                       "trace_benign_beta", "trace_recon_gamma"]
            val_sc = ["trace_multistage_01", "trace_benign_alpha"]
            train_sc = [sc for sc in states_df["scenario_id"].unique()
                        if sc not in test_sc and sc not in val_sc]

            train_df = states_df[states_df["scenario_id"].isin(train_sc)].reset_index(drop=True)

            self._scaler = FeatureScaler(scaler_type="robust").fit(train_df)
            self._states_df = states_df
            self._seq_builder = SequenceBuilder(sequence_length=8, pad_short_sequences=True)
            logger.info("ReplayService: dataset loaded (%d windows total)", len(states_df))

        except Exception as exc:
            logger.error("ReplayService: data load failed: %s", exc, exc_info=True)
            raise

    def start_session(
        self, scenario_id: str, k_steps: int = 4
    ) -> tuple[str, Dict[str, Any]]:
        """
        Initialise a new replay session for the given scenario.

        Returns (session_id, info_dict).
        """
        self._load_data()

        scenario_df = self._states_df[
            self._states_df["scenario_id"] == scenario_id
        ].reset_index(drop=True)

        if scenario_df.empty:
            available = sorted(self._states_df["scenario_id"].unique().tolist())
            raise ValueError(
                f"Scenario '{scenario_id}' not found. "
                f"Available: {available}"
            )

        X_scenario = self._scaler.transform(scenario_df)
        sequences = self._seq_builder.build_sequences(scenario_df, scaled_features=X_scenario)

        session_id = str(uuid.uuid4())[:8]
        true_stages = [_stage_name(int(y)) for y in sequences.y_current_stage]
        true_next = [_stage_name(int(y)) for y in sequences.y_next_stage]

        from ml.state.state_builder import FEATURE_NAMES
        telemetry_rows = []
        for _, row in scenario_df.iterrows():
            telemetry_rows.append({
                feat: round(float(row[feat]), 4) if feat in row else 0.0
                for feat in FEATURE_NAMES
            })

        session = ReplaySession(
            session_id=session_id,
            scenario_id=scenario_id,
            x_seqs=sequences.x_seq,
            masks=sequences.mask,
            true_stages=true_stages,
            true_next_stages=true_next,
            k_steps=k_steps,
            telemetry_rows=telemetry_rows,
        )
        self._sessions[session_id] = session
        logger.info("Replay session %s started: scenario=%s, windows=%d",
                    session_id, scenario_id, len(true_stages))

        return session_id, {
            "session_id": session_id,
            "scenario_id": scenario_id,
            "total_windows": len(true_stages),
            "stage_sequence": true_stages,
            "message": f"Session ready. Use POST /replay/step with session_id='{session_id}'.",
        }

    def step(self, session_id: str) -> Dict[str, Any]:
        """
        Advance the session by one window and return the model forecast.

        Returns dict matching ReplayStepResponse schema.
        """
        session = self._get_session(session_id)
        if session.is_complete:
            return {
                "session_id": session_id,
                "window_index": session.current_window,
                "total_windows": len(session.true_stages),
                "is_complete": True,
                "forecast": None,
            }

        idx = session.current_window
        x = session.x_seqs[idx:idx+1]    # (1, T, D)
        mask = session.masks[idx:idx+1]   # (1, T)

        # Use ModelService for inference (avoids duplication)
        from backend.services.model_service import ModelService
        svc = ModelService.get_instance()

        x_list = x[0].numpy().tolist()
        mask_list = mask[0].numpy().tolist()

        fc = svc.forecast(
            x_seq=x_list,
            mask_list=[bool(m) for m in mask_list],
            k_steps=session.k_steps,
            scenario_id=session.scenario_id,
        )

        # Inject ground-truth for UI comparison
        fc["ground_truth_stage"] = session.true_stages[idx]
        fc["ground_truth_next_stage"] = session.true_next_stages[idx]
        fc["window_index"] = idx
        fc["observed_stages"] = list(dict.fromkeys(session.true_stages[:idx+1]))

        # Attach real telemetry metrics from the scenario dataframe
        if session.telemetry_rows and idx < len(session.telemetry_rows):
            t_row = session.telemetry_rows[idx]
            fc["telemetry_features"] = t_row
            fc["flow_count"] = int(t_row.get("flow_count", 0))
            fc["flows_per_second"] = round(float(fc["flow_count"]) / 30.0, 2)
            fc["source_id"] = f"Replay:{session.scenario_id}"
            fc["source_kind"] = "Local Trace Replay"

        session.current_window += 1
        if session.current_window >= len(session.true_stages):
            session.is_complete = True

        return {
            "session_id": session_id,
            "window_index": idx,
            "total_windows": len(session.true_stages),
            "is_complete": session.is_complete,
            "forecast": fc,
        }

    def get_status(self, session_id: str) -> Dict[str, Any]:
        session = self._get_session(session_id)
        return {
            "session_id": session_id,
            "scenario_id": session.scenario_id,
            "current_window": session.current_window,
            "total_windows": len(session.true_stages),
            "is_complete": session.is_complete,
            "elapsed_seconds": round(time.time() - session.started_at, 2),
        }

    def list_available_scenarios(self) -> List[str]:
        """Return all scenario IDs available in the loaded dataset."""
        self._load_data()
        return sorted(self._states_df["scenario_id"].unique().tolist())

    def _get_session(self, session_id: str) -> ReplaySession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found or expired.")
        return session

    def cleanup_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
