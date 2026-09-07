"""
CyberSentinel AI — Model Service (Phase 10).

Singleton that owns the CyberWorldModelV2 instance and exposes
all ML operations as Python-native calls (no raw tensors exposed).

Startup order:
  1. Try to load checkpoint from models/world_model_v2.pt
  2. If missing: log warning, attempt to train on available datasets
  3. Load temperature from artifacts/calibration/temperature.json
  4. All subsequent requests share a single model instance (thread-safe for inference)
"""

from __future__ import annotations

import datetime
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Project imports
from ml.world_model.world_model_v2 import (
    CyberWorldModelV2,
    CyberWorldModelTrainerV2,
    WorldModelV2,
    RolloutResult,
)
from ml.world_model.explainability import build_explanation
from ml.defense.risk_engine import (
    ForecastEvent,
    RiskEngine,
    RiskEngineConfig,
    STAGE_TAXONOMY,
)
from ml.calibration.temperature_scaling import load_temperature
from mitre.mappings.mitre_mapper import get_mitre_summary
from agent.tools.forecasting_tools import CyberSentinelAgentTools, _PHASE9_BENCHMARK
from ml.state.state_builder import FEATURE_NAMES
from ml.world_model.world_model_v2 import INPUT_DIM

logger = logging.getLogger(__name__)

_WORKSPACE = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CHECKPOINT = _WORKSPACE / "models" / "world_model_v2.pt"
_CALIBRATION_ARTIFACT = _WORKSPACE / "artifacts" / "calibration" / "temperature.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stage_name(idx: int) -> str:
    if 0 <= idx < len(STAGE_TAXONOMY):
        return STAGE_TAXONOMY[idx]
    return "UNKNOWN"


def _stage_idx(name: str) -> int:
    try:
        return STAGE_TAXONOMY.index(name)
    except ValueError:
        return STAGE_TAXONOMY.index("UNKNOWN")


# ---------------------------------------------------------------------------
# ModelService singleton
# ---------------------------------------------------------------------------

class ModelService:
    """
    Thread-safe singleton wrapping CyberWorldModelV2.

    All public methods accept and return plain Python objects (lists, dicts).
    No raw PyTorch tensors are ever exposed outside this class.
    """

    _instance: Optional["ModelService"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.trainer: Optional[CyberWorldModelTrainerV2] = None
        self.temperature: float = 1.0
        self.is_loaded: bool = False
        self.checkpoint_path: Optional[str] = None
        self.calibration_loaded: bool = False
        self.risk_engine: RiskEngine = RiskEngine()
        self.tools: CyberSentinelAgentTools = CyberSentinelAgentTools()
        self._infer_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    svc = cls()
                    svc.startup()
                    cls._instance = svc
        return cls._instance

    def startup(self) -> None:
        """Load checkpoint and calibration. Train on-the-fly if no checkpoint."""
        logger.info("ModelService startup...")
        try:
            self._load_or_train()
            self._load_calibration()
        except Exception as exc:
            logger.error("ModelService startup failed: %s", exc, exc_info=True)
            self.is_loaded = False

    # ------------------------------------------------------------------
    # Loading / training
    # ------------------------------------------------------------------

    def _load_or_train(self) -> None:
        if _DEFAULT_CHECKPOINT.exists():
            logger.info("Loading V2 checkpoint from %s", _DEFAULT_CHECKPOINT)
            self.trainer = CyberWorldModelTrainerV2.load(_DEFAULT_CHECKPOINT)
            self.checkpoint_path = str(_DEFAULT_CHECKPOINT)
            self.is_loaded = True
            logger.info("CyberWorldModelV2 loaded (temperature=%.4f)", self.trainer.temperature)
        else:
            logger.warning("No checkpoint found at %s. Attempting on-the-fly training.", _DEFAULT_CHECKPOINT)
            self._train_fresh()

    def _train_fresh(self) -> None:
        """Train V2 from scratch on available datasets. Used when no checkpoint exists."""
        try:
            from network.flow.csv_loader import CSVFlowLoader
            from ml.state.state_builder import NetworkStateBuilder
            from ml.preprocessing.scaler import FeatureScaler
            from ml.preprocessing.sequence_builder import SequenceBuilder
            from ml.preprocessing.stage_labeler import StageLabeler

            dataset_dir = _WORKSPACE / "datasets" / "sample"
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

            loader = CSVFlowLoader()
            all_flows = []
            for p in sorted(dataset_dir.glob("*.csv")):
                all_flows.extend(loader.load_flows(p))

            if not all_flows:
                raise ValueError("No flows loaded from dataset directory")

            states_df = NetworkStateBuilder(window_size_seconds=30.0).build_states(all_flows)
            states_df = StageLabeler(fallback_to_heuristics=True).attach_labels_to_dataframe(states_df)

            test_sc = ["trace_multistage_03", "trace_multistage_theta", "trace_benign_beta", "trace_recon_gamma"]
            val_sc = ["trace_multistage_01", "trace_benign_alpha"]
            train_sc = [sc for sc in states_df["scenario_id"].unique() if sc not in test_sc and sc not in val_sc]

            train_df = states_df[states_df["scenario_id"].isin(train_sc)].reset_index(drop=True)
            val_df = states_df[states_df["scenario_id"].isin(val_sc)].reset_index(drop=True)

            scaler = FeatureScaler(scaler_type="robust").fit(train_df)
            X_tr = scaler.transform(train_df)
            X_val = scaler.transform(val_df)

            seq = SequenceBuilder(sequence_length=8, pad_short_sequences=True)
            tr = seq.build_sequences(train_df, scaled_features=X_tr)
            va = seq.build_sequences(val_df, scaled_features=X_val)

            # Transition weighting (same as Phase 8C canonical training)
            is_trans = (tr.y_current_stage != tr.y_next_stage).float()
            weights = 1.0 + 9.0 * is_trans

            trainer = CyberWorldModelTrainerV2(
                input_dim=INPUT_DIM,
                hidden_dim=128, num_heads=4, num_layers=2,
                num_stages=len(STAGE_TAXONOMY),
                dropout=0.1, learning_rate=1e-3, weight_decay=1e-4,
                batch_size=32, epochs=60, patience=10,
                lambda_stage=1.0, lambda_attack=1.0,
                lambda_state=1.0, lambda_next_stage=2.0, random_seed=42,
            )
            trainer.fit(
                tr.x_seq, tr.mask, tr.y_current_stage, tr.y_next_stage,
                tr.y_attack.float(), y_next_state_train=tr.y_next_state,
                sample_weights_train=weights,
                x_val=va.x_seq, mask_val=va.mask,
                y_current_stage_val=va.y_current_stage,
                y_next_stage_val=va.y_next_stage,
                y_attack_val=va.y_attack.float(),
                y_next_state_val=va.y_next_state,
            )

            _DEFAULT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            trainer.save(_DEFAULT_CHECKPOINT)
            self.trainer = trainer
            self.checkpoint_path = str(_DEFAULT_CHECKPOINT)
            self.is_loaded = True
            logger.info("On-the-fly training complete. Checkpoint saved.")

        except Exception as exc:
            logger.error("On-the-fly training failed: %s", exc, exc_info=True)
            self.is_loaded = False

    def _load_calibration(self) -> None:
        try:
            self.temperature = load_temperature(_CALIBRATION_ARTIFACT)
            if self.trainer is not None:
                self.trainer.temperature = self.temperature
            self.calibration_loaded = True
            logger.info("Calibration temperature T=%.4f loaded.", self.temperature)
        except Exception as exc:
            logger.warning("Calibration load failed: %s. Using T=1.0.", exc)
            self.temperature = 1.0
            self.calibration_loaded = False

    # ------------------------------------------------------------------
    # Core inference (returns plain Python dicts, no tensors)
    # ------------------------------------------------------------------

    def _to_tensor(self, x_seq: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert x_seq list-of-lists → (1, T, D) tensor + all-valid mask."""
        arr = np.array(x_seq, dtype=np.float32)   # (T, D)
        x = torch.tensor(arr).unsqueeze(0)          # (1, T, D)
        mask = torch.ones(1, arr.shape[0], dtype=torch.bool)
        return x, mask

    def forecast(
        self,
        x_seq: List[List[float]],
        mask_list: Optional[List[bool]] = None,
        k_steps: int = 4,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full forward pass + rollout + explain + MITRE + risk.

        Args:
            x_seq:    (T, D) — scaled feature vectors
            mask_list: (T,) bool — True=valid timestep
            k_steps:  rollout horizon

        Returns:
            Dict matching CyberSentinelForecast schema (no tensors).
        """
        if not self.is_loaded or self.trainer is None:
            raise RuntimeError("Model not loaded. Check server logs.")

        with self._infer_lock:
            x, mask = self._to_tensor(x_seq)
            if mask_list is not None:
                mask = torch.tensor([mask_list], dtype=torch.bool)

            # 1. Forward pass
            import torch.nn.functional as F
            with torch.no_grad():
                out = self.trainer._forward(x, mask)

            T_scale = max(self.trainer.temperature, 1e-4)
            p_next = F.softmax(out.logits_next_stage / T_scale, dim=-1).cpu().numpy()[0]
            p_curr = F.softmax(out.logits_current_stage, dim=-1).cpu().numpy()[0]
            p_atk = float(torch.sigmoid(out.logits_attack_prob[0]).item())

            pred_next_idx = int(np.argmax(p_next))
            pred_curr_idx = int(np.argmax(p_curr))
            confidence = float(np.max(p_next))

            eps = 1e-12
            raw_ent = float(-np.sum(p_next * np.log(p_next + eps)))
            norm_ent = raw_ent / max(np.log(len(p_next)), eps)

            cur_state = x[0, -1, :].numpy()
            pred_state = out.pred_next_state[0].detach().cpu().numpy()

            cur_stage = _stage_name(pred_curr_idx)
            nxt_stage = _stage_name(pred_next_idx)
            transition = nxt_stage != cur_stage

            # 2. Explanation
            explanation = build_explanation(
                cur_state, pred_state, cur_stage, nxt_stage, top_k=5
            )

            # 3. ForecastEvent
            sp = {_stage_name(i): round(float(p_next[i]), 4) for i in range(len(p_next))}
            event = ForecastEvent(
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                model_version="CyberWorldModelV2",
                horizon_seconds=30,
                current_stage=cur_stage,
                current_state=cur_state.tolist(),
                predicted_stage=nxt_stage,
                predicted_next_state=pred_state.tolist(),
                attack_probability=p_atk,
                stage_probabilities=sp,
                confidence=confidence,
                uncertainty_entropy=norm_ent,
                transition_detected=transition,
                top_features=explanation["top_k_changed_features"][:5],
                scenario_id=scenario_id,
            )

            # 4. MITRE
            mitre_summary = get_mitre_summary(nxt_stage)

            # 5. Risk
            risk = self.risk_engine.evaluate(event, horizon_steps=1)

            # 6. Rollout
            rollout: RolloutResult = self.trainer.rollout(x, mask, k_steps=k_steps)
            rollout_steps = [
                {
                    "step": k + 1,
                    "predicted_stage": _stage_name(int(rollout.predicted_stages[k][0])),
                    "confidence": round(float(rollout.confidence[k][0]), 4),
                    "uncertainty": round(float(rollout.uncertainty[k][0]), 4),
                    "attack_probability": round(float(rollout.attack_probabilities[k][0]), 4),
                }
                for k in range(rollout.horizon)
            ]

            return {
                "timestamp": event.timestamp,
                "model_version": "CyberWorldModelV2",
                "forecast_horizon": 30,
                "current_stage": cur_stage,
                "attack_probability": round(p_atk, 4),
                "predicted_next_stage": nxt_stage,
                "next_stage_probability": round(float(p_next[pred_next_idx]), 4),
                "confidence": round(confidence, 4),
                "transition_detected": transition,
                "transition_confidence": round(confidence if transition else 0.0, 4),
                "uncertainty_entropy": round(norm_ent, 4),
                "stage_probabilities": sp,
                "rollout_steps": rollout_steps,
                "top_features": explanation["top_k_changed_features"][:5],
                "stage_relevant_features": explanation["stage_relevant_features"][:5],
                "explanation_narrative": explanation["narrative"],
                "mitre_techniques": mitre_summary["techniques"],
                "primary_technique_id": mitre_summary["primary_technique_id"],
                "primary_technique_name": mitre_summary["primary_technique_name"],
                "risk_score": round(risk.risk_score, 2),
                "risk_level": risk.severity,
                "recommended_priority": risk.recommended_priority,
                "time_to_transition_hint": risk.time_to_transition_hint,
                "safety_flags": rollout.safety_flags,
                "provenance": {
                    "prediction": "CyberWorldModelV2",
                    "explanation": "physical_state_delta",
                    "mitre": "MITRE_ATT&CK_v14_static_mapping",
                    "risk": "RiskEngine_deterministic",
                    "narrative": "CyberSentinel_defensive_agent",
                },
                "_event": event,   # internal; stripped before API response
            }

    def get_model_info(self) -> Dict[str, Any]:
        if self.trainer is None:
            return {"error": "Model not loaded"}
        cfg = self.trainer.config
        return {
            "model_version": "CyberWorldModelV2",
            "model_class": CyberWorldModelV2.__name__,
            "input_dim": cfg.get("input_dim", INPUT_DIM),
            "hidden_dim": cfg.get("hidden_dim", 128),
            "num_stages": cfg.get("num_stages", len(STAGE_TAXONOMY)),
            "num_heads": cfg.get("num_heads", 4),
            "num_layers": cfg.get("num_layers", 2),
            "temperature": self.temperature,
            "is_calibrated": self.calibration_loaded,
            "checkpoint_path": self.checkpoint_path,
            "benchmark": _PHASE9_BENCHMARK,
        }

    def get_event_from_forecast(self, fc: Dict[str, Any]) -> Optional[ForecastEvent]:
        """Extract the ForecastEvent embedded in a forecast dict."""
        return fc.get("_event")
