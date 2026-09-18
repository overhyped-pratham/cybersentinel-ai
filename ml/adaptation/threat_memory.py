"""
CyberSentinel AI — Threat Memory Store.

PRD Requirement (Section 12, Priority 5 WOW Feature):
- Stores human-validated security samples and analyst metadata in a persistent Threat Memory.
- Strict human-in-the-loop: only human-approved events enter memory.
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_STORAGE = _ROOT / "artifacts" / "adaptation" / "threat_memory.json"


@dataclass
class ValidatedSample:
    """A single human-validated network telemetry sample."""
    sample_id: str
    timestamp: str
    feature_vector: List[float]             # 24-D scaled state vector
    original_stage: str                     # What the pipeline originally classified
    original_risk_score: float
    original_anomaly_score: float
    validated_label: str                    # Human-assigned label (e.g. EXFILTRATION, BENIGN_FP)
    is_malicious: bool                      # Human verdict: True=attack, False=benign
    analyst_notes: str
    analyst_id: str
    incorporated_in_model: bool = False
    incorporated_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ValidatedSample":
        return cls(**d)


class ThreatMemory:
    """
    Persistent memory storing validated threat cases for human-in-the-loop adaptive learning.
    """

    def __init__(self, storage_path: Optional[Union[str, Path]] = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else _DEFAULT_STORAGE
        self._samples: Dict[str, ValidatedSample] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("samples", []):
                s = ValidatedSample.from_dict(item)
                self._samples[s.sample_id] = s
            logger.info("Loaded %d validated samples from Threat Memory (%s)", len(self._samples), self.storage_path)
        except Exception as exc:
            logger.error("Failed to load Threat Memory: %s", exc)

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_samples": len(self._samples),
            "samples": [s.to_dict() for s in self._samples.values()],
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_validation(
        self,
        feature_vector: Union[np.ndarray, List[float]],
        validated_label: str,
        is_malicious: bool,
        original_stage: str = "UNKNOWN",
        original_risk_score: float = 0.0,
        original_anomaly_score: float = 0.0,
        analyst_notes: str = "",
        analyst_id: str = "soc_lead",
        sample_id: Optional[str] = None,
    ) -> ValidatedSample:
        """
        Records human analyst validation of an alert or novel event.
        """
        s_id = sample_id or f"val-{str(uuid.uuid4())[:8]}"
        vec = [float(v) for v in np.asarray(feature_vector).flatten()]
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        sample = ValidatedSample(
            sample_id=s_id,
            timestamp=now_str,
            feature_vector=vec,
            original_stage=original_stage,
            original_risk_score=original_risk_score,
            original_anomaly_score=original_anomaly_score,
            validated_label=validated_label,
            is_malicious=is_malicious,
            analyst_notes=analyst_notes,
            analyst_id=analyst_id,
            incorporated_in_model=False,
            incorporated_version=None,
        )

        self._samples[s_id] = sample
        self.save()
        logger.info("Added validated sample %s ('%s', malicious=%s)", s_id, validated_label, is_malicious)
        return sample

    def get_unincorporated_samples(self) -> List[ValidatedSample]:
        """Returns validated samples that have not yet been trained into a model update."""
        return [s for s in self._samples.values() if not s.incorporated_in_model]

    def mark_incorporated(self, sample_ids: List[str], version: str) -> None:
        """Marks samples as incorporated into a given model version."""
        for sid in sample_ids:
            if sid in self._samples:
                self._samples[sid].incorporated_in_model = True
                self._samples[sid].incorporated_version = version
        self.save()

    def get_all_samples(self) -> List[ValidatedSample]:
        return list(self._samples.values())

    def get_sample(self, sample_id: str) -> Optional[ValidatedSample]:
        return self._samples.get(sample_id)

    def __len__(self) -> int:
        return len(self._samples)

