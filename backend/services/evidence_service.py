"""
CyberSentinel AI — Grounded Evidence Aggregator Service.

Extracts, standardizes, and validates empirical model telemetry and predictions
from LiveIngestService, ModelService, and ReplayService.

Strict Grounding Rule:
  Gemini must NEVER invent attack stages, risk scores, probabilities, or MITRE IDs.
  All facts passed to Gemini originate strictly from the validated output of this service.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class EvidenceService:
    """Aggregates and formats real-time CyberSentinel model evidence for LLM reasoning."""

    def __init__(self, live_ingest_service=None, replay_service=None) -> None:
        self._live_svc = live_ingest_service
        self._replay_svc = replay_service

    def get_latest_evidence(
        self,
        current_forecast: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve structured model evidence from request payload or live services.

        Priority:
          1. Explicit forecast payload provided in request
          2. Latest window from LiveIngestService event log
          3. Active replay session state
        """
        fc = dict(current_forecast) if current_forecast else None

        # Fallback to LiveIngestService latest event if available
        if not fc and self._live_svc is not None:
            try:
                events = getattr(self._live_svc, "event_log", [])
                if events:
                    if session_id:
                        matching = [e for e in reversed(events) if e.get("session_id") == session_id]
                        if matching:
                            fc = matching[0]
                    if not fc:
                        fc = events[-1]
            except Exception as exc:
                logger.warning("[EvidenceService] Live service query error: %s", exc)

        # Fallback to ReplayService active session if available
        if not fc and self._replay_svc is not None and session_id:
            try:
                session = self._replay_svc.get_session(session_id)
                if session and getattr(session, "latest_forecast", None):
                    fc = session.latest_forecast
            except Exception as exc:
                logger.warning("[EvidenceService] Replay service query error: %s", exc)

        if not fc:
            return {
                "has_sufficient_evidence": False,
                "status": "NO_TELEMETRY_AVAILABLE",
                "timestamp": _utcnow(),
                "detail": "Insufficient telemetry/model evidence is available to determine this.",
                "current_stage": None,
                "predicted_next_stage": None,
                "attack_probability": None,
                "confidence": None,
                "risk_score": None,
                "risk_level": None,
                "features": [],
                "mitre": [],
                "rollout": [],
                "telemetry": {},
                "recent_history": [],
            }

        # Extract normalized structured evidence
        cur = fc.get("current_stage")
        nxt = fc.get("predicted_next_stage")
        atk_prob = fc.get("attack_probability")
        conf = fc.get("confidence")
        risk_score = fc.get("risk_score")
        risk_level = fc.get("risk_level") or "LOW"
        priority = fc.get("recommended_priority") or "P4 — Monitor"
        trans = fc.get("transition_detected", False)
        hint = fc.get("time_to_transition_hint") or ""
        narrative = fc.get("explanation_narrative") or ""

        # Feature deltas
        feats = fc.get("top_features") or fc.get("stage_relevant_features") or []
        cleaned_features = []
        for f in feats[:5]:
            if isinstance(f, dict) and "feature" in f:
                try:
                    cleaned_features.append({
                        "feature": str(f.get("feature", "")),
                        "current": round(float(f.get("current", 0.0)), 4),
                        "predicted": round(float(f.get("predicted", 0.0)), 4),
                        "rel_change_pct": round(float(f.get("rel_change_pct", 0.0)), 1),
                    })
                except (ValueError, TypeError):
                    continue

        # MITRE techniques
        mitre_list = fc.get("mitre_techniques") or []
        cleaned_mitre = []
        for m in mitre_list:
            if isinstance(m, dict):
                cleaned_mitre.append({
                    "technique_id": m.get("technique_id", ""),
                    "name": m.get("name", ""),
                    "tactic": m.get("tactic", ""),
                })
            elif isinstance(m, str):
                cleaned_mitre.append({"technique_id": m, "name": "Active Technique", "tactic": "Unknown"})

        primary_tid = fc.get("primary_technique_id") or (cleaned_mitre[0]["technique_id"] if cleaned_mitre else None)
        primary_name = fc.get("primary_technique_name") or (cleaned_mitre[0]["name"] if cleaned_mitre else None)

        # Rollout simulation steps
        rollout_steps = fc.get("rollout_steps") or []
        cleaned_rollout = []
        for r in rollout_steps[:4]:
            if isinstance(r, dict):
                try:
                    cleaned_rollout.append({
                        "step": r.get("step", 1),
                        "predicted_stage": r.get("predicted_stage", ""),
                        "confidence": round(float(r.get("confidence", 0.0)), 3),
                    })
                except (ValueError, TypeError):
                    continue

        # Telemetry metrics
        telem = fc.get("telemetry_features") or {}
        cleaned_telem = {
            "flow_count": fc.get("flow_count", 0),
            "flows_per_second": round(float(fc.get("flows_per_second", 0.0)), 1) if fc.get("flows_per_second") is not None else None,
            "pkt_rate": round(float(telem.get("pkt_rate", 0.0)), 1) if "pkt_rate" in telem else None,
            "byte_rate": round(float(telem.get("byte_rate", 0.0)), 0) if "byte_rate" in telem else None,
            "syn_ratio": round(float(telem.get("syn_ratio", 0.0)), 3) if "syn_ratio" in telem else None,
            "rst_ratio": round(float(telem.get("rst_ratio", 0.0)), 3) if "rst_ratio" in telem else None,
        }

        # Recent history
        recent_history = []
        if self._live_svc is not None:
            try:
                hist_events = getattr(self._live_svc, "event_log", [])
                recent_history = [
                    {
                        "window_id": str(e.get("window_id", ""))[:8],
                        "stage": e.get("current_stage", ""),
                        "risk_score": e.get("risk_score", 0.0),
                        "timestamp": e.get("timestamp", ""),
                    }
                    for e in list(hist_events)[-5:]
                    if e.get("current_stage")
                ]
            except Exception:
                pass

        return {
            "has_sufficient_evidence": bool(cur),
            "status": fc.get("status", "FORECAST"),
            "timestamp": fc.get("timestamp") or _utcnow(),
            "window_id": fc.get("window_id"),
            "session_id": fc.get("session_id") or session_id,
            "source_id": fc.get("source_id") or "LiveTelemetry",
            "current_stage": cur,
            "predicted_next_stage": nxt,
            "attack_probability": round(float(atk_prob), 4) if atk_prob is not None else None,
            "confidence": round(float(conf), 4) if conf is not None else None,
            "risk_score": round(float(risk_score), 1) if risk_score is not None else None,
            "risk_level": risk_level,
            "recommended_priority": priority,
            "transition_detected": trans,
            "time_to_transition_hint": hint,
            "explanation_narrative": narrative,
            "features": cleaned_features,
            "mitre": cleaned_mitre,
            "primary_technique_id": primary_tid,
            "primary_technique_name": primary_name,
            "rollout": cleaned_rollout,
            "telemetry": cleaned_telem,
            "recent_history": recent_history,
        }

    def format_grounding_prompt(self, evidence: Dict[str, Any], user_message: str) -> str:
        """
        Build the immutable factual grounding block for Gemini.
        """
        if not evidence.get("has_sufficient_evidence"):
            return (
                "=== CYBERSENTINEL SYSTEM EVIDENCE ===\n"
                "STATUS: INSUFFICIENT_TELEMETRY\n"
                "NOTICE: No active telemetry stream or model forecast is currently available.\n"
                "=== END EVIDENCE ===\n\n"
                f"User Question: {user_message}\n\n"
                "Instruction: Explicitly state: 'Insufficient telemetry/model evidence is available to determine this.' Do NOT fabricate values."
            )

        cur = evidence.get("current_stage", "UNKNOWN")
        nxt = evidence.get("predicted_next_stage", "UNKNOWN")
        atk = evidence.get("attack_probability")
        atk_str = f"{atk:.1%}" if atk is not None else "N/A"
        conf = evidence.get("confidence")
        conf_str = f"{conf:.1%}" if conf is not None else "N/A"
        score = evidence.get("risk_score")
        score_str = f"{score:.1f}/100" if score is not None else "N/A"
        level = evidence.get("risk_level", "UNKNOWN")
        priority = evidence.get("recommended_priority", "P4")
        trans = evidence.get("transition_detected", False)
        tid = evidence.get("primary_technique_id") or "None"
        tname = evidence.get("primary_technique_name") or "None"
        hint = evidence.get("time_to_transition_hint") or "None"
        narrative = evidence.get("explanation_narrative") or "None"

        feat_lines = [
            f"  • {f['feature']}: {f['current']} -> {f['predicted']} ({f['rel_change_pct']:+.1f}% shift)"
            for f in evidence.get("features", [])
        ]
        feats_str = "\n".join(feat_lines) if feat_lines else "  • Baseline telemetry within normal bounds"

        mitre_lines = [
            f"  • {m['technique_id']}: {m['name']} ({m['tactic']})"
            for m in evidence.get("mitre", [])
        ]
        mitre_str = "\n".join(mitre_lines) if mitre_lines else f"  • {tid}: {tname}"

        rollout_lines = [
            f"  • Step +{r['step']}: {r['predicted_stage']} (conf {r['confidence']:.0%})"
            for r in evidence.get("rollout", [])
        ]
        rollout_str = "\n".join(rollout_lines) if rollout_lines else "  • No forward rollout available"

        telem = evidence.get("telemetry", {})
        telem_str = (
            f"Flows: {telem.get('flow_count', 'N/A')}, Rate: {telem.get('flows_per_second', 'N/A')} flows/s, "
            f"SYN Ratio: {telem.get('syn_ratio', 'N/A')}, RST Ratio: {telem.get('rst_ratio', 'N/A')}"
        )

        hist = evidence.get("recent_history", [])
        hist_lines = [
            f"  • W-{h['window_id']}: {h['stage']} (Risk: {h['risk_score']})"
            for h in hist
        ]
        hist_str = "\n".join(hist_lines) if hist_lines else "  • Initial window in sequence"

        return f"""=== CYBERSENTINEL VERIFIED EVIDENCE (DO NOT OVERRIDE OR INVENT) ===
Current Classified Stage: {cur}
Predicted Next Stage: {nxt}
Attack Probability: {atk_str}
Model Confidence: {conf_str}
Composite Risk Score: {score_str} ({level})
Recommended Priority: {priority}
Stage Transition Detected: {trans}
Transition Hint: {hint}
Primary MITRE Technique: {tid} — {tname}
Mapped MITRE Techniques:
{mitre_str}
Top Feature Deltas (S_t -> S_t+1):
{feats_str}
Forward Rollout Simulation (K=4):
{rollout_str}
Physical Telemetry Summary:
  {telem_str}
Recent Telemetry Window History:
{hist_str}
Physical State Explanation Narrative:
  {narrative}
Evidence Timestamp: {evidence.get('timestamp', 'Live')}
=== END VERIFIED EVIDENCE ===

Analyst Question: {user_message}
"""
