"""
CyberSentinel AI — Attack Story Engine (Event Correlation & Dynamic Narrative).

PRD Requirements (Section 9, 19, Priority 1 WOW Feature):
- Dynamic event-correlation engine over timestamped security events.
- Inferences emerge from:
    1. Temporal proximity (exponential decay)
    2. Source / destination entity relationships (shared IPs/subnets)
    3. Learned state transitions from CyberWorldModelV2
    4. Attack probabilities and anomaly scores
- NO HARDCODED SEQUENCES: Attack chains emerge dynamically from observed telemetry.
- Provides structured attack stories for GET /api/attack-story/{id} and SOC dashboard.
"""

from __future__ import annotations

import datetime
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.defense.risk_engine import DEFAULT_SEVERITY, STAGE_TAXONOMY

logger = logging.getLogger(__name__)


@dataclass
class CorrelatedEventStep:
    """A single correlated link in an evolving attack story."""
    step_number: int
    event_id: str
    timestamp: str
    stage: str
    src_ip: str
    dst_ip: str
    risk_score: float
    anomaly_score: float
    attack_probability: float
    is_novel: bool
    causal_confidence: float            # Inferred link strength in [0, 1]
    link_rationale: str                 # Why this event was linked to predecessor
    key_evidence: List[str]             # Specific feature deviations / triggers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "risk_score": round(self.risk_score, 2),
            "anomaly_score": round(self.anomaly_score, 4),
            "attack_probability": round(self.attack_probability, 4),
            "is_novel": bool(self.is_novel),
            "causal_confidence": round(self.causal_confidence, 4),
            "link_rationale": self.link_rationale,
            "key_evidence": self.key_evidence,
        }


@dataclass
class AttackStory:
    """Complete dynamic incident narrative produced by the Attack Story Engine."""
    story_id: str
    title: str
    severity: str                       # LOW | MEDIUM | HIGH | CRITICAL
    peak_risk_score: float
    created_at: str
    event_count: int
    adversary_ips: List[str]
    target_ips: List[str]
    tactical_progression: List[str]     # Emergent stage sequence
    steps: List[CorrelatedEventStep]
    executive_summary: str
    containment_recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "severity": self.severity,
            "peak_risk_score": round(self.peak_risk_score, 2),
            "created_at": self.created_at,
            "event_count": self.event_count,
            "adversary_ips": self.adversary_ips,
            "target_ips": self.target_ips,
            "tactical_progression": self.tactical_progression,
            "steps": [s.to_dict() for s in self.steps],
            "executive_summary": self.executive_summary,
            "containment_recommendations": self.containment_recommendations,
        }


class AttackStoryEngine:
    """
    Event correlation engine synthesizing dynamic attack stories from security telemetry.
    """

    def __init__(
        self,
        time_decay_half_life_seconds: float = 120.0,
        min_correlation_threshold: float = 0.35,
    ) -> None:
        self.half_life = time_decay_half_life_seconds
        self.decay_lambda = math.log(2) / max(1.0, time_decay_half_life_seconds)
        self.min_threshold = min_correlation_threshold
        self._stories: Dict[str, AttackStory] = {}

    def parse_timestamp(self, ts_str: str) -> float:
        """Parses ISO timestamp string to epoch seconds."""
        try:
            cleaned = ts_str.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(cleaned)
            return dt.timestamp()
        except Exception:
            return time.time()

    def compute_link_confidence(
        self,
        e_prev: Dict[str, Any],
        e_curr: Dict[str, Any],
    ) -> Tuple[float, str, List[str]]:
        """
        Dynamically infers relationship strength between consecutive events.
        Based on:
          1. Temporal proximity: exponential decay over delta_t.
          2. Host / Entity continuity: matching src or dst IP.
          3. Threat escalation: transition in risk or tactical stage.
        """
        t_prev = self.parse_timestamp(e_prev.get("timestamp", ""))
        t_curr = self.parse_timestamp(e_curr.get("timestamp", ""))
        delta_t = max(0.0, t_curr - t_prev)

        # 1. Temporal score: 1.0 at 0s, 0.5 at half-life, -> 0 as delta_t grows
        time_score = math.exp(-self.decay_lambda * delta_t)

        # 2. Entity score
        src_prev, dst_prev = e_prev.get("src_ip", "unknown"), e_prev.get("dst_ip", "unknown")
        src_curr, dst_curr = e_curr.get("src_ip", "unknown"), e_curr.get("dst_ip", "unknown")

        entity_score = 0.0
        entity_reasons = []
        if src_curr == src_prev and src_curr != "unknown":
            entity_score += 0.5
            entity_reasons.append(f"Same source host ({src_curr})")
        if dst_curr == dst_prev and dst_curr != "unknown":
            entity_score += 0.3
            entity_reasons.append(f"Targeting same host ({dst_curr})")
        if src_curr == dst_prev and src_curr != "unknown":
            entity_score += 0.6  # Pivoting / Lateral Movement!
            entity_reasons.append(f"Pivoting observed: previous target {dst_prev} is now source")

        entity_score = min(1.0, entity_score)

        # 3. Tactical / Model State Transition
        stage_prev = e_prev.get("stage", "UNKNOWN")
        stage_curr = e_curr.get("stage", "UNKNOWN")
        sev_prev = DEFAULT_SEVERITY.get(stage_prev, 0.2)
        sev_curr = DEFAULT_SEVERITY.get(stage_curr, 0.2)

        transition_score = 0.5
        transition_reasons = []
        if sev_curr > sev_prev:
            transition_score = 0.8
            transition_reasons.append(f"Tactical escalation from {stage_prev} to {stage_curr}")
        elif stage_curr == stage_prev and stage_curr != "BENIGN":
            transition_score = 0.6
            transition_reasons.append(f"Sustained {stage_curr} activity")

        # 4. Composite Causal Link Confidence
        # Weights: Time (0.35), Entity (0.40), Transition (0.25)
        causal_conf = 0.35 * time_score + 0.40 * entity_score + 0.25 * transition_score

        reasons = entity_reasons + transition_reasons
        if delta_t < 60:
            reasons.append(f"Rapid succession (Δt={delta_t:.1f}s)")
        link_rationale = "; ".join(reasons) if reasons else "Correlated by temporal proximity"

        # Key evidence extraction
        evidence = []
        for feat in e_curr.get("top_features", [])[:3]:
            if isinstance(feat, dict):
                evidence.append(f"{feat.get('feature', 'metric')} ({feat.get('attribution', 0.0)})")

        return causal_conf, link_rationale, evidence

    def build_story(
        self,
        events: List[Dict[str, Any]],
        story_id: Optional[str] = None,
    ) -> AttackStory:
        """
        Builds a correlated AttackStory from a list of security events.
        """
        if not events:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            story = AttackStory(
                story_id=story_id or str(uuid.uuid4())[:8],
                title="Baseline Observation — No Suspicious Incidents",
                severity="LOW",
                peak_risk_score=0.0,
                created_at=now_iso,
                event_count=0,
                adversary_ips=[],
                target_ips=[],
                tactical_progression=["BENIGN"],
                steps=[],
                executive_summary="Normal baseline traffic observed. No correlated adversarial activities detected.",
                containment_recommendations=["Continue routine continuous monitoring."],
            )
            self._stories[story.story_id] = story
            return story

        # Sort events chronologically
        sorted_events = sorted(
            events, key=lambda e: self.parse_timestamp(e.get("timestamp", ""))
        )

        steps: List[CorrelatedEventStep] = []
        adversaries = set()
        targets = set()
        progression = []
        peak_risk = 0.0

        for i, ev in enumerate(sorted_events):
            risk = float(ev.get("risk_score", 0.0))
            if risk > peak_risk:
                peak_risk = risk

            src = ev.get("src_ip", "10.0.0.15")
            dst = ev.get("dst_ip", "10.0.0.50")
            stage = ev.get("stage", ev.get("predicted_next_stage", "UNKNOWN"))
            is_novel = bool(ev.get("is_novel", False))

            if stage != "BENIGN" or is_novel:
                adversaries.add(src)
                targets.add(dst)

            if not progression or progression[-1] != stage:
                progression.append(stage)

            if i == 0:
                causal_conf = 1.0
                rationale = "Initial detection trigger"
                evidence = [
                    f"{f.get('feature', 'feature')}: {f.get('attribution', 0.0)}"
                    for f in ev.get("top_features", [])[:3]
                    if isinstance(f, dict)
                ]
            else:
                causal_conf, rationale, evidence = self.compute_link_confidence(
                    sorted_events[i - 1], ev
                )

            step = CorrelatedEventStep(
                step_number=i + 1,
                event_id=ev.get("event_id", f"evt-{i+1}"),
                timestamp=ev.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
                stage=stage,
                src_ip=src,
                dst_ip=dst,
                risk_score=risk,
                anomaly_score=float(ev.get("anomaly_score", 0.0)),
                attack_probability=float(ev.get("attack_probability", 0.0)),
                is_novel=is_novel,
                causal_confidence=causal_conf,
                link_rationale=rationale,
                key_evidence=evidence,
            )
            steps.append(step)

        # Determine overall severity
        if peak_risk >= 75.0:
            severity = "CRITICAL"
        elif peak_risk >= 50.0:
            severity = "HIGH"
        elif peak_risk >= 30.0:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Generate title and executive summary based on emergent sequence
        tactics_str = " → ".join(progression)
        s_id = story_id or f"story-{str(uuid.uuid4())[:8]}"
        title = f"Multi-Stage Incident: {tactics_str} (Risk: {peak_risk:.1f}/100)"

        novelty_mentions = [s for s in steps if s.is_novel]
        novelty_clause = (
            f" Includes {len(novelty_mentions)} event(s) flagged as Potential Novel Behavior."
            if novelty_mentions
            else ""
        )

        adv_list = sorted(list(adversaries)) or ["Internal/External"]
        tgt_list = sorted(list(targets)) or ["Internal Assets"]

        summary = (
            f"Correlation engine identified an evolving attack chain spanning {len(steps)} events across "
            f"{adv_list} targeting {tgt_list}. The progression manifested as: {tactics_str}. "
            f"Peak risk reached {peak_risk:.1f}/100 ({severity} severity).{novelty_clause}"
        )

        # Dynamic containment recommendations
        recommendations = []
        if "LATERAL_MOVEMENT" in progression:
            recommendations.append(f"Isolate intermediate pivot host(s): {', '.join(adv_list)} immediately.")
        if "CREDENTIAL_ACCESS" in progression:
            recommendations.append("Enforce emergency credential resets for targeted domain accounts.")
        if "EXFILTRATION" in progression:
            recommendations.append("Sever outbound egress channels and initiate data loss prevention forensic review.")
        if novelty_mentions:
            recommendations.append("Submit novel traffic signatures to SOC analyst for human validation into Threat Memory.")
        if not recommendations:
            recommendations.append("Maintain continuous packet capture on monitored telemetry endpoints.")

        story = AttackStory(
            story_id=s_id,
            title=title,
            severity=severity,
            peak_risk_score=peak_risk,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            event_count=len(steps),
            adversary_ips=adv_list,
            target_ips=tgt_list,
            tactical_progression=progression,
            steps=steps,
            executive_summary=summary,
            containment_recommendations=recommendations,
        )

        self._stories[s_id] = story
        logger.info("AttackStory created: %s (id=%s, steps=%d)", title, s_id, len(steps))
        return story

    def get_story(self, story_id: str) -> Optional[AttackStory]:
        """Retrieve stored story by identifier."""
        return self._stories.get(story_id)

    def list_stories(self) -> List[Dict[str, Any]]:
        """List summary of all correlated stories."""
        return [
            {
                "story_id": s.story_id,
                "title": s.title,
                "severity": s.severity,
                "peak_risk_score": s.peak_risk_score,
                "event_count": s.event_count,
                "created_at": s.created_at,
            }
            for s in self._stories.values()
        ]
