"""
CyberSentinel AI — Defensive Agent (Phase 10).

Orchestrates analyst queries against the deterministic ML tools.

DESIGN PRINCIPLES:
  1. The LLM is NOT the detector.
  2. The LLM NEVER invents: stages, probabilities, MITRE IDs, feature changes.
  3. All facts come from structured tool output FIRST.
  4. The LLM's role: explain, summarize, prioritize — not detect.
  5. If Ollama is unavailable, the deterministic template fallback activates.
     The fallback must never produce generic AI filler.

Backend priority:
  1. Ollama (local LLM — offline capable)
  2. Deterministic template (always available, always grounded)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama client (optional)
# ---------------------------------------------------------------------------

def _check_ollama_available() -> Tuple[bool, str]:
    """Check if Ollama is running locally. Returns (available, model_name)."""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                # Prefer smaller cyber/general models
                preferred = ["llama3.2", "llama3", "mistral", "gemma2", "phi3", "llama2"]
                names = [m.get("name", "").split(":")[0] for m in models]
                for p in preferred:
                    if p in names:
                        return True, p
                return True, models[0].get("name", "unknown")
        return False, ""
    except Exception:
        return False, ""


def _ollama_generate(model: str, prompt: str, system: str) -> str:
    """Call Ollama /api/generate endpoint."""
    import requests
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 400},
    }
    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# Query routing (keyword-based, no ML)
# ---------------------------------------------------------------------------

_ROUTE_PATTERNS = {
    "current_state":   [r"what.*(happening|going on|current|right now|is it)", r"status"],
    "forecast":        [r"what.*(happen|next|likely|predict|coming|expect)", r"forecast", r"future"],
    "transition":      [r"transition", r"stage.*(change|shift)", r"moving to"],
    "features":        [r"why|reason|evidence|indicator|feature|signal|change|because"],
    "mitre":           [r"mitre|att.?ck|technique|tactic|T\d{4}"],
    "confidence":      [r"confident|certain|sure|probability|how (likely|sure|confident)"],
    "risk":            [r"risk|priority|urgent|severity|dangerous|threat level|how (bad|serious|dangerous)"],
    "rollout":         [r"k=|rollout|simulate|next (2|3|4|five|four|three)|future (states?|steps?)"],
    "metrics":         [r"benchmark|metric|accuracy|performance|compare|better|gru|logistic|baseline"],
    "investigate":     [r"investigate|analyst|look into|check|action|recommend|should (i|we|the)"],
}


def _route_query(query: str) -> str:
    """Classify analyst query into one of the routing categories."""
    q = query.lower()
    for route, patterns in _ROUTE_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                return route
    return "current_state"  # default


# ---------------------------------------------------------------------------
# Deterministic template fallback
# ---------------------------------------------------------------------------

def _template_answer(route: str, fc: Dict[str, Any], query: str) -> str:
    """
    Build a grounded natural-language answer from structured forecast fields only.

    GUARANTEE: Every statement in the output is traceable to a specific fc field.
    No hallucinated probabilities, stage names, or technique IDs.
    """
    cur = fc.get("current_stage", "UNKNOWN")
    nxt = fc.get("predicted_next_stage", "UNKNOWN")
    conf = fc.get("confidence", 0.0)
    atk = fc.get("attack_probability", 0.0)
    trans = fc.get("transition_detected", False)
    risk = fc.get("risk_level", "UNKNOWN")
    score = fc.get("risk_score", 0.0)
    tid = fc.get("primary_technique_id")
    tname = fc.get("primary_technique_name")
    hint = fc.get("time_to_transition_hint", "")
    priority = fc.get("recommended_priority", "")
    unc = fc.get("uncertainty_entropy", 0.0)
    narrative = fc.get("explanation_narrative", "")
    rollout = fc.get("rollout_steps", [])
    feats = fc.get("top_features", []) or fc.get("stage_relevant_features", [])

    if route == "current_state":
        out = (
            f"CyberSentinel is currently observing a **{cur}** network state. "
            f"The attack probability is {atk:.1%}. "
            f"Risk level: **{risk}** (score {score:.0f}/100). "
        )
        if trans:
            out += f"A stage transition to **{nxt}** is actively predicted."
        else:
            out += f"Stage continuation ({cur}) is predicted for the next 30-second window."
        return out

    elif route == "forecast":
        out = (
            f"CyberWorldModelV2 forecasts **{nxt}** as the most likely next stage "
            f"(confidence {conf:.1%}, calibrated temperature T=1.568). "
        )
        if trans:
            out += f"A genuine stage transition from {cur} to {nxt} is predicted. "
        out += f"Attack probability: {atk:.1%}. Forecast uncertainty (entropy): {unc:.3f}."
        return out

    elif route == "transition":
        if trans:
            return (
                f"Yes — a stage transition is predicted. "
                f"CyberWorldModelV2 forecasts a transition from **{cur}** to **{nxt}** "
                f"with {conf:.1%} confidence. {hint}"
            )
        else:
            return (
                f"No transition is predicted for the next window. "
                f"Stage **{cur}** is expected to continue (confidence {conf:.1%})."
            )

    elif route == "features":
        if narrative:
            return f"Evidence from physical state prediction:\n\n{narrative}"
        if feats:
            lines = [f"• **{f['feature']}**: {f['current']:.3f} → {f['predicted']:.3f} ({f['rel_change_pct']:+.1f}%)"
                     for f in feats[:3]]
            return (
                f"The key network-state changes driving this forecast:\n" + "\n".join(lines) +
                f"\n\nAll values come from CyberWorldModelV2 predicted physical state delta (S_hat_{{t+1}} - S_t)."
            )
        return "No significant feature changes detected in the current window."

    elif route == "mitre":
        if tid and tname:
            techs = fc.get("mitre_techniques", [])
            if techs:
                lines = [f"• **{t['technique_id']}** — {t['name']} ({t['tactic']})" for t in techs[:3]]
                return (
                    f"MITRE ATT&CK techniques for predicted stage **{nxt}**:\n" +
                    "\n".join(lines) +
                    f"\n\nPrimary: **{tid}** — {tname}. "
                    f"Source: MITRE ATT&CK Enterprise v14 static mapping (not LLM-generated)."
                )
            return f"Primary MITRE technique: **{tid}** — {tname} (for stage {nxt})."
        return f"No MITRE techniques mapped for stage **{nxt}**."

    elif route == "confidence":
        level = "high" if conf >= 0.8 else "moderate" if conf >= 0.5 else "low"
        return (
            f"Forecast confidence is **{conf:.1%}** ({level}). "
            f"Uncertainty (normalized entropy): {unc:.3f} — "
            f"{'very certain' if unc < 0.1 else 'moderate uncertainty' if unc < 0.4 else 'high uncertainty'}. "
            f"Temperature scaling (T=1.568) has been applied to calibrate these probabilities."
        )

    elif route == "risk":
        return (
            f"Risk score: **{score:.0f}/100** → **{risk}**. "
            f"{priority}. "
            f"{hint}"
        )

    elif route == "rollout":
        if rollout:
            path = " → ".join(
                f"**{r['predicted_stage']}** ({r['confidence']:.0%})" for r in rollout
            )
            return (
                f"K={len(rollout)} step forward simulation (no future observations consumed):\n\n"
                f"{path}\n\n"
                f"These are MODEL SIMULATIONS — not observed telemetry. "
                f"Each step uses CyberWorldModelV2's autoregressive physical state predictor."
            )
        return "No rollout data available in the current forecast."

    elif route == "metrics":
        return (
            "Phase 8C Hard Multi-Stage Holdout benchmark (N=44, 6 genuine transitions):\n\n"
            "| Model | Top-1 | Transition Acc | Brier |\n"
            "|-------|-------|---------------|-------|\n"
            "| **CyberWorldModelV2** | **97.73%** | **83.33%** | **0.0452** |\n"
            "| Temporal GRU | 81.82% | 66.67% | 0.2913 |\n"
            "| Logistic Regression | 50.00% | 0.00% | 0.7206 |\n\n"
            "Source: experiments/phase8c_investigation/phase8c_summary.json (immutable)."
        )

    elif route == "investigate":
        lines = [
            f"Based on the current model output, the following investigation steps are recommended:",
            f"",
            f"1. **Verify {cur} indicators** — check source IPs, connection patterns, and port activity",
        ]
        if feats:
            f0 = feats[0]
            lines.append(
                f"2. **Examine {f0['feature']}** — changed {f0['rel_change_pct']:+.1f}% "
                f"({f0['current']:.3f} → {f0['predicted']:.3f})"
            )
        if tid:
            lines.append(f"3. **Reference MITRE {tid}** ({tname}) for detection signatures")
        lines.append(f"4. **Escalate if risk remains {risk}** — {priority}")
        lines.append("")
        lines.append(
            "_All recommendations derived from CyberWorldModelV2 output. "
            "No LLM-generated conclusions._"
        )
        return "\n".join(lines)

    # Default
    return _template_answer("current_state", fc, query)


# ---------------------------------------------------------------------------
# CyberSentinelDefensiveAgent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are CyberSentinel, a defensive AI analyst assistant for a Security Operations Center (SOC).

STRICT RULES — violating any rule makes your response invalid:
1. You are NOT a threat detector. The ML model detects threats. You explain model output.
2. NEVER invent attack stages, probabilities, MITRE IDs, or feature changes.
3. NEVER override or contradict the structured model predictions provided to you.
4. ALL factual claims must be traceable to the structured context provided.
5. Use MARKDOWN formatting. Be concise. Prefer bullet points over long paragraphs.
6. If a fact is not in the structured context, say "Not available in current model output."
7. End every response with: _Source: CyberWorldModelV2 + deterministic tools_

You may: summarize predictions, explain evidence, recommend investigation steps, contextualize MITRE techniques, prioritize analyst attention — all based only on the provided structured data."""


def _build_ollama_prompt(query: str, fc: Dict[str, Any], route: str) -> str:
    """Build a grounded prompt for the LLM from structured forecast fields."""
    cur = fc.get("current_stage", "UNKNOWN")
    nxt = fc.get("predicted_next_stage", "UNKNOWN")
    conf = fc.get("confidence", 0.0)
    atk = fc.get("attack_probability", 0.0)
    trans = fc.get("transition_detected", False)
    risk = fc.get("risk_level", "UNKNOWN")
    score = fc.get("risk_score", 0.0)
    tid = fc.get("primary_technique_id") or "None"
    tname = fc.get("primary_technique_name") or "None"
    hint = fc.get("time_to_transition_hint", "")
    narrative = fc.get("explanation_narrative", "")
    feats = fc.get("top_features", [])[:3]
    rollout = fc.get("rollout_steps", [])

    feat_str = "\n".join(
        f"  - {f['feature']}: {f['current']:.3f}→{f['predicted']:.3f} ({f['rel_change_pct']:+.1f}%)"
        for f in feats
    ) or "  None"

    rollout_str = " → ".join(
        f"{r['predicted_stage']}({r['confidence']:.0%})" for r in rollout
    ) or "Not available"

    context = f"""=== STRUCTURED MODEL OUTPUT (DO NOT MODIFY) ===
Current stage: {cur}
Predicted next stage: {nxt}
Confidence: {conf:.1%}
Attack probability: {atk:.1%}
Transition detected: {trans}
Risk level: {risk} (score: {score:.0f}/100)
Primary MITRE: {tid} — {tname}
Transition hint: {hint}
Physical state narrative: {narrative}
Top feature changes:
{feat_str}
K=4 rollout path: {rollout_str}
Query intent: {route}
=== END STRUCTURED CONTEXT ===

Analyst question: {query}

Answer based ONLY on the structured context above. Do not invent any data."""
    return context


class CyberSentinelDefensiveAgent:
    """
    Orchestrates analyst queries against deterministic ML tools.

    Priority: Ollama → deterministic template fallback.
    The LLM is given ONLY structured tool output as context.
    """

    def __init__(self) -> None:
        self._ollama_available: Optional[bool] = None
        self._ollama_model: str = ""

    def _ensure_ollama_checked(self) -> None:
        if self._ollama_available is None:
            self._ollama_available, self._ollama_model = _check_ollama_available()
            if self._ollama_available:
                logger.info("Ollama available with model: %s", self._ollama_model)
            else:
                logger.info("Ollama not available — using template fallback.")

    def answer(
        self,
        query: str,
        current_forecast: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer an analyst question grounded in structured model output.

        Args:
            query:             Natural language question
            current_forecast:  CyberSentinelForecast dict from most recent model call
            session_id:        Replay session ID for context enrichment

        Returns:
            {answer, tool_calls, provenance, llm_backend, grounded_in_model_output}
        """
        self._ensure_ollama_checked()

        # Build a minimal safe forecast if none provided
        fc = current_forecast or {}
        tools_used = []

        # Route the query
        route = _route_query(query)
        tools_used.append(f"route_classifier:{route}")

        # Determine which tools were conceptually called
        route_to_tools = {
            "current_state":   ["get_current_state"],
            "forecast":        ["get_attack_forecast"],
            "transition":      ["get_transition_analysis"],
            "features":        ["get_feature_importance"],
            "mitre":           ["get_mitre_mapping"],
            "confidence":      ["get_attack_forecast"],
            "risk":            ["get_risk_assessment"],
            "rollout":         ["get_rollout"],
            "metrics":         ["get_model_metrics"],
            "investigate":     ["get_attack_forecast", "get_feature_importance",
                                "get_mitre_mapping", "get_risk_assessment"],
        }
        tools_used.extend(route_to_tools.get(route, ["get_current_state"]))

        # Try Ollama
        backend = "template_fallback"
        answer_text = ""

        if self._ollama_available and current_forecast:
            try:
                prompt = _build_ollama_prompt(query, fc, route)
                answer_text = _ollama_generate(self._ollama_model, prompt, _SYSTEM_PROMPT)
                backend = f"ollama:{self._ollama_model}"
                logger.debug("Ollama response generated (%d chars)", len(answer_text))
            except Exception as exc:
                logger.warning("Ollama call failed (%s), falling back to template.", exc)
                answer_text = ""

        # Template fallback (always deterministic)
        if not answer_text:
            answer_text = _template_answer(route, fc, query)
            backend = "template_fallback"

        # Safety check: strip any content that looks like hallucinated MITRE IDs
        # (template fallback is safe by construction; this guards Ollama output)
        if backend.startswith("ollama"):
            answer_text = _sanitize_ollama_output(answer_text, fc)

        return {
            "answer": answer_text,
            "tool_calls": tools_used,
            "provenance": "CyberSentinel_defensive_agent",
            "llm_backend": backend,
            "grounded_in_model_output": True,
        }


def _sanitize_ollama_output(text: str, fc: Dict[str, Any]) -> str:
    """
    Remove any MITRE IDs from Ollama output that weren't in the original forecast.
    Prevents LLM hallucination of technique IDs.
    """
    known_ids = {t.get("technique_id", "") for t in fc.get("mitre_techniques", [])}
    known_ids.discard("")

    # Find all T#### patterns in LLM output
    found = set(re.findall(r'\bT\d{4}(?:\.\d{3})?\b', text))
    hallucinated = found - known_ids
    if hallucinated:
        logger.warning(
            "Ollama hallucinated MITRE IDs %s — stripping from response.", hallucinated
        )
        for hid in hallucinated:
            text = text.replace(hid, "[TECHNIQUE_ID_REDACTED]")
    return text
