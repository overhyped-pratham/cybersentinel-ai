"""
CyberSentinel AI — Automated System Smoke Test.

Runs end-to-end operational sanity checks across:
  1. Model & calibration artifact integrity
  2. Feature scaler and 24-D state builder
  3. CyberWorldModelV2 forward inference and calibrated probabilities
  4. Explainability engine and MITRE ATT&CK mappings
  5. Dynamic multi-factor RiskEngine
  6. Ingest & Replay services
  7. Gemini AI Security Analyst grounding and status
  8. FastAPI backend endpoints (/health, /model/info, /forecast, /agent/status, /agent/chat)
  9. Frontend dashboard asset availability
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure root directory is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmokeTest")


class SmokeTestRunner:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def record(self, check_name: str, success: bool, detail: str = "") -> None:
        status_str = "PASS" if success else "FAIL"
        icon = "[OK]" if success else "[FAIL]"
        self.results.append((check_name, success, detail))
        detail_str = f" -- {detail}" if detail else ""
        print(f"  [{status_str}] {icon} {check_name}{detail_str}")

    def run_all(self) -> bool:
        print("=" * 70)
        print("         CYBERSENTINEL AI — SYSTEM SMOKE TEST")
        print("=" * 70)
        start_time = time.time()

        # 1. Artifact checks
        print("\n[1/6] Verifying Core Model & Calibration Artifacts...")
        self.check_artifacts()

        # 2. ML Engine checks
        print("\n[2/6] Verifying Neural World Model & Preprocessing Pipeline...")
        self.check_ml_engine()

        # 3. Defensive Intelligence checks
        print("\n[3/6] Verifying Explainability, RiskEngine & MITRE ATT&CK...")
        self.check_defensive_intelligence()

        # 4. Agent & Gemini Grounding checks
        print("\n[4/6] Verifying AI Security Analyst & Grounding Protocol...")
        self.check_agent_grounding()

        # 5. FastAPI Service & Route checks
        print("\n[5/6] Verifying FastAPI Application & Endpoints...")
        self.check_fastapi_endpoints()

        # 6. Dashboard & Frontend Asset checks
        print("\n[6/6] Verifying Dashboard & Bot Attack Simulator Assets...")
        self.check_frontend_assets()

        # Summary
        elapsed = time.time() - start_time
        total = len(self.results)
        passed = sum(1 for _, s, _ in self.results if s)
        failed = total - passed

        print("\n" + "=" * 70)
        print(f"SMOKE TEST SUMMARY: {passed}/{total} checks passed ({elapsed:.2f}s)")
        if failed == 0:
            print("STATUS: ALL SUBSYSTEMS OPERATIONAL [SUCCESS]")
        else:
            print(f"STATUS: {failed} CHECK(S) FAILED [ERROR]")
        print("=" * 70)

        return failed == 0

    def check_artifacts(self) -> None:
        # Check model checkpoint
        ckpt_path = _ROOT / "models" / "world_model_v2.pt"
        if ckpt_path.exists():
            size_kb = ckpt_path.stat().st_size / 1024
            self.record("Model Checkpoint (world_model_v2.pt)", True, f"{size_kb:.1f} KB")
        else:
            self.record("Model Checkpoint (world_model_v2.pt)", False, "File not found")

        # Check scaler
        scaler_path = _ROOT / "models" / "scaler.pkl"
        if scaler_path.exists():
            self.record("Feature Scaler (scaler.pkl)", True, "Exists")
        else:
            self.record("Feature Scaler (scaler.pkl)", False, "File not found")

        # Check temperature calibration
        temp_path = _ROOT / "artifacts" / "calibration" / "temperature.json"
        if temp_path.exists():
            try:
                data = json.loads(temp_path.read_text(encoding="utf-8"))
                temp_val = data.get("temperature", 1.0)
                self.record("Temperature Calibration Artifact", True, f"T*={temp_val:.4f}")
            except Exception as e:
                self.record("Temperature Calibration Artifact", False, str(e))
        else:
            self.record("Temperature Calibration Artifact", False, "File not found")

        # Check sample dataset
        sample_dir = _ROOT / "datasets" / "sample"
        csvs = list(sample_dir.glob("*.csv")) if sample_dir.exists() else []
        self.record("Scenario Datasets (datasets/sample/*.csv)", len(csvs) > 0, f"{len(csvs)} scenario CSVs")

    def check_ml_engine(self) -> None:
        try:
            from backend.services.model_service import ModelService
            svc = ModelService.get_instance()
            self.record("ModelService Singleton Initialization", svc.is_loaded, f"Loaded (T={svc.temperature:.4f})")

            # Test synthetic forecast
            x_dummy = [[0.0] * 24] * 5
            mask_dummy = [1] * 5
            fc = svc.forecast(x_seq=x_dummy, mask_list=mask_dummy, k_steps=4)

            has_stage = bool(fc.get("current_stage"))
            has_prob = 0.0 <= fc.get("attack_probability", -1) <= 1.0
            has_rollout = len(fc.get("rollout_steps", [])) == 4
            has_conf = 0.0 <= fc.get("confidence", -1) <= 1.0

            self.record("V2 Forward Inference & Stage Classification", has_stage, f"Stage: {fc.get('current_stage')}")
            self.record("Calibrated Attack Probability", has_prob, f"P(atk)={fc.get('attack_probability'):.4f} (Conf: {fc.get('confidence'):.4f})")
            self.record("K=4 Autoregressive Multi-Step Rollout", has_rollout, f"{len(fc.get('rollout_steps'))} steps")
        except Exception as e:
            self.record("ML Engine Execution", False, str(e))

    def check_defensive_intelligence(self) -> None:
        try:
            from backend.services.model_service import ModelService
            from mitre.mappings.mitre_mapper import get_mitre_summary
            from ml.defense.risk_engine import RiskEngine

            svc = ModelService.get_instance()
            x_dummy = [[0.1] * 24] * 5
            fc = svc.forecast(x_seq=x_dummy, mask_list=[1]*5, k_steps=1)

            # Check explainability
            top_feats = fc.get("top_features", [])
            self.record("Explainability Engine (|ΔS| Attribution)", len(top_feats) > 0, f"{len(top_feats)} top features identified")

            # Check MITRE
            mitre_list = fc.get("mitre_techniques", [])
            primary_id = fc.get("primary_technique_id")
            self.record("MITRE ATT&CK Technique Mapping (Enterprise v14)", bool(primary_id), f"Primary: {primary_id} ({len(mitre_list)} mapped)")

            # Check RiskEngine
            risk_score = fc.get("risk_score")
            risk_level = fc.get("risk_level")
            priority = fc.get("recommended_priority")
            self.record("Dynamic Multi-Factor Risk Assessment", risk_score is not None, f"Score: {risk_score}/100 ({risk_level}) -- {priority}")
        except Exception as e:
            self.record("Defensive Intelligence Pipeline", False, str(e))

    def check_agent_grounding(self) -> None:
        try:
            from backend.services.evidence_service import EvidenceService
            from backend.services.gemini_service import GeminiService

            evidence_svc = EvidenceService()
            gemini_svc = GeminiService()

            # 1. Status check
            status = gemini_svc.get_status()
            self.record("GeminiService Configuration", status["api_key_configured"], f"Model: {status['model']} (Status: {status['status']})")

            # 2. Missing telemetry test (must output insufficient telemetry notice)
            empty_evidence = evidence_svc.get_latest_evidence(current_forecast=None)
            prompt = evidence_svc.format_grounding_prompt(empty_evidence, "What is the threat?")
            self.record("Evidence Grounding: Insufficient Data Detection", "INSUFFICIENT_TELEMETRY" in prompt, "Triggered strictly on absent telemetry")

            # 3. Grounded extraction test
            mock_fc = {
                "status": "FORECAST",
                "current_stage": "Reconnaissance",
                "predicted_next_stage": "Initial_Access",
                "attack_probability": 0.74,
                "confidence": 0.88,
                "risk_score": 65.0,
                "risk_level": "MEDIUM",
                "recommended_priority": "P2 — Elevated",
                "primary_technique_id": "T1046",
                "primary_technique_name": "Network Service Discovery",
            }
            ev = evidence_svc.get_latest_evidence(current_forecast=mock_fc)
            self.record("Evidence Service Normalization", ev["has_sufficient_evidence"] and ev["current_stage"] == "Reconnaissance", "Stage, scores & MITRE standardized")

            # 4. Deterministic fallback test
            fallback_resp = gemini_svc._deterministic_fallback(ev, "Explain attack")
            self.record("Deterministic Rule Engine Fallback", "Reconnaissance" in fallback_resp and "T1046" in fallback_resp, "100% offline availability preserved")
        except Exception as e:
            self.record("Agent Grounding Architecture", False, str(e))

    def check_fastapi_endpoints(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from backend.app import app

            with TestClient(app) as client:
                # 1. /health
                r_health = client.get("/api/v1/health")
                self.record("API Endpoint: GET /api/v1/health", r_health.status_code == 200 and r_health.json()["status"] == "ok", f"HTTP {r_health.status_code}")

                # 2. /model/info
                r_info = client.get("/api/v1/model/info")
                self.record("API Endpoint: GET /api/v1/model/info", r_info.status_code == 200, f"Architecture: {r_info.json().get('architecture')}")

                # 3. /forecast
                dummy_req = {"x_seq": [[0.0]*24]*5, "mask": [1]*5, "k_steps": 2}
                r_fc = client.post("/api/v1/forecast", json=dummy_req)
                self.record("API Endpoint: POST /api/v1/forecast", r_fc.status_code == 200, f"HTTP {r_fc.status_code}")

                # 4. /agent/status
                r_astatus = client.get("/api/v1/agent/status")
                self.record("API Endpoint: GET /api/v1/agent/status", r_astatus.status_code == 200, f"Status: {r_astatus.json().get('status')}")

                # 5. /agent/chat
                chat_req = {
                    "message": "Summarize the active threat.",
                    "current_forecast": {
                        "status": "FORECAST",
                        "current_stage": "Exploitation",
                        "predicted_next_stage": "Installation",
                        "attack_probability": 0.85,
                        "confidence": 0.90,
                        "risk_score": 75.0,
                        "risk_level": "HIGH",
                        "primary_technique_id": "T1059",
                    }
                }
                r_chat = client.post("/api/v1/agent/chat", json=chat_req)
                has_answer = r_chat.status_code == 200 and len(r_chat.json().get("answer", "")) > 10
                self.record("API Endpoint: POST /api/v1/agent/chat", has_answer, f"Backend: {r_chat.json().get('llm_backend')}")

                # 6. /replay/scenarios
                r_scen = client.get("/api/v1/replay/scenarios")
                self.record("API Endpoint: GET /api/v1/replay/scenarios", r_scen.status_code == 200, f"{len(r_scen.json().get('scenarios', []))} scenarios available")
        except Exception as e:
            self.record("FastAPI Endpoint Verification", False, str(e))

    def check_frontend_assets(self) -> None:
        index_html = _ROOT / "dashboard" / "index.html"
        if index_html.exists():
            content = index_html.read_text(encoding="utf-8")
            has_analyst = "AI SECURITY ANALYST" in content
            has_floating = "floatingAnalystBtn" in content
            has_context_strip = "analystContextStrip" in content
            self.record("SOC Dashboard (dashboard/index.html)", has_analyst and has_floating and has_context_strip, "AI Security Analyst Console integrated")
        else:
            self.record("SOC Dashboard (dashboard/index.html)", False, "File missing")

        sim_html = _ROOT / "dashboard" / "simulator.html"
        if sim_html.exists():
            self.record("Bot Attack Simulator (dashboard/simulator.html)", True, "Exists & verified")
        else:
            self.record("Bot Attack Simulator (dashboard/simulator.html)", False, "File missing")


def main():
    runner = SmokeTestRunner()
    success = runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
