import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.app import app

sys.stdout.reconfigure(encoding="utf-8")

with TestClient(app) as client:
    r_status = client.get("/api/demo/status")
    assert r_status.status_code == 200, f"status error: {r_status.text}"
    print("GET /api/demo/status ->", r_status.json())

    r_metrics = client.get("/api/demo/metrics")
    assert r_metrics.status_code == 200, f"metrics error: {r_metrics.text}"
    print("GET /api/demo/metrics -> accuracy:", r_metrics.json().get("classifier_accuracy"))

    for step in ["benign", "known_attack", "disagreement", "held_out"]:
        r_step = client.post(f"/api/demo/step/{step}")
        assert r_step.status_code == 200, f"step {step} error: {r_step.text}"
        data = r_step.json()
        det = data["detection_result"]
        print(f"POST /api/demo/step/{step} -> stage: {det['current_stage']}, risk: {det['risk_score']:.1f}, verdict: {det['threat_classification']}")

    r_reset = client.post("/api/demo/reset")
    assert r_reset.status_code == 200
    print("POST /api/demo/reset -> OK")
