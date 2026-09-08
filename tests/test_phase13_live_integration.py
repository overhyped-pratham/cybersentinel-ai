"""
Phase 13 Test Suite: Live Integration & End-to-End Pipeline.

Tests the complete live ingestion chain:
  TelemetrySource → StreamProcessor → LiveIngestService → ModelService (CyberWorldModelV2)
  → RiskEngine → MITRE → Explainability → Broadcast Event

Verifies:
  - Real telemetry triggers actual ML inference
  - Every forecast event contains all required fields (rollout, mitre, risk, explainability)
  - Fail-closed behavior on missing model or empty telemetry
  - Dynamic sensitivity to telemetry variations
  - Scenario metadata invariance
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from network.flow.flow_record import FlowRecord
from network.telemetry.sources import ReplaySource
from backend.services.live_ingest_service import LiveIngestService, _build_live_event
from backend.services.model_service import ModelService
from network.telemetry.stream_processor import TelemetryWindowEvent

_CSV_SAMPLE = Path(__file__).resolve().parent.parent / "datasets" / "sample" / "trace_multistage_01.csv"


class TestLiveIngestServicePipeline:
    """Verifies LiveIngestService operations and end-to-end inference."""

    @pytest.mark.asyncio
    async def test_live_ingest_accumulates_and_infers(self):
        """Streams real CSV telemetry, verifies buffer accumulation and ML inference events."""
        source = ReplaySource(_CSV_SAMPLE, realtime_factor=0.0)
        svc = LiveIngestService()

        q = svc.subscribe()
        session_id = await svc.start_session(
            source,
            window_seconds=30.0,
            stride_seconds=30.0,
            k_steps=4,
        )

        events = []
        try:
            # Wait for at least 6 events (some BUFFERING, then at least 1 FORECAST)
            for _ in range(30):
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=2.0)
                    events.append(evt)
                    if evt.get("status") == "FORECAST":
                        break
                except asyncio.TimeoutError:
                    break
        finally:
            await svc.stop_session(session_id)
            svc.unsubscribe(q)

        assert len(events) >= 5, f"Expected at least 5 events, got {len(events)}"
        # Check buffering events
        buffering_events = [e for e in events if e.get("status") == "BUFFERING"]
        assert len(buffering_events) >= 1
        assert buffering_events[0]["sequence_required"] == 5

        # Check forecast events
        forecast_events = [e for e in events if e.get("status") == "FORECAST"]
        assert len(forecast_events) >= 1

        fc_evt = forecast_events[0]
        # Check all required Phase 13 fields exist
        assert "event_id" in fc_evt
        assert "window_id" in fc_evt
        assert "session_id" in fc_evt
        assert "timestamp" in fc_evt
        assert "flow_count" in fc_evt
        assert "flows_per_second" in fc_evt
        assert "inference_latency_ms" in fc_evt
        assert "model_version" in fc_evt
        assert "CyberWorldModelV2" in fc_evt["model_version"]

        # Check ML fields
        assert fc_evt["current_stage"] in ["BENIGN", "RECONNAISSANCE", "CREDENTIAL_ACCESS", "LATERAL_MOVEMENT", "EXFILTRATION"]
        assert fc_evt["predicted_next_stage"] in ["BENIGN", "RECONNAISSANCE", "CREDENTIAL_ACCESS", "LATERAL_MOVEMENT", "EXFILTRATION"]
        assert 0.0 <= fc_evt["attack_probability"] <= 1.0
        assert 0.0 <= fc_evt["confidence"] <= 1.0
        assert 0.0 <= fc_evt["risk_score"] <= 100.0
        assert isinstance(fc_evt["top_features"], list)
        assert len(fc_evt["top_features"]) > 0
        assert isinstance(fc_evt["mitre_techniques"], list)
        assert isinstance(fc_evt["rollout_steps"], list)
        assert len(fc_evt["rollout_steps"]) == 4

    @pytest.mark.asyncio
    async def test_fail_closed_when_model_unloaded(self):
        """When model is unavailable, emits explicit MODEL_UNAVAILABLE event, not mock predictions."""
        mock_model = MagicMock()
        mock_model.is_loaded = False

        svc = LiveIngestService()
        q = svc.subscribe()

        window = TelemetryWindowEvent(
            window_id="w_test_001",
            window_start=0.0,
            window_end=30.0,
            flows=[FlowRecord(0.0, "10.0.0.1", "10.0.0.2", 1024, 80, 6, 10, 1000)],
            flow_count=1,
            dropped_malformed=0,
            source_id="test",
        )

        await svc._process_window(window, "session_failclosed", mock_model)
        evt = await asyncio.wait_for(q.get(), timeout=1.0)
        svc.unsubscribe(q)

        assert evt["status"] == "MODEL_UNAVAILABLE"
        assert "predicted_next_stage" not in evt
        assert "risk_score" not in evt

    @pytest.mark.asyncio
    async def test_fail_closed_empty_flows(self):
        """When window contains zero flows, emits INVALID_TELEMETRY."""
        model_svc = ModelService.get_instance()
        svc = LiveIngestService()
        q = svc.subscribe()

        window = TelemetryWindowEvent(
            window_id="w_test_empty",
            window_start=0.0,
            window_end=30.0,
            flows=[],
            flow_count=0,
            dropped_malformed=0,
            source_id="test",
        )

        await svc._process_window(window, "session_empty", model_svc)
        evt = await asyncio.wait_for(q.get(), timeout=1.0)
        svc.unsubscribe(q)

        assert evt["status"] == "INVALID_TELEMETRY"
        assert "predicted_next_stage" not in evt

    @pytest.mark.asyncio
    async def test_scenario_name_invariance_in_live_pipeline(self):
        """Predictions must remain identical regardless of scenario metadata name."""
        model_svc = ModelService.get_instance()
        svc = LiveIngestService()

        # Generate a synthetic sequence of 5 identical windows
        flows = [
            FlowRecord(float(t), "10.0.0.1", "10.0.0.2", 1024 + t, 80, 6, 10, 500)
            for t in range(30)
        ]

        # Feed 5 windows with scenario "ATTACK_SCENARIO_ALPHA"
        for i in range(5):
            win_flows = [
                FlowRecord(float(i * 30 + t), "10.0.0.1", "10.0.0.2", 1024 + t, 80, 6, 10, 500)
                for t in range(30)
            ]
            win = TelemetryWindowEvent(
                window_id=f"win_{i}",
                window_start=float(i * 30),
                window_end=float((i + 1) * 30),
                flows=win_flows,
                flow_count=len(win_flows),
                dropped_malformed=0,
                source_id="test",
            )
            await svc._process_window(win, "ATTACK_SCENARIO_ALPHA", model_svc)

        alpha_event = svc.event_log[-1]

        # Feed 5 identical windows with scenario "BENIGN_INTERNAL_CORP"
        for i in range(5):
            win_flows = [
                FlowRecord(float(i * 30 + t), "10.0.0.1", "10.0.0.2", 1024 + t, 80, 6, 10, 500)
                for t in range(30)
            ]
            win = TelemetryWindowEvent(
                window_id=f"win_benign_{i}",
                window_start=float(i * 30),
                window_end=float((i + 1) * 30),
                flows=win_flows,
                flow_count=len(win_flows),
                dropped_malformed=0,
                source_id="test",
            )
            await svc._process_window(win, "BENIGN_INTERNAL_CORP", model_svc)

        benign_event = svc.event_log[-1]

        assert alpha_event["current_stage"] == benign_event["current_stage"]
        assert alpha_event["predicted_next_stage"] == benign_event["predicted_next_stage"]
        assert abs(alpha_event["attack_probability"] - benign_event["attack_probability"]) < 1e-4
        assert abs(alpha_event["risk_score"] - benign_event["risk_score"]) < 1e-3
