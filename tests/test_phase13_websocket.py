"""
Phase 13 Test Suite: WebSocket and SSE Streaming API.

Tests WebSocket connection, bidirectional ping/pong, event broadcast,
reconnection cleanup, SSE endpoint, and session status/history APIs.
"""

import json
import pytest
from starlette.testclient import TestClient

from backend.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestWebSocketStreaming:
    """Verifies WebSocket endpoint protocol and behavior."""

    def test_ws_connect_greeting(self, client):
        with client.websocket_connect("/api/v1/stream/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert "timestamp" in msg
            assert "CyberSentinel" in msg.get("message", "")

    def test_ws_ping_pong(self, client):
        with client.websocket_connect("/api/v1/stream/ws") as ws:
            # First is greeting
            ws.receive_json()
            # Send ping
            ws.send_json({"type": "ping"})
            pong = ws.receive_json()
            assert pong["type"] == "pong"
            assert "timestamp" in pong

    def test_ws_disconnect_cleans_up_subscriber(self, client):
        live_svc = app.state.live_ingest_service
        initial_count = len(live_svc._subscribers)

        with client.websocket_connect("/api/v1/stream/ws") as ws:
            assert len(live_svc._subscribers) == initial_count + 1

        # After exiting context manager, subscriber count must return to baseline
        assert len(live_svc._subscribers) == initial_count

    def test_ws_broadcast_event_received(self, client):
        live_svc = app.state.live_ingest_service
        with client.websocket_connect("/api/v1/stream/ws") as ws:
            # greeting
            ws.receive_json()

            # Broadcast a synthetic event via LiveIngestService
            synthetic_event = {
                "status": "FORECAST",
                "current_stage": "RECONNAISSANCE",
                "predicted_next_stage": "CREDENTIAL_ACCESS",
                "attack_probability": 0.95,
                "risk_score": 75.0,
            }
            # Broadcast puts into all subscribers
            for q in live_svc._subscribers:
                q.put_nowait(synthetic_event)

            received = ws.receive_json()
            assert received["status"] == "FORECAST"
            assert received["predicted_next_stage"] == "CREDENTIAL_ACCESS"
            assert received["risk_score"] == 75.0


class TestStreamingRestEndpoints:
    """Verifies REST endpoints associated with streaming."""

    def test_stream_status_endpoint(self, client):
        resp = client.get("/api/v1/stream/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "stats" in data
        assert "subscriber_count" in data
        assert "timestamp" in data

    def test_stream_history_endpoint(self, client):
        resp = client.get("/api/v1/stream/history?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert isinstance(data["events"], list)
        assert "count" in data

    @pytest.mark.asyncio
    async def test_sse_endpoint_initial_connect(self):
        """Verifies SSE endpoint constructs proper text/event-stream frames."""
        from backend.api.stream_endpoints import sse_stream
        from unittest.mock import MagicMock

        req = MagicMock()
        req.app.state.live_ingest_service = app.state.live_ingest_service
        resp = await sse_stream(req)
        assert resp.media_type == "text/event-stream"
        gen = resp.body_iterator
        first_chunk = await anext(gen)
        assert "event: connected" in first_chunk
        assert "data:" in first_chunk
