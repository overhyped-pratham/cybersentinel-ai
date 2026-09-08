"""
Phase 14: Security Hardening Test Suite.

Verifies:
1. Sliding-window rate limiting per client IP (120 req/min).
2. Rate limit exhaustion returns HTTP 429 Too Many Requests with Retry-After header.
3. WebSocket connection limiter caps concurrent connections (max 50).
4. API key authentication enforcement via X-API-Key (HTTP 401 on unauthorized).
5. Request payload size bounding (HTTP 413 on payloads > 10MB).
6. Safe HTTP security headers (nosniff, DENY, 1; mode=block).
7. Stream health endpoint reporting accurate security and operational metrics.
"""

import os
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.middleware.security import SecurityManager, security_manager


class TestSecurityManagerUnit:
    """Direct unit tests for SecurityManager class."""

    def test_rate_limiter_allows_and_blocks(self):
        sm = SecurityManager(rate_limit_per_minute=5, max_ws_connections=10)
        ip = "192.168.1.50"

        # First 5 calls allowed
        for _ in range(5):
            assert sm.check_rate_limit(ip) is True

        # 6th call blocked
        assert sm.check_rate_limit(ip) is False

    def test_rate_limiter_ip_isolation(self):
        sm = SecurityManager(rate_limit_per_minute=3, max_ws_connections=10)
        ip1 = "10.0.0.1"
        ip2 = "10.0.0.2"

        # Exhaust ip1
        for _ in range(3):
            assert sm.check_rate_limit(ip1) is True
        assert sm.check_rate_limit(ip1) is False

        # ip2 must still be unaffected
        assert sm.check_rate_limit(ip2) is True

    def test_websocket_connection_limiter(self):
        sm = SecurityManager(rate_limit_per_minute=100, max_ws_connections=3)

        assert sm.active_ws_count == 0
        assert sm.acquire_ws_slot() is True
        assert sm.acquire_ws_slot() is True
        assert sm.acquire_ws_slot() is True
        assert sm.active_ws_count == 3

        # 4th connection rejected
        assert sm.acquire_ws_slot() is False
        assert sm.active_ws_count == 3

        # Release a slot
        sm.release_ws_slot()
        assert sm.active_ws_count == 2
        assert sm.acquire_ws_slot() is True
        assert sm.active_ws_count == 3

    def test_api_key_verification(self):
        sm = SecurityManager()
        # When auth not enabled (default empty key)
        sm._api_key = ""
        assert sm.is_auth_enabled is False
        assert sm.verify_api_key(None) is True
        assert sm.verify_api_key("anything") is True

        # When auth enabled
        sm._api_key = "test-secret-token-777"
        assert sm.is_auth_enabled is True
        assert sm.verify_api_key(None) is False
        assert sm.verify_api_key("wrong-token") is False
        assert sm.verify_api_key("test-secret-token-777") is True


class TestSecurityMiddlewareIntegration:
    """HTTP integration tests for SecurityMiddleware with FastAPI app."""

    @pytest.fixture
    def client(self):
        # Reset security_manager state for predictable tests
        security_manager._request_history.clear()
        security_manager._active_ws_count = 0
        original_key = security_manager._api_key
        original_limit = security_manager.rate_limit_per_minute
        yield TestClient(app)
        security_manager._api_key = original_key
        security_manager.rate_limit_per_minute = original_limit
        security_manager._request_history.clear()

    def test_security_headers_present_on_responses(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert "1; mode=block" in resp.headers.get("X-XSS-Protection", "")

    def test_rate_limit_http_429_response(self, client):
        security_manager.rate_limit_per_minute = 10
        security_manager._request_history.clear()

        # Send 10 requests -> 200 OK
        for _ in range(10):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

        # 11th request -> 429 Too Many Requests
        resp = client.get("/api/v1/health")
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "60"
        data = resp.json()
        assert "Rate limit exceeded" in data.get("detail", "")

    def test_payload_size_limit_http_413(self, client):
        # Send header claiming body > 10MB
        oversized_bytes = 11 * 1024 * 1024  # 11 MB
        resp = client.post(
            "/api/v1/forecast",
            headers={"Content-Length": str(oversized_bytes)},
            json={"x_seq": []},
        )
        assert resp.status_code == 413
        data = resp.json()
        assert "Payload Too Large" in data.get("detail", "")

    def test_api_key_enforcement_when_enabled(self, client):
        security_manager._api_key = "production-guard-key-999"

        # Missing key -> 401
        resp = client.get("/api/v1/health")
        assert resp.status_code == 401
        assert "Invalid or missing X-API-Key" in resp.json().get("detail", "")

        # Invalid key -> 401
        resp = client.get("/api/v1/health", headers={"X-API-Key": "bad-key"})
        assert resp.status_code == 401

        # Valid key -> 200
        resp = client.get("/api/v1/health", headers={"X-API-Key": "production-guard-key-999"})
        assert resp.status_code == 200

    def test_stream_health_endpoint(self, client):
        resp = client.get("/api/v1/stream/health")
        assert resp.status_code == 200
        data = resp.json()

        assert "status" in data
        assert "mode" in data
        assert data["mode"] in ("LIVE", "REPLAY")
        assert "model_version" in data
        assert "model_available" in data
        assert "scaler_available" in data
        assert "active_ws_connections" in data
        assert "max_ws_connections" in data
        assert data["max_ws_connections"] > 0
        assert "dropped_malformed_count" in data
        assert "queue_backpressure_status" in data
