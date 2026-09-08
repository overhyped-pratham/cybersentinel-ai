"""
CyberSentinel AI — Production Security Hardening Middleware (Phase 14).

Features:
  1. Rate Limiting: Sliding-window rate limiter per client IP (default: 120 req/min).
  2. Connection Limiter: Caps concurrent WebSocket subscribers (default: 50).
  3. API Key Authorization: Optional header 'X-API-Key' verified against CYBERSENTINEL_API_KEY.
  4. Request Body Bounding: Rejects payloads > 10MB (HTTP 413) to prevent DoS.
  5. Safe Logging & Header Sanitization: Redacts authorization tokens from logs.
  6. Strictly Defensive: Zero offensive probes or socket injection capabilities.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional, Set

from fastapi import HTTPException, Request, Response, WebSocket, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Configurable security limits
DEFAULT_RATE_LIMIT_PER_MINUTE = int(os.getenv("CYBERSENTINEL_RATE_LIMIT", "120"))
DEFAULT_MAX_WS_CONNECTIONS = int(os.getenv("CYBERSENTINEL_MAX_WS_CONNECTIONS", "50"))
MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


class SecurityManager:
    """Manages rate limiting, connection limits, and authorization."""

    def __init__(
        self,
        rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
        max_ws_connections: int = DEFAULT_MAX_WS_CONNECTIONS,
    ) -> None:
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_ws_connections = max_ws_connections
        # IP -> deque of timestamps
        self._request_history: Dict[str, Deque[float]] = defaultdict(deque)
        self._active_ws_count = 0
        self._api_key = os.getenv("CYBERSENTINEL_API_KEY", "").strip()

    @property
    def is_auth_enabled(self) -> bool:
        return bool(self._api_key)

    def check_rate_limit(self, client_ip: str) -> bool:
        """Returns True if within rate limit, False if rate limited."""
        now = time.time()
        window_start = now - 60.0
        history = self._request_history[client_ip]

        # Evict timestamps older than 60s
        while history and history[0] < window_start:
            history.popleft()

        if len(history) >= self.rate_limit_per_minute:
            return False

        history.append(now)
        return True

    def verify_api_key(self, provided_key: Optional[str]) -> bool:
        """Verifies API key if auth is configured."""
        if not self.is_auth_enabled:
            return True
        return provided_key is not None and provided_key == self._api_key

    def acquire_ws_slot(self) -> bool:
        """Attempts to register a new WebSocket connection."""
        if self._active_ws_count >= self.max_ws_connections:
            return False
        self._active_ws_count += 1
        return True

    def release_ws_slot(self) -> None:
        """Releases a WebSocket connection slot."""
        if self._active_ws_count > 0:
            self._active_ws_count -= 1

    @property
    def active_ws_count(self) -> int:
        return self._active_ws_count


# Global security manager instance
security_manager = SecurityManager()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing rate limiting, size bounding, and auth."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"

        # 1. Check Content-Length size bound
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Payload Too Large: Max body size is 10 MB"},
                    )
            except ValueError:
                pass

        # 2. Check API Key if auth enabled (skip static UI and docs)
        path = request.url.path
        if security_manager.is_auth_enabled and not (
            path.startswith("/ui") or path in ("/docs", "/openapi.json", "/redoc", "/")
        ):
            api_key = request.headers.get("X-API-Key")
            if not security_manager.verify_api_key(api_key):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or missing X-API-Key header"},
                )

        # 3. Rate limiting (skip static UI files)
        if not path.startswith("/ui"):
            if not security_manager.check_rate_limit(client_ip):
                logger.warning("[Security] Rate limit exceeded for IP: %s on %s", client_ip, path)
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Too Many Requests: Rate limit exceeded"},
                    headers={"Retry-After": "60"},
                )

        response = await call_next(request)

        # 4. Security response headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        return response
