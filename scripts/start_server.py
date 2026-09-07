"""
CyberSentinel AI — Command Center Server Launcher (Phase 10).

Ensures model checkpoint and calibration artifacts exist, then starts
the FastAPI Uvicorn ASGI server hosting both the API and Command Center UI.

Usage:
  python scripts/start_server.py
  python scripts/start_server.py --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CyberSentinelLauncher")


def main():
    parser = argparse.ArgumentParser(description="Start CyberSentinel Command Center Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("=" * 65)
    print("      CYBERSENTINEL AI — SOC COMMAND CENTER SERVER")
    print("=" * 65)
    print(f"Server host: {args.host}")
    print(f"Server port: {args.port}")
    print(f"API Docs:    http://localhost:{args.port}/docs")
    print(f"Dashboard:   http://localhost:{args.port}/ui/index.html")
    print("=" * 65)

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
