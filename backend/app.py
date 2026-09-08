"""
CyberSentinel AI — FastAPI Application (Phase 10).

Start with:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

Or via the convenience launcher:
    python scripts/start_server.py
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure workspace root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.api.endpoints import router
from backend.api.stream_endpoints import stream_router
from backend.services.model_service import ModelService
from backend.services.replay_service import ReplayService
from backend.services.live_ingest_service import LiveIngestService
from backend.agents.defensive_agent import CyberSentinelDefensiveAgent
from backend.middleware.security import SecurityMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("CyberSentinel AI Command Center — Starting Up")
    logger.info("=" * 60)

    # Initialize singletons
    app.state.model_service = ModelService.get_instance()
    app.state.replay_service = ReplayService()
    app.state.live_ingest_service = LiveIngestService()
    app.state.agent = CyberSentinelDefensiveAgent()
    app.state.agent._ensure_ollama_checked()

    if app.state.model_service.is_loaded:
        logger.info("✓ CyberWorldModelV2 loaded (T=%.4f)", app.state.model_service.temperature)
    else:
        logger.warning("✗ Model NOT loaded — /forecast endpoints will return 503")

    logger.info("✓ ReplayService ready")
    logger.info("✓ LiveIngestService ready")
    logger.info("✓ DefensiveAgent ready")
    logger.info("API docs available at: http://localhost:8000/docs")
    logger.info("Dashboard available at: http://localhost:8000/ui/index.html")

    yield

    logger.info("CyberSentinel AI Command Center — Shutting Down")
    if hasattr(app.state, "live_ingest_service"):
        await app.state.live_ingest_service.stop_all()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CyberSentinel AI Command Center",
    description=(
        "SOC-grade temporal cyber threat forecasting system. "
        "Powered by CyberWorldModelV2 (Phase 8C empirical benchmark: "
        "Next-stage Top-1=97.73%, Transition Acc=83.33%, Brier=0.0452)."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Default state initialization so client access works even before startup
app.state.model_service = ModelService.get_instance()
app.state.replay_service = ReplayService()
app.state.live_ingest_service = LiveIngestService()
app.state.agent = CyberSentinelDefensiveAgent()

# CORS — allow dashboard (same host or file://) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityMiddleware)

# Mount dashboard as static files at /ui
_DASHBOARD = _ROOT / "dashboard"
if _DASHBOARD.exists():
    app.mount("/ui", StaticFiles(directory=str(_DASHBOARD), html=True), name="dashboard")

# Include API routes
app.include_router(router, prefix="/api/v1")
app.include_router(stream_router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/index.html")
