"""
app_people_counting.py
======================
FastAPI WebSocket server for real-time people counting + occupancy on top of
the existing CrossCamReid (ReID) pipeline.

Start:
    uvicorn app_people_counting:app --host 0.0.0.0 --port 8002

WebSocket endpoint:
    ws://<host>:8002/ws/people_counting/{client_id}?token=<JWT>

Set ``PC_JWT_SECRET`` (env) to enforce HS256 JWTs. If unset, any non-empty
token is accepted (dev mode; a warning is logged once).

Inbound messages: see crosscamreid/websocket/people_counting_handler.py
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crosscamreid.config import load_config
from crosscamreid.counting.org_registry import OrgRegistry
from crosscamreid.websocket.people_counting_handler import (
    people_counting_websocket_handler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app_people_counting")

# ── load config ───────────────────────────────────────────────────────────────

CONFIG_PATH = THIS_DIR / "config" / "config.yaml"
config = load_config(str(CONFIG_PATH))

# Per-org registry: shared SIDStore + ReID backend per org_id.
org_registry = OrgRegistry(config, dispose_on_zero=False)

# Active sessions keyed by client_id.
sessions: dict = {}

# ── FastAPI app ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("People-Counting API starting up.")
    logger.info("Config           : %s", CONFIG_PATH)
    logger.info("ReID backend     : %s", config.runtime.reid_backend)
    logger.info("Qdrant mode      : %s", config.database.qdrant.mode)
    logger.info("Postgres enabled : %s", config.database.postgres.enabled)
    yield
    # Best-effort shutdown.
    for client_id, session in list(sessions.items()):
        if session is None:
            continue
        try:
            session.close()
        except Exception:
            logger.exception("shutdown: close failed for %s", client_id)
    sessions.clear()
    logger.info("People-Counting API stopped.")


app = FastAPI(
    title="CrossCamReid People-Counting API",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0",
        "active_sessions": len(sessions),
        "reid_backend": config.runtime.reid_backend,
        "qdrant_mode": config.database.qdrant.mode,
    }


@app.websocket("/ws/people_counting/{client_id}")
async def ws_people_counting(
    websocket: WebSocket,
    client_id: str,
    token: str | None = Query(default=None),
):
    await people_counting_websocket_handler(
        websocket,
        client_id=client_id,
        token=token,
        config=config,
        org_registry=org_registry,
        sessions=sessions,
    )
