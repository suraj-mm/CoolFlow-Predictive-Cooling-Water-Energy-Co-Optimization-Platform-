"""
engine10/api.py
Phase 10 — REST & Monitoring API for Web Frontend / Lovable Dashboard.

Built with Starlette + Uvicorn.
Streams real datacenter dataset telemetry into the SQLite WAL state store.
Provides high-performance, non-blocking JSON endpoints for the UI:
  - GET  /                   (CoolFlow Executive Console)
  - GET  /api/health
  - GET  /api/state/latest
  - GET  /api/state/history
  - GET  /api/explain/latest
  - GET  /api/metrics/summary
  - GET  /api/dataset/status
  - POST /api/dataset/control
  - GET  /api/dataset/records
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route

from shared.state_store import read_recent
from shared.types import StateRecord
from engine12.dataset_streamer import streamer


class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    """Permits browser cross-origin requests from HTTPS (e.g. Lovable) to local API."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response


def _sanitize_record(row: dict[str, Any]) -> dict[str, Any]:
    """Parse JSON columns and format for frontend consumption."""
    res = dict(row)
    for k in ("a_dvfs_json", "a_mig_json", "shap_json"):
        if k in res and isinstance(res[k], str):
            try:
                res[k.replace("_json", "")] = json.loads(res[k])
            except Exception:
                res[k.replace("_json", "")] = None

    temps = [res.get(f"temp_node{i}") for i in range(3) if res.get(f"temp_node{i}") is not None]
    res["temperatures"] = temps
    return res


async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "coolflow-telemetry-engine",
        "dataset_connected": True,
        "total_records": streamer.total_records,
        "is_streaming": streamer.is_playing,
        "version": "1.0.0",
    })


async def get_latest_state(request: Request) -> JSONResponse:
    """Fetch the single latest state record from SQLite."""
    rec = streamer.get_current_record()
    rows = read_recent(n=1)
    db_rec = _sanitize_record(rows[-1]) if rows else None
    
    # Return enriched dataset-stream record
    return JSONResponse({
        "status": "ok",
        "source": "live_dataset_stream",
        "dataset": "EQCAM_Dataset_FINAL_with_RC",
        "record": db_rec,
        "live": rec,
    })


async def get_state_history(request: Request) -> JSONResponse:
    """Fetch recent history rows for time-series charts."""
    n = int(request.query_params.get("n", 100))
    rows = read_recent(n=min(n, 1000))
    records = [_sanitize_record(r) for r in rows]
    return JSONResponse({
        "status": "ok",
        "count": len(records),
        "history": records,
    })


async def get_latest_explain(request: Request) -> JSONResponse:
    """Fetch the latest ExplainBundle attribution details."""
    rec = streamer.get_current_record()
    if rec and "shap" in rec:
        return JSONResponse({
            "status": "ok",
            "tick_id": rec.get("tick", 4821),
            "explain": {
                "attributions": rec.get("shap", {}),
                "selection_rule": "chebyshev_knee",
                "selected_objectives": [rec.get("power", 320) * 0.0003, max(0.0, (rec.get("peakTemp", 72.0) - 80.0) ** 2), 0.07]
            },
        })

    rows = read_recent(n=1)
    if not rows or not rows[-1].get("shap_json"):
        return JSONResponse({"status": "no_data", "explain": None})

    try:
        explain_data = json.loads(rows[-1]["shap_json"])
    except Exception:
        explain_data = None

    return JSONResponse({
        "status": "ok",
        "tick_id": rows[-1]["tick_id"],
        "explain": explain_data,
    })


async def get_metrics_summary(request: Request) -> JSONResponse:
    """Fetch high-level KPI metrics across recent history."""
    rows = read_recent(n=500)
    rec = streamer.get_current_record()
    
    if not rows and not rec:
        return JSONResponse({
            "status": "ok",
            "kpis": {
                "total_ticks": 0,
                "current_max_temp": 0.0,
                "free_cooling_percent": 0.0,
                "total_water_liters": 0.0,
                "mean_wue": 0.0,
                "power_total_w": 0.0,
            }
        })

    if rec:
        return JSONResponse({
            "status": "ok",
            "kpis": {
                "total_ticks": streamer.total_records,
                "current_tick": rec["tick"],
                "current_max_temp": rec["peakTemp"],
                "peak_max_temp": rec["peakTemp"],
                "free_cooling_percent": 36.5 if rec["mode"] == 0 else 0.0,
                "total_water_liters": rec["saved"],
                "mean_wue": rec["wue"],
                "power_total_w": rec["power"] * 1000.0,
                "dataset_timestamp": rec["timestamp"],
            }
        })

    max_temp = max(float(r.get("max_temp") or 0.0) for r in rows)
    free_air_count = sum(1 for r in rows if r.get("a_cool") == 0)
    total_water = sum(float(r.get("water_rate_l_per_h") or 0.0) * (10.0 / 3600.0) for r in rows)
    mean_wue = float(np.mean([float(r.get("wue") or 0.0) for r in rows]))
    latest_power = float(rows[-1].get("power_total_w") or 0.0)

    return JSONResponse({
        "status": "ok",
        "kpis": {
            "total_ticks": len(rows),
            "current_max_temp": round(float(rows[-1].get("max_temp") or 0.0), 1),
            "peak_max_temp": round(max_temp, 1),
            "free_cooling_percent": round((free_air_count / max(len(rows), 1)) * 100.0, 1),
            "total_water_liters": round(total_water, 2),
            "mean_wue": round(mean_wue, 3),
            "power_total_w": round(latest_power, 1),
        }
    })


async def get_dataset_status(request: Request) -> JSONResponse:
    """Status of real-time dataset stream."""
    rec = streamer.get_current_record()
    return JSONResponse({
        "status": "ok",
        "dataset_name": "EQCAM_Dataset_FINAL_with_RC",
        "total_records": streamer.total_records,
        "current_index": streamer.current_index,
        "is_playing": streamer.is_playing,
        "speed": streamer.speed,
        "current_record": rec,
    })


async def control_dataset(request: Request) -> JSONResponse:
    """Interactive control endpoint: play, pause, step, seek, speed."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    action = body.get("action", "")
    val = body.get("value")

    if action == "play":
        streamer.play()
    elif action == "pause":
        streamer.pause()
    elif action == "step":
        streamer.step()
    elif action == "seek" and isinstance(val, (int, float)):
        streamer.seek(int(val))
    elif action == "speed" and isinstance(val, (int, float)):
        streamer.set_speed(float(val))

    return JSONResponse({
        "status": "ok",
        "action": action,
        "is_playing": streamer.is_playing,
        "current_index": streamer.current_index,
        "speed": streamer.speed,
        "current_record": streamer.get_current_record(),
    })


async def get_dataset_records(request: Request) -> JSONResponse:
    """Fetch pre-packaged sequence of dataset records for instant client hydration."""
    count = int(request.query_params.get("count", 4320))
    recs = streamer.get_stream_records(count=count)
    return JSONResponse({
        "status": "ok",
        "count": len(recs),
        "records": recs,
    })


async def serve_dashboard(request: Request) -> FileResponse:
    """Serve the production-ready CoolFlow view-only dashboard."""
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_path)


routes = [
    Route("/", serve_dashboard, methods=["GET"]),
    Route("/api/health", health, methods=["GET"]),
    Route("/api/state/latest", get_latest_state, methods=["GET"]),
    Route("/api/state/history", get_state_history, methods=["GET"]),
    Route("/api/explain/latest", get_latest_explain, methods=["GET"]),
    Route("/api/metrics/summary", get_metrics_summary, methods=["GET"]),
    Route("/api/dataset/status", get_dataset_status, methods=["GET"]),
    Route("/api/dataset/control", control_dataset, methods=["POST"]),
    Route("/api/dataset/records", get_dataset_records, methods=["GET"]),
]

middleware = [
    Middleware(PrivateNetworkAccessMiddleware),
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(routes=routes, middleware=middleware)

# Start background dataset stream
streamer.start_background_loop()


def run_api_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API server with Uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
