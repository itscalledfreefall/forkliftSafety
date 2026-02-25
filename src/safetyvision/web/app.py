"""SafetyVision Web UI – FastAPI backend."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from safetyvision.config import SafetyVisionConfig, load_config, validate, ConfigError

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

app = FastAPI(title="SafetyVision UI", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
CONFIG_PATH: str = os.environ.get("SAFETYVISION_CONFIG", "config/safetyvision.yaml")
BACKUP_SUFFIX = ".last-good"
SESSION_TOKENS: dict[str, float] = {}  # token -> expiry timestamp
SESSION_TTL = 3600 * 8  # 8 hours

# Default admin credentials (override via env)
ADMIN_USER = os.environ.get("SAFETYVISION_UI_USER", "admin")
ADMIN_PASS_HASH = hashlib.sha256(
    os.environ.get("SAFETYVISION_UI_PASS", "safetyvision").encode()
).hexdigest()

# Camera handle for live stream
_cap: Optional[cv2.VideoCapture] = None
_cap_lock = asyncio.Lock()

# Rate limiter for apply
_last_apply_ts: float = 0.0
APPLY_COOLDOWN = 5.0  # seconds


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


def _check_session(request: Request) -> str:
    token = request.cookies.get("sv_session")
    if not token or token not in SESSION_TOKENS:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if time.time() > SESSION_TOKENS[token]:
        SESSION_TOKENS.pop(token, None)
        raise HTTPException(status_code=401, detail="Session expired")
    return token


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.post("/api/auth/login")
async def login(body: LoginRequest):
    pw_hash = hashlib.sha256(body.password.encode()).hexdigest()
    if body.username != ADMIN_USER or pw_hash != ADMIN_PASS_HASH:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = secrets.token_urlsafe(32)
    SESSION_TOKENS[token] = time.time() + SESSION_TTL
    resp = JSONResponse({"ok": True})
    resp.set_cookie("sv_session", token, httponly=True, max_age=SESSION_TTL, samesite="strict")
    return resp


@app.post("/api/auth/logout")
async def logout(request: Request):
    token = request.cookies.get("sv_session")
    SESSION_TOKENS.pop(token, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("sv_session")
    return resp


@app.get("/api/auth/check")
async def auth_check(request: Request):
    token = request.cookies.get("sv_session")
    ok = token in SESSION_TOKENS and time.time() < SESSION_TOKENS.get(token, 0)
    return {"authenticated": ok}


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------
class ZoneConfig(BaseModel):
    yellow_start_y: float
    red_start_y: float


class AlertTimingConfig(BaseModel):
    repeat_interval_sec: float
    min_clear_sec: float
    min_alert_confidence: float


@app.get("/api/config")
async def get_config(_token: str = Depends(_check_session)):
    raw = _load_raw_config()
    return raw


@app.post("/api/config/zones")
async def set_zones(body: ZoneConfig, _token: str = Depends(_check_session)):
    """Validate and save zone cut lines."""
    try:
        cfg = _build_config_with_overrides({"alert": {
            "yellow_start_y": body.yellow_start_y,
            "red_start_y": body.red_start_y,
        }})
        validate(cfg)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _update_config_section("alert", {
        "yellow_start_y": round(body.yellow_start_y, 4),
        "red_start_y": round(body.red_start_y, 4),
    })
    return {"ok": True, "yellow_start_y": body.yellow_start_y, "red_start_y": body.red_start_y}


@app.post("/api/config/timing")
async def set_timing(body: AlertTimingConfig, _token: str = Depends(_check_session)):
    try:
        cfg = _build_config_with_overrides({"alert": {
            "repeat_interval_sec": body.repeat_interval_sec,
            "min_clear_sec": body.min_clear_sec,
            "min_alert_confidence": body.min_alert_confidence,
        }})
        validate(cfg)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _update_config_section("alert", {
        "repeat_interval_sec": body.repeat_interval_sec,
        "min_clear_sec": body.min_clear_sec,
        "min_alert_confidence": body.min_alert_confidence,
    })
    return {"ok": True}


@app.post("/api/config/validate")
async def validate_config(_token: str = Depends(_check_session)):
    try:
        load_config(CONFIG_PATH)
        return {"valid": True}
    except (ConfigError, Exception) as e:
        return {"valid": False, "error": str(e)}


@app.post("/api/config/restore")
async def restore_config(_token: str = Depends(_check_session)):
    """Restore last-known-good config."""
    backup = Path(CONFIG_PATH + BACKUP_SUFFIX)
    if not backup.exists():
        raise HTTPException(status_code=404, detail="No backup config found")
    shutil.copy2(str(backup), CONFIG_PATH)
    return {"ok": True, "message": "Previous config restored"}


# ---------------------------------------------------------------------------
# Service control
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status(_token: str = Depends(_check_session)):
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "is-active", "safetyvision"],
            capture_output=True, text=True, timeout=5,
        )
        state = result.stdout.strip()
    except Exception:
        state = "unknown"
    return {"service": state}


@app.post("/api/apply")
async def apply_config(_token: str = Depends(_check_session)):
    """Validate config, backup current, restart safetyvision service."""
    global _last_apply_ts
    now = time.time()
    if now - _last_apply_ts < APPLY_COOLDOWN:
        raise HTTPException(status_code=429, detail="Please wait before applying again")
    _last_apply_ts = now

    # Validate first
    try:
        load_config(CONFIG_PATH)
    except (ConfigError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Config invalid: {e}")

    # Backup current config
    backup = Path(CONFIG_PATH + BACKUP_SUFFIX)
    shutil.copy2(CONFIG_PATH, str(backup))

    # Restart service
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", "safetyvision"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Restart timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {"ok": True, "message": "Config applied and service restarted"}


# ---------------------------------------------------------------------------
# Live MJPEG stream
# ---------------------------------------------------------------------------
async def _get_camera() -> cv2.VideoCapture:
    global _cap
    async with _cap_lock:
        if _cap is None or not _cap.isOpened():
            raw = _load_raw_config()
            inp = raw.get("input", {})
            mode = inp.get("mode", "usb")
            if mode == "usb":
                dev = inp.get("usb_device", "/dev/video0")
                _cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            else:
                _cap = cv2.VideoCapture(inp.get("rtsp_url", ""))
            _cap.set(cv2.CAP_PROP_FRAME_WIDTH, inp.get("width", 640))
            _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, inp.get("height", 480))
            _cap.set(cv2.CAP_PROP_FPS, inp.get("fps", 30))
            _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return _cap


def _draw_zone_overlay(frame: np.ndarray, yellow_y: float, red_y: float) -> np.ndarray:
    """Draw semi-transparent zone bands on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    y_yel = int(yellow_y * h)
    y_red = int(red_y * h)

    cv2.rectangle(overlay, (0, 0), (w, y_yel), (0, 180, 0), -1)
    cv2.rectangle(overlay, (0, y_yel), (w, y_red), (0, 220, 255), -1)
    cv2.rectangle(overlay, (0, y_red), (w, h), (0, 0, 255), -1)

    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    cv2.line(frame, (0, y_yel), (w, y_yel), (0, 220, 255), 2)
    cv2.line(frame, (0, y_red), (w, y_red), (0, 0, 255), 2)
    return frame


async def _mjpeg_generator(overlay: bool = True):
    raw = _load_raw_config()
    alert = raw.get("alert", {})
    yellow_y = alert.get("yellow_start_y", 0.33)
    red_y = alert.get("red_start_y", 0.66)

    cap = await _get_camera()
    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.1)
            continue

        if overlay:
            frame = _draw_zone_overlay(frame, yellow_y, red_y)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        await asyncio.sleep(0.033)  # ~30fps


@app.get("/api/stream.mjpg")
async def stream(request: Request, overlay: bool = True):
    # Auth check via cookie
    token = request.cookies.get("sv_session")
    if not token or token not in SESSION_TOKENS:
        raise HTTPException(status_code=401)
    return StreamingResponse(
        _mjpeg_generator(overlay),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Frontend pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _load_raw_config() -> dict:
    p = Path(CONFIG_PATH)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_config_with_overrides(overrides: dict) -> SafetyVisionConfig:
    """Load current config, apply overrides, and return for validation.

    Does NOT validate — caller is responsible for calling validate().
    """
    from safetyvision.config import _merge, InputConfig, ModelConfig, AlertConfig, PerfConfig, LoggingConfig, HealthConfig

    raw = _load_raw_config()
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)

    return SafetyVisionConfig(
        input=_merge(InputConfig, raw.get("input")),
        model=_merge(ModelConfig, raw.get("model")),
        alert=_merge(AlertConfig, raw.get("alert")),
        perf=_merge(PerfConfig, raw.get("perf")),
        logging=_merge(LoggingConfig, raw.get("logging")),
        health=_merge(HealthConfig, raw.get("health")),
    )


def _update_config_section(section: str, values: dict) -> None:
    """Merge values into a section of the config YAML."""
    raw = _load_raw_config()
    raw.setdefault(section, {}).update(values)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(raw, f, default_flow_style=None, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="SafetyVision Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("-c", "--config", default=None)
    args = parser.parse_args()

    global CONFIG_PATH
    if args.config:
        CONFIG_PATH = args.config

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
