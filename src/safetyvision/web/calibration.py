"""Calibration API: save/load homography points and toggle distance mode.

Single-camera (back) scope. The router is constructed by ``app.py`` via
``create_calibration_router`` so the auth dependency and config path can
be injected without circular imports.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from safetyvision.workers.capture import CALIBRATION_FRAME_PATH

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
REPROJ_ERR_PX = 2.0
DET_MIN = 1e-6


class CalibrationPayload(BaseModel):
    source_points: list[list[float]]
    target_points: list[list[float]]
    frame_width: int
    frame_height: int


class CalibrationError(ValueError):
    """Raised for invalid calibration submissions."""


def _validate(payload: CalibrationPayload) -> np.ndarray:
    """Validate calibration. Returns the 3x3 homography matrix."""
    if len(payload.source_points) != 4 or len(payload.target_points) != 4:
        raise CalibrationError("Exactly 4 source and 4 target points required")

    for i, p in enumerate(payload.source_points):
        if len(p) != 2:
            raise CalibrationError(f"source_points[{i}] must be [x, y]")
        x, y = p
        if not (0 <= x <= payload.frame_width and 0 <= y <= payload.frame_height):
            raise CalibrationError(
                f"source_points[{i}] = ({x:.0f}, {y:.0f}) is outside "
                f"the {payload.frame_width}x{payload.frame_height} frame"
            )

    for i, p in enumerate(payload.target_points):
        if len(p) != 2:
            raise CalibrationError(f"target_points[{i}] must be [x, y]")

    src = np.array(payload.source_points, dtype=np.float32)
    tgt = np.array(payload.target_points, dtype=np.float32)

    h, _ = cv2.findHomography(src, tgt)
    if h is None:
        raise CalibrationError(
            "Homography is degenerate — points may be collinear or duplicated"
        )
    if abs(float(np.linalg.det(h))) < DET_MIN:
        raise CalibrationError(
            f"Homography determinant near zero (|det| < {DET_MIN}) — degenerate calibration"
        )

    # Round-trip reprojection: src -> tgt -> src must be < 2 px on average.
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
    h_inv = np.linalg.inv(h)
    back = cv2.perspectiveTransform(
        projected.reshape(-1, 1, 2).astype(np.float64), h_inv
    ).reshape(-1, 2)
    err = float(np.linalg.norm(back - src, axis=1).mean())
    if err > REPROJ_ERR_PX:
        raise CalibrationError(
            f"Reprojection error {err:.2f}px exceeds {REPROJ_ERR_PX}px threshold"
        )
    return h


# ---------------------------------------------------------------------------
# YAML helpers (kept local to avoid a circular import with app.py)
# ---------------------------------------------------------------------------
def _load_yaml(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _write_yaml(config_path: str, data: dict) -> None:
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)


def _alert_section(raw: dict) -> dict:
    return raw.get("alert", {}) or {}


def _calibration_path(raw: dict) -> str:
    return _alert_section(raw).get("calibration_path", "config/calibration_back.json")


def _restart_service() -> tuple[bool, str]:
    """Restart safetyvision via systemctl. Returns (ok, message)."""
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", "safetyvision"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return False, result.stderr.strip() or "systemctl restart failed"
    except subprocess.TimeoutExpired:
        return False, "systemctl restart timed out"
    except Exception as e:
        return False, str(e)
    return True, "service restarted"


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------
def create_calibration_router(
    check_session: Callable,
    get_config_path: Callable[[], str],
) -> APIRouter:
    """Build the calibration APIRouter with injected auth + config-path deps.

    ``get_config_path`` is a callable so the router picks up the live value
    of the app's CONFIG_PATH (which can be overridden by ``--config`` after
    the router has been constructed).
    """

    router = APIRouter(prefix="/api/calibration")
    _last_toggle: dict[str, float] = {"ts": 0.0}
    TOGGLE_COOLDOWN = 5.0  # seconds

    @router.get("/frame")
    async def get_frame(_t: str = Depends(check_session)):
        """Latest decoded frame from the back camera (tmpfs)."""
        path = Path(CALIBRATION_FRAME_PATH)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="No frame available yet — is the capture worker running?",
            )
        return FileResponse(str(path), media_type="image/jpeg")

    @router.get("/status")
    async def get_status(_t: str = Depends(check_session)):
        raw = _load_yaml(get_config_path())
        cal_path = Path(_calibration_path(raw))
        return {
            "zone_mode": _alert_section(raw).get("zone_mode", "bands"),
            "calibrated": cal_path.exists(),
            "calibration_path": str(cal_path),
        }

    @router.get("")
    async def get_calibration(_t: str = Depends(check_session)):
        raw = _load_yaml(get_config_path())
        cal_path = Path(_calibration_path(raw))
        if not cal_path.exists():
            raise HTTPException(status_code=404, detail="No calibration saved")
        with open(cal_path) as f:
            return json.load(f)

    @router.post("")
    async def save_calibration(
        payload: CalibrationPayload, _t: str = Depends(check_session)
    ):
        try:
            _validate(payload)
        except CalibrationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        raw = _load_yaml(get_config_path())
        cal_path = Path(_calibration_path(raw))
        cal_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "camera_id": "back",
            "source_points": payload.source_points,
            "target_points": payload.target_points,
            "frame_width": payload.frame_width,
            "frame_height": payload.frame_height,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(cal_path, "w") as f:
            json.dump(record, f, indent=2)
        return {"ok": True, "path": str(cal_path)}

    @router.delete("")
    async def delete_calibration(_t: str = Depends(check_session)):
        raw = _load_yaml(get_config_path())
        cal_path = Path(_calibration_path(raw))
        if not cal_path.exists():
            raise HTTPException(status_code=404, detail="No calibration to delete")
        cal_path.unlink()
        return {"ok": True}

    def _toggle(target_mode: str) -> JSONResponse:
        now = time.time()
        if now - _last_toggle["ts"] < TOGGLE_COOLDOWN:
            raise HTTPException(
                status_code=429, detail="Please wait before toggling mode again"
            )
        _last_toggle["ts"] = now

        raw = _load_yaml(get_config_path())
        if target_mode == "distance":
            cal_path = Path(_calibration_path(raw))
            if not cal_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot enable distance mode: calibration file not found at {cal_path}",
                )

        raw.setdefault("alert", {})["zone_mode"] = target_mode
        _write_yaml(get_config_path(), raw)

        ok, msg = _restart_service()
        if not ok:
            return JSONResponse({"ok": False, "error": msg}, status_code=500)
        return JSONResponse({"ok": True, "zone_mode": target_mode, "message": msg})

    @router.post("/enable")
    async def enable(_t: str = Depends(check_session)):
        return _toggle("distance")

    @router.post("/disable")
    async def disable(_t: str = Depends(check_session)):
        return _toggle("bands")

    return router
