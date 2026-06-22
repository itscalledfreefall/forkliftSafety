"""Thermal API: read/write the thermal config, expose status + violation snapshots.

Constructed by ``app.py`` via ``create_thermal_router`` so the auth dependency,
config path, and the live ``ThermalMonitor`` instance are injected without
circular imports. The camera is streaming-only — no measurement writes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import yaml
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from safetyvision.config import ConfigError, ThermalConfig, _merge, load_config
from safetyvision.web.thermal_monitor import FlirClient

_SNAPSHOT_RE = re.compile(r"^thermal_\d+\.jpg$")


class ThermalConfigPayload(BaseModel):
    enabled: bool = False
    host: str = ""
    username: str = "admin"
    password: str = ""  # blank means "keep existing"
    rtsp_url: str = ""
    max_temp_c: float = 40.0
    poll_interval_sec: float = 1.0
    repeat_interval_sec: float = 10.0
    snapshot_dir: str = "thermal_violations"
    max_snapshots: int = 200


class ThermalTestPayload(BaseModel):
    host: str = ""
    username: str = "admin"
    password: str = ""  # blank means "use saved password"


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def _write_yaml(path: str, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)


def create_thermal_router(
    check_session: Callable,
    get_config_path: Callable[[], str],
    get_monitor: Callable,
) -> APIRouter:
    router = APIRouter(prefix="/api/thermal")

    def _cfg() -> ThermalConfig:
        return _merge(ThermalConfig, _load_yaml(get_config_path()).get("thermal"))

    @router.get("/config")
    async def get_config(_t: str = Depends(check_session)):
        cfg = _cfg()
        return {
            "enabled": cfg.enabled,
            "host": cfg.host,
            "username": cfg.username,
            "password": "",  # never expose the stored password
            "password_set": bool(cfg.password),
            "rtsp_url": cfg.rtsp_url,
            "max_temp_c": cfg.max_temp_c,
            "poll_interval_sec": cfg.poll_interval_sec,
            "repeat_interval_sec": cfg.repeat_interval_sec,
            "snapshot_dir": cfg.snapshot_dir,
            "max_snapshots": cfg.max_snapshots,
        }

    @router.post("/config")
    async def save_config(payload: ThermalConfigPayload, _t: str = Depends(check_session)):
        path = get_config_path()
        raw = _load_yaml(path)
        prev = raw.get("thermal") or {}

        section = payload.model_dump()
        # Blank password means keep the existing one.
        if not section.get("password"):
            section["password"] = prev.get("password", "")

        raw["thermal"] = section
        _write_yaml(path, raw)
        try:
            load_config(path)  # full validation against the written file
        except ConfigError as e:
            raw["thermal"] = prev  # roll back
            _write_yaml(path, raw)
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True}

    @router.get("/status")
    async def status(_t: str = Depends(check_session)):
        return get_monitor().status()

    @router.get("/violations")
    async def violations(_t: str = Depends(check_session)):
        mon = get_monitor()
        return {"count": mon.status()["violation_count"], "items": mon.violations()}

    @router.get("/snapshot/{name}")
    async def snapshot(name: str, _t: str = Depends(check_session)):
        if not _SNAPSHOT_RE.match(name):
            raise HTTPException(status_code=400, detail="invalid snapshot name")
        path = Path(_cfg().snapshot_dir) / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="snapshot not found")
        return FileResponse(str(path), media_type="image/jpeg")

    @router.post("/test")
    async def test(payload: ThermalTestPayload, _t: str = Depends(check_session)):
        cfg = _cfg()
        host = payload.host or cfg.host
        username = payload.username or cfg.username
        password = payload.password or cfg.password
        if not host:
            raise HTTPException(status_code=400, detail="host is required")
        ok, msg = FlirClient(host, username, password).test()
        return JSONResponse({"ok": ok, "message": msg})

    return router
