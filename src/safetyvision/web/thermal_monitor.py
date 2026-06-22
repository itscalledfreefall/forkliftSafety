"""FLIR A50 thermal scene-max poller + violation snapshots.

This camera is streaming-only (measurement functions are not licensed), so the
violation threshold is checked against the live scene-max temperature read from
the FLIR REST API: ``GET /api/image/adjustment`` returns ``{"high": <Kelvin>, ...}``
which, in the camera's default auto level/span mode, tracks the hottest part of
the scene. No camera writes are performed.

``FlirClient`` does the HTTP (stdlib only). ``ThermalMonitor`` is a daemon thread
that polls scene-max, and on a rising edge above the threshold (rate-limited by
``repeat_interval_sec``) saves a tagged JPEG snapshot to disk for the web gallery.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..config import ThermalConfig, k_to_c

logger = logging.getLogger(__name__)


class FlirClient:
    """Minimal authenticated REST client for the FLIR web API."""

    def __init__(self, host: str, username: str, password: str, timeout: float = 8.0):
        self._base = f"http://{host}"
        self._user = username
        self._password = password
        self._timeout = timeout
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar())
        )
        self._logged_in = False

    def _open(self, path: str, data: Optional[bytes] = None) -> bytes:
        req = urllib.request.Request(self._base + path, data=data)
        with self._opener.open(req, timeout=self._timeout) as resp:
            return resp.read()

    def login(self) -> None:
        """Establish a session cookie via the Symfony form login."""
        self._open("/login")  # seed session cookie
        body = urllib.parse.urlencode(
            {"_username": self._user, "_password": self._password}
        ).encode()
        self._open("/login?redirectTo=%2F", data=body)
        self._logged_in = True

    def scene_max_c(self) -> float:
        """Return the live scene-max temperature in Celsius.

        Re-authenticates once on an auth failure.
        """
        for attempt in range(2):
            if not self._logged_in:
                self.login()
            try:
                raw = self._open("/api/image/adjustment")
                payload = json.loads(raw.decode("utf-8", "replace"))
                return k_to_c(float(payload["high"]))
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403) and attempt == 0:
                    self._logged_in = False
                    continue
                raise
        raise RuntimeError("scene_max_c: unreachable")

    def test(self) -> tuple[bool, str]:
        """Verify login + a scene-max read for the config tab.

        Returns a coarse, sanitized message; the detailed exception is logged
        only (so the endpoint can't be used as an SSRF/probe oracle).
        """
        try:
            self._logged_in = False
            temp = self.scene_max_c()
            return True, f"Connected. Scene max {temp:.1f} C."
        except urllib.error.HTTPError as exc:
            logger.warning("thermal test HTTP error: %s", exc)
            if exc.code in (401, 403):
                return False, "authentication failed"
            return False, "unexpected response from camera"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            logger.warning("thermal test connection error: %s", exc)
            return False, "camera unreachable"
        except Exception as exc:  # noqa: BLE001 - surfaced coarsely to the UI
            logger.warning("thermal test error: %s", exc)
            return False, "test failed"


def _tag_frame(frame: np.ndarray, temp_c: float, when: datetime) -> np.ndarray:
    """Burn an OVER-TEMP banner with the measured temperature onto a copy."""
    img = frame.copy()
    h, w = img.shape[:2]
    banner_h = max(28, h // 12)
    cv2.rectangle(img, (0, 0), (w, banner_h), (0, 0, 0), -1)
    text = f"OVER TEMP  {temp_c:.1f} C  {when:%Y-%m-%d %H:%M:%S}"
    scale = max(0.4, w / 900.0)
    cv2.putText(
        img, text, (8, int(banner_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2, cv2.LINE_AA,
    )
    return img


class ThermalMonitor:
    """Polls scene-max temperature and snapshots violations to disk."""

    def __init__(
        self,
        get_cfg: Callable[[], ThermalConfig],
        get_latest_frame: Callable[[], Optional[np.ndarray]],
    ):
        self._get_cfg = get_cfg
        self._get_frame = get_latest_frame
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[FlirClient] = None
        self._client_key: tuple = ()
        # State exposed to the web layer.
        self._scene_temp_c: Optional[float] = None
        self._in_violation = False
        self._violation_count = 0
        self._was_over = False
        self._last_snapshot_mono = 0.0

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="thermal-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- helpers ---------------------------------------------------------
    def _client_for(self, cfg: ThermalConfig) -> FlirClient:
        key = (cfg.host, cfg.username, cfg.password)
        if self._client is None or key != self._client_key:
            self._client = FlirClient(cfg.host, cfg.username, cfg.password)
            self._client_key = key
        return self._client

    def _snapshot_dir(self, cfg: ThermalConfig) -> Path:
        d = Path(cfg.snapshot_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_snapshot(self, cfg: ThermalConfig, temp_c: float) -> None:
        frame = self._get_frame()
        if frame is None:
            logger.warning("thermal violation %.1fC but no frame available to snapshot", temp_c)
            return
        when = datetime.now()
        tagged = _tag_frame(frame, temp_c, when)
        d = self._snapshot_dir(cfg)
        stem = f"thermal_{time.time_ns()}"
        ok, buf = cv2.imencode(".jpg", tagged, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        (d / f"{stem}.jpg").write_bytes(buf.tobytes())
        (d / f"{stem}.json").write_text(
            json.dumps(
                {
                    "temp_c": round(temp_c, 1),
                    "threshold_c": cfg.max_temp_c,
                    "timestamp": when.isoformat(timespec="seconds"),
                }
            )
        )
        self._prune(d, cfg.max_snapshots)

    @staticmethod
    def _prune(d: Path, max_snapshots: int) -> None:
        jpgs = sorted(d.glob("thermal_*.jpg"), key=lambda p: p.stat().st_mtime)
        for old in jpgs[:-max_snapshots] if len(jpgs) > max_snapshots else []:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)

    # -- poll loop -------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            cfg = self._get_cfg()
            interval = max(0.2, cfg.poll_interval_sec)
            if not cfg.enabled or not cfg.host:
                self._stop.wait(interval)
                continue
            try:
                temp = self._client_for(cfg).scene_max_c()
                self._evaluate(cfg, temp)
            except Exception as exc:  # noqa: BLE001 - camera may be transiently down
                logger.debug("thermal poll failed: %s", exc)
                with self._lock:
                    self._scene_temp_c = None
            self._stop.wait(interval)

    def _evaluate(self, cfg: ThermalConfig, temp_c: float) -> None:
        over = temp_c >= cfg.max_temp_c
        now = time.monotonic()
        snapshot = False
        if over:
            if not self._was_over:
                snapshot = True  # rising edge
            elif cfg.repeat_interval_sec > 0 and (
                now - self._last_snapshot_mono >= cfg.repeat_interval_sec
            ):
                snapshot = True  # still over, periodic re-trip
        self._was_over = over

        if snapshot:
            self._save_snapshot(cfg, temp_c)
            self._last_snapshot_mono = now
            with self._lock:
                self._violation_count += 1
        with self._lock:
            self._scene_temp_c = temp_c
            self._in_violation = over

    # -- web-facing accessors -------------------------------------------
    def status(self) -> dict:
        cfg = self._get_cfg()
        with self._lock:
            return {
                "enabled": cfg.enabled,
                "scene_temp_c": (
                    round(self._scene_temp_c, 1) if self._scene_temp_c is not None else None
                ),
                "threshold_c": cfg.max_temp_c,
                "in_violation": self._in_violation,
                "violation_count": self._violation_count,
            }

    def violations(self, limit: int = 60) -> list[dict]:
        cfg = self._get_cfg()
        d = Path(cfg.snapshot_dir)
        if not d.exists():
            return []
        jpgs = sorted(d.glob("thermal_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        out: list[dict] = []
        for p in jpgs[:limit]:
            meta = {}
            sidecar = p.with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                except (ValueError, OSError):
                    meta = {}
            out.append({"name": p.name, **meta})
        return out
