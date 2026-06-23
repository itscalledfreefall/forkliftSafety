"""Metrics worker – records FPS, latency per stage, dropped frames, alert events."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from loguru import logger

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import DetectionEvent, PipelineMetrics


class MetricsCollector:
    """Thread-safe metrics collection with rolling windows."""

    def __init__(self, window_sec: float = 5.0, state_path: str | Path | None = None):
        self._lock = threading.Lock()
        self._window = window_sec
        self._state_path = Path(state_path) if state_path else None
        self._capture_frame_times: deque[float] = deque()
        self._inference_frame_times: deque[float] = deque()
        self._decision_frame_times: deque[float] = deque()
        self._capture_latencies: deque[float] = deque()
        self._inference_latencies: deque[float] = deque()
        self._decision_latencies: deque[float] = deque()
        self._frames_dropped = 0
        self._alert_count = 0
        self._yellow_zone_entries = 0
        self._red_zone_entries = 0
        self._start_time = time.monotonic()
        # Last detection event values (overwritten each event)
        self._last_distance_m: Optional[float] = None
        self._last_zone_level: str = ""
        self._load_state()

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load metrics state from {}: {}", self._state_path, e)
            return

        self._alert_count = max(0, int(data.get("alert_count", 0)))
        self._yellow_zone_entries = max(0, int(data.get("yellow_zone_entries", 0)))
        self._red_zone_entries = max(0, int(data.get("red_zone_entries", 0)))

    def _persist_state_locked(self) -> None:
        if self._state_path is None:
            return

        data = {
            "alert_count": self._alert_count,
            "yellow_zone_entries": self._yellow_zone_entries,
            "red_zone_entries": self._red_zone_entries,
            "updated_at": time.time(),
        }
        tmp_path = self._state_path.with_name(f"{self._state_path.name}.tmp")
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            tmp_path.replace(self._state_path)
        except Exception as e:
            logger.warning("Failed to persist metrics state to {}: {}", self._state_path, e)

    def _record_stage_frame(self, dq: deque[float]) -> None:
        now = time.monotonic()
        dq.append(now)
        cutoff = now - self._window
        while dq and dq[0] < cutoff:
            dq.popleft()

    def record_frame(self) -> None:
        """Backward-compatible alias for inference stage FPS."""
        self.record_inference_frame()

    def record_capture_frame(self) -> None:
        with self._lock:
            self._record_stage_frame(self._capture_frame_times)

    def record_inference_frame(self) -> None:
        with self._lock:
            self._record_stage_frame(self._inference_frame_times)

    def record_decision_frame(self) -> None:
        with self._lock:
            self._record_stage_frame(self._decision_frame_times)

    def record_capture_latency(self, ms: float) -> None:
        with self._lock:
            self._capture_latencies.append(ms)
            if len(self._capture_latencies) > 300:
                self._capture_latencies.popleft()

    def record_inference_latency(self, ms: float) -> None:
        with self._lock:
            self._inference_latencies.append(ms)
            if len(self._inference_latencies) > 300:
                self._inference_latencies.popleft()

    def record_decision_latency(self, ms: float) -> None:
        with self._lock:
            self._decision_latencies.append(ms)
            if len(self._decision_latencies) > 300:
                self._decision_latencies.popleft()

    def record_drop(self) -> None:
        with self._lock:
            self._frames_dropped += 1

    def record_alert(self) -> None:
        with self._lock:
            self._alert_count += 1
            self._persist_state_locked()

    def record_yellow_entry(self) -> None:
        with self._lock:
            self._yellow_zone_entries += 1
            self._persist_state_locked()

    def record_red_entry(self) -> None:
        with self._lock:
            self._red_zone_entries += 1
            self._persist_state_locked()

    def record_detection_event(self, event: DetectionEvent) -> None:
        """Store the latest distance / zone for dashboard display."""
        with self._lock:
            self._last_distance_m = event.distance_m
            self._last_zone_level = event.zone_level

    def snapshot(self) -> PipelineMetrics:
        with self._lock:
            now = time.monotonic()

            def _fps(dq: deque[float]) -> float:
                if not dq:
                    return 0.0
                span = now - dq[0]
                if span <= 0:
                    return 0.0
                return len(dq) / span

            def _median(dq: deque) -> float:
                if not dq:
                    return 0.0
                s = sorted(dq)
                mid = len(s) // 2
                return s[mid]

            capture_fps = _fps(self._capture_frame_times)
            inference_fps = _fps(self._inference_frame_times)
            decision_fps = _fps(self._decision_frame_times)

            distance = self._last_distance_m
            return PipelineMetrics(
                capture_fps=round(capture_fps, 1),
                inference_fps=round(inference_fps, 1),
                decision_fps=round(decision_fps, 1),
                fps=round(inference_fps or capture_fps, 1),
                capture_latency_ms=round(_median(self._capture_latencies), 2),
                inference_latency_ms=round(_median(self._inference_latencies), 2),
                decision_latency_ms=round(_median(self._decision_latencies), 2),
                total_latency_ms=round(
                    _median(self._capture_latencies)
                    + _median(self._inference_latencies)
                    + _median(self._decision_latencies),
                    2,
                ),
                frames_dropped=self._frames_dropped,
                alert_count=self._alert_count,
                yellow_zone_entries=self._yellow_zone_entries,
                red_zone_entries=self._red_zone_entries,
                uptime_sec=round(now - self._start_time, 1),
                last_distance_m=round(distance, 2) if distance is not None else None,
                last_zone_level=self._last_zone_level,
            )


class MetricsWorker:
    """Periodically logs metrics snapshots."""

    def __init__(
        self,
        cfg: SafetyVisionConfig,
        collector: MetricsCollector,
        stop_event: threading.Event,
    ):
        self._cfg = cfg
        self._collector = collector
        self._stop = stop_event
        self._thread: Optional[threading.Thread] = None
        self._interval = cfg.health.heartbeat_interval_sec

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="metrics_worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            if self._stop.is_set():
                break

            snap = self._collector.snapshot()

            if self._cfg.logging.json_output:
                record = {
                    "type": "metrics",
                    "ts": time.time(),
                    "fps": snap.fps,
                    "capture_fps": snap.capture_fps,
                    "inference_fps": snap.inference_fps,
                    "decision_fps": snap.decision_fps,
                    "latency_capture_ms": snap.capture_latency_ms,
                    "latency_inference_ms": snap.inference_latency_ms,
                    "latency_decision_ms": snap.decision_latency_ms,
                    "latency_total_ms": snap.total_latency_ms,
                    "frames_dropped": snap.frames_dropped,
                    "alerts": snap.alert_count,
                    "yellow_zone_entries": snap.yellow_zone_entries,
                    "red_zone_entries": snap.red_zone_entries,
                    "uptime_s": snap.uptime_sec,
                    "last_distance_m": snap.last_distance_m,
                    "last_zone_level": snap.last_zone_level,
                }
                logger.info(json.dumps(record))
            else:
                logger.info(
                    "FPS(inf)={} capture={} decision={} lat={:.1f}ms dropped={} alerts={} yel={} red={} up={:.0f}s",
                    snap.fps,
                    snap.capture_fps,
                    snap.decision_fps,
                    snap.total_latency_ms,
                    snap.frames_dropped,
                    snap.alert_count,
                    snap.yellow_zone_entries,
                    snap.red_zone_entries,
                    snap.uptime_sec,
                )
