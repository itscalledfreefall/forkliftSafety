"""Decision worker – alert state machine with debounce and hysteresis."""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Optional

from loguru import logger

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import AlertEvent, AlertState, DetectionEvent


class DecisionWorker:
    """Consumes DetectionEvents and produces AlertEvents based on state machine."""

    def __init__(
        self,
        cfg: SafetyVisionConfig,
        in_queue: Queue,
        alert_queue: Queue,
        stop_event: threading.Event,
        latency_cb=None,
        frame_cb=None,
        alert_cb=None,
    ):
        self._cfg = cfg
        self._in_queue = in_queue
        self._alert_queue = alert_queue
        self._stop = stop_event
        self._latency_cb = latency_cb
        self._frame_cb = frame_cb
        self._alert_cb = alert_cb
        self._thread: Optional[threading.Thread] = None

        self._state = AlertState.IDLE
        self._last_trigger_ns: int = 0
        self._last_person_ns: int = 0
        self._alert_count: int = 0
        self._last_sound_key: str = ""
        self._smoothed_area_ratio: float = 0.0

    @property
    def state(self) -> AlertState:
        return self._state

    @property
    def alert_count(self) -> int:
        return self._alert_count

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="decision_worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def process_event(self, event: DetectionEvent) -> Optional[AlertEvent]:
        """Pure state-machine logic, testable without threads."""
        now_ns = event.timestamp_ns
        repeat_ns = int(self._cfg.alert.repeat_interval_sec * 1e9)
        clear_ns = int(self._cfg.alert.min_clear_sec * 1e9)
        self._update_area_ratio(event)
        sound_key = self._classify_sound_key(event)
        in_alert_zone = sound_key in ("danger", "medium")

        if in_alert_zone:
            self._last_person_ns = now_ns

        if self._state == AlertState.IDLE:
            if in_alert_zone:
                self._state = AlertState.TRIGGERED
                self._last_trigger_ns = now_ns
                self._alert_count += 1
                self._last_sound_key = sound_key
                return AlertEvent(
                    timestamp_ns=now_ns,
                    trigger_reason="person_detected",
                    cooldown_active=False,
                    sound_key=sound_key,
                )

        elif self._state == AlertState.TRIGGERED:
            if not in_alert_zone:
                # Check if clear period elapsed
                elapsed = now_ns - self._last_person_ns
                if elapsed >= clear_ns:
                    self._state = AlertState.IDLE
                    self._last_sound_key = ""
                    logger.info("Alert cleared after {:.1f}s of no person", elapsed / 1e9)
                    return None
            else:
                # Person remains in alert zone – switch clip immediately if zone changed.
                self._last_sound_key = sound_key

                # Person still present – check repeat interval
                elapsed = now_ns - self._last_trigger_ns
                if elapsed >= repeat_ns:
                    self._last_trigger_ns = now_ns
                    self._alert_count += 1
                    return AlertEvent(
                        timestamp_ns=now_ns,
                        trigger_reason="repeat_while_present",
                        cooldown_active=True,
                        sound_key=sound_key,
                    )

        return None

    def _update_area_ratio(self, event: DetectionEvent) -> None:
        """Smooth bbox area ratio to reduce near-threshold zone jitter."""
        if not event.person_detected:
            return
        alpha = self._cfg.alert.distance_smoothing_alpha
        self._smoothed_area_ratio = (
            alpha * event.max_bbox_area_ratio + (1.0 - alpha) * self._smoothed_area_ratio
        )

    def _classify_sound_key(self, event: DetectionEvent) -> str:
        """Map detection size to alert level."""
        if not event.person_detected:
            return ""

        if self._cfg.alert.use_zone_polygons:
            if event.zone_level not in ("danger", "medium"):
                return "medium" if self._cfg.alert.always_announce_person else ""
            if event.zone_confidence_max < self._cfg.alert.min_alert_confidence:
                return ""
            return event.zone_level

        if event.confidence_max < self._cfg.alert.min_alert_confidence:
            return ""
        ratio = self._smoothed_area_ratio
        h = self._cfg.alert.zone_hysteresis_ratio
        danger_enter = self._cfg.alert.close_area_ratio + h
        danger_exit = max(self._cfg.alert.close_area_ratio - h, 0.0)
        medium_enter = self._cfg.alert.medium_area_ratio + h
        medium_exit = max(self._cfg.alert.medium_area_ratio - h, 0.0)

        if self._last_sound_key == "danger":
            if ratio >= danger_exit:
                return "danger"
            if ratio >= medium_exit:
                return "medium"
            return "medium" if self._cfg.alert.always_announce_person else ""

        if self._last_sound_key == "medium":
            if ratio >= danger_enter:
                return "danger"
            if ratio >= medium_exit:
                return "medium"
            return "medium" if self._cfg.alert.always_announce_person else ""

        # First entry uses configured threshold directly; hysteresis applies after entry.
        if ratio >= self._cfg.alert.close_area_ratio:
            return "danger"
        if ratio >= self._cfg.alert.medium_area_ratio:
            return "medium"
        return "medium" if self._cfg.alert.always_announce_person else ""

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event: DetectionEvent = self._in_queue.get(timeout=0.1)
            except Empty:
                continue

            t0 = time.time_ns()
            alert = self.process_event(event)
            t1 = time.time_ns()

            if alert is not None:
                try:
                    self._alert_queue.put_nowait(alert)
                    if self._alert_cb:
                        self._alert_cb()
                except Exception:
                    logger.warning("Alert queue full, dropping alert event")

            if self._latency_cb:
                self._latency_cb((t1 - t0) / 1e6)
            if self._frame_cb:
                self._frame_cb()

        logger.info("Decision worker stopped")
