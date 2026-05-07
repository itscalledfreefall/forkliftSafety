"""Decision worker – alert state machine with horizontal band zones."""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Optional

from loguru import logger

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import AlertEvent, AlertState, DetectionEvent


class DecisionWorker:
    """Consumes DetectionEvents and produces AlertEvents based on state machine.

    Zone classification (green/yellow/red) is already resolved by inference
    worker into ``event.zone_level``.  This worker only manages the alert
    state machine: trigger, repeat-throttle, and clear.
    """

    def __init__(
        self,
        cfg: SafetyVisionConfig,
        in_queue: Queue,
        alert_queue: Queue,
        stop_event: threading.Event,
        latency_cb=None,
        frame_cb=None,
        alert_cb=None,
        event_cb=None,
    ):
        self._cfg = cfg
        self._in_queue = in_queue
        self._alert_queue = alert_queue
        self._stop = stop_event
        self._latency_cb = latency_cb
        self._frame_cb = frame_cb
        self._alert_cb = alert_cb
        self._event_cb = event_cb
        self._thread: Optional[threading.Thread] = None

        self._state = AlertState.IDLE
        self._last_trigger_ns: int = 0
        self._last_person_ns: int = 0
        self._alert_count: int = 0

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

        sound_key = self._classify_sound_key(event)
        in_alert_zone = sound_key in ("danger", "medium")

        if in_alert_zone:
            self._last_person_ns = now_ns

        if self._state == AlertState.IDLE:
            if in_alert_zone:
                self._state = AlertState.TRIGGERED
                self._last_trigger_ns = now_ns
                self._alert_count += 1
                return AlertEvent(
                    timestamp_ns=now_ns,
                    trigger_reason="person_detected",
                    cooldown_active=False,
                    sound_key=sound_key,
                )

        elif self._state == AlertState.TRIGGERED:
            if not in_alert_zone:
                elapsed = now_ns - self._last_person_ns
                if elapsed >= clear_ns:
                    self._state = AlertState.IDLE
                    logger.info("Alert cleared after {:.1f}s of no person", elapsed / 1e9)
                    return None
            else:
                # Person still in alert zone – check repeat interval
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

    def _classify_sound_key(self, event: DetectionEvent) -> str:
        """Map zone_level to sound key with confidence gate."""
        if not event.person_detected:
            return ""
        if event.confidence_max < self._cfg.alert.min_alert_confidence:
            return ""
        if event.zone_level == "danger":
            return "danger"
        if event.zone_level == "medium":
            return "medium"
        # Green zone or no zone → silent
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

            if self._event_cb is not None:
                self._event_cb(event)

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
