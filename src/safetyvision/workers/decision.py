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
    ):
        self._cfg = cfg
        self._in_queue = in_queue
        self._alert_queue = alert_queue
        self._stop = stop_event
        self._latency_cb = latency_cb
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

        if event.person_detected:
            self._last_person_ns = now_ns

        if self._state == AlertState.IDLE:
            if event.person_detected:
                self._state = AlertState.TRIGGERED
                self._last_trigger_ns = now_ns
                self._alert_count += 1
                return AlertEvent(
                    timestamp_ns=now_ns,
                    trigger_reason="person_detected",
                    cooldown_active=False,
                )

        elif self._state == AlertState.TRIGGERED:
            if not event.person_detected:
                # Check if clear period elapsed
                elapsed = now_ns - self._last_person_ns
                if elapsed >= clear_ns:
                    self._state = AlertState.IDLE
                    logger.info("Alert cleared after {:.1f}s of no person", elapsed / 1e9)
                    return None
            else:
                # Person still present – check repeat interval
                elapsed = now_ns - self._last_trigger_ns
                if elapsed >= repeat_ns:
                    self._last_trigger_ns = now_ns
                    self._alert_count += 1
                    return AlertEvent(
                        timestamp_ns=now_ns,
                        trigger_reason="repeat_while_present",
                        cooldown_active=True,
                    )

        return None

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
                except Exception:
                    logger.warning("Alert queue full, dropping alert event")

            if self._latency_cb:
                self._latency_cb((t1 - t0) / 1e6)

        logger.info("Decision worker stopped")
