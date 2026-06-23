"""Decision worker – per-camera alert state machine."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Optional

from loguru import logger

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import AlertEvent, AlertState, DetectionEvent


@dataclass
class _CameraState:
    state: AlertState = AlertState.IDLE
    last_trigger_ns: int = 0
    last_person_ns: int = 0
    alert_count: int = 0
    last_sound_key: str = ""


class DecisionWorker:
    """Consumes DetectionEvents and produces AlertEvents.

    Each camera maintains its own state machine so an active person on one
    camera does not block another camera from clearing.
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
        self._states: dict[str, _CameraState] = {}

    @property
    def alert_count(self) -> int:
        return sum(s.alert_count for s in self._states.values())

    def state_for(self, camera_id: str) -> AlertState:
        return self._states.get(camera_id, _CameraState()).state

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
        cam_state = self._states.setdefault(event.camera_id or "default", _CameraState())

        now_ns = event.timestamp_ns
        repeat_ns = int(self._cfg.alert.repeat_interval_sec * 1e9)
        clear_ns = int(self._cfg.alert.min_clear_sec * 1e9)

        sound_key = self._classify_sound_key(event)
        in_alert_zone = sound_key in ("danger", "medium")

        if in_alert_zone:
            cam_state.last_person_ns = now_ns

        if cam_state.state == AlertState.IDLE:
            if in_alert_zone:
                cam_state.state = AlertState.TRIGGERED
                cam_state.last_trigger_ns = now_ns
                cam_state.last_sound_key = sound_key
                cam_state.alert_count += 1
                return AlertEvent(
                    timestamp_ns=now_ns,
                    trigger_reason="person_detected",
                    cooldown_active=False,
                    sound_key=sound_key,
                    camera_id=event.camera_id,
                )

        elif cam_state.state == AlertState.TRIGGERED:
            if not in_alert_zone:
                elapsed = now_ns - cam_state.last_person_ns
                if elapsed >= clear_ns:
                    cam_state.state = AlertState.IDLE
                    cam_state.last_sound_key = ""
                    logger.info(
                        "[{}] Alert cleared after {:.1f}s of no person",
                        event.camera_id or "default",
                        elapsed / 1e9,
                    )
                    return None
            else:
                if self._sound_priority(sound_key) > self._sound_priority(cam_state.last_sound_key):
                    cam_state.last_trigger_ns = now_ns
                    cam_state.last_sound_key = sound_key
                    cam_state.alert_count += 1
                    return AlertEvent(
                        timestamp_ns=now_ns,
                        trigger_reason="zone_escalated",
                        cooldown_active=False,
                        sound_key=sound_key,
                        camera_id=event.camera_id,
                    )
                elapsed = now_ns - cam_state.last_trigger_ns
                if elapsed >= repeat_ns:
                    cam_state.last_trigger_ns = now_ns
                    cam_state.alert_count += 1
                    # Repeat at the highest zone seen this episode. Do NOT lower
                    # last_sound_key to the current frame's zone: a person on the
                    # danger/medium boundary flips zones frame-to-frame, and a
                    # downgrade here would make the next danger frame look like a
                    # fresh escalation and fire immediately, bypassing the throttle.
                    return AlertEvent(
                        timestamp_ns=now_ns,
                        trigger_reason="repeat_while_present",
                        cooldown_active=True,
                        sound_key=cam_state.last_sound_key,
                        camera_id=event.camera_id,
                    )

        return None

    @staticmethod
    def _sound_priority(sound_key: str) -> int:
        if sound_key == "danger":
            return 2
        if sound_key == "medium":
            return 1
        return 0

    def _classify_sound_key(self, event: DetectionEvent) -> str:
        if not event.person_detected:
            return ""
        if event.confidence_max < self._cfg.alert.min_alert_confidence:
            return ""
        if event.zone_level == "danger":
            return "danger"
        if event.zone_level == "medium":
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
