"""Inference worker – delegates model I/O to a backend, handles zones + smoothing."""

from __future__ import annotations

import os
import threading
import time
from queue import Empty, Queue
from typing import Optional

from loguru import logger

from safetyvision.config import SafetyVisionConfig
from safetyvision.inference.backends import InferenceBackend, load_backend
from safetyvision.types import Detection, DetectionEvent, FramePacket


def _pin_to_cores(cores: list[int]) -> None:
    try:
        os.sched_setaffinity(0, set(cores))
        logger.debug("Inference thread pinned to cores {}", cores)
    except (AttributeError, OSError):
        pass


def _classify_detection_zone(
    det: Detection,
    frame_h: int,
    cfg: SafetyVisionConfig,
) -> str:
    """Classify a detection into a horizontal band zone by footpoint Y.

    Bands (top-to-bottom):
        0.0  .. yellow_start_y  = green  (no sound)
        yellow_start_y .. red_start_y = medium
        red_start_y .. 1.0      = danger
    """
    if frame_h <= 0:
        return ""
    foot_y = float(det.y2) / frame_h
    foot_y = min(max(foot_y, 0.0), 1.0)

    if foot_y >= cfg.alert.red_start_y:
        return "danger"
    if foot_y >= cfg.alert.yellow_start_y:
        return "medium"
    return ""


class InferenceWorker:
    """Runs a single Hailo inference thread over the shared frame queue."""

    def __init__(
        self,
        cfg: SafetyVisionConfig,
        in_queue: Queue,
        out_queue: Queue,
        stop_event: threading.Event,
        latency_cb=None,
        frame_cb=None,
        backend: Optional[InferenceBackend] = None,
    ):
        self._cfg = cfg
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._stop = stop_event
        self._latency_cb = latency_cb
        self._frame_cb = frame_cb
        self._backend = backend
        self._thread: Optional[threading.Thread] = None
        self._smoothing: dict[str, list[bool]] = {}

    def start(self) -> None:
        if self._backend is None:
            self._backend = load_backend(self._cfg)
        self._thread = threading.Thread(
            target=self._run, name="inference_worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception as e:
                logger.warning("Backend close error: {}", e)
            self._backend = None

    def _apply_temporal_smoothing(self, camera_id: str, person_detected: bool) -> bool:
        """Require majority of recent frames to agree per-camera."""
        n = max(1, self._cfg.perf.temporal_smoothing_frames)
        buf = self._smoothing.setdefault(camera_id, [])
        buf.append(person_detected)
        if len(buf) > n:
            buf.pop(0)
        return sum(buf) >= (n + 1) // 2

    def _run(self) -> None:
        _pin_to_cores(self._cfg.perf.inference_cpu_cores)

        while not self._stop.is_set():
            try:
                pkt: FramePacket = self._in_queue.get(timeout=0.1)
            except Empty:
                continue

            t0 = time.time_ns()
            try:
                dets = self._backend.infer(pkt.frame)
            except Exception as e:
                logger.error("Inference failure: {}", e)
                continue
            t1 = time.time_ns()

            frame_h = pkt.frame.shape[0]
            raw_detected = len(dets) > 0
            smoothed = self._apply_temporal_smoothing(pkt.camera_id, raw_detected)
            max_conf = max((d.confidence for d in dets), default=0.0)

            # Multi-person: highest-risk band wins (danger > medium > green)
            zone_level = ""
            for d in dets:
                zone = _classify_detection_zone(d, frame_h, self._cfg)
                if zone == "danger":
                    zone_level = "danger"
                    break
                if zone == "medium":
                    zone_level = "medium"

            event = DetectionEvent(
                timestamp_ns=pkt.timestamp_ns,
                person_detected=smoothed,
                confidence_max=max_conf,
                bbox_count=len(dets),
                zone_level=zone_level,
                camera_id=pkt.camera_id,
            )

            # Drain and push latest event to keep decision queue fresh.
            while not self._out_queue.empty():
                try:
                    self._out_queue.get_nowait()
                except Exception:
                    break
            try:
                self._out_queue.put_nowait(event)
            except Exception:
                pass

            if self._latency_cb:
                self._latency_cb((t1 - t0) / 1e6)
            if self._frame_cb:
                self._frame_cb()

        logger.info("Inference worker stopped")
