"""Inference worker - delegates model I/O to a backend, handles zones + smoothing."""

from __future__ import annotations

import os
import threading
import time
from queue import Empty, Queue
from typing import Optional

from loguru import logger

from safetyvision.config import CameraConfig, SafetyVisionConfig, get_effective_zone_thresholds
from safetyvision.inference.backends import InferenceBackend, load_backend
from safetyvision.types import Detection, DetectionEvent, FramePacket
from safetyvision.zones.distance import DistanceZoneStrategy


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
    camera: CameraConfig | None = None,
) -> str:
    """Classify a detection into a horizontal band zone by footpoint Y."""
    if frame_h <= 0:
        return ""
    foot_y = float(det.y2) / frame_h
    foot_y = min(max(foot_y, 0.0), 1.0)
    yellow_start_y, red_start_y = get_effective_zone_thresholds(cfg, camera)

    if foot_y >= red_start_y:
        return "danger"
    if foot_y >= yellow_start_y:
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
        self._camera_by_id = {camera.id: camera for camera in cfg.input.cameras}
        self._distance_strategies: dict[str, DistanceZoneStrategy] = {}
        for camera in cfg.input.cameras:
            uses_global_distance = cfg.alert.zone_mode == "distance" and camera.id == "back"
            if camera.mode != "distance" and not uses_global_distance:
                continue
            calibration_path = (
                camera.distance.calibration_path
                if camera.mode == "distance" and camera.distance.calibration_path
                else cfg.alert.calibration_path
            )
            danger_m = (
                camera.distance.danger_distance_m
                if camera.mode == "distance"
                else cfg.alert.danger_threshold_m
            )
            warning_m = (
                camera.distance.warning_distance_m
                if camera.mode == "distance"
                else cfg.alert.warning_threshold_m
            )
            self._distance_strategies[camera.id] = DistanceZoneStrategy(
                calibration_path=calibration_path,
                danger_m=danger_m,
                warning_m=warning_m,
                smoothing_frames=cfg.alert.distance_smoothing_frames,
            )

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

    def _classify_zone(
        self,
        camera_id: str,
        camera_cfg: CameraConfig | None,
        dets: list[Detection],
        frame_h: int,
        frame_w: int,
    ) -> tuple[str, float | None]:
        uses_global_distance = self._cfg.alert.zone_mode == "distance" and camera_id == "back"
        if camera_cfg is not None and (camera_cfg.mode == "distance" or uses_global_distance):
            strategy = self._distance_strategies.get(camera_id)
            if strategy is not None:
                result = strategy.classify(dets, frame_h, frame_w)
                return result.zone_level, result.distance_m

        zone_level = ""
        for det in dets:
            zone = _classify_detection_zone(det, frame_h, self._cfg, camera_cfg)
            if zone == "danger":
                return "danger", None
            if zone == "medium":
                zone_level = "medium"
        return zone_level, None

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

            frame_h, frame_w = pkt.frame.shape[:2]
            raw_detected = len(dets) > 0
            smoothed = self._apply_temporal_smoothing(pkt.camera_id, raw_detected)
            max_conf = max((d.confidence for d in dets), default=0.0)
            camera_cfg = self._camera_by_id.get(pkt.camera_id)
            zone_level, distance_m = self._classify_zone(
                pkt.camera_id, camera_cfg, dets, frame_h, frame_w
            )

            event = DetectionEvent(
                timestamp_ns=pkt.timestamp_ns,
                person_detected=smoothed,
                confidence_max=max_conf,
                bbox_count=len(dets),
                zone_level=zone_level,
                camera_id=pkt.camera_id,
                distance_m=distance_m,
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
