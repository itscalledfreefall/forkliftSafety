"""Capture worker – one per camera. RTSP via GStreamer, tagged FramePackets."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from queue import Full, Queue
from typing import Optional

import cv2
from loguru import logger

from safetyvision.config import CameraConfig, SafetyVisionConfig
from safetyvision.types import FramePacket


def _pin_to_cores(cores: list[int]) -> None:
    try:
        os.sched_setaffinity(0, set(cores))
    except (AttributeError, OSError):
        logger.warning("CPU pinning unavailable on this platform")


def _build_gst_pipeline(url: str, width: int, height: int) -> str:
    return (
        f"rtspsrc location={url} protocols=tcp latency=50 drop-on-latency=true "
        f"! decodebin ! videoconvert "
        f"! video/x-raw,width={width},height={height} "
        f"! appsink sync=false max-buffers=1 drop=true"
    )


def _open_rtsp(url: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    gst = _build_gst_pipeline(url, width, height)
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap
    logger.warning("GStreamer open failed for {}, falling back to FFmpeg", url)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|framedrop;1|"
        "probesize;32|analyzeduration;0"
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CaptureWorker:
    """Continuously captures frames from a single RTSP camera."""

    def __init__(
        self,
        cfg: SafetyVisionConfig,
        camera: CameraConfig,
        out_queue: Queue,
        stop_event: threading.Event,
        latency_cb=None,
        frame_cb=None,
        drop_cb=None,
    ):
        self._cfg = cfg
        self._camera = camera
        self._out_queue = out_queue
        self._stop = stop_event
        self._latency_cb = latency_cb
        self._frame_cb = frame_cb
        self._drop_cb = drop_cb
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self._frames_dropped = 0
        self._connected = threading.Event()

        self._snapshot_dir = Path(cfg.perf.shm_snapshot_dir)
        self._snapshot_path = self._snapshot_dir / f"frame_{camera.id}.jpg"
        self._snapshot_interval_ns = int(cfg.perf.shm_snapshot_interval_sec * 1e9)
        self._last_snapshot_ns = 0
        try:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Cannot create snapshot dir {}: {}", self._snapshot_dir, e)

    @property
    def camera_id(self) -> str:
        return self._camera.id

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"capture_{self._camera.id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _open(self) -> cv2.VideoCapture:
        return _open_rtsp(
            self._camera.rtsp_url,
            self._cfg.input.width,
            self._cfg.input.height,
            self._cfg.input.target_fps,
        )

    def _reconnect(self) -> None:
        self._connected.clear()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        backoff = 1.0
        max_backoff = self._cfg.health.camera_reconnect_max_backoff_sec
        while not self._stop.is_set():
            logger.info(
                "[{}] Reconnecting RTSP (backoff={:.1f}s)", self._camera.id, backoff
            )
            try:
                self._cap = self._open()
                if self._cap.isOpened():
                    self._connected.set()
                    logger.info("[{}] Camera connected", self._camera.id)
                    return
            except Exception as e:
                logger.error("[{}] Reconnect error: {}", self._camera.id, e)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, max_backoff)

    def _maybe_write_snapshot(self, frame, now_ns: int) -> None:
        if now_ns - self._last_snapshot_ns < self._snapshot_interval_ns:
            return
        self._last_snapshot_ns = now_ns
        try:
            cv2.imwrite(str(self._snapshot_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        except Exception as e:
            logger.debug("[{}] Snapshot write failed: {}", self._camera.id, e)

    def _run(self) -> None:
        _pin_to_cores(self._cfg.perf.capture_cpu_cores)

        self._cap = self._open()
        if not self._cap.isOpened():
            logger.error("[{}] Initial open failed, entering reconnect loop", self._camera.id)
            self._reconnect()
        else:
            self._connected.set()

        while not self._stop.is_set():
            if not self._connected.is_set():
                self._reconnect()
                if self._stop.is_set():
                    break
                continue

            t0 = time.time_ns()
            ret, frame = self._cap.read()
            t1 = time.time_ns()
            if not ret or frame is None:
                logger.warning("[{}] Frame read failed, reconnecting", self._camera.id)
                self._reconnect()
                continue

            if self._latency_cb:
                self._latency_cb((t1 - t0) / 1e6)

            self._seq += 1
            pkt = FramePacket(
                frame=frame,
                timestamp_ns=time.time_ns(),
                camera_id=self._camera.id,
                seq=self._seq,
            )

            try:
                while not self._out_queue.empty():
                    try:
                        self._out_queue.get_nowait()
                        self._frames_dropped += 1
                        if self._drop_cb:
                            self._drop_cb()
                    except Exception:
                        break
                self._out_queue.put_nowait(pkt)
                if self._frame_cb:
                    self._frame_cb()
            except Full:
                self._frames_dropped += 1
                if self._drop_cb:
                    self._drop_cb()

            self._maybe_write_snapshot(frame, pkt.timestamp_ns)

        if self._cap is not None:
            self._cap.release()
            logger.info(
                "[{}] Capture stopped (dropped {} frames)",
                self._camera.id,
                self._frames_dropped,
            )
