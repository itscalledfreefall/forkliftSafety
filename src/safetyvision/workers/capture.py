"""Capture worker – grabs frames from RTSP or USB with minimal latency."""

from __future__ import annotations

import os
import threading
import time
from queue import Full, Queue
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import FramePacket


def _pin_to_cores(cores: list[int]) -> None:
    """Best-effort CPU affinity pinning (Linux only)."""
    try:
        os.sched_setaffinity(0, set(cores))
        logger.debug("Capture thread pinned to cores {}", cores)
    except (AttributeError, OSError):
        logger.warning("CPU pinning unavailable on this platform")


def _build_gst_pipeline(cfg: SafetyVisionConfig) -> str:
    """Build a GStreamer pipeline string for low-latency RTSP capture."""
    return (
        f"rtspsrc location={cfg.input.rtsp_url} latency=0 drop-on-latency=true "
        f"! decodebin ! videoconvert "
        f"! video/x-raw,width={cfg.input.width},height={cfg.input.height} "
        f"! appsink max-buffers=1 drop=true"
    )


def _open_usb(cfg: SafetyVisionConfig) -> cv2.VideoCapture:
    """Open a USB/V4L2 camera with explicit settings."""
    cap = cv2.VideoCapture(cfg.input.usb_device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.input.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.input.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.input.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _open_rtsp(cfg: SafetyVisionConfig) -> cv2.VideoCapture:
    """Open an RTSP stream via GStreamer pipeline."""
    gst = _build_gst_pipeline(cfg)
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        logger.warning("GStreamer RTSP failed, falling back to FFmpeg backend")
        cap = cv2.VideoCapture(cfg.input.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CaptureWorker:
    """Continuously captures frames and pushes the latest to a bounded queue."""

    def __init__(
        self,
        cfg: SafetyVisionConfig,
        out_queue: Queue,
        stop_event: threading.Event,
        latency_cb=None,
        frame_cb=None,
        drop_cb=None,
    ):
        self._cfg = cfg
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

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="capture_worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _open_camera(self) -> cv2.VideoCapture:
        if self._cfg.input.mode == "rtsp":
            return _open_rtsp(self._cfg)
        return _open_usb(self._cfg)

    def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        self._connected.clear()
        if self._cap:
            self._cap.release()
            self._cap = None

        backoff = 1.0
        max_backoff = self._cfg.health.camera_reconnect_max_backoff_sec
        while not self._stop.is_set():
            logger.info("Attempting camera reconnect (backoff={:.1f}s)", backoff)
            try:
                self._cap = self._open_camera()
                if self._cap.isOpened():
                    self._connected.set()
                    logger.info("Camera reconnected")
                    return
            except Exception as e:
                logger.error("Reconnect failed: {}", e)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, max_backoff)

    def _run(self) -> None:
        _pin_to_cores(self._cfg.perf.capture_cpu_cores)

        self._cap = self._open_camera()
        if not self._cap.isOpened():
            logger.error("Initial camera open failed, entering reconnect loop")
            self._reconnect()

        if self._cap and self._cap.isOpened():
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
                logger.warning("Frame read failed, reconnecting")
                self._reconnect()
                continue
            if self._latency_cb:
                self._latency_cb((t1 - t0) / 1e6)

            ts = time.time_ns()
            self._seq += 1
            pkt = FramePacket(
                frame=frame,
                timestamp_ns=ts,
                source_id=self._cfg.input.mode,
                seq=self._seq,
            )

            try:
                # Non-blocking put: drop oldest if full
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

        if self._cap:
            self._cap.release()
            logger.info("Capture worker stopped (dropped {} frames)", self._frames_dropped)
