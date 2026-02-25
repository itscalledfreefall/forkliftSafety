"""Core event types for SafetyVision pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


@dataclass(slots=True)
class FramePacket:
    """A captured frame moving through the pipeline."""

    frame: np.ndarray
    timestamp_ns: int = field(default_factory=time.time_ns)
    source_id: str = ""
    seq: int = 0


@dataclass(slots=True)
class DetectionEvent:
    """Output of inference + zone classification.

    zone_level is the highest-risk horizontal band any detected person's
    footpoint falls into: "danger" | "medium" | "" (green / no person).
    """

    timestamp_ns: int
    person_detected: bool
    confidence_max: float
    bbox_count: int
    zone_level: str = ""          # "danger" | "medium" | ""
    source_id: str = ""


@dataclass(slots=True)
class AlertEvent:
    """Issued when the alert worker acts on a detection."""

    timestamp_ns: int
    trigger_reason: str
    cooldown_active: bool
    sound_key: str = "danger"
    audio_started_ms: float = 0.0


class AlertState(Enum):
    """Alert state-machine states."""

    IDLE = auto()
    TRIGGERED = auto()
    COOLDOWN = auto()


@dataclass(slots=True)
class Detection:
    """Single bounding-box detection."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


@dataclass
class PipelineMetrics:
    """Rolling metrics snapshot."""

    capture_fps: float = 0.0
    inference_fps: float = 0.0
    decision_fps: float = 0.0
    fps: float = 0.0
    capture_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    decision_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    frames_dropped: int = 0
    alert_count: int = 0
    uptime_sec: float = 0.0
