"""Configuration loader with schema validation and defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class CameraZoneConfig:
    """Optional per-camera zone thresholds.

    When omitted, the camera falls back to the global alert thresholds.
    """

    yellow_start_y: Optional[float] = None
    red_start_y: Optional[float] = None


@dataclass
class CameraDistanceConfig:
    """Placeholder per-camera distance configuration for rear camera mode."""

    warning_distance_m: float = 2.0
    danger_distance_m: float = 1.0
    calibration_path: str = ""


@dataclass
class CameraConfig:
    """A single RTSP camera source feeding the detection pipeline."""

    id: str = "default"
    rtsp_url: str = ""
    rtsp_url_main: str = ""  # optional high-res preview for the web UI
    mode: str = "zone"  # "zone" | "distance"
    zone: CameraZoneConfig = field(default_factory=CameraZoneConfig)
    distance: CameraDistanceConfig = field(default_factory=CameraDistanceConfig)


@dataclass
class InputConfig:
    """Capture-side settings. RTSP-only; Hailo target has no USB path."""

    cameras: List[CameraConfig] = field(default_factory=list)
    width: int = 640
    height: int = 480
    target_fps: int = 15


@dataclass
class ModelConfig:
    """Hailo-only inference configuration."""

    runtime: str = "hailo"
    path_hef: str = "/usr/share/hailo-models/yolov6n_h8l.hef"
    conf_threshold: float = 0.45
    iou_threshold: float = 0.50
    person_class_id: int = 0


@dataclass
class AlertConfig:
    siren_wav: str = "assets/audio/siren.wav"
    voice_wav: str = "assets/audio/warning_voice.wav"
    # Horizontal band zone cut lines (normalized Y, 0=top, 1=bottom).
    yellow_start_y: float = 0.33
    red_start_y: float = 0.66
    min_alert_confidence: float = 0.60
    always_announce_person: bool = False
    repeat_interval_sec: float = 0.75
    min_clear_sec: float = 3.0


@dataclass
class PerfConfig:
    max_queue_size: int = 1
    capture_cpu_cores: List[int] = field(default_factory=lambda: [0])
    inference_cpu_cores: List[int] = field(default_factory=lambda: [1, 2, 3])
    temporal_smoothing_frames: int = 3
    shm_snapshot_dir: str = "/dev/shm/safetyvision"
    shm_snapshot_interval_sec: float = 1.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    json_output: bool = True
    log_dir: str = "/var/log/safetyvision"
    max_size_mb: int = 50


@dataclass
class HealthConfig:
    heartbeat_interval_sec: float = 1.0
    camera_reconnect_max_backoff_sec: float = 30.0


@dataclass
class SafetyVisionConfig:
    input: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    health: HealthConfig = field(default_factory=HealthConfig)


def _merge(dc_class, data: dict):
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    if data is None:
        return dc_class()
    valid = {f.name for f in dc_class.__dataclass_fields__.values()}
    return dc_class(**{k: v for k, v in data.items() if k in valid})


def _parse_cameras(raw_input: dict) -> List[CameraConfig]:
    raw_list = raw_input.get("cameras") or []
    cams: List[CameraConfig] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        cam = _merge(CameraConfig, entry)
        cam.zone = _merge(CameraZoneConfig, entry.get("zone"))
        cam.distance = _merge(CameraDistanceConfig, entry.get("distance"))
        cams.append(cam)
    return cams


def get_effective_zone_thresholds(
    cfg: "SafetyVisionConfig", camera: CameraConfig | None = None
) -> tuple[float, float]:
    """Return zone cut lines for a camera, falling back to global alert defaults."""
    yellow = cfg.alert.yellow_start_y
    red = cfg.alert.red_start_y
    if camera is not None:
        if camera.zone.yellow_start_y is not None:
            yellow = camera.zone.yellow_start_y
        if camera.zone.red_start_y is not None:
            red = camera.zone.red_start_y
    return yellow, red


def load_config(path: str | Path | None = None) -> SafetyVisionConfig:
    """Load and validate configuration from YAML file."""
    if path is None:
        path = os.environ.get("SAFETYVISION_CONFIG", "config/safetyvision.yaml")
    path = Path(path)

    raw: dict = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    raw_input = raw.get("input") or {}
    input_cfg = _merge(InputConfig, raw_input)
    input_cfg.cameras = _parse_cameras(raw_input)

    cfg = SafetyVisionConfig(
        input=input_cfg,
        model=_merge(ModelConfig, raw.get("model")),
        alert=_merge(AlertConfig, raw.get("alert")),
        perf=_merge(PerfConfig, raw.get("perf")),
        logging=_merge(LoggingConfig, raw.get("logging")),
        health=_merge(HealthConfig, raw.get("health")),
    )
    validate(cfg)
    return cfg


class ConfigError(ValueError):
    """Raised for invalid configuration."""


def validate(cfg: SafetyVisionConfig) -> None:
    """Raise ConfigError on invalid values."""
    if not cfg.input.cameras:
        raise ConfigError("input.cameras must contain at least one entry")

    ids: set[str] = set()
    for cam in cfg.input.cameras:
        if not cam.id:
            raise ConfigError("camera entries require a non-empty 'id'")
        if cam.id in ids:
            raise ConfigError(f"duplicate camera id: '{cam.id}'")
        ids.add(cam.id)
        if not cam.rtsp_url:
            raise ConfigError(f"camera '{cam.id}' requires rtsp_url")
        if cam.mode not in {"zone", "distance"}:
            raise ConfigError(f"camera '{cam.id}' mode must be 'zone' or 'distance'")

        cam_yellow, cam_red = get_effective_zone_thresholds(cfg, cam)
        if not 0.0 < cam_yellow < 1.0:
            raise ConfigError(f"camera '{cam.id}' yellow_start_y must be between 0 and 1")
        if not 0.0 < cam_red < 1.0:
            raise ConfigError(f"camera '{cam.id}' red_start_y must be between 0 and 1")
        if cam_yellow >= cam_red:
            raise ConfigError(
                f"camera '{cam.id}' yellow_start_y must be less than red_start_y"
            )

        if cam.distance.warning_distance_m <= 0:
            raise ConfigError(f"camera '{cam.id}' warning_distance_m must be positive")
        if cam.distance.danger_distance_m <= 0:
            raise ConfigError(f"camera '{cam.id}' danger_distance_m must be positive")
        if cam.distance.warning_distance_m <= cam.distance.danger_distance_m:
            raise ConfigError(
                f"camera '{cam.id}' warning_distance_m must be greater than danger_distance_m"
            )

    if cfg.model.runtime != "hailo":
        raise ConfigError(
            f"model.runtime must be 'hailo' (CPU runtimes removed), got '{cfg.model.runtime}'"
        )
    if not cfg.model.path_hef:
        raise ConfigError("model.path_hef is required")

    if not 0.0 < cfg.model.conf_threshold < 1.0:
        raise ConfigError("model.conf_threshold must be between 0 and 1")
    if not 0.0 < cfg.model.iou_threshold < 1.0:
        raise ConfigError("model.iou_threshold must be between 0 and 1")

    if not 0.0 < cfg.alert.yellow_start_y < 1.0:
        raise ConfigError("alert.yellow_start_y must be between 0 and 1")
    if not 0.0 < cfg.alert.red_start_y < 1.0:
        raise ConfigError("alert.red_start_y must be between 0 and 1")
    if cfg.alert.yellow_start_y >= cfg.alert.red_start_y:
        raise ConfigError("alert.yellow_start_y must be less than alert.red_start_y")
    if not 0.0 < cfg.alert.min_alert_confidence < 1.0:
        raise ConfigError("alert.min_alert_confidence must be between 0 and 1")
    if cfg.alert.repeat_interval_sec <= 0:
        raise ConfigError("alert.repeat_interval_sec must be positive")
    if cfg.alert.min_clear_sec <= 0:
        raise ConfigError("alert.min_clear_sec must be positive")
    if cfg.perf.max_queue_size < 1:
        raise ConfigError("perf.max_queue_size must be >= 1")
