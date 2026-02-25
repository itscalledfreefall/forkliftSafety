"""Configuration loader with schema validation and defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class InputConfig:
    mode: str = "usb"
    rtsp_url: str = ""
    usb_device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class ModelConfig:
    path_onnx: str = "models/yolo26n.onnx"
    path_openvino: str = "models/yolo26n_openvino_model/yolo26n.xml"
    path_pt: str = "models/yolo26n.pt"
    runtime: str = "onnxruntime"
    input_size: int = 512
    conf_threshold: float = 0.45
    iou_threshold: float = 0.50
    person_class_id: int = 0


@dataclass
class AlertConfig:
    siren_wav: str = "assets/audio/siren.wav"
    voice_wav: str = "assets/audio/warning_voice.wav"
    use_zone_polygons: bool = False
    danger_zone_polygon: List[List[float]] = field(default_factory=list)
    medium_zone_polygon: List[List[float]] = field(default_factory=list)
    close_area_ratio: float = 0.20
    medium_area_ratio: float = 0.08
    min_alert_confidence: float = 0.60
    zone_hysteresis_ratio: float = 0.02
    distance_smoothing_alpha: float = 0.4
    always_announce_person: bool = False
    repeat_interval_sec: float = 0.75
    min_clear_sec: float = 3.0


@dataclass
class PerfConfig:
    max_queue_size: int = 1
    capture_cpu_cores: List[int] = field(default_factory=lambda: [0])
    inference_cpu_cores: List[int] = field(default_factory=lambda: [1, 2, 3])
    inference_threads: int = 4
    temporal_smoothing_frames: int = 3


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


def load_config(path: str | Path | None = None) -> SafetyVisionConfig:
    """Load and validate configuration from YAML file.

    Falls back to defaults when *path* is None or the file is missing.
    """
    if path is None:
        path = os.environ.get("SAFETYVISION_CONFIG", "config/safetyvision.yaml")
    path = Path(path)

    raw: dict = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    cfg = SafetyVisionConfig(
        input=_merge(InputConfig, raw.get("input")),
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


def _validate_polygon(name: str, polygon: List[List[float]]) -> None:
    if not polygon:
        return
    if len(polygon) < 3:
        raise ConfigError(f"alert.{name} must contain at least 3 points")
    for idx, pt in enumerate(polygon):
        if not isinstance(pt, list) or len(pt) != 2:
            raise ConfigError(f"alert.{name}[{idx}] must be [x, y]")
        x, y = pt
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            raise ConfigError(f"alert.{name}[{idx}] coordinates must be normalized between 0 and 1")


def validate(cfg: SafetyVisionConfig) -> None:
    """Raise ConfigError on invalid values."""
    if cfg.input.mode not in ("rtsp", "usb"):
        raise ConfigError(f"input.mode must be 'rtsp' or 'usb', got '{cfg.input.mode}'")
    if cfg.input.mode == "rtsp" and not cfg.input.rtsp_url:
        raise ConfigError("input.rtsp_url is required when mode is 'rtsp'")
    if cfg.model.runtime not in ("onnxruntime", "openvino", "ultralytics"):
        raise ConfigError(
            "model.runtime must be 'onnxruntime', 'openvino', or 'ultralytics', "
            f"got '{cfg.model.runtime}'"
        )
    if cfg.model.runtime == "onnxruntime" and not cfg.model.path_onnx:
        raise ConfigError("model.path_onnx is required when runtime is 'onnxruntime'")
    if cfg.model.runtime == "openvino" and not (cfg.model.path_openvino or cfg.model.path_onnx):
        raise ConfigError(
            "model.path_openvino or model.path_onnx is required when runtime is 'openvino'"
        )
    if cfg.model.runtime == "ultralytics" and not cfg.model.path_pt:
        raise ConfigError("model.path_pt is required when runtime is 'ultralytics'")
    if not 0.0 < cfg.model.conf_threshold < 1.0:
        raise ConfigError("model.conf_threshold must be between 0 and 1")
    if not 0.0 < cfg.model.iou_threshold < 1.0:
        raise ConfigError("model.iou_threshold must be between 0 and 1")
    if cfg.alert.repeat_interval_sec <= 0:
        raise ConfigError("alert.repeat_interval_sec must be positive")
    if cfg.alert.min_clear_sec <= 0:
        raise ConfigError("alert.min_clear_sec must be positive")
    if not 0.0 < cfg.alert.medium_area_ratio < 1.0:
        raise ConfigError("alert.medium_area_ratio must be between 0 and 1")
    if not 0.0 < cfg.alert.close_area_ratio < 1.0:
        raise ConfigError("alert.close_area_ratio must be between 0 and 1")
    if cfg.alert.close_area_ratio <= cfg.alert.medium_area_ratio:
        raise ConfigError("alert.close_area_ratio must be greater than alert.medium_area_ratio")
    if not 0.0 < cfg.alert.min_alert_confidence < 1.0:
        raise ConfigError("alert.min_alert_confidence must be between 0 and 1")
    if not 0.0 <= cfg.alert.zone_hysteresis_ratio < 0.5:
        raise ConfigError("alert.zone_hysteresis_ratio must be between 0 and 0.5")
    if not 0.0 < cfg.alert.distance_smoothing_alpha <= 1.0:
        raise ConfigError("alert.distance_smoothing_alpha must be between 0 and 1")
    _validate_polygon("danger_zone_polygon", cfg.alert.danger_zone_polygon)
    _validate_polygon("medium_zone_polygon", cfg.alert.medium_zone_polygon)
    if cfg.alert.use_zone_polygons:
        has_danger = len(cfg.alert.danger_zone_polygon) >= 3
        has_medium = len(cfg.alert.medium_zone_polygon) >= 3
        if not (has_danger or has_medium):
            raise ConfigError(
                "alert.use_zone_polygons is true, but no valid zone polygons are configured"
            )
    if cfg.perf.max_queue_size < 1:
        raise ConfigError("perf.max_queue_size must be >= 1")
