"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from safetyvision.config import ConfigError, SafetyVisionConfig, load_config, validate


class TestLoadConfig:
    def test_defaults_when_no_file(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.input.mode == "usb"
        assert cfg.model.runtime == "onnxruntime"
        assert cfg.model.conf_threshold == 0.45
        assert cfg.perf.max_queue_size == 1

    def test_loads_yaml(self, tmp_path):
        data = {
            "input": {"mode": "rtsp", "rtsp_url": "rtsp://cam:554/live"},
            "model": {"conf_threshold": 0.6},
            "alert": {"repeat_interval_sec": 10.0},
        }
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_config(p)
        assert cfg.input.mode == "rtsp"
        assert cfg.input.rtsp_url == "rtsp://cam:554/live"
        assert cfg.model.conf_threshold == 0.6
        assert cfg.alert.repeat_interval_sec == 10.0

    def test_ignores_unknown_keys(self, tmp_path):
        data = {"input": {"mode": "usb", "unknown_key": "value"}}
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_config(p)
        assert cfg.input.mode == "usb"


class TestValidation:
    def test_invalid_mode(self):
        cfg = SafetyVisionConfig()
        cfg.input.mode = "invalid"
        with pytest.raises(ConfigError, match="input.mode"):
            validate(cfg)

    def test_rtsp_requires_url(self):
        cfg = SafetyVisionConfig()
        cfg.input.mode = "rtsp"
        cfg.input.rtsp_url = ""
        with pytest.raises(ConfigError, match="rtsp_url"):
            validate(cfg)

    def test_invalid_runtime(self):
        cfg = SafetyVisionConfig()
        cfg.model.runtime = "tensorrt"
        with pytest.raises(ConfigError, match="model.runtime"):
            validate(cfg)

    def test_onnxruntime_requires_onnx_path(self):
        cfg = SafetyVisionConfig()
        cfg.model.runtime = "onnxruntime"
        cfg.model.path_onnx = ""
        with pytest.raises(ConfigError, match="path_onnx"):
            validate(cfg)

    def test_openvino_requires_some_model_path(self):
        cfg = SafetyVisionConfig()
        cfg.model.runtime = "openvino"
        cfg.model.path_onnx = ""
        cfg.model.path_openvino = ""
        with pytest.raises(ConfigError, match="path_openvino"):
            validate(cfg)

    def test_ultralytics_requires_pt_path(self):
        cfg = SafetyVisionConfig()
        cfg.model.runtime = "ultralytics"
        cfg.model.path_pt = ""
        with pytest.raises(ConfigError, match="path_pt"):
            validate(cfg)

    def test_conf_threshold_range(self):
        cfg = SafetyVisionConfig()
        cfg.model.conf_threshold = 1.5
        with pytest.raises(ConfigError, match="conf_threshold"):
            validate(cfg)

    def test_zero_repeat_interval(self):
        cfg = SafetyVisionConfig()
        cfg.alert.repeat_interval_sec = 0
        with pytest.raises(ConfigError, match="repeat_interval"):
            validate(cfg)

    def test_invalid_hysteresis_ratio(self):
        cfg = SafetyVisionConfig()
        cfg.alert.zone_hysteresis_ratio = 0.7
        with pytest.raises(ConfigError, match="zone_hysteresis_ratio"):
            validate(cfg)

    def test_invalid_smoothing_alpha(self):
        cfg = SafetyVisionConfig()
        cfg.alert.distance_smoothing_alpha = 0
        with pytest.raises(ConfigError, match="distance_smoothing_alpha"):
            validate(cfg)

    def test_invalid_min_alert_confidence(self):
        cfg = SafetyVisionConfig()
        cfg.alert.min_alert_confidence = 1.2
        with pytest.raises(ConfigError, match="min_alert_confidence"):
            validate(cfg)

    def test_zone_polygons_require_points_when_enabled(self):
        cfg = SafetyVisionConfig()
        cfg.alert.use_zone_polygons = True
        cfg.alert.danger_zone_polygon = []
        cfg.alert.medium_zone_polygon = []
        with pytest.raises(ConfigError, match="use_zone_polygons"):
            validate(cfg)

    def test_invalid_zone_polygon_point(self):
        cfg = SafetyVisionConfig()
        cfg.alert.danger_zone_polygon = [[0.1, 0.2], [0.3], [0.5, 0.6]]
        with pytest.raises(ConfigError, match="danger_zone_polygon"):
            validate(cfg)

    def test_valid_config_passes(self):
        cfg = SafetyVisionConfig()
        validate(cfg)  # should not raise
