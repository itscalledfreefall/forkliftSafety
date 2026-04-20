"""Tests for configuration loading and validation (Hailo-only schema)."""

import pytest
import yaml

from safetyvision.config import (
    CameraConfig,
    ConfigError,
    SafetyVisionConfig,
    load_config,
    validate,
)


def _valid_cfg() -> SafetyVisionConfig:
    """A minimally-valid config for mutation tests."""
    cfg = SafetyVisionConfig()
    cfg.input.cameras = [CameraConfig(id="back", rtsp_url="rtsp://cam:554/live")]
    return cfg


class TestLoadConfig:
    def test_loads_yaml_with_cameras_list(self, tmp_path):
        data = {
            "input": {
                "cameras": [
                    {
                        "id": "back",
                        "rtsp_url": "rtsp://cam:554/sub",
                        "rtsp_url_main": "rtsp://cam:554/main",
                        "mode": "distance",
                        "zone": {
                            "yellow_start_y": 0.25,
                            "red_start_y": 0.70,
                        },
                        "distance": {
                            "warning_distance_m": 2.5,
                            "danger_distance_m": 1.2,
                            "calibration_path": "config/calibration/back.yaml",
                        },
                    }
                ],
                "target_fps": 20,
            },
            "model": {
                "runtime": "hailo",
                "path_hef": "/usr/share/hailo-models/yolov6n_h8l.hef",
                "conf_threshold": 0.6,
            },
            "alert": {"repeat_interval_sec": 2.0},
        }
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_config(p)
        assert len(cfg.input.cameras) == 1
        assert cfg.input.cameras[0].id == "back"
        assert cfg.input.cameras[0].rtsp_url == "rtsp://cam:554/sub"
        assert cfg.input.cameras[0].rtsp_url_main == "rtsp://cam:554/main"
        assert cfg.input.cameras[0].mode == "distance"
        assert cfg.input.cameras[0].zone.yellow_start_y == 0.25
        assert cfg.input.cameras[0].zone.red_start_y == 0.70
        assert cfg.input.cameras[0].distance.warning_distance_m == 2.5
        assert cfg.input.cameras[0].distance.danger_distance_m == 1.2
        assert cfg.input.cameras[0].distance.calibration_path == "config/calibration/back.yaml"
        assert cfg.input.target_fps == 20
        assert cfg.model.conf_threshold == 0.6
        assert cfg.alert.repeat_interval_sec == 2.0

    def test_loads_multiple_cameras(self, tmp_path):
        data = {
            "input": {
                "cameras": [
                    {"id": "back", "rtsp_url": "rtsp://a:554/live"},
                    {"id": "front", "rtsp_url": "rtsp://b:554/live"},
                ]
            },
        }
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_config(p)
        assert [c.id for c in cfg.input.cameras] == ["back", "front"]

    def test_ignores_unknown_keys(self, tmp_path):
        data = {
            "input": {
                "cameras": [{"id": "back", "rtsp_url": "rtsp://x/y", "extra": "value"}],
                "unknown_key": "value",
            },
        }
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_config(p)
        assert cfg.input.cameras[0].id == "back"


class TestValidation:
    def test_empty_cameras_list_rejected(self):
        cfg = SafetyVisionConfig()
        with pytest.raises(ConfigError, match="cameras"):
            validate(cfg)

    def test_camera_requires_rtsp_url(self):
        cfg = SafetyVisionConfig()
        cfg.input.cameras = [CameraConfig(id="back", rtsp_url="")]
        with pytest.raises(ConfigError, match="rtsp_url"):
            validate(cfg)

    def test_camera_requires_id(self):
        cfg = SafetyVisionConfig()
        cfg.input.cameras = [CameraConfig(id="", rtsp_url="rtsp://x/y")]
        with pytest.raises(ConfigError, match="id"):
            validate(cfg)

    def test_duplicate_camera_ids_rejected(self):
        cfg = SafetyVisionConfig()
        cfg.input.cameras = [
            CameraConfig(id="back", rtsp_url="rtsp://a/1"),
            CameraConfig(id="back", rtsp_url="rtsp://b/1"),
        ]
        with pytest.raises(ConfigError, match="duplicate camera id"):
            validate(cfg)

    def test_runtime_must_be_hailo(self):
        cfg = _valid_cfg()
        cfg.model.runtime = "onnxruntime"
        with pytest.raises(ConfigError, match="model.runtime"):
            validate(cfg)

    def test_camera_mode_must_be_zone_or_distance(self):
        cfg = _valid_cfg()
        cfg.input.cameras[0].mode = "sensor"
        with pytest.raises(ConfigError, match="mode must be 'zone' or 'distance'"):
            validate(cfg)

    def test_path_hef_required(self):
        cfg = _valid_cfg()
        cfg.model.path_hef = ""
        with pytest.raises(ConfigError, match="path_hef"):
            validate(cfg)

    def test_conf_threshold_range(self):
        cfg = _valid_cfg()
        cfg.model.conf_threshold = 1.5
        with pytest.raises(ConfigError, match="conf_threshold"):
            validate(cfg)

    def test_zero_repeat_interval(self):
        cfg = _valid_cfg()
        cfg.alert.repeat_interval_sec = 0
        with pytest.raises(ConfigError, match="repeat_interval"):
            validate(cfg)

    def test_yellow_start_y_out_of_range(self):
        cfg = _valid_cfg()
        cfg.alert.yellow_start_y = 0.0
        with pytest.raises(ConfigError, match="yellow_start_y"):
            validate(cfg)

    def test_red_start_y_out_of_range(self):
        cfg = _valid_cfg()
        cfg.alert.red_start_y = 1.0
        with pytest.raises(ConfigError, match="red_start_y"):
            validate(cfg)

    def test_yellow_must_be_less_than_red(self):
        cfg = _valid_cfg()
        cfg.alert.yellow_start_y = 0.70
        cfg.alert.red_start_y = 0.50
        with pytest.raises(ConfigError, match="yellow_start_y.*less than.*red_start_y"):
            validate(cfg)

    def test_equal_cut_lines_invalid(self):
        cfg = _valid_cfg()
        cfg.alert.yellow_start_y = 0.50
        cfg.alert.red_start_y = 0.50
        with pytest.raises(ConfigError, match="yellow_start_y"):
            validate(cfg)

    def test_invalid_min_alert_confidence(self):
        cfg = _valid_cfg()
        cfg.alert.min_alert_confidence = 1.2
        with pytest.raises(ConfigError, match="min_alert_confidence"):
            validate(cfg)

    def test_invalid_camera_zone_override(self):
        cfg = _valid_cfg()
        cfg.input.cameras[0].zone.yellow_start_y = 0.8
        cfg.input.cameras[0].zone.red_start_y = 0.4
        with pytest.raises(ConfigError, match="camera 'back' yellow_start_y"):
            validate(cfg)

    def test_invalid_camera_distance_threshold_order(self):
        cfg = _valid_cfg()
        cfg.input.cameras[0].mode = "distance"
        cfg.input.cameras[0].distance.warning_distance_m = 1.0
        cfg.input.cameras[0].distance.danger_distance_m = 1.0
        with pytest.raises(ConfigError, match="warning_distance_m must be greater"):
            validate(cfg)

    def test_valid_config_passes(self):
        validate(_valid_cfg())

    def test_custom_band_lines(self):
        cfg = _valid_cfg()
        cfg.alert.yellow_start_y = 0.40
        cfg.alert.red_start_y = 0.75
        validate(cfg)
