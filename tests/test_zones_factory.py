"""Tests for the zone-strategy factory."""

import json

import pytest

from safetyvision.config import SafetyVisionConfig, AlertConfig
from safetyvision.zones import create_zone_strategy
from safetyvision.zones.bands import BandZoneStrategy
from safetyvision.zones.distance import DistanceZoneStrategy


SYNTHETIC_CAL = {
    "camera_id": "back",
    "source_points": [[0, 0], [100, 0], [0, 100], [100, 100]],
    "target_points": [[-1, 1], [1, 1], [-1, -1], [1, -1]],
    "frame_width": 100,
    "frame_height": 100,
    "created_at": "2026-04-29T00:00:00Z",
}


class TestCreateZoneStrategy:
    def test_bands_mode(self):
        cfg = SafetyVisionConfig()
        cfg.alert.zone_mode = "bands"
        strat = create_zone_strategy(cfg)
        assert isinstance(strat, BandZoneStrategy)

    def test_distance_mode(self, tmp_path):
        cal = tmp_path / "calibration_back.json"
        cal.write_text(json.dumps(SYNTHETIC_CAL))

        cfg = SafetyVisionConfig()
        cfg.alert.zone_mode = "distance"
        cfg.alert.calibration_path = str(cal)
        cfg.alert.danger_threshold_m = 1.0
        cfg.alert.warning_threshold_m = 5.0
        cfg.alert.distance_smoothing_frames = 3

        strat = create_zone_strategy(cfg)
        assert isinstance(strat, DistanceZoneStrategy)

    def test_distance_mode_missing_calibration_raises(self, tmp_path):
        cfg = SafetyVisionConfig()
        cfg.alert.zone_mode = "distance"
        cfg.alert.calibration_path = str(tmp_path / "missing.json")

        with pytest.raises(FileNotFoundError):
            create_zone_strategy(cfg)

    def test_unknown_mode_raises(self):
        cfg = SafetyVisionConfig()
        cfg.alert.zone_mode = "bogus"
        with pytest.raises(ValueError, match="Unknown zone_mode"):
            create_zone_strategy(cfg)
