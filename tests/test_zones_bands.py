"""Tests for BandZoneStrategy."""

import pytest

from safetyvision.types import Detection
from safetyvision.zones.bands import BandZoneStrategy
from safetyvision.zones.base import ZoneResult


def _det(y2: float) -> Detection:
    return Detection(x1=0.0, y1=0.0, x2=10.0, y2=y2, confidence=0.9, class_id=0)


class TestBandZoneStrategy:
    def setup_method(self):
        # Match the YAML defaults from config/safetyvision.yaml.
        self.strat = BandZoneStrategy(yellow_start_y=0.29, red_start_y=0.78)
        self.frame_h = 480
        self.frame_w = 640

    def test_no_detections_returns_green(self):
        result = self.strat.classify([], self.frame_h, self.frame_w)
        assert result == ZoneResult(zone_level="", distance_m=None)

    def test_zero_height_frame_returns_green(self):
        result = self.strat.classify([_det(100.0)], 0, self.frame_w)
        assert result.zone_level == ""

    def test_footpoint_above_yellow_is_green(self):
        # foot_y = 100/480 = 0.208 < 0.29
        result = self.strat.classify([_det(100.0)], self.frame_h, self.frame_w)
        assert result.zone_level == ""
        assert result.distance_m is None

    def test_footpoint_in_yellow_band_is_medium(self):
        # foot_y = 200/480 = 0.417, between 0.29 and 0.78
        result = self.strat.classify([_det(200.0)], self.frame_h, self.frame_w)
        assert result.zone_level == "medium"

    def test_footpoint_in_red_band_is_danger(self):
        # foot_y = 400/480 = 0.833 >= 0.78
        result = self.strat.classify([_det(400.0)], self.frame_h, self.frame_w)
        assert result.zone_level == "danger"

    def test_threshold_inclusive_at_red(self):
        # foot_y == red_start_y exactly
        red_y_pixel = 0.78 * self.frame_h
        result = self.strat.classify([_det(red_y_pixel)], self.frame_h, self.frame_w)
        assert result.zone_level == "danger"

    def test_threshold_inclusive_at_yellow(self):
        yellow_y_pixel = 0.29 * self.frame_h
        result = self.strat.classify([_det(yellow_y_pixel)], self.frame_h, self.frame_w)
        assert result.zone_level == "medium"

    def test_multi_person_max_risk_wins(self):
        # green + medium + danger together → danger
        dets = [_det(50.0), _det(200.0), _det(400.0)]
        result = self.strat.classify(dets, self.frame_h, self.frame_w)
        assert result.zone_level == "danger"

    def test_multi_person_medium_when_no_danger(self):
        dets = [_det(50.0), _det(200.0)]
        result = self.strat.classify(dets, self.frame_h, self.frame_w)
        assert result.zone_level == "medium"

    def test_distance_m_always_none(self):
        result = self.strat.classify([_det(400.0)], self.frame_h, self.frame_w)
        assert result.distance_m is None

    def test_footpoint_clamped_above_one(self):
        # y2 beyond frame_h → clamped to 1.0 → still in danger band
        result = self.strat.classify([_det(10000.0)], self.frame_h, self.frame_w)
        assert result.zone_level == "danger"
