"""Tests for DistanceZoneStrategy."""

import json
import math

import pytest

from safetyvision.types import Detection
from safetyvision.zones.distance import DistanceZoneStrategy, _Homography


# A synthetic calibration that maps the pixel square [0..100, 0..100] linearly
# to the meter square [-1..1, -1..1] (Y inverted so up = +Y).
#
#   pixel (50, 50)  -> meters (0, 0)         distance 0
#   pixel (50, 0)   -> meters (0, 1)         distance 1
#   pixel (100, 50) -> meters (1, 0)         distance 1
#   pixel (100, 100)-> meters (1, -1)        distance sqrt(2)
SYNTHETIC_CAL = {
    "camera_id": "back",
    "source_points": [[0, 0], [100, 0], [0, 100], [100, 100]],
    "target_points": [[-1, 1], [1, 1], [-1, -1], [1, -1]],
    "frame_width": 100,
    "frame_height": 100,
    "created_at": "2026-04-29T00:00:00Z",
}


def _write_cal(tmp_path, data=SYNTHETIC_CAL):
    p = tmp_path / "calibration_back.json"
    p.write_text(json.dumps(data))
    return str(p)


def _det_at_foot(cx: float, foot_y: float) -> Detection:
    return Detection(
        x1=cx - 5.0, y1=foot_y - 50.0, x2=cx + 5.0, y2=foot_y,
        confidence=0.9, class_id=0,
    )


class TestHomographyShim:
    def test_identity_like_transform(self, tmp_path):
        import numpy as np

        h = _Homography(
            source=np.array(SYNTHETIC_CAL["source_points"], dtype=np.float32),
            target=np.array(SYNTHETIC_CAL["target_points"], dtype=np.float32),
        )
        out = h.transform_points(np.array([[50.0, 50.0]], dtype=np.float32))
        assert out.shape == (1, 2)
        assert math.isclose(out[0, 0], 0.0, abs_tol=1e-6)
        assert math.isclose(out[0, 1], 0.0, abs_tol=1e-6)

    def test_degenerate_raises(self):
        import numpy as np

        # All four points collinear → findHomography returns None
        with pytest.raises(ValueError, match="degenerate"):
            _Homography(
                source=np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32),
                target=np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32),
            )


class TestDistanceZoneStrategy:
    def test_missing_calibration_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DistanceZoneStrategy(
                calibration_path=str(tmp_path / "missing.json"),
                danger_m=1.0, warning_m=5.0,
            )

    def test_wrong_point_count_raises(self, tmp_path):
        bad = {**SYNTHETIC_CAL, "source_points": [[0, 0], [10, 0]],
               "target_points": [[0, 0], [1, 0]]}
        path = _write_cal(tmp_path, bad)
        with pytest.raises(ValueError, match="exactly 4"):
            DistanceZoneStrategy(calibration_path=path, danger_m=1.0, warning_m=5.0)

    def test_no_detections_returns_green(self, tmp_path):
        strat = DistanceZoneStrategy(
            calibration_path=_write_cal(tmp_path),
            danger_m=1.0, warning_m=5.0,
        )
        result = strat.classify([], 100, 100)
        assert result.zone_level == ""
        assert result.distance_m is None

    def test_origin_footpoint_distance_zero_is_danger(self, tmp_path):
        strat = DistanceZoneStrategy(
            calibration_path=_write_cal(tmp_path),
            danger_m=1.0, warning_m=5.0,
        )
        # Footpoint at pixel (50, 50) → meters (0, 0)
        result = strat.classify([_det_at_foot(50.0, 50.0)], 100, 100)
        assert result.zone_level == "danger"
        assert math.isclose(result.distance_m, 0.0, abs_tol=1e-5)

    def test_footpoint_inside_danger_band(self, tmp_path):
        # Pixel (50, 0) → (0, 1) meters → distance ≈ 1.0
        # With danger_m=1.5 the boundary is well clear of float noise.
        strat = DistanceZoneStrategy(
            calibration_path=_write_cal(tmp_path),
            danger_m=1.5, warning_m=5.0,
        )
        result = strat.classify([_det_at_foot(50.0, 0.0)], 100, 100)
        assert result.zone_level == "danger"
        assert math.isclose(result.distance_m, 1.0, abs_tol=1e-5)

    def test_footpoint_in_warning_band(self, tmp_path):
        # Use thresholds where 1.0m falls into the warning band
        strat = DistanceZoneStrategy(
            calibration_path=_write_cal(tmp_path),
            danger_m=0.5, warning_m=5.0,
        )
        # Pixel (50, 0) → distance 1.0
        result = strat.classify([_det_at_foot(50.0, 0.0)], 100, 100)
        assert result.zone_level == "medium"

    def test_far_footpoint_is_green(self, tmp_path):
        # warning_m=0.5; 1.0m distance → beyond warning → green
        strat = DistanceZoneStrategy(
            calibration_path=_write_cal(tmp_path),
            danger_m=0.1, warning_m=0.5,
        )
        result = strat.classify([_det_at_foot(50.0, 0.0)], 100, 100)
        assert result.zone_level == ""

    def test_multi_person_min_distance_wins(self, tmp_path):
        strat = DistanceZoneStrategy(
            calibration_path=_write_cal(tmp_path),
            danger_m=1.0, warning_m=5.0,
        )
        # Person A at distance ~sqrt(2), Person B at distance 0
        a = _det_at_foot(100.0, 100.0)
        b = _det_at_foot(50.0, 50.0)
        result = strat.classify([a, b], 100, 100)
        assert math.isclose(result.distance_m, 0.0, abs_tol=1e-5)
        assert result.zone_level == "danger"


class TestSmoothing:
    def test_median_dampens_outlier(self, tmp_path):
        strat = DistanceZoneStrategy(
            calibration_path=_write_cal(tmp_path),
            danger_m=1.0, warning_m=5.0,
            smoothing_frames=3,
        )
        # Frame 1: pixel (50, 0) → distance 1.0
        r1 = strat.classify([_det_at_foot(50.0, 0.0)], 100, 100)
        assert math.isclose(r1.distance_m, 1.0, abs_tol=1e-5)

        # Frame 2: same → buffer [1.0, 1.0], median 1.0
        r2 = strat.classify([_det_at_foot(50.0, 0.0)], 100, 100)
        assert math.isclose(r2.distance_m, 1.0, abs_tol=1e-5)

        # Frame 3: outlier far away (~sqrt(2)) → buffer [1.0, 1.0, 1.414],
        # median 1.0 (outlier dampened)
        r3 = strat.classify([_det_at_foot(100.0, 100.0)], 100, 100)
        assert math.isclose(r3.distance_m, 1.0, abs_tol=1e-5)

    def test_buffer_resets_on_no_detection(self, tmp_path):
        strat = DistanceZoneStrategy(
            calibration_path=_write_cal(tmp_path),
            danger_m=1.0, warning_m=5.0,
            smoothing_frames=3,
        )
        # Prime buffer with three close readings
        for _ in range(3):
            strat.classify([_det_at_foot(50.0, 50.0)], 100, 100)

        # No detection → buffer cleared
        empty = strat.classify([], 100, 100)
        assert empty.distance_m is None

        # Next reading is single sample, not median with old close values
        result = strat.classify([_det_at_foot(50.0, 0.0)], 100, 100)
        assert math.isclose(result.distance_m, 1.0, abs_tol=1e-5)

    def test_buffer_capped_at_smoothing_frames(self, tmp_path):
        strat = DistanceZoneStrategy(
            calibration_path=_write_cal(tmp_path),
            danger_m=1.0, warning_m=5.0,
            smoothing_frames=2,
        )
        # Three readings: 0.0, 0.0, 1.0 → buffer drops first → [0.0, 1.0],
        # median = 0.5
        strat.classify([_det_at_foot(50.0, 50.0)], 100, 100)
        strat.classify([_det_at_foot(50.0, 50.0)], 100, 100)
        r3 = strat.classify([_det_at_foot(50.0, 0.0)], 100, 100)
        assert math.isclose(r3.distance_m, 0.5, abs_tol=1e-5)
