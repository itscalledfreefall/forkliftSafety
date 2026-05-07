"""Tests for the calibration validator."""

import pytest

from safetyvision.web.calibration import (
    CalibrationError,
    CalibrationPayload,
    _validate,
)


def _payload(source, target, w=640, h=480):
    return CalibrationPayload(
        source_points=source,
        target_points=target,
        frame_width=w,
        frame_height=h,
    )


class TestValidate:
    def test_valid_calibration(self):
        # Square pixel patch -> square meter patch
        h = _validate(_payload(
            source=[[100, 100], [500, 100], [100, 400], [500, 400]],
            target=[[-1, 1], [1, 1], [-1, -1], [1, -1]],
        ))
        assert h.shape == (3, 3)

    def test_wrong_source_count(self):
        with pytest.raises(CalibrationError, match="Exactly 4"):
            _validate(_payload(
                source=[[100, 100], [200, 200], [300, 300]],
                target=[[0, 0], [1, 0], [0, 1]],
            ))

    def test_wrong_target_count(self):
        with pytest.raises(CalibrationError, match="Exactly 4"):
            _validate(_payload(
                source=[[100, 100], [200, 100], [100, 200], [200, 200]],
                target=[[0, 0], [1, 0]],
            ))

    def test_source_point_outside_frame(self):
        with pytest.raises(CalibrationError, match="outside"):
            _validate(_payload(
                source=[[100, 100], [700, 100], [100, 400], [500, 400]],
                target=[[-1, 1], [1, 1], [-1, -1], [1, -1]],
                w=640, h=480,
            ))

    def test_negative_source_point_rejected(self):
        with pytest.raises(CalibrationError, match="outside"):
            _validate(_payload(
                source=[[-5, 100], [500, 100], [100, 400], [500, 400]],
                target=[[-1, 1], [1, 1], [-1, -1], [1, -1]],
            ))

    def test_collinear_source_points_rejected(self):
        # All four source points on the same horizontal line
        with pytest.raises(CalibrationError, match="degenerate|near zero|collinear"):
            _validate(_payload(
                source=[[100, 200], [200, 200], [300, 200], [400, 200]],
                target=[[-1, 1], [1, 1], [-1, -1], [1, -1]],
            ))

    def test_collinear_target_points_rejected(self):
        with pytest.raises(CalibrationError, match="degenerate|near zero|collinear"):
            _validate(_payload(
                source=[[100, 100], [500, 100], [100, 400], [500, 400]],
                target=[[0, 0], [1, 0], [2, 0], [3, 0]],
            ))

    def test_duplicate_source_points_rejected(self):
        with pytest.raises(CalibrationError):
            _validate(_payload(
                source=[[100, 100], [100, 100], [100, 400], [500, 400]],
                target=[[-1, 1], [1, 1], [-1, -1], [1, -1]],
            ))

    def test_malformed_source_point(self):
        with pytest.raises(CalibrationError, match=r"\[x, y\]"):
            _validate(_payload(
                source=[[100, 100, 50], [500, 100], [100, 400], [500, 400]],
                target=[[-1, 1], [1, 1], [-1, -1], [1, -1]],
            ))

    def test_source_at_frame_boundary_accepted(self):
        # Inclusive at edges
        h = _validate(_payload(
            source=[[0, 0], [640, 0], [0, 480], [640, 480]],
            target=[[-1, 1], [1, 1], [-1, -1], [1, -1]],
            w=640, h=480,
        ))
        assert h.shape == (3, 3)
