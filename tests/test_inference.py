"""Tests for inference preprocessing, postprocessing, and zone classification."""

import numpy as np
import pytest

from safetyvision.types import Detection
from safetyvision.config import SafetyVisionConfig
from safetyvision.workers.inference import (
    _classify_detection_zone,
    _compute_iou,
    _letterbox,
    _nms,
    _postprocess,
    _preprocess,
)


class TestLetterbox:
    def test_square_input(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, scale, pad = _letterbox(frame, 640)
        assert padded.shape == (640, 640, 3)
        assert scale == 1.0
        assert pad == (0, 0)

    def test_landscape_input(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, scale, pad = _letterbox(frame, 640)
        assert padded.shape == (640, 640, 3)
        assert scale == 1.0
        assert pad[0] == 0
        assert pad[1] == 80

    def test_small_input_scales_up(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        padded, scale, pad = _letterbox(frame, 640)
        assert padded.shape == (640, 640, 3)
        assert scale == 2.0


class TestPreprocess:
    def test_output_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blob, scale, pad = _preprocess(frame, 640)
        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32

    def test_normalized_range(self):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        blob, _, _ = _preprocess(frame, 640)
        assert blob.max() <= 1.0
        assert blob.min() >= 0.0


class TestNMS:
    def test_empty_input(self):
        assert _nms([], 0.5) == []

    def test_single_detection(self):
        d = Detection(0, 0, 100, 100, 0.9, 0)
        assert len(_nms([d], 0.5)) == 1

    def test_suppresses_overlapping(self):
        d1 = Detection(0, 0, 100, 100, 0.9, 0)
        d2 = Detection(5, 5, 105, 105, 0.7, 0)
        result = _nms([d1, d2], 0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_keeps_non_overlapping(self):
        d1 = Detection(0, 0, 50, 50, 0.9, 0)
        d2 = Detection(200, 200, 300, 300, 0.8, 0)
        assert len(_nms([d1, d2], 0.5)) == 2


class TestIoU:
    def test_identical_boxes(self):
        d = Detection(0, 0, 100, 100, 0.9, 0)
        assert _compute_iou(d, d) == pytest.approx(1.0)

    def test_no_overlap(self):
        d1 = Detection(0, 0, 50, 50, 0.9, 0)
        d2 = Detection(100, 100, 200, 200, 0.8, 0)
        assert _compute_iou(d1, d2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        d1 = Detection(0, 0, 100, 100, 0.9, 0)
        d2 = Detection(50, 50, 150, 150, 0.8, 0)
        assert 0.0 < _compute_iou(d1, d2) < 1.0


class TestZoneBands:
    """Horizontal band zone classification.

    Default bands: green [0, 0.33), yellow [0.33, 0.66), red [0.66, 1.0]
    Zone is decided by bbox footpoint Y (y2 / frame_h).
    """

    @pytest.fixture
    def cfg(self):
        c = SafetyVisionConfig()
        c.alert.yellow_start_y = 0.33
        c.alert.red_start_y = 0.66
        return c

    def test_green_zone_top(self, cfg):
        """Footpoint in top third => green (no zone)."""
        # y2=100 on 480h frame => foot_y = 100/480 ≈ 0.208
        d = Detection(x1=100, y1=50, x2=200, y2=100, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == ""

    def test_yellow_zone_middle(self, cfg):
        """Footpoint in middle third => medium."""
        # y2=240 on 480h frame => foot_y = 240/480 = 0.50
        d = Detection(x1=100, y1=150, x2=200, y2=240, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == "medium"

    def test_red_zone_bottom(self, cfg):
        """Footpoint in bottom third => danger."""
        # y2=400 on 480h frame => foot_y = 400/480 ≈ 0.833
        d = Detection(x1=100, y1=300, x2=200, y2=400, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == "danger"

    def test_exactly_at_yellow_boundary(self, cfg):
        """Footpoint exactly at yellow_start_y => medium."""
        # foot_y = 0.33 exactly
        y2 = 0.33 * 480  # 158.4
        d = Detection(x1=100, y1=100, x2=200, y2=y2, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == "medium"

    def test_exactly_at_red_boundary(self, cfg):
        """Footpoint exactly at red_start_y => danger."""
        y2 = 0.66 * 480  # 316.8
        d = Detection(x1=100, y1=200, x2=200, y2=y2, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == "danger"

    def test_footpoint_at_very_bottom(self, cfg):
        """Footpoint at frame bottom (y2=frame_h) => danger."""
        d = Detection(x1=100, y1=300, x2=200, y2=480, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == "danger"

    def test_footpoint_at_very_top(self, cfg):
        """Footpoint at frame top (y2≈0) => green."""
        d = Detection(x1=100, y1=0, x2=200, y2=10, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == ""

    def test_custom_cut_lines(self):
        """Custom yellow=0.50, red=0.80."""
        cfg = SafetyVisionConfig()
        cfg.alert.yellow_start_y = 0.50
        cfg.alert.red_start_y = 0.80

        # foot_y = 200/480 ≈ 0.417 => green
        d1 = Detection(x1=0, y1=100, x2=100, y2=200, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d1, 480, cfg) == ""

        # foot_y = 300/480 = 0.625 => medium
        d2 = Detection(x1=0, y1=200, x2=100, y2=300, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d2, 480, cfg) == "medium"

        # foot_y = 450/480 ≈ 0.9375 => danger
        d3 = Detection(x1=0, y1=350, x2=100, y2=450, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d3, 480, cfg) == "danger"

    def test_zero_frame_height(self, cfg):
        """frame_h=0 should return empty string safely."""
        d = Detection(x1=0, y1=0, x2=100, y2=100, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 0, cfg) == ""


class TestPostprocess:
    def test_yolov8_format(self):
        output = np.zeros((1, 84, 2), dtype=np.float32)
        output[0, 0, 0] = 320
        output[0, 1, 0] = 320
        output[0, 2, 0] = 100
        output[0, 3, 0] = 200
        output[0, 4, 0] = 0.9

        output[0, 0, 1] = 100
        output[0, 1, 1] = 100
        output[0, 2, 1] = 50
        output[0, 3, 1] = 50
        output[0, 6, 1] = 0.95  # class 2

        dets = _postprocess(output, 0.3, 0.5, person_class_id=0, scale=1.0, pad=(0, 0))
        assert len(dets) == 1
        assert dets[0].class_id == 0

    def test_filters_below_threshold(self):
        output = np.zeros((1, 84, 1), dtype=np.float32)
        output[0, 0, 0] = 320
        output[0, 1, 0] = 320
        output[0, 2, 0] = 100
        output[0, 3, 0] = 200
        output[0, 4, 0] = 0.2

        dets = _postprocess(output, 0.3, 0.5, person_class_id=0, scale=1.0, pad=(0, 0))
        assert len(dets) == 0

    def test_nx84_layout_with_small_n_is_not_transposed(self):
        output = np.zeros((1, 2, 84), dtype=np.float32)
        output[0, 0, 0] = 320
        output[0, 0, 1] = 320
        output[0, 0, 2] = 100
        output[0, 0, 3] = 200
        output[0, 0, 4] = 0.9

        output[0, 1, 0] = 100
        output[0, 1, 1] = 100
        output[0, 1, 2] = 50
        output[0, 1, 3] = 50
        output[0, 1, 6] = 0.95

        dets = _postprocess(output, 0.3, 0.5, person_class_id=0, scale=1.0, pad=(0, 0))
        assert len(dets) == 1
        assert dets[0].class_id == 0

    def test_nx6_nms_output_format(self):
        output = np.array(
            [
                [10.0, 20.0, 110.0, 220.0, 0.92, 0.0],
                [30.0, 40.0, 130.0, 240.0, 0.95, 2.0],
                [50.0, 60.0, 150.0, 260.0, 0.20, 0.0],
            ],
            dtype=np.float32,
        )
        dets = _postprocess(output, 0.3, 0.5, person_class_id=0, scale=1.0, pad=(0, 0))
        assert len(dets) == 1
        assert dets[0].class_id == 0
        assert dets[0].confidence == pytest.approx(0.92)
