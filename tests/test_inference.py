"""Tests for inference preprocessing and postprocessing."""

import numpy as np
import pytest

from safetyvision.types import Detection
from safetyvision.workers.inference import (
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
        assert pad[0] == 0  # no horizontal padding
        assert pad[1] == 80  # vertical padding

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
        result = _nms([d], 0.5)
        assert len(result) == 1

    def test_suppresses_overlapping(self):
        d1 = Detection(0, 0, 100, 100, 0.9, 0)
        d2 = Detection(5, 5, 105, 105, 0.7, 0)  # high overlap with d1
        result = _nms([d1, d2], 0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_keeps_non_overlapping(self):
        d1 = Detection(0, 0, 50, 50, 0.9, 0)
        d2 = Detection(200, 200, 300, 300, 0.8, 0)
        result = _nms([d1, d2], 0.5)
        assert len(result) == 2


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
        iou = _compute_iou(d1, d2)
        assert 0.0 < iou < 1.0


class TestPostprocess:
    def test_yolov8_format(self):
        # (1, 84, N) format: 4 box + 80 classes, 2 detections
        output = np.zeros((1, 84, 2), dtype=np.float32)
        # Det 1: person with high confidence
        output[0, 0, 0] = 320  # cx
        output[0, 1, 0] = 320  # cy
        output[0, 2, 0] = 100  # w
        output[0, 3, 0] = 200  # h
        output[0, 4, 0] = 0.9  # person class score (class 0)

        # Det 2: car with high confidence (should be filtered)
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
        output[0, 4, 0] = 0.2  # below threshold

        dets = _postprocess(output, 0.3, 0.5, person_class_id=0, scale=1.0, pad=(0, 0))
        assert len(dets) == 0

    def test_nx84_layout_with_small_n_is_not_transposed(self):
        # Shape (N, 84) with N < 84 should still be treated as N detections.
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
