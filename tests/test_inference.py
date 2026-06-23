"""Tests for zone classification and HailoBackend pre/postprocessing."""

import threading
from queue import Queue

import numpy as np
import pytest

from safetyvision.config import CameraConfig, SafetyVisionConfig
from safetyvision.inference.hailo_backend import HailoBackend
from safetyvision.types import Detection
from safetyvision.workers.inference import InferenceWorker, _classify_detection_zone


class TestZoneBands:
    """Horizontal band zone classification.

    Default bands: green [0, 0.33), yellow [0.33, 0.66), red [0.66, 1.0]
    Zone is decided by bbox footpoint Y (y2 / frame_h).
    """

    @pytest.fixture
    def cfg(self):
        c = SafetyVisionConfig()
        c.input.cameras = [CameraConfig(id="back", rtsp_url="rtsp://x/y")]
        c.alert.yellow_start_y = 0.33
        c.alert.red_start_y = 0.66
        return c

    def test_green_zone_top(self, cfg):
        d = Detection(x1=100, y1=50, x2=200, y2=100, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == ""

    def test_yellow_zone_middle(self, cfg):
        d = Detection(x1=100, y1=150, x2=200, y2=240, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == "medium"

    def test_red_zone_bottom(self, cfg):
        d = Detection(x1=100, y1=300, x2=200, y2=400, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == "danger"

    def test_exactly_at_yellow_boundary(self, cfg):
        y2 = 0.33 * 480
        d = Detection(x1=100, y1=100, x2=200, y2=y2, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == "medium"

    def test_exactly_at_red_boundary(self, cfg):
        y2 = 0.66 * 480
        d = Detection(x1=100, y1=200, x2=200, y2=y2, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == "danger"

    def test_footpoint_at_very_bottom(self, cfg):
        d = Detection(x1=100, y1=300, x2=200, y2=480, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == "danger"

    def test_footpoint_at_very_top(self, cfg):
        d = Detection(x1=100, y1=0, x2=200, y2=10, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg) == ""

    def test_custom_cut_lines(self):
        cfg = SafetyVisionConfig()
        cfg.input.cameras = [CameraConfig(id="back", rtsp_url="rtsp://x/y")]
        cfg.alert.yellow_start_y = 0.50
        cfg.alert.red_start_y = 0.80

        d1 = Detection(x1=0, y1=100, x2=100, y2=200, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d1, 480, cfg) == ""

        d2 = Detection(x1=0, y1=200, x2=100, y2=300, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d2, 480, cfg) == "medium"

        d3 = Detection(x1=0, y1=350, x2=100, y2=450, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d3, 480, cfg) == "danger"

    def test_zero_frame_height(self, cfg):
        d = Detection(x1=0, y1=0, x2=100, y2=100, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 0, cfg) == ""

    def test_camera_specific_cut_lines_override_global_defaults(self):
        cfg = SafetyVisionConfig()
        cam = CameraConfig(id="back", rtsp_url="rtsp://x/y")
        cam.zone.yellow_start_y = 0.20
        cam.zone.red_start_y = 0.40
        cfg.input.cameras = [cam]
        cfg.alert.yellow_start_y = 0.50
        cfg.alert.red_start_y = 0.80

        d = Detection(x1=0, y1=100, x2=100, y2=220, confidence=0.9, class_id=0)
        assert _classify_detection_zone(d, 480, cfg, cam) == "danger"


class TestInferenceWorkerConstruction:
    def test_accepts_zone_callbacks(self):
        cfg = SafetyVisionConfig()
        cfg.input.cameras = [CameraConfig(id="back", rtsp_url="rtsp://x/y")]
        worker = InferenceWorker(
            cfg,
            Queue(),
            Queue(),
            threading.Event(),
            zone_yellow_cb=lambda: None,
            zone_red_cb=lambda: None,
        )
        assert worker._zone_yellow_cb is not None
        assert worker._zone_red_cb is not None
        assert worker._prev_zone_levels == {}

    def test_zone_entry_transitions_count_once_per_episode(self):
        cfg = SafetyVisionConfig()
        cfg.input.cameras = [CameraConfig(id="back", rtsp_url="rtsp://x/y")]
        counts = {"yellow": 0, "red": 0}
        worker = InferenceWorker(
            cfg,
            Queue(),
            Queue(),
            threading.Event(),
            zone_yellow_cb=lambda: counts.__setitem__("yellow", counts["yellow"] + 1),
            zone_red_cb=lambda: counts.__setitem__("red", counts["red"] + 1),
        )

        worker._record_zone_entry_transition("back", "")
        worker._record_zone_entry_transition("back", "medium")
        worker._record_zone_entry_transition("back", "medium")
        worker._record_zone_entry_transition("back", "danger")
        worker._record_zone_entry_transition("back", "danger")
        worker._record_zone_entry_transition("back", "medium")
        worker._record_zone_entry_transition("back", "")
        worker._record_zone_entry_transition("back", "danger")

        assert counts == {"yellow": 1, "red": 2}


class TestHailoPreprocess:
    """HailoBackend._preprocess doesn't require hardware or the hailo_platform module."""

    def _backend(self, input_shape=(640, 640, 3)):
        b = HailoBackend()
        b._input_shape = input_shape
        return b

    def test_output_is_uint8_nhwc(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        blob = self._backend()._preprocess(frame)
        assert blob.dtype == np.uint8
        assert blob.shape == (1, 640, 640, 3)

    def test_resizes_to_model_input(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        blob = self._backend(input_shape=(416, 416, 3))._preprocess(frame)
        assert blob.shape == (1, 416, 416, 3)

    def test_channel_order_bgr_to_rgb(self):
        # Solid blue BGR frame — channel 0 is B in the source.
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[..., 0] = 255  # B in BGR
        blob = self._backend(input_shape=(32, 32, 3))._preprocess(frame)
        # After BGR->RGB, blue sits at channel 2 and channel 0 (R) is zero.
        assert blob[0, 0, 0, 0] == 0
        assert blob[0, 0, 0, 2] == 255


class TestHailoPostprocess:
    """Hailo NMS output parsing — shape (1, num_classes, max_boxes, 5).

    Row layout: [y_min, x_min, y_max, x_max, score] in normalized [0, 1] coords,
    sorted by score descending; trailing zero-filled rows stop iteration.
    """

    def _backend(self, conf=0.45, person_class=0):
        b = HailoBackend()
        b._conf_threshold = conf
        b._person_class_id = person_class
        return b

    def test_empty_output_returns_no_detections(self):
        output = np.zeros((1, 80, 20, 5), dtype=np.float32)
        dets = self._backend()._postprocess(output, src_h=480, src_w=640)
        assert dets == []

    def test_scales_normalized_coords_to_source_frame(self):
        output = np.zeros((1, 80, 20, 5), dtype=np.float32)
        # [y_min, x_min, y_max, x_max, score]
        output[0, 0, 0] = [0.1, 0.2, 0.5, 0.6, 0.9]
        dets = self._backend()._postprocess(output, src_h=480, src_w=640)
        assert len(dets) == 1
        d = dets[0]
        assert d.x1 == pytest.approx(0.2 * 640)
        assert d.y1 == pytest.approx(0.1 * 480)
        assert d.x2 == pytest.approx(0.6 * 640)
        assert d.y2 == pytest.approx(0.5 * 480)
        assert d.confidence == pytest.approx(0.9)
        assert d.class_id == 0

    def test_stops_at_first_below_threshold(self):
        output = np.zeros((1, 80, 20, 5), dtype=np.float32)
        output[0, 0, 0] = [0.0, 0.0, 0.2, 0.2, 0.95]
        output[0, 0, 1] = [0.3, 0.3, 0.5, 0.5, 0.80]
        output[0, 0, 2] = [0.6, 0.6, 0.8, 0.8, 0.30]  # below threshold -> stops here
        output[0, 0, 3] = [0.0, 0.0, 0.1, 0.1, 0.99]  # would pass, but we stopped
        dets = self._backend(conf=0.45)._postprocess(output, src_h=480, src_w=640)
        assert len(dets) == 2
        assert all(d.confidence >= 0.45 for d in dets)

    def test_reads_configured_person_class_slice(self):
        output = np.zeros((1, 80, 20, 5), dtype=np.float32)
        # Put a high-score box in class 5 only — class 0 slice is empty.
        output[0, 5, 0] = [0.1, 0.1, 0.3, 0.3, 0.9]
        dets_default = self._backend(person_class=0)._postprocess(output, 480, 640)
        dets_custom = self._backend(person_class=5)._postprocess(output, 480, 640)
        assert dets_default == []
        assert len(dets_custom) == 1
        assert dets_custom[0].class_id == 5

    def test_unexpected_rank_returns_empty(self):
        output = np.zeros((80, 20, 5), dtype=np.float32)
        assert self._backend()._postprocess(output, 480, 640) == []

    def test_list_output_uses_person_class_entry(self):
        output = [np.zeros((20, 5), dtype=np.float32) for _ in range(80)]
        output[0][0] = [0.1, 0.2, 0.5, 0.6, 0.9]
        dets = self._backend()._postprocess(output, src_h=480, src_w=640)
        assert len(dets) == 1
        assert dets[0].confidence == pytest.approx(0.9)

    def test_singleton_classwise_output_is_unwrapped(self):
        output = [np.zeros((1, 20, 5), dtype=np.float32)]
        output[0][0, 0] = [0.1, 0.2, 0.5, 0.6, 0.9]
        dets = self._backend(person_class=0)._postprocess(output, src_h=480, src_w=640)
        assert len(dets) == 1
        assert dets[0].x1 == pytest.approx(0.2 * 640)

    def test_singleton_outer_list_with_class_tensor_is_unwrapped(self):
        output = [np.zeros((80, 20, 5), dtype=np.float32)]
        output[0][0, 0] = [0.1, 0.2, 0.5, 0.6, 0.9]
        dets = self._backend(person_class=0)._postprocess(output, src_h=480, src_w=640)
        assert len(dets) == 1
        assert dets[0].y2 == pytest.approx(0.5 * 480)
