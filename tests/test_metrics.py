"""Tests for the metrics collector."""

import time

from safetyvision.workers.metrics import MetricsCollector


class TestMetricsCollector:
    def test_initial_snapshot(self):
        m = MetricsCollector()
        snap = m.snapshot()
        assert snap.fps == 0.0
        assert snap.capture_fps == 0.0
        assert snap.inference_fps == 0.0
        assert snap.decision_fps == 0.0
        assert snap.frames_dropped == 0
        assert snap.alert_count == 0

    def test_inference_frame_counting(self):
        m = MetricsCollector(window_sec=10.0)
        for _ in range(10):
            m.record_inference_frame()
        snap = m.snapshot()
        assert snap.fps > 0
        assert snap.inference_fps > 0

    def test_stage_fps_counting(self):
        m = MetricsCollector(window_sec=10.0)
        for _ in range(5):
            m.record_capture_frame()
            m.record_decision_frame()
        snap = m.snapshot()
        assert snap.capture_fps > 0
        assert snap.decision_fps > 0

    def test_latency_recording(self):
        m = MetricsCollector()
        for ms in [10.0, 20.0, 30.0]:
            m.record_inference_latency(ms)
        snap = m.snapshot()
        assert snap.inference_latency_ms == 20.0  # median

    def test_drop_counting(self):
        m = MetricsCollector()
        m.record_drop()
        m.record_drop()
        assert m.snapshot().frames_dropped == 2

    def test_alert_counting(self):
        m = MetricsCollector()
        m.record_alert()
        assert m.snapshot().alert_count == 1

    def test_zone_entry_counting(self):
        m = MetricsCollector()
        m.record_yellow_entry()
        m.record_red_entry()
        m.record_red_entry()
        snap = m.snapshot()
        assert snap.yellow_zone_entries == 1
        assert snap.red_zone_entries == 2

    def test_uptime_increases(self):
        m = MetricsCollector()
        time.sleep(0.05)
        assert m.snapshot().uptime_sec > 0
