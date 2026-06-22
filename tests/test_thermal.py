"""Tests for the thermal monitor (FLIR scene-max heat check)."""

import numpy as np

from safetyvision.config import ThermalConfig, c_to_k, k_to_c
from safetyvision.web.thermal_monitor import ThermalMonitor


def _cfg(tmp_path, **kw) -> ThermalConfig:
    c = ThermalConfig(
        enabled=True,
        host="cam",
        rtsp_url="rtsp://cam/avc",
        max_temp_c=40.0,
        repeat_interval_sec=10.0,
        snapshot_dir=str(tmp_path),
        max_snapshots=5,
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _monitor(cfg) -> ThermalMonitor:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    return ThermalMonitor(get_cfg=lambda: cfg, get_latest_frame=lambda: frame)


class TestKelvinCelsius:
    def test_roundtrip(self):
        assert k_to_c(c_to_k(37.5)) == 37.5

    def test_known(self):
        assert round(k_to_c(306.62), 2) == 33.47


class TestViolationGating:
    def test_rising_edge_takes_one_snapshot(self, tmp_path):
        cfg = _cfg(tmp_path)
        m = _monitor(cfg)
        m._evaluate(cfg, 30.0)  # below
        assert m.status()["violation_count"] == 0
        m._evaluate(cfg, 45.0)  # crosses above -> 1 snapshot
        assert m.status()["violation_count"] == 1
        m._evaluate(cfg, 46.0)  # still above, within repeat window -> no new
        assert m.status()["violation_count"] == 1
        assert m.status()["in_violation"] is True

    def test_drop_then_rise_retriggers(self, tmp_path):
        cfg = _cfg(tmp_path)
        m = _monitor(cfg)
        m._evaluate(cfg, 45.0)  # rising -> 1
        m._evaluate(cfg, 30.0)  # drops below
        assert m.status()["in_violation"] is False
        m._evaluate(cfg, 50.0)  # rises again -> 2
        assert m.status()["violation_count"] == 2

    def test_repeat_interval_zero_only_rising_edge(self, tmp_path):
        cfg = _cfg(tmp_path, repeat_interval_sec=0.0)
        m = _monitor(cfg)
        m._evaluate(cfg, 45.0)
        m._evaluate(cfg, 46.0)
        m._evaluate(cfg, 47.0)
        assert m.status()["violation_count"] == 1

    def test_periodic_retrigger_after_interval(self, tmp_path):
        cfg = _cfg(tmp_path, repeat_interval_sec=10.0)
        m = _monitor(cfg)
        m._evaluate(cfg, 45.0)  # rising -> 1
        m._last_snapshot_mono -= 11.0  # simulate >interval elapsed
        m._evaluate(cfg, 45.0)  # still over, interval passed -> 2
        assert m.status()["violation_count"] == 2

    def test_snapshots_written_and_pruned(self, tmp_path):
        cfg = _cfg(tmp_path, max_snapshots=3, repeat_interval_sec=0.0)
        m = _monitor(cfg)
        # Force 6 distinct rising-edge snapshots.
        for _ in range(6):
            m._evaluate(cfg, 30.0)  # reset below
            m._evaluate(cfg, 50.0)  # rising -> snapshot
        jpgs = list(tmp_path.glob("thermal_*.jpg"))
        assert len(jpgs) == 3  # pruned to max_snapshots
        assert len(m.violations()) == 3


class TestStatus:
    def test_status_reports_temp_and_threshold(self, tmp_path):
        cfg = _cfg(tmp_path)
        m = _monitor(cfg)
        m._evaluate(cfg, 33.3)
        s = m.status()
        assert s["scene_temp_c"] == 33.3
        assert s["threshold_c"] == 40.0
        assert s["enabled"] is True
