"""Unit tests for web metrics parsing helpers."""

from __future__ import annotations

import json

import safetyvision.web.app as web_app


def test_extract_metrics_from_direct_json_line():
    line = json.dumps(
        {
            "type": "metrics",
            "fps": 24.9,
            "capture_fps": 25.0,
            "inference_fps": 24.9,
            "decision_fps": 24.9,
            "latency_capture_ms": 39.1,
            "latency_inference_ms": 24.0,
            "latency_decision_ms": 0.01,
            "latency_total_ms": 63.11,
            "frames_dropped": 0,
            "alerts": 2,
            "uptime_s": 12.5,
        }
    )
    data = web_app._extract_metrics_from_text(line)
    assert data is not None
    assert data["fps"] == 24.9
    assert data["latency_total_ms"] == 63.11
    assert data["alerts"] == 2


def test_extract_metrics_from_wrapped_msg_json():
    msg = json.dumps(
        {
            "type": "metrics",
            "fps": 15.5,
            "capture_fps": 15.5,
            "inference_fps": 15.5,
            "decision_fps": 15.5,
            "latency_capture_ms": 67.9,
            "latency_inference_ms": 25.2,
            "latency_decision_ms": 0.01,
            "latency_total_ms": 93.11,
            "frames_dropped": 0,
            "alerts": 1,
            "uptime_s": 99.9,
        }
    )
    outer = json.dumps({"ts": "2026-01-01T00:00:00Z", "level": "INFO", "msg": msg})
    data = web_app._extract_metrics_from_text(outer)
    assert data is not None
    assert data["fps"] == 15.5
    assert data["latency_total_ms"] == 93.11
    assert data["alerts"] == 1


def test_extract_metrics_from_loguru_style_unescaped_msg():
    line = (
        '{"ts":"2026-02-25T10:55:36.431+00:00","level":"INFO","logger":"safetyvision.workers.metrics",'
        '"fn":"_run","line":176,"msg":"{"type": "metrics", "ts": 1772016936.4316103, '
        '"fps": 15.4, "capture_fps": 16.1, "inference_fps": 15.4, "decision_fps": 15.4, '
        '"latency_capture_ms": 67.98, "latency_inference_ms": 25.6, "latency_decision_ms": 0.0, '
        '"latency_total_ms": 93.58, "frames_dropped": 0, "alerts": 0, "uptime_s": 1.1}"}'
    )
    data = web_app._extract_metrics_from_text(line)
    assert data is not None
    assert data["fps"] == 15.4
    assert data["latency_total_ms"] == 93.58
    assert data["alerts"] == 0


def test_read_latest_metrics_uses_env_log_path(tmp_path, monkeypatch):
    log_path = tmp_path / "safetyvision.log"
    first = json.dumps(
        {
            "type": "metrics",
            "fps": 10.0,
            "latency_capture_ms": 10.0,
            "latency_inference_ms": 10.0,
            "latency_decision_ms": 0.0,
            "latency_total_ms": 20.0,
            "frames_dropped": 0,
            "alerts": 0,
        }
    )
    second = json.dumps(
        {
            "type": "metrics",
            "fps": 20.0,
            "capture_fps": 21.0,
            "inference_fps": 20.0,
            "decision_fps": 20.0,
            "latency_capture_ms": 30.0,
            "latency_inference_ms": 15.0,
            "latency_decision_ms": 0.0,
            "latency_total_ms": 45.0,
            "frames_dropped": 1,
            "alerts": 3,
            "uptime_s": 50.0,
        }
    )
    log_path.write_text(f"{first}\nnot-metrics\n{second}\n", encoding="utf-8")
    monkeypatch.setenv("SAFETYVISION_METRICS_LOG", str(log_path))

    data = web_app._read_latest_metrics()
    assert data is not None
    assert data["fps"] == 20.0
    assert data["capture_fps"] == 21.0
    assert data["alerts"] == 3
