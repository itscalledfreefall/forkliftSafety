"""Tests for alert worker queue collapse behavior."""

import threading
from queue import Queue

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import AlertEvent
from safetyvision.workers.alert import AlertWorker


def test_collapse_prefers_newer_danger_over_stale_medium_repeats():
    worker = AlertWorker(SafetyVisionConfig(), Queue(), threading.Event())
    first = AlertEvent(
        timestamp_ns=1,
        trigger_reason="repeat_while_present",
        cooldown_active=True,
        sound_key="medium",
    )
    worker._alert_queue.put(
        AlertEvent(
            timestamp_ns=2,
            trigger_reason="repeat_while_present",
            cooldown_active=True,
            sound_key="medium",
        )
    )
    worker._alert_queue.put(
        AlertEvent(
            timestamp_ns=3,
            trigger_reason="zone_escalated",
            cooldown_active=False,
            sound_key="danger",
        )
    )

    selected = worker._collapse_pending_alerts(first)

    assert selected.sound_key == "danger"
    assert selected.trigger_reason == "zone_escalated"
