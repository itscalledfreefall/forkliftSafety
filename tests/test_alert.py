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


def test_antispam_gate():
    """Same/lower zone within the cooldown is suppressed; escalation breaks through."""
    DANGER, MEDIUM = 2, 1
    cooldown = 3_000_000_000  # 3s
    # Within cooldown, same zone -> suppress (flicker re-trigger spam).
    assert AlertWorker._suppress(DANGER, DANGER, 1_000_000_000, cooldown) is True
    # Within cooldown, lower zone -> suppress (de-escalation re-announce).
    assert AlertWorker._suppress(MEDIUM, DANGER, 1_000_000_000, cooldown) is True
    # Within cooldown, higher zone -> play (genuine escalation).
    assert AlertWorker._suppress(DANGER, MEDIUM, 1_000_000_000, cooldown) is False
    # After cooldown, same zone -> play (legitimate repeat).
    assert AlertWorker._suppress(DANGER, DANGER, 3_500_000_000, cooldown) is False
    # First alarm ever (last_priority=-1) -> play.
    assert AlertWorker._suppress(MEDIUM, -1, 0, cooldown) is False
