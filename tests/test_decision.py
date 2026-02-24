"""Tests for the decision worker alert state machine."""

import threading
from queue import Queue

import pytest

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import AlertState, DetectionEvent
from safetyvision.workers.decision import DecisionWorker


def _make_event(person: bool, ts_ns: int = 0, conf: float = 0.8) -> DetectionEvent:
    return DetectionEvent(
        timestamp_ns=ts_ns,
        person_detected=person,
        confidence_max=conf if person else 0.0,
        bbox_count=1 if person else 0,
        source_id="test",
    )


@pytest.fixture
def worker():
    cfg = SafetyVisionConfig()
    cfg.alert.repeat_interval_sec = 5.0
    cfg.alert.min_clear_sec = 3.0
    w = DecisionWorker(cfg, Queue(), Queue(), threading.Event())
    return w


class TestAlertStateMachine:
    def test_initial_state_idle(self, worker):
        assert worker.state == AlertState.IDLE

    def test_no_alert_on_no_person(self, worker):
        event = _make_event(person=False, ts_ns=1_000_000_000)
        alert = worker.process_event(event)
        assert alert is None
        assert worker.state == AlertState.IDLE

    def test_triggers_on_person(self, worker):
        event = _make_event(person=True, ts_ns=1_000_000_000)
        alert = worker.process_event(event)
        assert alert is not None
        assert alert.trigger_reason == "person_detected"
        assert worker.state == AlertState.TRIGGERED

    def test_no_repeat_before_interval(self, worker):
        # Trigger
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000))
        assert worker.state == AlertState.TRIGGERED

        # Still person but before repeat interval (5s)
        alert = worker.process_event(_make_event(person=True, ts_ns=3_000_000_000))
        assert alert is None  # Not yet time to repeat

    def test_repeats_after_interval(self, worker):
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000))
        # After 5s repeat interval
        alert = worker.process_event(_make_event(person=True, ts_ns=6_500_000_000))
        assert alert is not None
        assert alert.trigger_reason == "repeat_while_present"
        assert alert.cooldown_active is True

    def test_clears_after_min_clear(self, worker):
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000))
        assert worker.state == AlertState.TRIGGERED

        # Person gone, but not enough time
        alert = worker.process_event(_make_event(person=False, ts_ns=2_000_000_000))
        assert alert is None
        # State stays TRIGGERED because min_clear_sec not met

        # After min_clear_sec (3s from last person seen at 1s)
        alert = worker.process_event(_make_event(person=False, ts_ns=5_000_000_000))
        assert alert is None
        assert worker.state == AlertState.IDLE

    def test_alert_count_increments(self, worker):
        assert worker.alert_count == 0
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000))
        assert worker.alert_count == 1
        worker.process_event(_make_event(person=True, ts_ns=7_000_000_000))
        assert worker.alert_count == 2

    def test_retrigger_after_clear(self, worker):
        # Trigger
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000))
        assert worker.state == AlertState.TRIGGERED

        # Clear
        worker.process_event(_make_event(person=False, ts_ns=5_000_000_000))
        assert worker.state == AlertState.IDLE

        # New trigger
        alert = worker.process_event(_make_event(person=True, ts_ns=10_000_000_000))
        assert alert is not None
        assert alert.trigger_reason == "person_detected"
        assert worker.alert_count == 2
