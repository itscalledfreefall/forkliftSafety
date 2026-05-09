"""Tests for the decision worker alert state machine."""

import threading
from queue import Queue

import pytest

from safetyvision.config import CameraConfig, SafetyVisionConfig
from safetyvision.types import AlertState, DetectionEvent
from safetyvision.workers.decision import DecisionWorker


TEST_CAM = "test"


def _make_event(
    person: bool,
    ts_ns: int = 0,
    conf: float = 0.8,
    zone_level: str = "",
    camera_id: str = TEST_CAM,
) -> DetectionEvent:
    return DetectionEvent(
        timestamp_ns=ts_ns,
        person_detected=person,
        confidence_max=conf if person else 0.0,
        bbox_count=1 if person else 0,
        zone_level=zone_level if person else "",
        camera_id=camera_id,
    )


class _WorkerProxy:
    """Thin wrapper exposing per-camera state as attributes for test ergonomics."""

    def __init__(self, worker: DecisionWorker, camera_id: str) -> None:
        self._w = worker
        self._cam = camera_id

    @property
    def state(self):
        return self._w.state_for(self._cam)

    @property
    def alert_count(self):
        return self._w.alert_count

    def process_event(self, event):
        return self._w.process_event(event)


@pytest.fixture
def worker():
    """Decision worker with default band config (0.33/0.66 cut lines)."""
    cfg = SafetyVisionConfig()
    cfg.input.cameras = [CameraConfig(id=TEST_CAM, rtsp_url="rtsp://x/y")]
    cfg.alert.repeat_interval_sec = 5.0
    cfg.alert.min_clear_sec = 3.0
    cfg.alert.min_alert_confidence = 0.60
    cfg.alert.always_announce_person = False
    return _WorkerProxy(
        DecisionWorker(cfg, Queue(), Queue(), threading.Event()), TEST_CAM
    )


class TestAlertStateMachine:
    def test_initial_state_idle(self, worker):
        assert worker.state == AlertState.IDLE

    def test_no_alert_on_no_person(self, worker):
        alert = worker.process_event(_make_event(person=False, ts_ns=1_000_000_000))
        assert alert is None
        assert worker.state == AlertState.IDLE

    def test_danger_zone_triggers(self, worker):
        alert = worker.process_event(
            _make_event(person=True, ts_ns=1_000_000_000, zone_level="danger")
        )
        assert alert is not None
        assert alert.trigger_reason == "person_detected"
        assert alert.sound_key == "danger"
        assert worker.state == AlertState.TRIGGERED

    def test_medium_zone_triggers(self, worker):
        alert = worker.process_event(
            _make_event(person=True, ts_ns=1_000_000_000, zone_level="medium")
        )
        assert alert is not None
        assert alert.sound_key == "medium"

    def test_green_zone_no_alert(self, worker):
        """Person in green zone (zone_level='') => no alert."""
        alert = worker.process_event(
            _make_event(person=True, ts_ns=1_000_000_000, zone_level="")
        )
        assert alert is None
        assert worker.state == AlertState.IDLE

    def test_low_confidence_no_alert(self, worker):
        """Person detected but below min_alert_confidence => no alert."""
        alert = worker.process_event(
            _make_event(person=True, ts_ns=1_000_000_000, conf=0.50, zone_level="danger")
        )
        assert alert is None
        assert worker.state == AlertState.IDLE

    def test_no_repeat_before_interval(self, worker):
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000, zone_level="danger"))
        assert worker.state == AlertState.TRIGGERED

        alert = worker.process_event(
            _make_event(person=True, ts_ns=3_000_000_000, zone_level="danger")
        )
        assert alert is None  # throttled

    def test_repeats_after_interval(self, worker):
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000, zone_level="danger"))

        alert = worker.process_event(
            _make_event(person=True, ts_ns=6_500_000_000, zone_level="danger")
        )
        assert alert is not None
        assert alert.trigger_reason == "repeat_while_present"
        assert alert.cooldown_active is True
        assert alert.sound_key == "danger"

    def test_clears_after_min_clear(self, worker):
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000, zone_level="danger"))
        assert worker.state == AlertState.TRIGGERED

        # Person gone, not enough time
        worker.process_event(_make_event(person=False, ts_ns=2_000_000_000))
        assert worker.state == AlertState.TRIGGERED

        # After min_clear_sec
        worker.process_event(_make_event(person=False, ts_ns=5_000_000_000))
        assert worker.state == AlertState.IDLE

    def test_alert_count_increments(self, worker):
        assert worker.alert_count == 0
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000, zone_level="danger"))
        assert worker.alert_count == 1
        worker.process_event(_make_event(person=True, ts_ns=7_000_000_000, zone_level="danger"))
        assert worker.alert_count == 2

    def test_retrigger_after_clear(self, worker):
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000, zone_level="medium"))
        assert worker.state == AlertState.TRIGGERED

        # Clear
        worker.process_event(_make_event(person=False, ts_ns=5_000_000_000))
        assert worker.state == AlertState.IDLE

        # New trigger
        alert = worker.process_event(
            _make_event(person=True, ts_ns=10_000_000_000, zone_level="danger")
        )
        assert alert is not None
        assert alert.sound_key == "danger"
        assert worker.alert_count == 2


class TestMultiPersonZones:
    """Multi-person zone priority is resolved by inference worker.

    These tests verify that the decision worker correctly maps the
    already-resolved zone_level to the right sound key.
    """

    def test_danger_wins_over_medium(self, worker):
        """Inference reports danger (any person in red) => danger sound."""
        alert = worker.process_event(
            _make_event(person=True, ts_ns=1_000_000_000, zone_level="danger")
        )
        assert alert is not None
        assert alert.sound_key == "danger"

    def test_medium_only(self, worker):
        """All persons in yellow only => medium sound."""
        alert = worker.process_event(
            _make_event(person=True, ts_ns=1_000_000_000, zone_level="medium")
        )
        assert alert is not None
        assert alert.sound_key == "medium"

    def test_all_in_green(self, worker):
        """All persons in green => no alert."""
        alert = worker.process_event(
            _make_event(person=True, ts_ns=1_000_000_000, zone_level="")
        )
        assert alert is None

    def test_zone_switch_during_repeat(self, worker):
        """Zone can change between repeats."""
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000, zone_level="medium"))
        assert worker.state == AlertState.TRIGGERED

        # After repeat interval, now in danger
        alert = worker.process_event(
            _make_event(person=True, ts_ns=7_000_000_000, zone_level="danger")
        )
        assert alert is not None
        assert alert.sound_key == "danger"

    def test_person_moves_to_green_then_clears(self, worker):
        """Person moves from red to green, eventually clears."""
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000, zone_level="danger"))
        assert worker.state == AlertState.TRIGGERED

        # Person moves to green
        worker.process_event(_make_event(person=True, ts_ns=2_000_000_000, zone_level=""))
        assert worker.state == AlertState.TRIGGERED  # not yet cleared

        # After min_clear_sec
        worker.process_event(_make_event(person=False, ts_ns=5_000_000_000))
        assert worker.state == AlertState.IDLE
