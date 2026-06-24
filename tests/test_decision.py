"""Tests for the decision worker alert state machine."""

import threading
from queue import Queue

import pytest

from safetyvision.config import CameraConfig, SafetyVisionConfig
from safetyvision.types import AlertEvent, AlertState, DetectionEvent
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

    def test_zone_escalation_triggers_immediately(self, worker):
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000, zone_level="medium"))

        alert = worker.process_event(
            _make_event(person=True, ts_ns=2_000_000_000, zone_level="danger")
        )

        assert alert is not None
        assert alert.trigger_reason == "zone_escalated"
        assert alert.cooldown_active is False
        assert alert.sound_key == "danger"

    def test_repeat_gap_measured_from_audio_end(self, worker):
        """repeat_interval_sec is the silence between alarms, not start-to-start.

        With a 5s interval, a clip that finishes at t=4s must push the next repeat
        out to t=9s (4s audio end + 5s gap), not t=6s (trigger + 5s).
        """
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000, zone_level="danger"))
        # Alarm audio finishes at t=4s.
        worker._w.record_audio_done(TEST_CAM, 4_000_000_000)

        # t=7s: 6s since trigger but only 3s since audio ended -> still throttled.
        assert worker.process_event(
            _make_event(person=True, ts_ns=7_000_000_000, zone_level="danger")
        ) is None

        # t=9.5s: 5.5s since audio ended -> repeat fires.
        alert = worker.process_event(
            _make_event(person=True, ts_ns=9_500_000_000, zone_level="danger")
        )
        assert alert is not None
        assert alert.trigger_reason == "repeat_while_present"

    def test_boundary_flapping_does_not_respawn_escalation(self, worker):
        """Person on the danger/medium boundary flips zone frame-to-frame.

        Once danger has fired, a repeat must keep the highest zone (danger) and
        the next danger frame must NOT be treated as a fresh escalation, or the
        alert audio spams back-to-back.
        """
        worker.process_event(_make_event(person=True, ts_ns=1_000_000_000, zone_level="danger"))

        # Repeat fires on a medium frame after the interval; it must announce the
        # highest zone of the episode (danger), not downgrade to medium.
        rep = worker.process_event(
            _make_event(person=True, ts_ns=6_500_000_000, zone_level="medium")
        )
        assert rep is not None
        assert rep.trigger_reason == "repeat_while_present"
        assert rep.sound_key == "danger"

        # The very next danger frame must not re-fire as an escalation.
        nxt = worker.process_event(
            _make_event(person=True, ts_ns=6_600_000_000, zone_level="danger")
        )
        assert nxt is None

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


class TestZoneEntryCounting:
    """Zone-entry counters must count one entry per presence, not per frame.

    Counting lives in the decision worker so it inherits the debounced episode
    state machine: a person standing in the red zone (band flickers
    danger<->medium, or a detection drops for a frame) counts as a single red
    entry until they clear.
    """

    def _kind(self, alert):
        return DecisionWorker._zone_entry_kind(alert)

    def test_kind_episode_open_danger_is_red(self):
        alert = AlertEvent(0, "person_detected", False, "danger")
        assert DecisionWorker._zone_entry_kind(alert) == "red"

    def test_kind_episode_open_medium_is_yellow(self):
        alert = AlertEvent(0, "person_detected", False, "medium")
        assert DecisionWorker._zone_entry_kind(alert) == "yellow"

    def test_kind_escalation_to_danger_is_red(self):
        alert = AlertEvent(0, "zone_escalated", False, "danger")
        assert DecisionWorker._zone_entry_kind(alert) == "red"

    def test_kind_repeat_does_not_count(self):
        alert = AlertEvent(0, "repeat_while_present", True, "danger")
        assert DecisionWorker._zone_entry_kind(alert) is None

    def _tally(self, worker, frames):
        """Drive frames through process_event, tallying zone-entry counters the
        same way DecisionWorker._run does."""
        counts = {"yellow": 0, "red": 0}
        for ts_ns, person, zone in frames:
            alert = worker.process_event(_make_event(person=person, ts_ns=ts_ns, zone_level=zone))
            if alert is not None:
                kind = self._kind(alert)
                if kind is not None:
                    counts[kind] += 1
        return counts

    def test_red_counts_once_through_flicker(self, worker):
        """A single presence that flickers danger<->medium and drops a frame
        counts exactly one red entry (and the opening yellow)."""
        ms = 1_000_000
        frames = [
            (1000 * ms, True, "medium"),  # episode opens -> yellow
            (1040 * ms, True, "danger"),  # escalation -> red
            (1080 * ms, True, "medium"),  # boundary flicker, no count
            (1120 * ms, True, "danger"),  # back to danger, no recount
            (1160 * ms, True, ""),        # dropped detection (< min_clear), no clear
            (1200 * ms, True, "danger"),  # reacquired, no recount
            (1240 * ms, True, "medium"),
            (1280 * ms, True, "danger"),
        ]
        assert self._tally(worker, frames) == {"yellow": 1, "red": 1}

    def test_separate_presences_count_again(self, worker):
        """After the person clears (min_clear_sec of no person), a new entry
        into the red zone counts a fresh red entry."""
        ms = 1_000_000
        frames = [
            (1000 * ms, True, "danger"),   # presence 1 -> red
            (1040 * ms, True, "danger"),   # repeat-throttled, no count
            (5000 * ms, False, ""),        # 3.96s of no person -> clears
            (6000 * ms, True, "danger"),   # presence 2 -> red
        ]
        assert self._tally(worker, frames) == {"yellow": 0, "red": 2}

    def test_medium_only_presence_counts_yellow_only(self, worker):
        ms = 1_000_000
        frames = [
            (1000 * ms, True, "medium"),
            (1040 * ms, True, "medium"),
            (1080 * ms, True, ""),
            (1120 * ms, True, "medium"),
        ]
        assert self._tally(worker, frames) == {"yellow": 1, "red": 0}
