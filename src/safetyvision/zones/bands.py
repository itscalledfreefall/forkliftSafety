"""Horizontal-band zone strategy (preserves the pre-distance-mode logic)."""

from __future__ import annotations

from safetyvision.types import Detection
from safetyvision.zones.base import ZoneResult


class BandZoneStrategy:
    """Classify each detection by its footpoint Y position.

    Bands (top-to-bottom, normalized Y):
        0.0  .. yellow_start_y       = green   ("")
        yellow_start_y .. red_start_y = medium
        red_start_y .. 1.0           = danger

    Multi-person rule: the highest-risk band wins (danger > medium > green).
    """

    def __init__(self, yellow_start_y: float, red_start_y: float):
        self._yellow_y = yellow_start_y
        self._red_y = red_start_y

    def classify(
        self,
        detections: list[Detection],
        frame_h: int,
        frame_w: int,
    ) -> ZoneResult:
        if frame_h <= 0 or not detections:
            return ZoneResult(zone_level="", distance_m=None)

        zone_level = ""
        for det in detections:
            band = self._classify_one(det, frame_h)
            if band == "danger":
                return ZoneResult(zone_level="danger", distance_m=None)
            if band == "medium":
                zone_level = "medium"
        return ZoneResult(zone_level=zone_level, distance_m=None)

    def _classify_one(self, det: Detection, frame_h: int) -> str:
        foot_y = float(det.y2) / frame_h
        foot_y = min(max(foot_y, 0.0), 1.0)
        if foot_y >= self._red_y:
            return "danger"
        if foot_y >= self._yellow_y:
            return "medium"
        return ""
