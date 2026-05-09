"""Zone strategy interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from safetyvision.types import Detection


@dataclass(slots=True)
class ZoneResult:
    """Result of classifying a frame's detections into a zone level.

    zone_level: "danger" | "medium" | "" (no person / green).
    distance_m: meters from forklift origin to the closest person in
                distance mode. ``None`` in band mode.
    """

    zone_level: str
    distance_m: Optional[float] = None


class ZoneStrategy(Protocol):
    """Classifies a frame's detections into a single ZoneResult."""

    def classify(
        self,
        detections: list[Detection],
        frame_h: int,
        frame_w: int,
    ) -> ZoneResult: ...
