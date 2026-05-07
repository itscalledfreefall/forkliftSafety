"""Factory: select a ZoneStrategy from config."""

from __future__ import annotations

from safetyvision.config import SafetyVisionConfig
from safetyvision.zones.base import ZoneStrategy


def create_zone_strategy(cfg: SafetyVisionConfig) -> ZoneStrategy:
    """Build the strategy named by ``cfg.alert.zone_mode``.

    Imports are deferred so band-mode deployments don't pay the cv2
    homography import cost on startup.
    """
    mode = cfg.alert.zone_mode
    if mode == "bands":
        from safetyvision.zones.bands import BandZoneStrategy

        return BandZoneStrategy(
            yellow_start_y=cfg.alert.yellow_start_y,
            red_start_y=cfg.alert.red_start_y,
        )
    if mode == "distance":
        from safetyvision.zones.distance import DistanceZoneStrategy

        return DistanceZoneStrategy(
            calibration_path=cfg.alert.calibration_path,
            danger_m=cfg.alert.danger_threshold_m,
            warning_m=cfg.alert.warning_threshold_m,
            smoothing_frames=cfg.alert.distance_smoothing_frames,
        )
    raise ValueError(f"Unknown zone_mode: {mode!r}")
