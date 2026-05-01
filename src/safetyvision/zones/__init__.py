"""Zone classification strategies (band vs. distance)."""

from safetyvision.zones.base import ZoneResult, ZoneStrategy
from safetyvision.zones.factory import create_zone_strategy

__all__ = ["ZoneResult", "ZoneStrategy", "create_zone_strategy"]
