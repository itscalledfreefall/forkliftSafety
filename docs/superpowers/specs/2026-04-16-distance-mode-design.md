# Distance Mode — Design Spec

## Overview

Add a distance-based zone classification mode to SafetyVision using the `supervision` library's `ViewTransformer` (homography). The camera is mounted on the forklift (top or back — configurable via calibration). Detected persons' footpoints are transformed from pixel coordinates to real-world meters, and distance from the camera origin (= forklift position) determines the alert zone.

The existing horizontal band mode is preserved. Operators choose between `bands` and `distance` via config.

## Architecture: Zone Strategy Pattern

Extract zone classification from `InferenceWorker` into a strategy interface. Two implementations:

- **BandZoneStrategy** — current Y-band logic (`yellow_start_y` / `red_start_y`), extracted as-is.
- **DistanceZoneStrategy** — homography-based, uses `supervision.ViewTransformer` to convert footpoints to meters, then compares against meter thresholds.

`InferenceWorker` delegates zone classification to whichever strategy is configured. Everything downstream (Decision, Alert workers) is unchanged — they receive the same `zone_level` string.

### Pipeline (unchanged)

```
Capture → Inference → Decision → Alert
               ↑ delegates to
     BandZoneStrategy  OR  DistanceZoneStrategy
```

## New Files

| File | Purpose |
|------|---------|
| `src/safetyvision/zones/__init__.py` | Package init |
| `src/safetyvision/zones/base.py` | `ZoneStrategy` protocol + `ZoneResult` dataclass |
| `src/safetyvision/zones/bands.py` | `BandZoneStrategy` — extracted from `inference.py:92-113` |
| `src/safetyvision/zones/distance.py` | `DistanceZoneStrategy` — supervision + homography |
| `src/safetyvision/zones/factory.py` | `create_zone_strategy(config)` factory function |
| `src/safetyvision/web/calibration.py` | FastAPI calibration API endpoints |
| `src/safetyvision/web/static/calibration.html` | Calibration UI page |

## Modified Files

| File | Change |
|------|--------|
| `src/safetyvision/workers/inference.py` | Remove inline `_classify_detection_zone`, delegate to strategy |
| `src/safetyvision/config.py` | Add `zone_mode`, `calibration_file`, `danger_threshold_m`, `warning_threshold_m` to `AlertConfig` |
| `config/safetyvision.yaml` | Add distance mode config fields |
| `config/safetyvision.raspberry.yaml` | Add distance mode config fields |
| `src/safetyvision/web/app.py` | Mount calibration router |
| `src/safetyvision/types.py` | Add `distance_m: float | None = None` to `DetectionEvent` |
| `src/safetyvision/workers/metrics.py` | Add `last_distance_m`, `last_zone_level` to snapshot |
| `src/safetyvision/web/static/js/app.js` | Render distance readout |
| `src/safetyvision/web/templates/index.html` | Add distance display element |

## Zone Strategy Interface

```python
# src/safetyvision/zones/base.py
from dataclasses import dataclass
from typing import Protocol

@dataclass
class ZoneResult:
    zone_level: str        # "danger" | "medium" | ""
    distance_m: float | None  # None for band mode

class ZoneStrategy(Protocol):
    def classify(self, detections: list, frame_h: int, frame_w: int) -> ZoneResult:
        ...
```

## BandZoneStrategy

Extracted directly from `inference.py:92-113`. Same logic — footpoint Y normalized against `yellow_start_y` / `red_start_y`. Returns `ZoneResult(zone_level=..., distance_m=None)`.

## DistanceZoneStrategy

```python
# src/safetyvision/zones/distance.py
import supervision as sv
import numpy as np

class DistanceZoneStrategy:
    def __init__(self, calibration_path: str, danger_m: float, warning_m: float):
        data = json.load(open(calibration_path))
        self.transformer = sv.ViewTransformer(
            source=np.array(data["source_points"], dtype=np.float32),
            target=np.array(data["target_points"], dtype=np.float32),
        )
        # Thresholds come from safetyvision.yaml (single source of truth)
        self.danger_m = danger_m
        self.warning_m = warning_m

    def classify(self, detections: list, frame_h: int, frame_w: int) -> ZoneResult:
        if not detections:
            return ZoneResult(zone_level="", distance_m=None)

        footpoints = np.array(
            [((d.x1 + d.x2) / 2, d.y2) for d in detections], dtype=np.float32
        )
        world_pts = self.transformer.transform_points(footpoints)
        distances = np.linalg.norm(world_pts, axis=1)
        min_dist = float(distances.min())

        if min_dist <= self.danger_m:
            zone = "danger"
        elif min_dist <= self.warning_m:
            zone = "medium"
        else:
            zone = ""

        return ZoneResult(zone_level=zone, distance_m=min_dist)
```

### Runtime performance

`ViewTransformer.transform_points()` is a single `cv2.perspectiveTransform` call — one matrix multiply. Cost is ~0.01ms per frame regardless of detection count. No impact on the 25 FPS pipeline.

## Config Changes

```yaml
# safetyvision.yaml — alert section
alert:
  zone_mode: "bands"              # "bands" or "distance"
  # Band mode (existing)
  yellow_start_y: 0.29
  red_start_y: 0.78
  # Distance mode (new)
  calibration_file: "config/calibration.json"
  danger_threshold_m: 2.0
  warning_threshold_m: 5.0
```

### AlertConfig dataclass additions

```python
zone_mode: str = "bands"
calibration_file: str = "config/calibration.json"
danger_threshold_m: float = 2.0
warning_threshold_m: float = 5.0
```

### Validation rules

- `zone_mode` must be `"bands"` or `"distance"`
- If `zone_mode == "distance"`, `calibration_file` must exist
- `danger_threshold_m > 0`
- `danger_threshold_m < warning_threshold_m`

## Calibration File Format

```json
{
  "source_points": [[160, 96], [480, 96], [128, 256], [512, 256]],
  "target_points": [[0.0, 8.0], [6.0, 8.0], [0.0, 2.0], [6.0, 2.0]],
  "created_at": "2026-04-16T12:00:00Z"
}
```

- `source_points`: 4 pixel coordinates clicked in the camera frame
- `target_points`: 4 corresponding real-world coordinates in meters, in a forklift-relative coordinate frame (see Coordinate Frame below)
- Thresholds (`danger_threshold_m`, `warning_threshold_m`) live only in `safetyvision.yaml` — single source of truth. The calibration file stores only the point mapping.

## Coordinate Frame

All real-world target coordinates are in a **forklift-relative frame**:

- **Origin (0, 0)** = the camera/forklift position (where the camera is mounted)
- **X axis** = lateral (left/right from the forklift's perspective)
- **Y axis** = forward distance from the forklift

The calibration UI must enforce this: the operator measures each ground point's distance *from the camera mount position*. The UI instructions will say: "Measure each point's distance from the camera in meters. X = left/right, Y = forward."

This means `np.linalg.norm(world_pts, axis=1)` correctly computes the Euclidean distance from the forklift to each detected person.

If the operator's reference points are not naturally centered on the forklift (e.g. they use fixed floor markings), they must offset their measurements so the camera mount is at (0, 0). The calibration UI will include a note about this.

## Calibration Web UI

### Page: `/calibration`

Split layout:
- **Left**: frozen camera frame where operator clicks 4 ground points
- **Right**: coordinate input panel — pixel coords auto-filled on click, operator types real-world X/Y meters for each point. Alert thresholds (danger_m, warning_m) at the bottom.
- **Top bar**: "Capture Frame" button (grabs snapshot), "Save & Apply" button (saves + switches mode)

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/calibration/frame` | Returns JPEG snapshot from live camera |
| `GET` | `/api/calibration` | Returns current calibration.json (404 if none) |
| `POST` | `/api/calibration` | Validates, saves calibration.json, sets `zone_mode: distance` in YAML, restarts safetyvision service |
| `DELETE` | `/api/calibration` | Removes calibration, sets `zone_mode: bands` in YAML, restarts safetyvision service |

### Validation on POST

- Exactly 4 source and 4 target points
- Source points within frame dimensions
- Points are not collinear (form a valid quadrilateral)
- Target points have positive meter values
- `danger_threshold_m < warning_threshold_m`
- `danger_threshold_m > 0`

## Applying Calibration

The web UI (`safetyvision-ui`) and detection pipeline (`safetyvision`) run as separate systemd services in separate processes. They cannot share memory.

When the operator clicks "Save & Apply":

1. Web app validates and writes `calibration.json`
2. Web app updates `zone_mode: "distance"` in `safetyvision.yaml`
3. Web app restarts the `safetyvision` service via `systemctl restart safetyvision` (same pattern as the existing `POST /api/apply` endpoint)
4. On startup, `InferenceWorker` reads config, factory creates the appropriate `ZoneStrategy`

This follows the existing config-apply pattern: write config file, restart service. No in-process signaling needed.

## DetectionEvent Update

```python
@dataclass
class DetectionEvent:
    timestamp_ns: int
    person_detected: bool
    confidence_max: float
    bbox_count: int
    zone_level: str
    source_id: str
    distance_m: float | None = None  # NEW — populated in distance mode
```

The `distance_m` field is:
- `None` when using band mode
- The closest person's distance in meters when using distance mode

## Metrics & Dashboard Data Path

Currently, `MetricsWorker` logs aggregate counters (FPS, latency, alert count) and the web dashboard polls `GET /api/metrics`. The `distance_m` value needs to reach both metrics and the dashboard.

### Metrics extension

Add `distance_m` and `zone_level` to the metrics snapshot. The `MetricsCollector` already receives `DetectionEvent` — extend its snapshot to include `last_distance_m` and `last_zone_level` from the most recent event.

### Dashboard display

The existing `GET /api/metrics` response gains two fields: `last_distance_m` (float or null) and `last_zone_level` (string). The dashboard JS already polls this endpoint — add a distance readout element that shows the value when `last_distance_m` is not null.

### Modified files for this data path

| File | Change |
|------|--------|
| `src/safetyvision/workers/metrics.py` | Add `last_distance_m`, `last_zone_level` to snapshot |
| `src/safetyvision/web/app.py` | Include new fields in `/api/metrics` response |
| `src/safetyvision/web/static/js/app.js` | Render distance readout when available |
| `src/safetyvision/web/templates/index.html` | Add distance display element |

## Web Dashboard Update

When distance mode is active, the existing dashboard shows:
- Current closest person distance (e.g. "3.2m") via the metrics poll
- Zone status with color (green / warning / danger)
- Link to `/calibration` page to recalibrate

## Dependencies

- `supervision` — `pip install supervision` (adds `ViewTransformer`, lightweight)
- No other new dependencies. `supervision` uses numpy and opencv which are already in the project.

## Testing Strategy

- **Unit tests**: `BandZoneStrategy` and `DistanceZoneStrategy` with known inputs/outputs
- **Integration test**: Calibration API endpoint save/load/delete cycle
- **Manual test**: Calibration UI with live camera, verify distance readout against tape measure
