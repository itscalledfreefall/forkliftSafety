# Hailo Integration — Design Spec

## Overview

Add a Hailo-8L inference backend to SafetyVision for deployment on Raspberry Pi 5 with the official AI HAT+. Replaces CPU-based ONNX Runtime with the Hailo PCIe accelerator for YOLO person detection. Also adds multi-camera capture for the 3-camera forklift deployment.

**Target device:** Raspberry Pi 5 + Hailo-8L (13 TOPS) via PCIe.
**Target model:** `yolov6n_h8l.hef` — pre-compiled YOLO v6 nano for Hailo-8L. COCO-trained, class 0 = person. No model compilation needed.
**Target throughput:** 3 cameras × 15 FPS = 45 FPS aggregate (well within Hailo-8L's ~60 FPS capacity for YOLOv6n).

## Why YOLOv6n and not yolo26n

Hailo accelerators only run pre-compiled `.hef` files. Compiling a custom model (yolo26n) requires Hailo's Dataflow Compiler toolchain on x86 Linux, a calibration dataset, hours of compilation, and accuracy validation. `yolov6n_h8l.hef` ships with Raspberry Pi OS, detects persons identically (both COCO-trained), and at Hailo-8L speeds the model choice is irrelevant for throughput. Future work can revisit custom model compilation if accuracy gaps appear.

## Architecture

Extend the existing `InferenceWorker` runtime selector. The worker already branches on `model.runtime` (onnxruntime / openvino / pytorch). Add a new branch for `hailo`. Everything downstream of the worker is unchanged — detections still land in the same `Detection` dataclass.

```
3 Capture Workers ──► frame queue ──► Inference Worker ──► Decision ──► Alert
(one per camera,                         (runtime: hailo)
 tagged by camera_id)                    yolov6n_h8l.hef
```

## New Files

| File | Purpose |
|------|---------|
| `src/safetyvision/inference/__init__.py` | Package init |
| `src/safetyvision/inference/backends.py` | Backend protocol, registry |
| `src/safetyvision/inference/hailo_backend.py` | Hailo inference backend |
| `src/safetyvision/inference/onnx_backend.py` | Extracted from `inference.py` |
| `src/safetyvision/inference/openvino_backend.py` | Extracted from `inference.py` |
| `src/safetyvision/inference/pytorch_backend.py` | Extracted from `inference.py` |

## Modified Files

| File | Change |
|------|--------|
| `src/safetyvision/workers/inference.py` | Slim down — delegate model loading/inference to a backend. Keep pre/postprocessing and the per-frame loop. |
| `src/safetyvision/workers/capture.py` | Support multiple named cameras. One `CaptureWorker` instance per camera, each with its own RTSP stream. |
| `src/safetyvision/supervisor.py` | Start N capture workers from `input.cameras` list |
| `src/safetyvision/config.py` | Add `input.cameras` list, `model.path_hef`, extend `runtime` options |
| `src/safetyvision/types.py` | Add `camera_id` to `FramePacket` and `Detection` |
| `config/safetyvision.yaml` | Add Hailo config, multi-camera list |
| `config/safetyvision.raspberry.yaml` | Same, with Pi-specific paths |

## Backend Protocol

```python
# src/safetyvision/inference/backends.py
from typing import Protocol
import numpy as np

class InferenceBackend(Protocol):
    """Unified interface for all inference backends."""

    def load(self, config) -> None:
        """Load model from disk. Called once at startup."""

    def infer(self, frame: np.ndarray) -> list:
        """
        Run inference on a single BGR frame.
        Returns list of Detection objects (person class only).
        Preprocessing (resize, color conversion) happens inside the backend
        because different backends expect different formats.
        """

    def close(self) -> None:
        """Release resources."""
```

Each backend handles its own preprocessing because formats differ:
- ONNX/OpenVINO: BGR → RGB → letterbox → NCHW float32 [0,1]
- Hailo: BGR → RGB → resize to 640×640 → UINT8 NHWC (no normalization)

## Hailo Backend Implementation

```python
# src/safetyvision/inference/hailo_backend.py
import numpy as np
import cv2
from hailo_platform import (
    HEF, VDevice, FormatType, HailoStreamInterface,
    InferVStreams, InputVStreamParams, OutputVStreamParams,
    ConfigureParams,
)
from ..types import Detection

class HailoBackend:
    def __init__(self):
        self._device = None
        self._network_group = None
        self._input_vstreams_params = None
        self._output_vstreams_params = None
        self._input_name = None
        self._output_name = None
        self._input_shape = None  # (H, W, C)
        self._person_class_id = 0

    def load(self, config) -> None:
        hef = HEF(config.model.path_hef)
        self._device = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=hef, interface=HailoStreamInterface.PCIe
        )
        network_groups = self._device.configure(hef, configure_params)
        self._network_group = network_groups[0]

        self._input_vstreams_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.UINT8
        )
        self._output_vstreams_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )

        input_info = hef.get_input_vstream_infos()[0]
        output_info = hef.get_output_vstream_infos()[0]
        self._input_name = input_info.name
        self._output_name = output_info.name
        self._input_shape = input_info.shape  # (H, W, C)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR → RGB → resize 640x640 → uint8 NHWC."""
        h, w = self._input_shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.uint8)[None, ...]  # add batch dim

    def _postprocess(
        self, output: np.ndarray, src_h: int, src_w: int, conf_threshold: float
    ) -> list[Detection]:
        """
        Hailo NMS output format: shape (1, num_classes, max_bboxes, 5)
        Last dim: [y_min, x_min, y_max, x_max, score] in [0,1] normalized coords.
        Extract class 0 (person), scale to source image size.
        """
        detections = []
        # Select person class output
        person_boxes = output[0, self._person_class_id]  # (max_bboxes, 5)
        for box in person_boxes:
            y_min, x_min, y_max, x_max, score = box
            if score < conf_threshold:
                break  # Hailo returns sorted by score, empty rows filled with zeros
            x1 = float(x_min * src_w)
            y1 = float(y_min * src_h)
            x2 = float(x_max * src_w)
            y2 = float(y_max * src_h)
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                score=float(score),
                class_id=self._person_class_id,
            ))
        return detections

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        src_h, src_w = frame_bgr.shape[:2]
        preprocessed = self._preprocess(frame_bgr)

        with InferVStreams(
            self._network_group,
            self._input_vstreams_params,
            self._output_vstreams_params,
        ) as infer_pipeline:
            with self._network_group.activate():
                output_dict = infer_pipeline.infer({self._input_name: preprocessed})

        output = output_dict[self._output_name]
        return self._postprocess(
            output, src_h, src_w, conf_threshold=0.50
        )

    def close(self) -> None:
        if self._device is not None:
            self._device.release()
            self._device = None
```

### Hailo output format notes

The `yolov8s_h8l.hef` / `yolov6n_h8l.hef` models compile with `HAILO_NMS_BY_CLASS` post-processing baked in. The output is a 4-D tensor `(batch=1, num_classes=80, max_boxes_per_class=100, 5)` where the last dimension is `[y_min, x_min, y_max, x_max, score]` in normalized `[0, 1]` coordinates.

**Key behaviors:**
- Output is sorted by score within each class, descending
- Unused slots are zero-filled, so we stop iterating when score drops below threshold
- Score threshold and IoU threshold are baked into the HEF at compile time (0.2 and 0.7 respectively for our chosen model); our `conf_threshold` filters again at a higher value

### InferVStreams context manager cost

Opening `InferVStreams` inside `infer()` is simple but adds ~2ms overhead per call. For production we can hoist it to `load()` and keep the stream open for the worker's lifetime. Spec keeps the simple version first; optimize only if benchmarks show it's a bottleneck.

## Multi-Camera Capture

### Current state

`CaptureWorker` in `src/safetyvision/workers/capture.py` is a single-camera class instantiated once by the `Supervisor`. The RTSP URL comes from `config.input.rtsp_url`.

### Changes

1. **Config change**: Replace the single `rtsp_url` field with an `input.cameras: list[CameraConfig]`. Each entry has `id` and `rtsp_url`.
2. **Supervisor change**: Iterate `config.input.cameras`, instantiate one `CaptureWorker` per entry, pass `camera_id` to each. All workers push to the same frame queue.
3. **FramePacket**: Add `camera_id: str` field.
4. **Per-camera frame writer** (for calibration UI): Each capture worker writes the latest decoded frame to `/dev/shm/safetyvision/frame_<camera_id>.jpg` at a throttled rate (once per second is fine — not for runtime, only for the calibration snapshot).
5. **GStreamer pipeline unchanged** (still H.264 substream). Hardware decode on Pi 5. 3 streams at 640×480 costs ~20% of one CPU core.

### Backwards compatibility

If the old `input.rtsp_url` field is present but `input.cameras` is absent, treat it as a single camera with `id: "default"`. Log a deprecation warning on startup.

## Config Changes

```yaml
# safetyvision.yaml
input:
  cameras:
    - id: back
      rtsp_url: "rtsp://admin:matrix18@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
    - id: left
      rtsp_url: "rtsp://admin:matrix18@192.168.1.109:554/cam/realmonitor?channel=1&subtype=1"
    - id: right
      rtsp_url: "rtsp://admin:matrix18@192.168.1.110:554/cam/realmonitor?channel=1&subtype=1"
  target_fps: 15
  mode: rtsp

model:
  runtime: hailo                                        # NEW: "hailo" option
  path_hef: "/usr/share/hailo-models/yolov6n_h8l.hef"   # NEW: Hailo model path
  path_onnx: "models/yolo26n.onnx"                      # kept for CPU fallback
  path_openvino: "models/yolo26n_openvino_model/yolo26n.xml"
  path_pt: "models/yolo26n.pt"
  conf_threshold: 0.50
  iou_threshold: 0.50   # unused for Hailo (baked in HEF)
  person_class_id: 0
```

### Validation rules

- `model.runtime` must be one of: `hailo`, `onnxruntime`, `openvino`, `pytorch`
- If `runtime == "hailo"`: `path_hef` must exist, `hailo_platform` must be importable
- `input.cameras` must have at least one entry
- All `camera_id` values must be unique
- `target_fps × len(cameras)` should be ≤ 60 (Hailo-8L capacity warning, non-fatal)

## Dependencies

- `hailo-all` apt package (already installed on target Pi — metapackage includes HailoRT, Python bindings, drivers)
- `python3-hailort` apt package (already installed)
- No pip dependencies added

### Fallback behavior

On non-Pi systems, the `hailo_platform` import will fail. The backend registry must handle this gracefully — if `runtime: hailo` is configured but the module is missing, log a clear error and exit. Other runtimes (onnxruntime, openvino, pytorch) remain available for development on other platforms.

## Performance Expectations

| Metric | Target | Expected |
|--------|--------|----------|
| FPS per camera | 15 | 15 sustained |
| Aggregate FPS | 45 | 45 (Hailo headroom to 60+) |
| Inference latency | <30ms | ~10ms on Hailo-8L for YOLOv6n |
| Total pipeline latency | <60ms | 30-40ms |
| CPU usage | <40% | <25% on Pi 5 |
| Power draw | <15W | ~10-12W total system |

## Testing Strategy

- **Unit tests**:
  - `HailoBackend._postprocess` — known fake output tensor → expected Detection list
  - `HailoBackend._preprocess` — shape, dtype, color channel order
  - Config validation — invalid runtime, missing paths, duplicate camera_ids
- **Hardware-in-loop tests** (run only on target Pi):
  - Load model, verify device identification
  - Run inference on a known-person image, verify detection
  - Sustained 45 FPS across 3 synthesized streams for 60 seconds
- **Manual tests**:
  - 3 real cameras connected via PoE switch
  - Verify person detection in each camera's feed
  - Check total CPU/memory/thermal under load (no throttling)
  - Confirm Hailo temperature stays within operating range

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Hailo device release/re-acquire issues on crash | Use `VDevice()` as context manager; systemd auto-restart handles the rest |
| `yolov6n_h8l.hef` output format differs from assumptions | Write postprocess against real output captured from `hailortcli run`; unit test with fixture |
| Frame format mismatch causes silent bad detections | Log input shape at startup, assert matches 640×640×3 |
| Only one VDevice at a time — can conflict with other Hailo apps | Document conflict; the `safetyvision` service has exclusive use |
| 3 cameras saturate Hailo | Reduce FPS per camera, or round-robin prioritization by last-alert-time |
