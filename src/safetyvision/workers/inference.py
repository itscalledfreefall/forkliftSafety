"""Inference worker – runs YOLO ONNX/OpenVINO on frames, outputs detections."""

from __future__ import annotations

import os
import threading
import time
from queue import Empty, Queue
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import Detection, DetectionEvent, FramePacket


def _pin_to_cores(cores: list[int]) -> None:
    try:
        os.sched_setaffinity(0, set(cores))
        logger.debug("Inference thread pinned to cores {}", cores)
    except (AttributeError, OSError):
        pass


def _letterbox(frame: np.ndarray, target: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with letterbox padding, return (padded, scale, (pad_w, pad_h))."""
    h, w = frame.shape[:2]
    scale = min(target / h, target / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2
    padded = np.full((target, target, 3), 114, dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return padded, scale, (pad_w, pad_h)


def _preprocess(frame: np.ndarray, input_size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """BGR frame -> NCHW float32 blob."""
    padded, scale, pad = _letterbox(frame, input_size)
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, 0), scale, pad


def _nms(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    """Simple NMS over a list of Detection objects."""
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        for d in dets:
            iou = _compute_iou(best, d)
            if iou < iou_threshold:
                remaining.append(d)
        dets = remaining
    return keep


def _compute_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _postprocess(
    output: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    person_class_id: int,
    scale: float,
    pad: tuple[int, int],
) -> List[Detection]:
    """Parse YOLO output tensor into filtered person detections.

    Supports two common YOLO output formats:
      - (1, 84, N) – YOLOv8/YOLO11 style: 4 box coords + 80 class scores
      - (1, N, 85) – YOLOv5 style: 4 box coords + objectness + 80 class scores
    """
    if output.ndim == 3:
        output = output[0]

    # Detect format: (84, N) vs (N, 85+)
    # YOLOv8 outputs (84, N) where 84 = 4 box + 80 classes, N = num detections
    # YOLOv5 outputs (N, 85) where 85 = 4 box + 1 obj_conf + 80 classes
    _FEATURE_DIMS = {84, 85, 80 + 4, 80 + 5}  # common YOLO feature counts
    if output.shape[0] in _FEATURE_DIMS or (
        output.shape[0] < output.shape[1]
    ):
        # (84, N) or similar -> transpose to (N, 84)
        output = output.T
        has_objectness = output.shape[1] > 84
    else:
        has_objectness = output.shape[1] > 84

    detections: List[Detection] = []
    for row in output:
        if has_objectness:
            # YOLOv5: [cx, cy, w, h, obj_conf, cls0, cls1, ...]
            obj_conf = row[4]
            class_scores = row[5:]
            cls_id = int(np.argmax(class_scores))
            conf = float(obj_conf * class_scores[cls_id])
        else:
            # YOLOv8: [cx, cy, w, h, cls0, cls1, ...]
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])

        if cls_id != person_class_id or conf < conf_thresh:
            continue

        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x1 = (cx - w / 2 - pad[0]) / scale
        y1 = (cy - h / 2 - pad[1]) / scale
        x2 = (cx + w / 2 - pad[0]) / scale
        y2 = (cy + h / 2 - pad[1]) / scale
        detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=cls_id))

    return _nms(detections, iou_thresh)


class InferenceWorker:
    """Runs YOLO inference on frames from input queue, pushes DetectionEvents."""

    def __init__(
        self,
        cfg: SafetyVisionConfig,
        in_queue: Queue,
        out_queue: Queue,
        stop_event: threading.Event,
        latency_cb=None,
    ):
        self._cfg = cfg
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._stop = stop_event
        self._latency_cb = latency_cb
        self._session = None
        self._thread: Optional[threading.Thread] = None
        # Temporal smoothing buffer
        self._recent_detections: list[bool] = []

    def start(self) -> None:
        self._load_model()
        self._thread = threading.Thread(
            target=self._run, name="inference_worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _load_model(self) -> None:
        model_path = self._cfg.model.path_onnx
        runtime = self._cfg.model.runtime

        if runtime == "openvino":
            self._load_openvino(model_path)
        else:
            self._load_onnxruntime(model_path)

    def _load_onnxruntime(self, model_path: str) -> None:
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = self._cfg.perf.inference_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.enable_cpu_mem_arena = True

        self._session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._runtime_type = "onnxruntime"
        logger.info("ONNX Runtime model loaded: {}", model_path)

    def _load_openvino(self, model_path: str) -> None:
        try:
            from openvino.runtime import Core

            core = Core()
            model = core.read_model(model_path)
            config = {"INFERENCE_NUM_THREADS": str(self._cfg.perf.inference_threads)}
            self._session = core.compile_model(model, "CPU", config)
            self._infer_request = self._session.create_infer_request()
            self._runtime_type = "openvino"
            logger.info("OpenVINO model loaded: {}", model_path)
        except Exception as e:
            logger.warning("OpenVINO load failed ({}), falling back to ONNX Runtime", e)
            self._load_onnxruntime(model_path)

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        if self._runtime_type == "openvino":
            self._infer_request.infer({0: blob})
            return self._infer_request.get_output_tensor(0).data.copy()
        else:
            return self._session.run(None, {self._input_name: blob})[0]

    def _apply_temporal_smoothing(self, person_detected: bool) -> bool:
        """Require N consecutive frames of detection to confirm presence."""
        n = self._cfg.perf.temporal_smoothing_frames
        self._recent_detections.append(person_detected)
        if len(self._recent_detections) > n:
            self._recent_detections.pop(0)
        # Person confirmed only if majority of recent frames agree
        return sum(self._recent_detections) >= (n + 1) // 2

    def _run(self) -> None:
        _pin_to_cores(self._cfg.perf.inference_cpu_cores)
        input_size = self._cfg.model.input_size

        while not self._stop.is_set():
            try:
                pkt: FramePacket = self._in_queue.get(timeout=0.1)
            except Empty:
                continue

            t0 = time.time_ns()
            blob, scale, pad = _preprocess(pkt.frame, input_size)
            raw_out = self._infer(blob)
            dets = _postprocess(
                raw_out,
                self._cfg.model.conf_threshold,
                self._cfg.model.iou_threshold,
                self._cfg.model.person_class_id,
                scale,
                pad,
            )
            t1 = time.time_ns()

            raw_detected = len(dets) > 0
            smoothed = self._apply_temporal_smoothing(raw_detected)
            max_conf = max((d.confidence for d in dets), default=0.0)

            event = DetectionEvent(
                timestamp_ns=pkt.timestamp_ns,
                person_detected=smoothed,
                confidence_max=max_conf,
                bbox_count=len(dets),
                source_id=pkt.source_id,
            )

            # Non-blocking push to decision queue
            while not self._out_queue.empty():
                try:
                    self._out_queue.get_nowait()
                except Exception:
                    break
            try:
                self._out_queue.put_nowait(event)
            except Exception:
                pass

            if self._latency_cb:
                self._latency_cb((t1 - t0) / 1e6)

        logger.info("Inference worker stopped")
