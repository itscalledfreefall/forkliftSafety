"""Hailo-8L inference backend using a pre-compiled HEF."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import Detection


class HailoBackend:
    """Runs YOLO person detection on a Hailo-8L accelerator via HailoRT."""

    def __init__(self) -> None:
        self._device = None
        self._network_group = None
        self._network_group_activation = None
        self._infer_pipeline = None
        self._infer_ctx = None
        self._input_vstreams_params = None
        self._output_vstreams_params = None
        self._input_name = ""
        self._output_name = ""
        self._input_shape: tuple[int, int, int] = (640, 640, 3)
        self._conf_threshold = 0.45
        self._person_class_id = 0

    def load(self, cfg: SafetyVisionConfig) -> None:
        path = cfg.model.path_hef
        if not Path(path).exists():
            raise FileNotFoundError(f"HEF not found: {path}")

        try:
            from hailo_platform import (  # type: ignore
                ConfigureParams,
                FormatType,
                HEF,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError as e:
            raise RuntimeError(
                "hailo_platform Python bindings not available. "
                "Install apt package 'hailo-all' on the target Pi."
            ) from e

        hef = HEF(path)
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
        self._input_shape = tuple(input_info.shape)  # (H, W, C)

        self._conf_threshold = cfg.model.conf_threshold
        self._person_class_id = cfg.model.person_class_id

        # Hoist activation + pipeline out of the hot path — saves ~2 ms per frame.
        self._network_group_activation = self._network_group.activate()
        self._network_group_activation.__enter__()

        self._infer_pipeline = InferVStreams(
            self._network_group,
            self._input_vstreams_params,
            self._output_vstreams_params,
        )
        self._infer_ctx = self._infer_pipeline.__enter__()

        logger.info(
            "Hailo model loaded: {} (input {}x{}x{})",
            path,
            self._input_shape[0],
            self._input_shape[1],
            self._input_shape[2],
        )

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        src_h, src_w = frame_bgr.shape[:2]
        blob = self._preprocess(frame_bgr)
        output = self._infer_ctx.infer({self._input_name: blob})[self._output_name]
        return self._postprocess(output, src_h, src_w)

    def close(self) -> None:
        try:
            if self._infer_pipeline is not None:
                self._infer_pipeline.__exit__(None, None, None)
        finally:
            self._infer_pipeline = None
            self._infer_ctx = None
        try:
            if self._network_group_activation is not None:
                self._network_group_activation.__exit__(None, None, None)
        finally:
            self._network_group_activation = None
        if self._device is not None:
            self._device.release()
            self._device = None

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR -> RGB -> resize to network input -> uint8 NHWC."""
        h, w = self._input_shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.uint8)[None, ...]

    def _postprocess(
        self,
        output: np.ndarray,
        src_h: int,
        src_w: int,
    ) -> list[Detection]:
        """Parse Hailo NMS output into Detection objects.

        Expected layout: (batch=1, num_classes, max_boxes, 5) where the last
        dim is [y_min, x_min, y_max, x_max, score] in [0, 1] normalized coords,
        sorted by score descending with zero-filled trailing rows.
        """
        if output.ndim != 4:
            logger.warning("Unexpected Hailo output rank: {}", output.ndim)
            return []

        person_boxes = output[0, self._person_class_id]  # (max_boxes, 5)
        detections: list[Detection] = []
        for row in person_boxes:
            score = float(row[4])
            if score < self._conf_threshold:
                break  # sorted; zero-filled trailing rows
            y_min, x_min, y_max, x_max = (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
            )
            detections.append(
                Detection(
                    x1=x_min * src_w,
                    y1=y_min * src_h,
                    x2=x_max * src_w,
                    y2=y_max * src_h,
                    confidence=score,
                    class_id=self._person_class_id,
                )
            )
        return detections
