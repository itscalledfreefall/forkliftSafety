"""Inference backend protocol and registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import Detection


@runtime_checkable
class InferenceBackend(Protocol):
    """Unified interface for inference backends."""

    def load(self, cfg: SafetyVisionConfig) -> None:
        """Load model resources. Called once at worker start."""

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame in original resolution."""

    def close(self) -> None:
        """Release device / model resources."""


def load_backend(cfg: SafetyVisionConfig) -> InferenceBackend:
    """Return a backend instance matching ``cfg.model.runtime``."""
    runtime = cfg.model.runtime
    if runtime == "hailo":
        from safetyvision.inference.hailo_backend import HailoBackend

        backend = HailoBackend()
        backend.load(cfg)
        return backend
    raise ValueError(f"Unsupported runtime: {runtime!r}")
