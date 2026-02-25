"""Alert worker – async audio playback with throttling."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

import numpy as np
from loguru import logger

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import AlertEvent


def _load_wav(path: str) -> Optional[tuple[np.ndarray, int, int]]:
    """Load WAV file into memory. Returns (samples, sample_rate, channels) or None."""
    p = Path(path)
    if not p.exists():
        logger.warning("WAV file not found: {}", path)
        return None
    try:
        with wave.open(str(p), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.uint8

        samples = np.frombuffer(raw, dtype=dtype)
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)
        logger.info("Loaded WAV: {} ({:.1f}s, {}Hz)", path, n_frames / sample_rate, sample_rate)
        return samples, sample_rate, n_channels
    except Exception as e:
        logger.error("Failed to load WAV {}: {}", path, e)
        return None


class AlertWorker:
    """Plays audio alerts asynchronously from a command queue."""

    def __init__(
        self,
        cfg: SafetyVisionConfig,
        alert_queue: Queue,
        stop_event: threading.Event,
    ):
        self._cfg = cfg
        self._alert_queue = alert_queue
        self._stop = stop_event
        self._thread: Optional[threading.Thread] = None
        self._siren_data = None
        self._voice_data = None
        self._clip_map: dict[str, Optional[tuple[np.ndarray, int, int]]] = {}
        self._sd = None  # sounddevice module, lazy-loaded
        self._playback_count: int = 0

    @property
    def playback_count(self) -> int:
        return self._playback_count

    def start(self) -> None:
        self._preload_audio()
        self._thread = threading.Thread(
            target=self._run, name="alert_worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _preload_audio(self) -> None:
        """Load WAV files into memory at startup."""
        self._siren_data = _load_wav(self._cfg.alert.siren_wav)
        self._voice_data = _load_wav(self._cfg.alert.voice_wav)
        self._clip_map = {
            "danger": self._siren_data,
            "medium": self._voice_data,
        }

    def _play_clip(self, clip_data) -> float:
        """Play a preloaded clip. Returns playback start time in ms since epoch."""
        if clip_data is None:
            return 0.0

        samples, sample_rate, channels = clip_data
        start_ms = time.time() * 1000

        try:
            if self._sd is None:
                import sounddevice as sd
                self._sd = sd

            self._sd.play(samples, samplerate=sample_rate, blocking=True)
        except Exception as e:
            logger.error("Audio playback failed: {}", e)
            # Fallback: try simpleaudio
            try:
                import simpleaudio as sa

                play_obj = sa.play_buffer(
                    samples.tobytes(),
                    num_channels=channels,
                    bytes_per_sample=samples.dtype.itemsize,
                    sample_rate=sample_rate,
                )
                play_obj.wait_done()
            except Exception as e2:
                logger.error("Fallback audio also failed: {}", e2)

        return start_ms

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                alert: AlertEvent = self._alert_queue.get(timeout=0.5)
            except Empty:
                continue

            logger.info(
                "Alert triggered: reason={}, cooldown={}, sound_key={}",
                alert.trigger_reason,
                alert.cooldown_active,
                alert.sound_key,
            )

            clip = self._clip_map.get(alert.sound_key) or self._siren_data
            start_ms = self._play_clip(clip)

            alert.audio_started_ms = start_ms
            self._playback_count += 1

            logger.info(
                "Alert audio complete: playback #{}, started_ms={:.0f}",
                self._playback_count,
                start_ms,
            )

        logger.info("Alert worker stopped")
