"""Supervisor – orchestrates all workers, health checks, graceful shutdown."""

from __future__ import annotations

import signal
import sys
import threading
from pathlib import Path
from queue import Queue
from typing import List

from loguru import logger

from safetyvision.config import SafetyVisionConfig, load_config
from safetyvision.workers.alert import AlertWorker
from safetyvision.workers.capture import CaptureWorker
from safetyvision.workers.decision import DecisionWorker
from safetyvision.workers.inference import InferenceWorker
from safetyvision.workers.metrics import MetricsCollector, MetricsWorker


def _setup_logging(cfg: SafetyVisionConfig) -> None:
    logger.remove()

    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"
    if cfg.logging.json_output:
        fmt = (
            '{{"ts":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}","level":"{level}",'
            '"logger":"{name}","fn":"{function}","line":{line},"msg":"{message}"}}'
        )

    logger.add(
        sys.stderr,
        level=cfg.logging.level,
        format=fmt,
        colorize=not cfg.logging.json_output,
    )
    logger.add(
        str(log_dir / "safetyvision.log"),
        level=cfg.logging.level,
        format=fmt,
        rotation=f"{cfg.logging.max_size_mb} MB",
        retention="7 days",
        compression="gz",
    )


def _startup_checks(cfg: SafetyVisionConfig) -> list[str]:
    errors: list[str] = []

    if not Path(cfg.model.path_hef).exists():
        errors.append(f"HEF model not found: {cfg.model.path_hef}")

    for label, path in [("siren", cfg.alert.siren_wav), ("voice", cfg.alert.voice_wav)]:
        if not Path(path).exists():
            logger.warning("Audio file missing ({}): {}", label, path)

    return errors


class Supervisor:
    """Main pipeline orchestrator."""

    def __init__(self, cfg: SafetyVisionConfig):
        self.cfg = cfg
        self._stop = threading.Event()
        self._metrics = MetricsCollector(
            state_path=Path(cfg.logging.log_dir) / "metrics_state.json"
        )

        qsize = cfg.perf.max_queue_size
        self._capture_q: Queue = Queue(maxsize=qsize * max(1, len(cfg.input.cameras)))
        self._detection_q: Queue = Queue(maxsize=qsize)
        self._alert_q: Queue = Queue(maxsize=qsize * 4)

        self._captures: List[CaptureWorker] = [
            CaptureWorker(
                cfg,
                camera,
                self._capture_q,
                self._stop,
                latency_cb=self._metrics.record_capture_latency,
                frame_cb=self._metrics.record_capture_frame,
                drop_cb=self._metrics.record_drop,
            )
            for camera in cfg.input.cameras
        ]
        self._inference = InferenceWorker(
            cfg,
            self._capture_q,
            self._detection_q,
            self._stop,
            latency_cb=self._metrics.record_inference_latency,
            frame_cb=self._metrics.record_inference_frame,
        )
        self._decision = DecisionWorker(
            cfg,
            self._detection_q,
            self._alert_q,
            self._stop,
            latency_cb=self._metrics.record_decision_latency,
            frame_cb=self._metrics.record_decision_frame,
            alert_cb=self._metrics.record_alert,
            event_cb=self._metrics.record_detection_event,
            zone_yellow_cb=self._metrics.record_yellow_entry,
            zone_red_cb=self._metrics.record_red_entry,
        )
        self._alert = AlertWorker(
            cfg, self._alert_q, self._stop, audio_done_cb=self._decision.record_audio_done
        )
        self._metrics_worker = MetricsWorker(cfg, self._metrics, self._stop)

    def run(self) -> None:
        _setup_logging(self.cfg)
        logger.info(
            "SafetyVision starting (runtime={}, cameras={})",
            self.cfg.model.runtime,
            [c.id for c in self.cfg.input.cameras],
        )

        errors = _startup_checks(self.cfg)
        if errors:
            for e in errors:
                logger.error("Startup check failed: {}", e)
            logger.error("Aborting due to startup check failures")
            sys.exit(1)

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self._alert.start()
        self._decision.start()
        self._inference.start()
        for cap in self._captures:
            cap.start()
        self._metrics_worker.start()

        logger.info("All workers started")

        try:
            while not self._stop.is_set():
                self._stop.wait(timeout=self.cfg.health.heartbeat_interval_sec)
                self._health_check()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        logger.info("Shutting down SafetyVision...")
        self._stop.set()

        for cap in self._captures:
            cap.stop()
        self._inference.stop()
        self._decision.stop()
        self._alert.stop()
        self._metrics_worker.stop()

        snap = self._metrics.snapshot()
        logger.info(
            "Final stats: inf_fps={}, cap_fps={}, dec_fps={}, total_latency={}ms, dropped={}, alerts={}, uptime={}s",
            snap.fps, snap.capture_fps, snap.decision_fps, snap.total_latency_ms,
            snap.frames_dropped, snap.alert_count, snap.uptime_sec,
        )
        logger.info("SafetyVision stopped cleanly")

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Received signal {}, initiating shutdown", signum)
        self._stop.set()

    def _health_check(self) -> None:
        for cap in self._captures:
            if not cap.is_connected:
                logger.warning("Health: camera {} disconnected", cap.camera_id)

        snap = self._metrics.snapshot()
        if snap.fps > 0 and snap.total_latency_ms > 120:
            logger.warning(
                "Health: latency {:.1f}ms exceeds 120ms target", snap.total_latency_ms
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SafetyVision Forklift Safety System")
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to config YAML (default: config/safetyvision.yaml or $SAFETYVISION_CONFIG)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    supervisor = Supervisor(cfg)
    supervisor.run()


if __name__ == "__main__":
    main()
