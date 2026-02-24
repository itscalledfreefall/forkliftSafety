#!/usr/bin/env python3
"""Benchmark script – measures RTSP vs USB capture + inference throughput."""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

from safetyvision.config import load_config
from safetyvision.workers.inference import _letterbox, _preprocess


def benchmark_capture(cfg, duration_sec: int = 60) -> dict:
    """Benchmark raw capture FPS and latency."""
    if cfg.input.mode == "usb":
        cap = cv2.VideoCapture(cfg.input.usb_device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(cfg.input.rtsp_url)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.input.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.input.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.input.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return {"error": "Cannot open camera"}

    frame_times = []
    drops = 0
    start = time.monotonic()

    while (time.monotonic() - start) < duration_sec:
        t0 = time.monotonic()
        ret, frame = cap.read()
        t1 = time.monotonic()
        if not ret:
            drops += 1
            continue
        frame_times.append((t1 - t0) * 1000)

    cap.release()

    if not frame_times:
        return {"error": "No frames captured"}

    elapsed = time.monotonic() - start
    return {
        "mode": cfg.input.mode,
        "duration_sec": round(elapsed, 1),
        "total_frames": len(frame_times),
        "fps": round(len(frame_times) / elapsed, 1),
        "latency_p50_ms": round(statistics.median(frame_times), 2),
        "latency_p95_ms": round(sorted(frame_times)[int(len(frame_times) * 0.95)], 2),
        "drops": drops,
    }


def benchmark_inference(cfg, n_frames: int = 300) -> dict:
    """Benchmark inference throughput on synthetic frames."""
    # Load model
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = cfg.perf.inference_threads
    session = ort.InferenceSession(
        cfg.model.path_onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # Synthetic frame
    dummy = np.random.randint(0, 255, (cfg.input.height, cfg.input.width, 3), dtype=np.uint8)
    blob, _, _ = _preprocess(dummy, cfg.model.input_size)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: blob})

    latencies = []
    for _ in range(n_frames):
        t0 = time.monotonic()
        session.run(None, {input_name: blob})
        t1 = time.monotonic()
        latencies.append((t1 - t0) * 1000)

    return {
        "runtime": cfg.model.runtime,
        "n_frames": n_frames,
        "fps": round(1000 / statistics.mean(latencies), 1),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
    }


def main():
    parser = argparse.ArgumentParser(description="SafetyVision Benchmark")
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--capture-sec", type=int, default=60)
    parser.add_argument("--inference-frames", type=int, default=300)
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not args.skip_capture:
        print(f"\n=== Capture Benchmark ({cfg.input.mode}) ===")
        result = benchmark_capture(cfg, args.capture_sec)
        for k, v in result.items():
            print(f"  {k}: {v}")

    if not args.skip_inference:
        print(f"\n=== Inference Benchmark ({cfg.model.runtime}) ===")
        result = benchmark_inference(cfg, args.inference_frames)
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
