#!/usr/bin/env python3
"""ROI Zone Calibration Utility for SafetyVision.

Displays a camera frame with the 3 horizontal band zones overlaid:
  - Green (top)   = no sound
  - Yellow (mid)  = medium sound
  - Red (bottom)  = danger sound

Usage:
    python scripts/calibrate_zones.py --source /dev/video0
    python scripts/calibrate_zones.py --source snapshot.jpg -c config/safetyvision.yaml

Controls:
    UP/DOWN arrows  – move yellow cut line
    LEFT/RIGHT      – move red cut line
    SHIFT + arrows  – fine-tune (1px instead of 5px)
    's'             – save to config YAML
    'r'             – reset to defaults (0.33 / 0.66)
    'q' / ESC       – quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# -- Colours (BGR) -----------------------------------------------------------
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
GREEN = (0, 180, 0)
WHITE = (255, 255, 255)
ALPHA = 0.25


def _draw_zones(frame: np.ndarray, yellow_y: float, red_y: float) -> np.ndarray:
    """Draw the three horizontal band zones on a copy of the frame."""
    h, w = frame.shape[:2]
    canvas = frame.copy()
    overlay = canvas.copy()

    y_yellow = int(yellow_y * h)
    y_red = int(red_y * h)

    # Green band (top)
    cv2.rectangle(overlay, (0, 0), (w, y_yellow), GREEN, -1)
    # Yellow band (middle)
    cv2.rectangle(overlay, (0, y_yellow), (w, y_red), YELLOW, -1)
    # Red band (bottom)
    cv2.rectangle(overlay, (0, y_red), (w, h), RED, -1)

    canvas = cv2.addWeighted(overlay, ALPHA, canvas, 1.0 - ALPHA, 0)

    # Cut lines
    cv2.line(canvas, (0, y_yellow), (w, y_yellow), YELLOW, 2)
    cv2.line(canvas, (0, y_red), (w, y_red), RED, 2)

    # Labels
    cv2.putText(canvas, f"GREEN (no sound)", (10, y_yellow // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    mid_y = (y_yellow + y_red) // 2
    cv2.putText(canvas, f"YELLOW (medium)", (10, mid_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    bot_y = (y_red + h) // 2
    cv2.putText(canvas, f"RED (danger)", (10, bot_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    # HUD
    hud = [
        f"yellow_start_y: {yellow_y:.3f}   red_start_y: {red_y:.3f}",
        "UP/DOWN=yellow  LEFT/RIGHT=red  SHIFT=fine  s=save  r=reset  q=quit",
    ]
    for i, line in enumerate(hud):
        cv2.putText(canvas, line, (10, h - 30 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    return canvas


def _grab_frame(source: str) -> np.ndarray:
    p = Path(source)
    if p.exists() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        frame = cv2.imread(str(p))
        if frame is not None:
            return frame

    if source.startswith("/dev/"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}", file=sys.stderr)
        sys.exit(1)

    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"ERROR: Could not read frame from: {source}", file=sys.stderr)
        sys.exit(1)
    return frame


def _save_yaml(config_path: str, yellow_y: float, red_y: float) -> None:
    path = Path(config_path)
    data = {}
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    alert = data.setdefault("alert", {})
    alert["yellow_start_y"] = round(yellow_y, 4)
    alert["red_start_y"] = round(red_y, 4)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)

    print(f"\nSaved to {path}:")
    print(f"  yellow_start_y: {yellow_y:.4f}")
    print(f"  red_start_y:    {red_y:.4f}")


def main():
    parser = argparse.ArgumentParser(description="SafetyVision Zone Calibrator")
    parser.add_argument("--source", "-s", required=True,
                        help="Camera device, RTSP URL, or image file")
    parser.add_argument("-c", "--config", default="config/safetyvision.yaml")
    args = parser.parse_args()

    frame = _grab_frame(args.source)
    h = frame.shape[0]

    yellow_y = 0.33
    red_y = 0.66

    # Load existing values if present
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
        alert = data.get("alert", {})
        yellow_y = alert.get("yellow_start_y", yellow_y)
        red_y = alert.get("red_start_y", red_y)

    win = "SafetyVision Zone Calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        canvas = _draw_zones(frame, yellow_y, red_y)
        cv2.imshow(win, canvas)

        key = cv2.waitKeyEx(30)
        step = 1.0 / h  # fine step = 1 pixel

        # Detect shift (platform-dependent, check both)
        coarse = 5.0 / h

        if key == -1:
            continue
        elif key in (ord("q"), 27):
            break
        elif key == ord("s"):
            _save_yaml(args.config, yellow_y, red_y)
        elif key == ord("r"):
            yellow_y, red_y = 0.33, 0.66
            print("Reset to defaults")
        # UP arrow = move yellow line up
        elif key == 0xFF52 or key == 82:  # up
            yellow_y = max(0.01, yellow_y - coarse)
        elif key == 0xFF54 or key == 84:  # down
            yellow_y = min(red_y - 0.01, yellow_y + coarse)
        elif key == 0xFF51 or key == 81:  # left = move red line up
            red_y = max(yellow_y + 0.01, red_y - coarse)
        elif key == 0xFF53 or key == 83:  # right = move red line down
            red_y = min(0.99, red_y + coarse)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
