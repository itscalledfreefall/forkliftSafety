#!/usr/bin/env bash
# SafetyVision – Continuous mainstream recorder
# Records RTSP mainstream as MP4 segments with no re-encoding (stream copy).
# Auto-rotates hourly, auto-deletes oldest files when disk usage exceeds threshold.
#
# Usage:  ./scripts/record.sh [config_path]
# Config: reads rtsp_url_main from safetyvision.yaml
set -euo pipefail

CONFIG="${1:-config/safetyvision.yaml}"
RECORD_DIR="${SAFETYVISION_RECORD_DIR:-/home/inofatech/forkliftSafety/recordings}"
SEGMENT_SEC="${SAFETYVISION_SEGMENT_SEC:-3600}"       # 1 hour per file
MAX_DISK_PERCENT="${SAFETYVISION_MAX_DISK_PERCENT:-80}" # delete oldest when above this
MIN_FREE_GB="${SAFETYVISION_MIN_FREE_GB:-5}"            # or when free space below this

# ---------------------------------------------------------------------------
# Parse RTSP URL from config
# ---------------------------------------------------------------------------
RTSP_URL=$(python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
url = cfg.get('input', {}).get('rtsp_url_main', '')
if not url:
    url = cfg.get('input', {}).get('rtsp_url', '')
print(url)
")

if [ -z "$RTSP_URL" ]; then
    echo "ERROR: No RTSP URL found in $CONFIG" >&2
    exit 1
fi

echo "Recording from: $RTSP_URL"
echo "Output dir:     $RECORD_DIR"
echo "Segment length: ${SEGMENT_SEC}s"
echo "Max disk usage: ${MAX_DISK_PERCENT}%"

mkdir -p "$RECORD_DIR"

# ---------------------------------------------------------------------------
# Cleanup old recordings
# ---------------------------------------------------------------------------
cleanup_old() {
    # Check disk usage
    local usage
    usage=$(df "$RECORD_DIR" --output=pcent | tail -1 | tr -d ' %')
    local free_gb
    free_gb=$(df "$RECORD_DIR" --output=avail -B1G | tail -1 | tr -d ' ')

    while (( usage > MAX_DISK_PERCENT )) || (( free_gb < MIN_FREE_GB )); do
        # Find oldest recording
        local oldest
        oldest=$(find "$RECORD_DIR" -name '*.mp4' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-)
        if [ -z "$oldest" ]; then
            echo "WARN: Disk full but no recordings to delete" >&2
            break
        fi
        echo "Cleanup: deleting $oldest (disk ${usage}%, free ${free_gb}GB)"
        rm -f "$oldest"
        usage=$(df "$RECORD_DIR" --output=pcent | tail -1 | tr -d ' %')
        free_gb=$(df "$RECORD_DIR" --output=avail -B1G | tail -1 | tr -d ' ')
    done
}

# ---------------------------------------------------------------------------
# Main recording loop
# ---------------------------------------------------------------------------
while true; do
    cleanup_old

    TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    OUTFILE="$RECORD_DIR/${TIMESTAMP}.mp4"

    echo "[$(date)] Starting segment: $OUTFILE"

    # FFmpeg: stream copy (no re-encoding), segment duration, TCP transport
    ffmpeg -hide_banner -loglevel warning \
        -rtsp_transport tcp \
        -i "$RTSP_URL" \
        -c copy \
        -movflags +frag_keyframe+empty_moov+faststart \
        -t "$SEGMENT_SEC" \
        -y "$OUTFILE" \
        2>&1 || true

    # If ffmpeg exits early (camera disconnect), wait before retry
    if [ ! -s "$OUTFILE" ]; then
        echo "WARN: Empty recording, camera may be offline. Retrying in 10s..."
        rm -f "$OUTFILE"
        sleep 10
    else
        SIZE=$(du -h "$OUTFILE" | cut -f1)
        echo "[$(date)] Segment complete: $OUTFILE ($SIZE)"
    fi
done
