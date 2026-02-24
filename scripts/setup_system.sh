#!/usr/bin/env bash
# SafetyVision – Ubuntu Server system setup script
# Run as root on a fresh Ubuntu Server LTS install.
set -euo pipefail

echo "=== SafetyVision System Setup ==="

# 1. CPU Performance Governor
echo "[1/6] Setting CPU performance governor..."
apt-get update -qq
apt-get install -y -qq linux-tools-common linux-tools-generic cpufrequtils
echo 'GOVERNOR="performance"' > /etc/default/cpufrequtils
systemctl restart cpufrequtils || true
# Also set immediately
cpupower frequency-set -g performance 2>/dev/null || true

# 2. Runtime dependencies
echo "[2/6] Installing runtime dependencies..."
apt-get install -y -qq \
    python3.11 python3.11-venv python3-pip \
    ffmpeg \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    v4l-utils \
    sox alsa-utils libportaudio2

# 3. System user
echo "[3/6] Creating safetyvision system user..."
if ! id safetyvision &>/dev/null; then
    useradd --system --create-home --shell /usr/sbin/nologin safetyvision
    usermod -aG audio,video safetyvision
fi

# 4. OS tuning
echo "[4/6] Applying OS tuning..."
# File descriptor limits
cat > /etc/security/limits.d/safetyvision.conf <<LIMITS
safetyvision soft nofile 8192
safetyvision hard nofile 16384
LIMITS

# Journald size cap
mkdir -p /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/safetyvision.conf <<JOURNAL
[Journal]
SystemMaxUse=200M
JOURNAL
systemctl restart systemd-journald

# 5. Deploy directory
echo "[5/6] Setting up /opt/safetyvision..."
mkdir -p /opt/safetyvision
# Copy project files (run from project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cp -r "$PROJECT_DIR"/src "$PROJECT_DIR"/config "$PROJECT_DIR"/assets \
      "$PROJECT_DIR"/requirements.txt "$PROJECT_DIR"/pyproject.toml \
      /opt/safetyvision/

# Create venv
python3.11 -m venv /opt/safetyvision/.venv
/opt/safetyvision/.venv/bin/pip install --upgrade pip
/opt/safetyvision/.venv/bin/pip install -e /opt/safetyvision/

chown -R safetyvision:safetyvision /opt/safetyvision
mkdir -p /var/log/safetyvision
chown safetyvision:safetyvision /var/log/safetyvision

# 6. Install systemd service
echo "[6/6] Installing systemd service..."
cp "$PROJECT_DIR"/deploy/safetyvision.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable safetyvision

echo ""
echo "=== Setup complete ==="
echo "Place your ONNX model at: /opt/safetyvision/models/yolo26n.onnx"
echo "Place WAV files at: /opt/safetyvision/assets/audio/"
echo "Edit config: /opt/safetyvision/config/safetyvision.yaml"
echo "Start: systemctl start safetyvision"
echo "Logs:  journalctl -u safetyvision -f"
