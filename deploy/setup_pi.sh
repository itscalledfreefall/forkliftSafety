#!/usr/bin/env bash
# SafetyVision Raspberry Pi setup (RTSP + Web UI)
# Run as root on Raspberry Pi OS / Ubuntu Server ARM.
set -euo pipefail

echo "=== SafetyVision Raspberry Pi Setup ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="/opt/safetyvision"

echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3 python3-venv python3-pip \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    alsa-utils libportaudio2

# Hailo runtime — provides the 'hailo_platform' Python bindings, PCIe driver,
# and /usr/share/hailo-models HEFs. Requires Raspberry Pi OS / Bookworm.
if ! dpkg -s hailo-all >/dev/null 2>&1; then
    echo "  -> installing hailo-all (requires apt repo configured)"
    apt-get install -y -qq hailo-all || {
        echo "  !! 'hailo-all' not found. Follow Raspberry Pi AI Kit docs:"
        echo "     https://www.raspberrypi.com/documentation/accessories/ai-kit.html"
        exit 1
    }
fi

echo "[2/6] Creating service user..."
if ! id safetyvision >/dev/null 2>&1; then
    useradd --system --create-home --shell /usr/sbin/nologin safetyvision
fi
usermod -aG audio,video safetyvision

echo "[3/6] Deploying project to ${TARGET_DIR}..."
mkdir -p "${TARGET_DIR}"
cp -r \
    "${PROJECT_DIR}/src" \
    "${PROJECT_DIR}/config" \
    "${PROJECT_DIR}/assets" \
    "${PROJECT_DIR}/scripts" \
    "${PROJECT_DIR}/deploy" \
    "${PROJECT_DIR}/pyproject.toml" \
    "${PROJECT_DIR}/requirements.txt" \
    "${TARGET_DIR}/"

mkdir -p "${TARGET_DIR}/models" "${TARGET_DIR}/logs" "${TARGET_DIR}/recordings"

echo "[4/6] Creating virtual environment..."
# --system-site-packages so the venv can import hailo_platform (apt-installed)
python3 -m venv --system-site-packages "${TARGET_DIR}/.venv"
"${TARGET_DIR}/.venv/bin/pip" install --upgrade pip setuptools wheel
"${TARGET_DIR}/.venv/bin/pip" install -e "${TARGET_DIR}"
"${TARGET_DIR}/.venv/bin/pip" install -e "${TARGET_DIR}[webui]"

echo "[5/6] Installing systemd services..."
cp "${PROJECT_DIR}/deploy/safetyvision.service" /etc/systemd/system/
cp "${PROJECT_DIR}/deploy/safetyvision-ui.service" /etc/systemd/system/
cp "${PROJECT_DIR}/deploy/safetyvision-sudoers" /etc/sudoers.d/safetyvision
chmod 440 /etc/sudoers.d/safetyvision

chown -R safetyvision:safetyvision "${TARGET_DIR}"
mkdir -p /var/log/safetyvision
chown -R safetyvision:safetyvision /var/log/safetyvision

echo "[6/6] Enabling services..."
systemctl daemon-reload
systemctl enable safetyvision.service safetyvision-ui.service

echo ""
echo "=== Raspberry Pi setup complete ==="
echo "1) Verify Hailo device: hailortcli fw-control identify"
echo "2) Verify HEF present:  ls /usr/share/hailo-models/yolov6n_h8l.hef"
echo "3) Edit RTSP URLs in:   ${TARGET_DIR}/config/safetyvision.raspberry.yaml"
echo "4) Point the service at the Pi config if needed:"
echo "   sudo systemctl edit safetyvision"
echo "   (add Environment=SAFETYVISION_CONFIG=${TARGET_DIR}/config/safetyvision.raspberry.yaml)"
echo "5) Start services:"
echo "   sudo systemctl restart safetyvision safetyvision-ui"
