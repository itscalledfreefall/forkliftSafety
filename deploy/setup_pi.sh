#!/usr/bin/env bash
# SafetyVision Raspberry Pi setup (RTSP + Web UI)
# Run as root on Raspberry Pi OS / Ubuntu Server ARM.
set -euo pipefail

echo "=== SafetyVision Raspberry Pi Setup ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="${TARGET_DIR:-$PROJECT_DIR}"
RUN_USER="${RUN_USER:-${SUDO_USER:-$(stat -c %U "$PROJECT_DIR")}}"
OFFLINE="${OFFLINE:-0}"
if [[ -z "${RUN_GROUP:-}" ]]; then
    if id "${RUN_USER}" >/dev/null 2>&1; then
        RUN_GROUP="$(id -gn "$RUN_USER")"
    else
        RUN_GROUP="$RUN_USER"
    fi
fi
CONFIG_PATH="${CONFIG_PATH:-$TARGET_DIR/config/safetyvision.raspberry.yaml}"
DEFAULT_VENV_DIR="${TARGET_DIR}/.venv"
if [[ "${OFFLINE}" == "1" && ! -x "${DEFAULT_VENV_DIR}/bin/python" && -x /opt/safetyvision/.venv/bin/python ]]; then
    VENV_DIR="${VENV_DIR:-/opt/safetyvision/.venv}"
else
    VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"
fi

render_template() {
    local template="$1"
    local output="$2"
    sed \
        -e "s|@TARGET_DIR@|${TARGET_DIR}|g" \
        -e "s|@RUN_USER@|${RUN_USER}|g" \
        -e "s|@RUN_GROUP@|${RUN_GROUP}|g" \
        -e "s|@VENV_DIR@|${VENV_DIR}|g" \
        -e "s|@CONFIG_PATH@|${CONFIG_PATH}|g" \
        "$template" > "$output"
}

echo "[1/6] Installing system packages..."
if [[ "${OFFLINE}" == "1" ]]; then
    echo "  -> offline mode; skipping apt package install"
    command -v python3 >/dev/null 2>&1 || {
        echo "  !! python3 is required in offline mode"
        exit 1
    }
    dpkg -s hailo-all >/dev/null 2>&1 || {
        echo "  !! hailo-all must already be installed in offline mode"
        exit 1
    }
else
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
fi

echo "[2/6] Preparing runtime user..."
if ! id "${RUN_USER}" >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash "${RUN_USER}"
fi
usermod -aG audio,video "${RUN_USER}"

echo "[3/6] Preparing project at ${TARGET_DIR}..."
mkdir -p "${TARGET_DIR}"
if [[ "${TARGET_DIR}" != "${PROJECT_DIR}" ]]; then
    apt-get install -y -qq rsync
    rsync -a \
        --exclude ".git" \
        --exclude ".venv" \
        --exclude ".pytest_cache" \
        --exclude "logs" \
        --exclude "recordings" \
        "${PROJECT_DIR}/" "${TARGET_DIR}/"
fi

mkdir -p "${TARGET_DIR}/models" "${TARGET_DIR}/logs" "${TARGET_DIR}/recordings"
chown -R "${RUN_USER}:${RUN_GROUP}" \
    "${TARGET_DIR}/config" \
    "${TARGET_DIR}/models" \
    "${TARGET_DIR}/logs" \
    "${TARGET_DIR}/recordings"

echo "[4/6] Creating virtual environment..."
if [[ "${OFFLINE}" == "1" ]]; then
    echo "  -> offline mode; using existing virtual environment at ${VENV_DIR}"
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        echo "  !! no usable virtual environment found at ${VENV_DIR}"
        echo "     online setup once, or pass VENV_DIR=/path/to/existing/.venv"
        exit 1
    fi
else
    # --system-site-packages so the venv can import hailo_platform (apt-installed)
    python3 -m venv --system-site-packages "${VENV_DIR}"
    chown -R "${RUN_USER}:${RUN_GROUP}" "${VENV_DIR}"
    sudo -u "${RUN_USER}" "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
    sudo -u "${RUN_USER}" "${VENV_DIR}/bin/pip" install -e "${TARGET_DIR}"
    sudo -u "${RUN_USER}" "${VENV_DIR}/bin/pip" install -e "${TARGET_DIR}[webui]"
fi

echo "[5/6] Installing systemd services..."
render_template "${PROJECT_DIR}/deploy/safetyvision.service" /etc/systemd/system/safetyvision.service
render_template "${PROJECT_DIR}/deploy/safetyvision-ui.service" /etc/systemd/system/safetyvision-ui.service
render_template "${PROJECT_DIR}/deploy/safetyvision-sudoers" /etc/sudoers.d/safetyvision
chmod 440 /etc/sudoers.d/safetyvision

mkdir -p /var/log/safetyvision
chown -R "${RUN_USER}:${RUN_GROUP}" /var/log/safetyvision

echo "[6/6] Enabling services..."
systemctl daemon-reload
systemctl enable safetyvision.service safetyvision-ui.service

echo ""
echo "=== Raspberry Pi setup complete ==="
echo "Project directory:      ${TARGET_DIR}"
echo "Virtualenv directory:   ${VENV_DIR}"
echo "Service user/group:     ${RUN_USER}:${RUN_GROUP}"
echo "Rendered config path:   ${CONFIG_PATH}"
echo "1) Verify Hailo device: hailortcli fw-control identify"
echo "2) Verify HEF present:  ls /usr/share/hailo-models/yolov6n_h8l.hef"
echo "3) Edit RTSP URLs in:   ${CONFIG_PATH}"
echo "4) Start services:"
echo "   sudo systemctl restart safetyvision safetyvision-ui"
