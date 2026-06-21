#!/usr/bin/env bash
# SafetyVision – Hotspot (Access Point) setup for Raspberry Pi 5
# Target OS: Raspberry Pi OS Bookworm / Debian 13 (uses NetworkManager, not dhcpcd)
# Run as root.
# eth0 is left untouched (used for PoE camera switch).
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Error: must run as root (use sudo)" >&2
    exit 1
fi

# -- Config (edit these or pass as args) -----------------------------------
AP_IFACE="${1:-wlan0}"
AP_SSID="${2:-SafetyVision}"
AP_PASS="${3:-safetyvision123}"
OFFLINE="${OFFLINE:-0}"
AP_IP="192.168.10.1"
AP_SUBNET="192.168.10.0/24"
DHCP_RANGE_START="192.168.10.10"
DHCP_RANGE_END="192.168.10.50"
DHCP_LEASE="12h"

have_pkg() {
    dpkg -s "$1" >/dev/null 2>&1
}

echo "=== SafetyVision Pi 5 Hotspot Setup ==="
echo "Interface: $AP_IFACE"
echo "SSID:      $AP_SSID"
echo "IP:        $AP_IP"
echo

# 1. Install packages (iptables is NOT preinstalled on Bookworm)
echo "[1/7] Installing packages..."
if [[ "${OFFLINE}" == "1" ]]; then
    echo "  -> offline mode; skipping apt package install"
    for pkg in hostapd dnsmasq iptables iptables-persistent; do
        if ! have_pkg "${pkg}"; then
            echo "  !! offline mode requires preinstalled package: ${pkg}" >&2
            exit 1
        fi
    done
else
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq hostapd dnsmasq iptables iptables-persistent
fi

# 2. Stop services during config
echo "[2/7] Stopping services for reconfiguration..."
systemctl stop hostapd 2>/dev/null || true
systemctl stop dnsmasq 2>/dev/null || true

# 3. Tell NetworkManager to ignore wlan0 (it would fight hostapd otherwise)
echo "[3/7] Excluding ${AP_IFACE} from NetworkManager..."
mkdir -p /etc/NetworkManager/conf.d
cat > /etc/NetworkManager/conf.d/safetyvision.conf <<NMCONF
[keyfile]
unmanaged-devices=interface-name:${AP_IFACE}
NMCONF
systemctl restart NetworkManager

# 4. hostapd config
echo "[4/7] Writing hostapd config..."
cat > /etc/hostapd/hostapd.conf <<HOSTAPD
interface=${AP_IFACE}
driver=nl80211
ssid=${AP_SSID}
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=${AP_PASS}
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
HOSTAPD

sed -i 's|^#DAEMON_CONF=.*|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd 2>/dev/null || true

# 5. dnsmasq config
echo "[5/7] Writing dnsmasq config..."
cat > /etc/dnsmasq.d/safetyvision.conf <<DNSMASQ
interface=${AP_IFACE}
dhcp-range=${DHCP_RANGE_START},${DHCP_RANGE_END},${DHCP_LEASE}
address=/safetyvision.local/${AP_IP}
DNSMASQ

# 6. IP forwarding + NAT (hotspot clients → eth0 cameras)
echo "[6/7] Enabling IP forwarding + iptables NAT..."
# Bookworm does not use /etc/sysctl.conf — use /etc/sysctl.d/
cat > /etc/sysctl.d/99-safetyvision.conf <<SYSCTL
net.ipv4.ip_forward=1
SYSCTL
sysctl -w net.ipv4.ip_forward=1 >/dev/null

# Add iptables rules (idempotent via -C check)
iptables -t nat -C POSTROUTING -o eth0 -j MASQUERADE 2>/dev/null || \
    iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
iptables -C FORWARD -i eth0 -o "${AP_IFACE}" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || \
    iptables -A FORWARD -i eth0 -o "${AP_IFACE}" -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -C FORWARD -i "${AP_IFACE}" -o eth0 -j ACCEPT 2>/dev/null || \
    iptables -A FORWARD -i "${AP_IFACE}" -o eth0 -j ACCEPT
netfilter-persistent save

# 7. Systemd service to assign static IP at boot (no dhcpcd on Bookworm)
echo "[7/7] Writing boot service for static IP..."
cat > /etc/systemd/system/safetyvision-hotspot.service <<SERVICE
[Unit]
Description=SafetyVision Hotspot - static IP on ${AP_IFACE}
Before=hostapd.service dnsmasq.service
After=network.target NetworkManager.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/sbin/ip addr flush dev ${AP_IFACE}
ExecStart=/sbin/ip addr add ${AP_IP}/24 dev ${AP_IFACE}
ExecStart=/sbin/ip link set ${AP_IFACE} up

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload

# Set static IP immediately
ip addr flush dev "${AP_IFACE}" 2>/dev/null || true
ip addr add "${AP_IP}/24" dev "${AP_IFACE}"
ip link set "${AP_IFACE}" up

# Enable all services for boot persistence
systemctl unmask hostapd
systemctl enable safetyvision-hotspot hostapd dnsmasq netfilter-persistent
systemctl start hostapd dnsmasq

echo
echo "=== Hotspot active ==="
echo "SSID:     $AP_SSID"
echo "Password: $AP_PASS"
echo "Pi IP:    $AP_IP"
echo "UI:       http://${AP_IP}:8080"
echo
echo "eth0 is untouched — plug your PoE camera switch there."
echo "All services enabled at boot."
