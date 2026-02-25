#!/usr/bin/env bash
# SafetyVision – Hotspot (Access Point) setup for IPC
# Run as root. Creates a WPA2-secured WiFi AP on a specified interface.
set -euo pipefail

# -- Config (edit these) ------------------------------------------------
AP_IFACE="${1:-wlan0}"
AP_SSID="${2:-SafetyVision}"
AP_PASS="${3:-safetyvision123}"
AP_IP="192.168.10.1"
AP_SUBNET="192.168.10.0/24"
DHCP_RANGE="192.168.10.10,192.168.10.50,12h"

echo "=== SafetyVision Hotspot Setup ==="
echo "Interface: $AP_IFACE"
echo "SSID:      $AP_SSID"
echo "IP:        $AP_IP"

# 1. Install packages
apt-get update -qq
apt-get install -y -qq hostapd dnsmasq

# 2. Stop services during config
systemctl stop hostapd || true
systemctl stop dnsmasq || true

# 3. Static IP on AP interface
cat > /etc/netplan/90-safetyvision-ap.yaml <<NETPLAN
network:
  version: 2
  wifis:
    ${AP_IFACE}:
      dhcp4: false
      addresses:
        - ${AP_IP}/24
NETPLAN
netplan apply || true

# Fallback for non-netplan systems
ip addr flush dev "$AP_IFACE" 2>/dev/null || true
ip addr add "${AP_IP}/24" dev "$AP_IFACE" 2>/dev/null || true
ip link set "$AP_IFACE" up

# 4. hostapd config
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

# Point hostapd to config
sed -i 's|^#DAEMON_CONF=.*|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd 2>/dev/null || true

# 5. dnsmasq config
cat > /etc/dnsmasq.d/safetyvision.conf <<DNSMASQ
interface=${AP_IFACE}
dhcp-range=${DHCP_RANGE}
address=/safetyvision.local/${AP_IP}
DNSMASQ

# 6. Enable and start
systemctl unmask hostapd
systemctl enable hostapd dnsmasq
systemctl start hostapd dnsmasq

echo ""
echo "=== Hotspot active ==="
echo "SSID:     $AP_SSID"
echo "Password: $AP_PASS"
echo "UI:       http://${AP_IP}:8080"
echo ""
echo "Connect a device to this WiFi and open the URL above."
