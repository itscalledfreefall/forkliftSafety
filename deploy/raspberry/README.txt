SafetyVision on Raspberry Pi
============================

This profile keeps the same RTSP camera workflow and tunes runtime defaults for ARM CPUs.

Quick Install
-------------

  cd ~/forkliftSafety
  sudo bash scripts/setup_raspberry_pi.sh

Configure
---------

1) Edit camera URLs:
   /opt/safetyvision/config/safetyvision.raspberry.yaml
2) Put model file:
   /opt/safetyvision/models/yolo26n.onnx
3) Put audio files:
   /opt/safetyvision/assets/audio/danger.wav
   /opt/safetyvision/assets/audio/medium.wav

Use Raspberry Profile
---------------------

  sudo systemctl edit safetyvision

Add:

  [Service]
  Environment=SAFETYVISION_CONFIG=/opt/safetyvision/config/safetyvision.raspberry.yaml

Then restart:

  sudo systemctl daemon-reload
  sudo systemctl restart safetyvision safetyvision-ui

Notes
-----

- If sounddevice fails on Pi, alert worker falls back to aplay automatically.
- If openvino is selected on ARM, runtime falls back to onnxruntime.
