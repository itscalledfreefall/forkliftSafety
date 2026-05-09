SafetyVision on Raspberry Pi
============================

This profile keeps the same RTSP camera workflow and tunes runtime defaults for ARM CPUs.
The default deploy model is now single-source: systemd runs directly from the
checkout you install from, instead of copying the project into `/opt/safetyvision`.

Quick Install
-------------

  cd ~/forkliftSafety
  sudo bash deploy/setup_pi.sh

This renders the systemd units to point at the current checkout and its local
`.venv`. If you still want a copied deploy tree, override `TARGET_DIR`:

  sudo TARGET_DIR=/opt/safetyvision RUN_USER=safetyvision RUN_GROUP=safetyvision bash deploy/setup_pi.sh

Configure
---------

1) Edit camera URLs:
   ~/forkliftSafety/config/safetyvision.raspberry.yaml
2) Put model file:
   ~/forkliftSafety/models/yolo26n.onnx
3) Put audio files:
   ~/forkliftSafety/assets/audio/danger.wav
   ~/forkliftSafety/assets/audio/medium.wav

Use Raspberry Profile
---------------------

  sudo systemctl daemon-reload
  sudo systemctl restart safetyvision safetyvision-ui

Notes
-----

- If sounddevice fails on Pi, alert worker falls back to aplay automatically.
- If openvino is selected on ARM, runtime falls back to onnxruntime.
- The generated units use the checkout path you installed from unless `TARGET_DIR`
  is overridden during setup.
