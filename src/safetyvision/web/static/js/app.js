/* SafetyVision UI - Main JS */
(function () {
  'use strict';

  // ── Auth check ──────────────────────────────────────────────
  async function checkAuth() {
    try {
      const res = await fetch('/api/auth/check');
      const data = await res.json();
      if (!data.authenticated) window.location.href = '/login';
    } catch { window.location.href = '/login'; }
  }
  checkAuth();

  // ── Toast ───────────────────────────────────────────────────
  function toast(msg, type) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = 'toast show ' + (type || 'success');
    setTimeout(() => el.className = 'toast', 3000);
  }

  // ── Tabs ────────────────────────────────────────────────────
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
      if (tab.dataset.tab === 'dashboard') startStream();
    });
  });

  // ── Service Status ──────────────────────────────────────────
  async function pollStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      const badge = document.getElementById('serviceStatus');
      badge.textContent = data.service;
      badge.className = 'status-badge ' +
        (data.service === 'active' ? 'active' :
         data.service === 'inactive' ? 'inactive' : 'unknown');
    } catch {}
  }
  pollStatus();
  setInterval(pollStatus, 5000);

  // ── Metrics ─────────────────────────────────────────────────
  function setMetricValue(id, value, digits) {
    const el = document.getElementById(id);
    if (!el) return;
    if (typeof value !== 'number' || Number.isNaN(value)) {
      el.textContent = '--';
      return;
    }
    el.textContent = value.toFixed(digits);
  }

  async function pollMetrics() {
    try {
      const res = await fetch('/api/metrics');
      if (res.status === 401) return;
      const data = await res.json();
      if (!data.available) {
        setMetricValue('metricFps', NaN, 1);
        setMetricValue('metricLatency', NaN, 1);
        setMetricValue('metricCaptureFps', NaN, 1);
        setMetricValue('metricInferenceFps', NaN, 1);
        setMetricValue('metricYellowEntries', NaN, 0);
        setMetricValue('metricRedEntries', NaN, 0);
        return;
      }
      setMetricValue('metricFps', data.fps, 1);
      setMetricValue('metricLatency', data.latency_total_ms, 1);
      setMetricValue('metricCaptureFps', data.capture_fps, 1);
      setMetricValue('metricInferenceFps', data.inference_fps, 1);
      setMetricValue('metricYellowEntries', data.yellow_zone_entries, 0);
      setMetricValue('metricRedEntries', data.red_zone_entries, 0);
    } catch {
      // Keep existing values on transient errors.
    }
  }
  pollMetrics();
  setInterval(pollMetrics, 1000);

  // ── Live Stream ─────────────────────────────────────────────
  function startStream() {
    const img = document.getElementById('liveStream');
    const offline = document.getElementById('streamOffline');
    img.src = '/api/stream.mjpg?overlay=true&t=' + Date.now();
    img.onload = () => { offline.style.display = 'none'; };
    img.onerror = () => { offline.style.display = 'flex'; };
  }
  startStream();

  function escapeHtml(value) {
    return String(value || '').replace(/[&<>"']/g, char => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[char]));
  }

  function cameraCardMarkup(camera) {
    const effectiveZone = camera.effective_zone || {};
    const zone = camera.zone || {};
    const distance = camera.distance || {};
    const yellow = Number(zone.yellow_start_y ?? effectiveZone.yellow_start_y ?? 0.33);
    const red = Number(zone.red_start_y ?? effectiveZone.red_start_y ?? 0.66);
    const warningDistance = Number(distance.warning_distance_m ?? 2.0);
    const dangerDistance = Number(distance.danger_distance_m ?? 1.0);
    const calibrationPath = distance.calibration_path || '';
    const mode = camera.mode || 'zone';

    return `
      <article class="camera-card" data-camera-id="${escapeHtml(camera.id)}">
        <div class="camera-card-header">
          <div>
            <h3>${escapeHtml(camera.id)}</h3>
            <p class="camera-card-subtitle">${escapeHtml(camera.rtsp_url_main || camera.rtsp_url || '')}</p>
          </div>
          <span class="camera-mode-badge">${escapeHtml(mode)}</span>
        </div>
        <div class="form-grid camera-grid">
          <div class="form-group">
            <label>Mode</label>
            <select class="camera-mode-select">
              <option value="zone" ${mode === 'zone' ? 'selected' : ''}>Zone</option>
              <option value="distance" ${mode === 'distance' ? 'selected' : ''}>Distance</option>
            </select>
          </div>
        </div>

        <div class="camera-mode-section camera-zone-fields">
          <div class="form-grid camera-grid">
            <div class="form-group">
              <label>Yellow Start Y</label>
              <input type="number" class="camera-zone-yellow" step="0.01" min="0.01" max="0.99" value="${yellow.toFixed(2)}">
            </div>
            <div class="form-group">
              <label>Red Start Y</label>
              <input type="number" class="camera-zone-red" step="0.01" min="0.01" max="0.99" value="${red.toFixed(2)}">
            </div>
          </div>
        </div>

        <div class="camera-mode-section camera-distance-fields">
          <div class="form-grid camera-grid">
            <div class="form-group">
              <label>Warning Distance (m)</label>
              <input type="number" class="camera-warning-distance" step="0.1" min="0.1" value="${warningDistance.toFixed(1)}">
            </div>
            <div class="form-group">
              <label>Danger Distance (m)</label>
              <input type="number" class="camera-danger-distance" step="0.1" min="0.1" value="${dangerDistance.toFixed(1)}">
            </div>
            <div class="form-group camera-wide-field">
              <label>Calibration Path</label>
              <input type="text" class="camera-calibration-path" value="${escapeHtml(calibrationPath)}" placeholder="config/calibration/back.yaml">
            </div>
          </div>
          <p class="camera-mode-note">Distance mode is stored in config now. The rear-camera metric distance calculation is the next implementation step.</p>
        </div>

        <div class="camera-card-actions">
          <button class="btn-primary save-camera-btn">Save Camera</button>
        </div>
      </article>
    `;
  }

  function bindCameraCard(card) {
    const modeSelect = card.querySelector('.camera-mode-select');
    const badge = card.querySelector('.camera-mode-badge');
    const zoneFields = card.querySelector('.camera-zone-fields');
    const distanceFields = card.querySelector('.camera-distance-fields');
    const saveButton = card.querySelector('.save-camera-btn');

    function refreshMode() {
      const mode = modeSelect.value;
      badge.textContent = mode;
      zoneFields.style.display = mode === 'zone' ? 'block' : 'none';
      distanceFields.style.display = mode === 'distance' ? 'block' : 'none';
    }

    modeSelect.addEventListener('change', refreshMode);
    refreshMode();

    saveButton.addEventListener('click', async () => {
      const cameraId = card.dataset.cameraId;
      const payload = {
        mode: modeSelect.value,
        zone: {
          yellow_start_y: parseFloat(card.querySelector('.camera-zone-yellow').value),
          red_start_y: parseFloat(card.querySelector('.camera-zone-red').value),
        },
        distance: {
          warning_distance_m: parseFloat(card.querySelector('.camera-warning-distance').value),
          danger_distance_m: parseFloat(card.querySelector('.camera-danger-distance').value),
          calibration_path: card.querySelector('.camera-calibration-path').value.trim(),
        },
      };

      try {
        const res = await fetch('/api/config/cameras/' + encodeURIComponent(cameraId), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) {
          toast(data.detail || 'Camera save failed', 'error');
          return;
        }
        toast(cameraId + ' saved');
        await loadCameraConfigs();
      } catch {
        toast('Connection error', 'error');
      }
    });
  }

  async function loadCameraConfigs() {
    const container = document.getElementById('cameraCards');
    if (!container) return;
    try {
      const res = await fetch('/api/config/cameras');
      if (res.status === 401) { window.location.href = '/login'; return; }
      const cameras = await res.json();
      if (!Array.isArray(cameras) || cameras.length === 0) {
        container.innerHTML = '<p class="help-text">No cameras configured.</p>';
        return;
      }
      container.innerHTML = cameras.map(cameraCardMarkup).join('');
      container.querySelectorAll('.camera-card').forEach(bindCameraCard);
    } catch {
      container.innerHTML = '<p class="help-text">Unable to load camera settings.</p>';
    }
  }

  // ── Load Config ─────────────────────────────────────────────
  async function loadConfig() {
    try {
      const res = await fetch('/api/config');
      if (res.status === 401) { window.location.href = '/login'; return; }
      const cfg = await res.json();
      const alert = cfg.alert || {};

      // Zone sliders
      const yy = alert.yellow_start_y || 0.33;
      const ry = alert.red_start_y || 0.66;
      document.getElementById('yellowSlider').value = yy;
      document.getElementById('redSlider').value = ry;
      document.getElementById('yellowVal').textContent = yy.toFixed(2);
      document.getElementById('redVal').textContent = ry.toFixed(2);
      updateZonePreview(yy, ry);

      // Timing
      document.getElementById('repeatInterval').value = alert.repeat_interval_sec || 1.5;
      document.getElementById('minClear').value = alert.min_clear_sec || 3.0;
      document.getElementById('minConfidence').value = alert.min_alert_confidence || 0.55;
      await loadCameraConfigs();
    } catch {}
  }
  loadConfig();

  // ── Zone Preview ────────────────────────────────────────────
  function updateZonePreview(yy, ry) {
    document.getElementById('greenBand').style.flex = yy;
    document.getElementById('yellowBand').style.flex = ry - yy;
    document.getElementById('redBand').style.flex = 1 - ry;
  }

  const yellowSlider = document.getElementById('yellowSlider');
  const redSlider = document.getElementById('redSlider');

  yellowSlider.addEventListener('input', () => {
    let yy = parseFloat(yellowSlider.value);
    let ry = parseFloat(redSlider.value);
    if (yy >= ry - 0.02) { yy = ry - 0.02; yellowSlider.value = yy; }
    document.getElementById('yellowVal').textContent = yy.toFixed(2);
    updateZonePreview(yy, ry);
  });

  redSlider.addEventListener('input', () => {
    let yy = parseFloat(yellowSlider.value);
    let ry = parseFloat(redSlider.value);
    if (ry <= yy + 0.02) { ry = yy + 0.02; redSlider.value = ry; }
    document.getElementById('redVal').textContent = ry.toFixed(2);
    updateZonePreview(yy, ry);
  });

  // ── Save Zones ──────────────────────────────────────────────
  document.getElementById('saveZonesBtn').addEventListener('click', async () => {
    const yy = parseFloat(yellowSlider.value);
    const ry = parseFloat(redSlider.value);
    try {
      const res = await fetch('/api/config/zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yellow_start_y: yy, red_start_y: ry }),
      });
      const data = await res.json();
      if (res.ok) {
        toast('Zones saved');
        await loadCameraConfigs();
      }
      else toast(data.detail || 'Save failed', 'error');
    } catch { toast('Connection error', 'error'); }
  });

  // ── Save Timing ─────────────────────────────────────────────
  document.getElementById('saveTimingBtn').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/config/timing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repeat_interval_sec: parseFloat(document.getElementById('repeatInterval').value),
          min_clear_sec: parseFloat(document.getElementById('minClear').value),
          min_alert_confidence: parseFloat(document.getElementById('minConfidence').value),
        }),
      });
      const data = await res.json();
      if (res.ok) toast('Timing saved');
      else toast(data.detail || 'Save failed', 'error');
    } catch { toast('Connection error', 'error'); }
  });

  // ── Validate ────────────────────────────────────────────────
  document.getElementById('validateBtn').addEventListener('click', async () => {
    const status = document.getElementById('applyStatus');
    status.textContent = 'Validating...';
    try {
      const res = await fetch('/api/config/validate', { method: 'POST' });
      const data = await res.json();
      if (data.valid) {
        status.textContent = 'Config valid';
        toast('Config is valid');
      } else {
        status.textContent = 'Invalid: ' + data.error;
        toast('Config invalid: ' + data.error, 'error');
      }
    } catch { toast('Connection error', 'error'); status.textContent = ''; }
  });

  // ── Apply & Restart ─────────────────────────────────────────
  document.getElementById('applyBtn').addEventListener('click', () => {
    document.getElementById('confirmModal').style.display = 'flex';
  });

  document.getElementById('cancelApply').addEventListener('click', () => {
    document.getElementById('confirmModal').style.display = 'none';
  });

  document.getElementById('confirmApply').addEventListener('click', async () => {
    document.getElementById('confirmModal').style.display = 'none';
    const status = document.getElementById('applyStatus');
    status.textContent = 'Applying...';
    try {
      const res = await fetch('/api/apply', { method: 'POST' });
      const data = await res.json();
      if (data.ok) {
        toast('Applied and restarted');
        status.textContent = 'Applied successfully';
        pollStatus();
      } else {
        toast(data.error || 'Apply failed', 'error');
        status.textContent = 'Failed: ' + (data.error || '');
      }
    } catch { toast('Connection error', 'error'); status.textContent = ''; }
  });

  // ── Restore ─────────────────────────────────────────────────
  document.getElementById('restoreBtn').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/config/restore', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        toast('Previous config restored');
        loadConfig();
      } else {
        toast(data.detail || 'Restore failed', 'error');
      }
    } catch { toast('Connection error', 'error'); }
  });

  // ── Logout ──────────────────────────────────────────────────
  document.getElementById('logoutBtn').addEventListener('click', async () => {
    await fetch('/api/auth/logout', { method: 'POST' });
    window.location.href = '/login';
  });

})();
