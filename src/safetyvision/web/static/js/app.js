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
      if (tab.dataset.tab === 'thermal') { startThermalStream(); loadThermalGallery(); }
    });
  });

  // ── Service Status ──────────────────────────────────────────
  async function pollStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      const badge = document.getElementById('serviceStatus');
      const txt = document.getElementById('serviceStatusText');
      if (txt) txt.textContent = data.service;
      badge.className = 'status-pill ' +
        (data.service === 'active' ? 'active' :
         data.service === 'inactive' ? 'inactive' : 'unknown');

      // Mirror service status into the live-feed badge
      const isOnline = data.service === 'active';
      const liveBadge = document.getElementById('liveBadge');
      const vstatus = document.getElementById('videoStatus');
      const vicon = document.getElementById('videoWifiIcon');
      if (liveBadge) liveBadge.classList.toggle('offline', !isOnline);
      if (vstatus) vstatus.textContent = isOnline ? 'Online' : 'Offline';
      if (vicon) {
        vicon.classList.toggle('online-icon', isOnline);
        vicon.classList.toggle('offline-icon', !isOnline);
      }
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

  function setMetricCount(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    if (typeof value !== 'number' || Number.isNaN(value)) {
      el.textContent = '--';
      return;
    }
    el.textContent = String(Math.trunc(value));
  }

  const ZONE_LABEL = { green: 'Safe Zone', medium: 'Medium Zone', danger: 'Danger Zone' };
  const ZONE_TO_VALUE_COLOR = { green: 'value-green', medium: 'value-yellow', danger: 'value-red' };
  const ZONE_TO_ACCENT = { green: 'accent-green', medium: 'accent-yellow', danger: 'accent-red' };

  function updateDistanceCard(data) {
    const card = document.getElementById('distanceCard');
    const row = document.getElementById('dashboardBottomRow');
    const distEl = document.getElementById('metricDistance');
    const zoneEl = document.getElementById('metricZoneLevel');
    if (!card) return;
    const isDistance = data && data.zone_mode === 'distance';
    if (row) row.classList.toggle('video-only', !isDistance);
    card.style.display = isDistance ? '' : 'none';
    if (!isDistance) return;

    const next = (typeof data.last_distance_m === 'number')
      ? data.last_distance_m.toFixed(2) : '--';
    // Pulse only on real value changes (skip transitions to/from "--").
    if (distEl.textContent !== next && next !== '--' && distEl.textContent !== '--') {
      distEl.classList.remove('pulse');
      void distEl.offsetWidth;  // force reflow to restart the keyframe
      distEl.classList.add('pulse');
    }
    distEl.textContent = next;

    const rawZone = data.last_zone_level || '';
    const zone = rawZone === '' ? 'green' : rawZone;
    zoneEl.textContent = ZONE_LABEL[zone] || 'Safe Zone';
    zoneEl.className = 'zone-tag ' + zone;

    // Tint the distance value + the card accent bar to match the zone
    distEl.classList.remove('value-green', 'value-yellow', 'value-red');
    distEl.classList.add(ZONE_TO_VALUE_COLOR[zone]);
    card.classList.remove('accent-green', 'accent-yellow', 'accent-red');
    card.classList.add(ZONE_TO_ACCENT[zone]);
  }

  async function pollMetrics() {
    try {
      const res = await fetch('/api/metrics');
      if (res.status === 401) return;
      const data = await res.json();
      updateDistanceCard(data);
      if (!data.available) {
        setMetricValue('metricFps', NaN, 1);
        setMetricValue('metricLatency', NaN, 1);
        setMetricValue('metricCaptureFps', NaN, 1);
        setMetricValue('metricInferenceFps', NaN, 1);
        setMetricCount('metricYellowEntries', NaN);
        setMetricCount('metricRedEntries', NaN);
        return;
      }
      setMetricValue('metricFps', data.fps, 1);
      setMetricValue('metricLatency', data.latency_total_ms, 1);
      setMetricValue('metricCaptureFps', data.capture_fps, 1);
      setMetricValue('metricInferenceFps', data.inference_fps, 1);
      setMetricCount('metricYellowEntries', data.yellow_zone_entries);
      setMetricCount('metricRedEntries', data.red_zone_entries);

      // Live-feed FPS readout (rounded to whole number, like the mockup)
      const vfps = document.getElementById('videoFps');
      if (vfps && typeof data.capture_fps === 'number')
        vfps.textContent = data.capture_fps.toFixed(0);
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

  // ── Load Config ─────────────────────────────────────────────
  async function loadConfig() {
    try {
      const res = await fetch('/api/config');
      if (res.status === 401) { window.location.href = '/login'; return; }
      const cfg = await res.json();
      const alert = cfg.alert || {};
      const input = cfg.input || {};

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

      // Live-feed video meta
      const w = input.width || 640;
      const h = input.height || 480;
      const resEl = document.getElementById('videoResolution');
      if (resEl) resEl.textContent = w + '×' + h;

      // Closest-person threshold readouts
      const danger = alert.danger_threshold_m || 1.0;
      const warning = alert.warning_threshold_m || 5.0;
      const yel = document.getElementById('cpYellowThreshold');
      const red = document.getElementById('cpRedThreshold');
      if (yel) yel.textContent = warning.toFixed(1);
      if (red) red.textContent = danger.toFixed(1);
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
  async function saveZones(silent) {
    const yy = parseFloat(yellowSlider.value);
    const ry = parseFloat(redSlider.value);
    try {
      const res = await fetch('/api/config/zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yellow_start_y: yy, red_start_y: ry }),
      });
      const data = await res.json();
      if (res.ok) toast(silent ? 'Zones updated — live view follows shortly' : 'Zones saved');
      else toast(data.detail || 'Save failed', 'error');
    } catch { toast('Connection error', 'error'); }
  }

  document.getElementById('saveZonesBtn').addEventListener('click', () => saveZones(false));

  // Auto-save when a slider is released so the live camera overlay follows the
  // new cut lines without needing a separate Save click. (Alerts still require
  // Apply & Restart.)
  yellowSlider.addEventListener('change', () => saveZones(true));
  redSlider.addEventListener('change', () => saveZones(true));

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

  // ── Thermal ─────────────────────────────────────────────────
  function startThermalStream() {
    const img = document.getElementById('thermalStream');
    const offline = document.getElementById('thermalOffline');
    if (!img) return;
    img.src = '/api/stream/thermal.mjpg?t=' + Date.now();
    img.onload = () => { offline.style.display = 'none'; };
    img.onerror = () => { offline.style.display = 'flex'; };
  }

  async function loadThermalConfig() {
    try {
      const res = await fetch('/api/thermal/config');
      if (!res.ok) return;
      const c = await res.json();
      document.getElementById('thermalEnabled').value = c.enabled ? 'true' : 'false';
      document.getElementById('thermalHost').value = c.host || '';
      document.getElementById('thermalUsername').value = c.username || 'admin';
      document.getElementById('thermalRtsp').value = c.rtsp_url || '';
      document.getElementById('thermalMaxTemp').value = c.max_temp_c;
      document.getElementById('thermalPoll').value = c.poll_interval_sec;
      document.getElementById('thermalRepeat').value = c.repeat_interval_sec;
      document.getElementById('thermalMaxSnaps').value = c.max_snapshots;
      document.getElementById('thermalPwSet').textContent = c.password_set ? '(saved)' : '(not set)';
    } catch {}
  }
  loadThermalConfig();

  async function pollThermal() {
    try {
      const res = await fetch('/api/thermal/status');
      if (!res.ok) return;
      const s = await res.json();
      document.getElementById('thermalSceneTemp').textContent =
        s.scene_temp_c == null ? '--' : s.scene_temp_c.toFixed(1);
      document.getElementById('thermalThreshold').textContent = (s.threshold_c || 0).toFixed(1);
      document.getElementById('thermalViolationCount').textContent = s.violation_count;
      const badge = document.getElementById('thermalStateBadge');
      const card = document.getElementById('thermalSceneCard');
      if (!s.enabled) {
        badge.textContent = 'Disabled'; badge.className = 'card-badge badge-dim';
        card.className = 'metric-card accent-primary';
      } else if (s.in_violation) {
        badge.textContent = 'OVER TEMP'; badge.className = 'card-badge badge-red';
        card.className = 'metric-card accent-red border-red';
      } else {
        badge.textContent = 'Normal'; badge.className = 'card-badge badge-green';
        card.className = 'metric-card accent-green';
      }
    } catch {}
  }
  pollThermal();
  setInterval(pollThermal, 2000);

  async function loadThermalGallery() {
    try {
      const res = await fetch('/api/thermal/violations');
      if (!res.ok) return;
      const data = await res.json();
      const grid = document.getElementById('thermalGallery');
      const items = data.items || [];
      grid.replaceChildren();
      if (!items.length) {
        const p = document.createElement('p');
        p.className = 'help-text';
        p.textContent = 'No violations recorded yet.';
        grid.appendChild(p);
        return;
      }
      items.forEach(it => {
        const card = document.createElement('div');
        card.className = 'metric-card';
        const bezel = document.createElement('div');
        bezel.className = 'video-bezel';
        bezel.style.aspectRatio = '4 / 3';
        const img = document.createElement('img');
        img.src = '/api/thermal/snapshot/' + encodeURIComponent(it.name);
        img.alt = 'violation';
        img.style.cssText = 'width:100%;height:100%;object-fit:cover';
        bezel.appendChild(img);
        const footer = document.createElement('div');
        footer.className = 'card-footer';
        const tag = document.createElement('span');
        tag.className = 'zone-tag danger';
        tag.textContent = (it.temp_c != null) ? it.temp_c.toFixed(1) + '°C' : '';
        const when = document.createElement('span');
        when.style.marginLeft = '8px';
        when.textContent = it.timestamp || '';
        footer.append(tag, when);
        card.append(bezel, footer);
        grid.appendChild(card);
      });
    } catch {}
  }

  const thermalTestBtn = document.getElementById('thermalTestBtn');
  if (thermalTestBtn) thermalTestBtn.addEventListener('click', async () => {
    const status = document.getElementById('thermalTestStatus');
    status.textContent = 'Testing…';
    try {
      const res = await fetch('/api/thermal/test', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          host: document.getElementById('thermalHost').value,
          username: document.getElementById('thermalUsername').value,
          password: document.getElementById('thermalPassword').value,
        }),
      });
      const data = await res.json();
      status.textContent = data.message || '';
      toast(data.ok ? 'Camera OK' : 'Camera test failed', data.ok ? 'success' : 'error');
    } catch { toast('Connection error', 'error'); status.textContent = ''; }
  });

  const saveThermalBtn = document.getElementById('saveThermalBtn');
  if (saveThermalBtn) saveThermalBtn.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/thermal/config', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: document.getElementById('thermalEnabled').value === 'true',
          host: document.getElementById('thermalHost').value,
          username: document.getElementById('thermalUsername').value,
          password: document.getElementById('thermalPassword').value,
          rtsp_url: document.getElementById('thermalRtsp').value,
          max_temp_c: parseFloat(document.getElementById('thermalMaxTemp').value),
          poll_interval_sec: parseFloat(document.getElementById('thermalPoll').value),
          repeat_interval_sec: parseFloat(document.getElementById('thermalRepeat').value),
          max_snapshots: parseInt(document.getElementById('thermalMaxSnaps').value, 10),
          snapshot_dir: 'thermal_violations',
        }),
      });
      const data = await res.json();
      if (res.ok) {
        toast('Thermal saved — Apply & Restart to take effect');
        document.getElementById('thermalPassword').value = '';
        loadThermalConfig();
      } else toast(data.detail || 'Save failed', 'error');
    } catch { toast('Connection error', 'error'); }
  });

  // Render lucide icons declared via data-lucide attributes
  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }

})();
