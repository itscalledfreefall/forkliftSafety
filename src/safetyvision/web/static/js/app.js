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
      if (res.ok) toast('Zones saved');
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
