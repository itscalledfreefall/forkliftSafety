// Distance-mode calibration UI for the back camera.

const canvas = document.getElementById('frameCanvas');
const ctx = canvas.getContext('2d');
const placeholder = document.getElementById('canvasPlaceholder');
const captureBtn = document.getElementById('captureBtn');
const resetBtn = document.getElementById('resetPointsBtn');
const saveBtn = document.getElementById('saveBtn');
const enableBtn = document.getElementById('enableBtn');
const disableBtn = document.getElementById('disableBtn');
const pointInputs = document.getElementById('pointInputs');
const frameStatus = document.getElementById('frameStatus');
const actionStatus = document.getElementById('actionStatus');
const modeBadge = document.getElementById('modeBadge');
const toast = document.getElementById('toast');
const logoutBtn = document.getElementById('logoutBtn');

const POINT_COLORS = ['#00c853', '#ffd600', '#448aff', '#ff1744'];
const state = {
  frameLoaded: false,
  frameImg: null,
  frameWidth: 640,
  frameHeight: 480,
  points: [], // [{px, py, xm, ym}]
};

function showToast(msg, kind = 'info') {
  toast.textContent = msg;
  toast.className = 'toast ' + kind;
  toast.style.display = 'block';
  setTimeout(() => { toast.style.display = 'none'; }, 2800);
}

async function refreshStatus() {
  const r = await fetch('/api/calibration/status');
  if (!r.ok) {
    modeBadge.textContent = 'auth?';
    if (r.status === 401) window.location.href = '/login';
    return;
  }
  const data = await r.json();
  modeBadge.textContent = data.zone_mode === 'distance' ? 'DISTANCE' : 'BANDS';
  modeBadge.className = 'status-badge ' + (data.zone_mode === 'distance' ? 'green' : 'yellow');

  enableBtn.disabled = !(data.calibrated && data.zone_mode !== 'distance');
  disableBtn.disabled = data.zone_mode !== 'distance';
}

function redraw() {
  if (!state.frameImg) return;
  ctx.drawImage(state.frameImg, 0, 0, canvas.width, canvas.height);
  state.points.forEach((p, i) => {
    ctx.fillStyle = POINT_COLORS[i];
    ctx.beginPath();
    ctx.arc(p.px, p.py, 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), p.px, p.py);
  });
}

function rebuildPointInputs() {
  if (state.points.length === 0) {
    pointInputs.innerHTML =
      '<div class="point-empty">Click 4 points on the captured frame to begin.</div>';
    saveBtn.disabled = true;
    return;
  }
  pointInputs.innerHTML = '';
  state.points.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'point-row';
    row.innerHTML = `
      <span class="point-badge" style="background:${POINT_COLORS[i]}">${i + 1}</span>
      <label>X meters
        <input type="number" step="0.01" data-idx="${i}" data-axis="xm" value="${p.xm ?? ''}">
      </label>
      <label>Y meters
        <input type="number" step="0.01" data-idx="${i}" data-axis="ym" value="${p.ym ?? ''}">
      </label>
    `;
    pointInputs.appendChild(row);
    const px = document.createElement('div');
    px.className = 'point-pixel';
    px.textContent = `pixel (${Math.round(p.px)}, ${Math.round(p.py)})`;
    row.appendChild(px);
  });

  pointInputs.querySelectorAll('input[type="number"]').forEach(el => {
    el.addEventListener('input', e => {
      const idx = Number(e.target.dataset.idx);
      const axis = e.target.dataset.axis;
      const v = e.target.value === '' ? null : parseFloat(e.target.value);
      state.points[idx][axis] = Number.isFinite(v) ? v : null;
      saveBtn.disabled = !canSave();
    });
  });

  saveBtn.disabled = !canSave();
}

function canSave() {
  if (state.points.length !== 4) return false;
  return state.points.every(p =>
    Number.isFinite(p.xm) && Number.isFinite(p.ym)
  );
}

canvas.addEventListener('click', (e) => {
  if (!state.frameLoaded) return;
  if (state.points.length >= 4) {
    showToast('4 points already placed. Reset to start over.', 'warn');
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const px = (e.clientX - rect.left) * sx;
  const py = (e.clientY - rect.top) * sy;
  state.points.push({ px, py, xm: null, ym: null });
  redraw();
  rebuildPointInputs();
});

captureBtn.addEventListener('click', async () => {
  frameStatus.textContent = 'Loading...';
  try {
    const r = await fetch('/api/calibration/frame', { cache: 'no-store' });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      frameStatus.textContent = 'Error: ' + detail;
      return;
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      state.frameImg = img;
      state.frameWidth = img.naturalWidth;
      state.frameHeight = img.naturalHeight;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      state.frameLoaded = true;
      placeholder.style.display = 'none';
      redraw();
      frameStatus.textContent = `Frame ${img.naturalWidth}×${img.naturalHeight}`;
      URL.revokeObjectURL(url);
    };
    img.src = url;
  } catch (err) {
    frameStatus.textContent = 'Error: ' + err.message;
  }
});

resetBtn.addEventListener('click', () => {
  state.points = [];
  redraw();
  rebuildPointInputs();
});

saveBtn.addEventListener('click', async () => {
  if (!canSave()) return;
  actionStatus.textContent = 'Saving...';
  const body = {
    source_points: state.points.map(p => [p.px, p.py]),
    target_points: state.points.map(p => [p.xm, p.ym]),
    frame_width: state.frameWidth,
    frame_height: state.frameHeight,
  };
  const r = await fetch('/api/calibration', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    actionStatus.textContent = 'Save failed: ' + (data.detail || r.statusText);
    showToast('Calibration rejected: ' + (data.detail || r.statusText), 'error');
    return;
  }
  actionStatus.textContent = 'Saved to ' + data.path;
  showToast('Calibration saved', 'success');
  refreshStatus();
});

async function postToggle(path, label) {
  if (!confirm(`${label}? This will restart the SafetyVision service.`)) return;
  actionStatus.textContent = label + '...';
  const r = await fetch(path, { method: 'POST' });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.ok === false) {
    actionStatus.textContent = 'Failed: ' + (data.detail || data.error || r.statusText);
    showToast('Mode change failed', 'error');
    return;
  }
  actionStatus.textContent = `Mode set to ${data.zone_mode}. ${data.message || ''}`;
  showToast('Mode changed — service restarting', 'success');
  setTimeout(refreshStatus, 1200);
}

enableBtn.addEventListener('click', () => postToggle('/api/calibration/enable', 'Enable distance mode'));
disableBtn.addEventListener('click', () => postToggle('/api/calibration/disable', 'Switch to band mode'));

logoutBtn.addEventListener('click', async () => {
  await fetch('/api/auth/logout', { method: 'POST' });
  window.location.href = '/login';
});

refreshStatus();
