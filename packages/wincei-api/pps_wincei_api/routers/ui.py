"""GET / and /ui — Customer-facing web GUI (drag-drop + inline preview).

Self-contained HTML+CSS+JS — không cần extra deps.
Mở browser: http://localhost:8000/  hoặc /ui
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from ..__version__ import __version__

router = APIRouter(tags=["ui"])

_HTML = r"""<!DOCTYPE html>
<html lang="vi"><head>
<meta charset="utf-8">
<title>WINCEI v__VER__ — Real Estate AI</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,Segoe UI,sans-serif;background:#0f1419;color:#e6edf3;min-height:100vh}
header{background:linear-gradient(90deg,#1a73e8,#4a90e2);padding:20px;text-align:center;color:#fff;box-shadow:0 4px 12px rgba(0,0,0,.3)}
header h1{font-size:24px;margin-bottom:4px}
header p{font-size:13px;opacity:.9}
main{max-width:1400px;margin:0 auto;padding:24px}
.controls{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px;margin-bottom:24px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.row.single{grid-template-columns:1fr}
label{display:block;font-size:13px;color:#8b949e;margin-bottom:6px;font-weight:600}
select,input[type=text]{width:100%;padding:10px 12px;background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:8px;font-size:14px}
.dropzone{border:2px dashed #30363d;border-radius:12px;padding:60px 20px;text-align:center;cursor:pointer;transition:all .2s;background:#0d1117}
.dropzone:hover,.dropzone.dragging{border-color:#1a73e8;background:#0d2440;transform:scale(1.01)}
.dropzone p{color:#8b949e;font-size:15px;margin:6px 0}
.dropzone .big{font-size:18px;color:#e6edf3;font-weight:600}
.dropzone input{display:none}
button{padding:14px 28px;background:#1a73e8;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:15px;font-weight:600;transition:all .2s}
button:hover:not(:disabled){background:#2986ff;transform:translateY(-1px);box-shadow:0 4px 12px rgba(26,115,232,.4)}
button:disabled{background:#30363d;cursor:not-allowed;color:#6e7681}
button.secondary{background:#30363d}
.btn-row{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap}
.status{padding:14px 16px;border-radius:8px;margin-top:16px;font-size:14px;display:none}
.status.show{display:block}
.status.info{background:#0d2440;border:1px solid #1a73e8;color:#79b8ff}
.status.success{background:#0a3622;border:1px solid #238636;color:#3fb950}
.status.error{background:#3c1419;border:1px solid #f85149;color:#f85149}
.progress{height:4px;background:#30363d;border-radius:2px;overflow:hidden;margin-top:10px}
.progress-bar{height:100%;background:linear-gradient(90deg,#1a73e8,#4a90e2);width:0;transition:width .3s ease}
.results{display:none}
.results.show{display:block}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:24px}
.compare .panel{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden}
.compare .panel h3{padding:12px 16px;font-size:14px;background:#21262d;border-bottom:1px solid #30363d;color:#79b8ff}
.compare img{width:100%;display:block}
.masks-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:24px}
.mask-tile{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;cursor:pointer;transition:all .2s}
.mask-tile:hover{border-color:#1a73e8;transform:translateY(-2px)}
.mask-tile img{width:100%;display:block;background:#000}
.mask-tile .label{padding:8px 10px;font-size:12px;color:#8b949e;display:flex;justify-content:space-between;align-items:center}
.mask-tile .label .verdict{font-size:10px;padding:2px 6px;border-radius:3px}
.verdict.pass{background:#0a3622;color:#3fb950}
.verdict.review{background:#3c2914;color:#d29922}
.verdict.fail{background:#3c1419;color:#f85149}
.verdict.no_target{background:#21262d;color:#6e7681}
.report{margin-top:24px;background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:16px;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#8b949e;overflow-x:auto;max-height:400px;overflow-y:auto}
.report pre{white-space:pre-wrap;word-break:break-word}
nav{background:#161b22;border-bottom:1px solid #30363d;padding:12px 24px;display:flex;gap:16px;font-size:13px}
nav a{color:#79b8ff;text-decoration:none}
nav a:hover{text-decoration:underline}
nav .sep{color:#30363d}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #30363d;border-top-color:#1a73e8;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.scorecard{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;margin-top:12px}
.scorecard .item{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px 10px;font-size:11px}
.scorecard .item .name{color:#8b949e;margin-bottom:2px}
.scorecard .item .val{font-weight:600;font-size:14px}
.preview-thumb{max-height:200px;object-fit:contain;background:#000;border-radius:8px;margin-top:10px;display:none}
.preview-thumb.show{display:block}
</style></head>
<body>
<header>
<h1>🪟🏠 WINCEI v__VER__ — Real Estate AI Studio</h1>
<p>Smart segmentation · HDR fusion · Window+ceiling fix · VLM+SAM2 hybrid</p>
</header>
<nav>
<a href="/docs">📖 Swagger API</a><span class="sep">·</span>
<a href="/redoc">📘 ReDoc</a><span class="sep">·</span>
<a href="/api/v1/health">💓 Health</a><span class="sep">·</span>
<a href="/api/v1/jobs">📋 Jobs</a><span class="sep">·</span>
<a href="https://github.com/NgVB1408/pro-photo-studio" target="_blank">🐙 GitHub</a>
</nav>
<main>

<div class="controls">
<div class="row">
<div>
<label>Pipeline mode</label>
<select id="mode">
<option value="segment-masks">🎯 Smart Segmentation (9 masks + AI eval) — RECOMMEND</option>
<option value="window-ceiling">🪟 Window + Ceiling Fix (tone correction)</option>
<option value="hdr-fuse">📸 HDR Bracket Fusion (Sony AEB)</option>
<option value="detect-regions">📋 Detect Regions (JSON bbox [0..1000])</option>
<option value="full-recovery-ceiling">⭐ Full Recovery Ceiling (VLM + SAM 2 PRO)</option>
</select>
</div>
<div>
<label>Mock mode (test fast, no CPU)</label>
<select id="mock">
<option value="false">OFF — chạy thật trên CPU (chậm 6-10 phút/ảnh)</option>
<option value="true">ON — trả stub instant</option>
</select>
</div>
</div>

<div class="dropzone" id="dropzone">
<div class="big">📤 Drop ảnh vào đây hoặc click chọn file</div>
<p>JPG / PNG / TIFF — tối đa 200MB</p>
<p style="margin-top:8px"><small>HDR mode: chọn nhiều ảnh bracket cùng cảnh</small></p>
<input type="file" id="fileInput" accept="image/*" multiple>
</div>
<img id="previewThumb" class="preview-thumb" alt="preview">

<div class="btn-row">
<button id="runBtn" disabled>▶ Run</button>
<button id="resetBtn" class="secondary">Reset</button>
</div>

<div class="status" id="status"></div>
<div class="progress" id="progressContainer" style="display:none">
<div class="progress-bar" id="progressBar"></div>
</div>
</div>

<div class="results" id="results">
<h2 style="margin-bottom:16px">📊 Kết quả</h2>
<div id="verdictBox" class="status info show" style="display:flex;justify-content:space-between;align-items:center"></div>
<div class="compare" id="compare" style="display:none">
<div class="panel"><h3>🔵 BEFORE</h3><img id="beforeImg" alt="before"></div>
<div class="panel"><h3>🟢 AFTER (Overlay)</h3><img id="afterImg" alt="after"></div>
</div>
<div class="masks-grid" id="masksGrid"></div>
<div class="scorecard" id="scorecard"></div>
<div class="report" id="report" style="display:none"><pre id="reportPre"></pre></div>
</div>

</main>

<script>
const API_BASE = '';
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewThumb = document.getElementById('previewThumb');
const runBtn = document.getElementById('runBtn');
const resetBtn = document.getElementById('resetBtn');
const status = document.getElementById('status');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const results = document.getElementById('results');
const compare = document.getElementById('compare');
const beforeImg = document.getElementById('beforeImg');
const afterImg = document.getElementById('afterImg');
const masksGrid = document.getElementById('masksGrid');
const verdictBox = document.getElementById('verdictBox');
const reportBox = document.getElementById('report');
const reportPre = document.getElementById('reportPre');
const scorecard = document.getElementById('scorecard');
const modeSelect = document.getElementById('mode');
const mockSelect = document.getElementById('mock');

let selectedFiles = [];

function setStatus(msg, type='info') {
  status.className = 'status show ' + type;
  status.innerHTML = type === 'info' ? `<span class="spin"></span>${msg}` : msg;
}
function clearStatus() { status.className = 'status'; }
function setProgress(pct) {
  progressContainer.style.display = pct > 0 ? 'block' : 'none';
  progressBar.style.width = pct + '%';
}

['dragenter','dragover'].forEach(e =>
  dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.add('dragging'); })
);
['dragleave','drop'].forEach(e =>
  dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.remove('dragging'); })
);
dropzone.addEventListener('drop', ev => {
  ev.preventDefault();
  handleFiles(ev.dataTransfer.files);
});
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', ev => handleFiles(ev.target.files));

function handleFiles(files) {
  selectedFiles = Array.from(files);
  if (!selectedFiles.length) return;
  const first = selectedFiles[0];
  const url = URL.createObjectURL(first);
  previewThumb.src = url;
  previewThumb.classList.add('show');
  beforeImg.src = url;
  dropzone.querySelector('.big').textContent = `✓ ${selectedFiles.length} file: ${first.name}${selectedFiles.length > 1 ? ` + ${selectedFiles.length - 1} khác` : ''}`;
  runBtn.disabled = false;
}

resetBtn.addEventListener('click', () => {
  selectedFiles = [];
  fileInput.value = '';
  previewThumb.classList.remove('show');
  results.classList.remove('show');
  clearStatus();
  setProgress(0);
  dropzone.querySelector('.big').textContent = '📤 Drop ảnh vào đây hoặc click chọn file';
  runBtn.disabled = true;
  masksGrid.innerHTML = '';
  scorecard.innerHTML = '';
});

runBtn.addEventListener('click', async () => {
  if (!selectedFiles.length) return;
  runBtn.disabled = true;
  results.classList.remove('show');
  setProgress(5);
  const mode = modeSelect.value;
  const mock = mockSelect.value === 'true';

  setStatus(`Uploading ${selectedFiles.length} file → ${mode}${mock ? ' (mock)' : ''}…`);

  const fd = new FormData();
  if (mode === 'window-ceiling' || mode === 'detect-regions' || mode === 'full-recovery-ceiling') {
    fd.append('file', selectedFiles[0]);
  } else {
    selectedFiles.forEach(f => fd.append('files', f));
  }
  fd.append('mock', mock ? 'true' : 'false');
  if (mode === 'window-ceiling') fd.append('mode', mock ? 'sync' : 'async');

  try {
    const resp = await fetch(`${API_BASE}/api/v1/${mode}`, { method: 'POST', body: fd });

    if (mode === 'full-recovery-ceiling' && !mock) {
      // PNG file response
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      afterImg.src = url;
      compare.style.display = 'grid';
      results.classList.add('show');
      verdictBox.innerHTML = `<div>✅ Recovery PNG downloaded — <a href="${url}" download="ceiling_full_recovery.png" style="color:#79b8ff">Save</a></div>`;
      setStatus('Done', 'success');
      setProgress(100);
      runBtn.disabled = false;
      return;
    }

    const json = await resp.json();
    if (!resp.ok) throw new Error(json.detail || JSON.stringify(json));

    // Mock instant or regions-direct
    if (json.mode === 'sync' || mode === 'detect-regions' || mock) {
      handleSync(json, mode);
      runBtn.disabled = false;
      return;
    }

    // Async: poll job
    setStatus(`Job ${json.job_id.slice(0,8)}… queued, polling status`);
    setProgress(10);
    const final = await pollJob(json.job_id);
    if (final.status === 'failed') throw new Error(final.error || 'Job failed');
    handleAsync(final);
    runBtn.disabled = false;
  } catch (err) {
    setStatus('❌ ' + err.message, 'error');
    setProgress(0);
    runBtn.disabled = false;
  }
});

async function pollJob(jobId) {
  for (let i = 0; i < 360; i++) { // up to 30 minutes (5s × 360)
    const r = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`);
    const j = await r.json();
    setProgress(j.progress_pct || 10);
    setStatus(`Job ${jobId.slice(0,8)}… ${j.status} (${(j.progress_pct||0).toFixed(0)}%) — ${j.message || ''}`);
    if (j.status === 'completed' || j.status === 'failed') return j;
    await new Promise(r => setTimeout(r, 5000));
  }
  throw new Error('Job timeout 30 phút');
}

function handleSync(json, mode) {
  results.classList.add('show');
  setProgress(100);
  if (json.mock) setStatus('✅ Mock response trả về', 'success');
  else setStatus('✅ Done', 'success');

  if (mode === 'detect-regions') {
    renderRegions(json);
  } else if (json.eval) {
    renderEval(json.eval, mode);
  } else {
    reportPre.textContent = JSON.stringify(json, null, 2);
    reportBox.style.display = 'block';
  }
}

function handleAsync(job) {
  results.classList.add('show');
  setProgress(100);
  setStatus(`✅ Done — ${job.message}`, 'success');

  const verdicts = job.metadata?.verdicts || [];
  const passCt = job.metadata?.pass_count || 0;
  const reviewCt = job.metadata?.review_count || 0;
  const failCt = job.metadata?.fail_count || 0;

  verdictBox.innerHTML = `
    <div>
      <strong>Job:</strong> ${job.job_id.slice(0,8)}…
      ${verdicts.length ? ` · Verdicts: ${passCt}✅ pass / ${reviewCt}⚠️ review / ${failCt}❌ fail` : ''}
    </div>
    <div>
      <a href="${API_BASE}/api/v1/jobs/${job.job_id}/download" style="color:#79b8ff;text-decoration:none">
        <button>⬇ Download Zip</button>
      </a>
    </div>`;

  reportPre.textContent = JSON.stringify(job, null, 2);
  reportBox.style.display = 'block';
}

function renderEval(ev, mode) {
  const verdictClass = ev.verdict || 'review';
  verdictBox.className = 'status show ' + (verdictClass === 'pass' ? 'success' : verdictClass === 'fail' ? 'error' : 'info');
  verdictBox.innerHTML = `
    <div>
      <strong>${(ev.verdict || '?').toUpperCase()}</strong> · overall score
      <strong>${(ev.overall_score || 0).toFixed(3)}</strong>
    </div>
    <div style="font-size:11px;opacity:.8">Mode: ${mode}</div>`;

  if (ev.per_mask) {
    masksGrid.innerHTML = '';
    scorecard.innerHTML = '';
    Object.entries(ev.per_mask).forEach(([name, m]) => {
      const cov = (m.coverage * 100).toFixed(1);
      const verdict = m.verdict || 'review';
      const sc = document.createElement('div');
      sc.className = 'item';
      sc.innerHTML = `
        <div class="name">${name}</div>
        <div class="val">${cov}%</div>
        <div style="font-size:10px"><span class="verdict ${verdict}" style="padding:1px 4px;border-radius:2px">${verdict}</span></div>`;
      scorecard.appendChild(sc);
    });
  }
  reportPre.textContent = JSON.stringify(ev, null, 2);
  reportBox.style.display = 'block';
}

function renderRegions(json) {
  reportPre.textContent = JSON.stringify(json, null, 2);
  reportBox.style.display = 'block';
  const el = json.detected_elements || {};
  verdictBox.innerHTML = `
    <div><strong>Real Estate Regions</strong> · ${Object.keys(el).length} elements</div>
    <div style="font-size:11px">${json.image_size?.width}×${json.image_size?.height} · ${json.camera_angle || 'standard'}</div>`;
  if (el.ceiling) {
    scorecard.innerHTML = `
      <div class="item"><div class="name">ceiling</div><div class="val">${(el.ceiling.area_pct||0).toFixed(1)}%</div><div style="font-size:10px">conf ${el.ceiling.confidence}</div></div>
      <div class="item"><div class="name">walls</div><div class="val">${(el.walls?.area_pct||0).toFixed(1)}%</div><div style="font-size:10px">conf ${el.walls?.confidence||'-'}</div></div>
      <div class="item"><div class="name">floor</div><div class="val">${(el.floor?.area_pct||0).toFixed(1)}%</div><div style="font-size:10px">conf ${el.floor?.confidence||'-'}</div></div>
      <div class="item"><div class="name">windows</div><div class="val">${(el.windows||[]).length}</div><div style="font-size:10px">panes</div></div>
      <div class="item"><div class="name">doors</div><div class="val">${(el.doors||[]).length}</div></div>`;
  }
}
</script>
</body></html>
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing() -> str:
    return _HTML.replace("__VER__", __version__)


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui_alias() -> str:
    return _HTML.replace("__VER__", __version__)
