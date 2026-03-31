/* ══════════════════════════════════════════════════════
   APU WATCH · FMS — main.js
   ══════════════════════════════════════════════════════ */

// ── Clock ────────────────────────────────────────────────
(function () {
  const el = document.getElementById('clock');
  const ds = document.getElementById('datestamp');
  function tick() {
    const n = new Date();
    el.textContent = n.toISOString().slice(11, 19) + 'Z';
    ds.textContent = n.toLocaleDateString('en-GB').replace(/\//g, '/');
  }
  tick(); setInterval(tick, 1000);
})();

// ── Panel switching ──────────────────────────────────────
const panels = {
  upload:     document.getElementById('panelUpload'),
  processing: document.getElementById('panelProcessing'),
  dashboard:  document.getElementById('panelDash'),
};
const navItems = document.querySelectorAll('.nav-item');

function showPanel(name) {
  Object.entries(panels).forEach(([k, el]) => el.style.display = k === name ? 'block' : 'none');
  navItems.forEach(n => n.classList.toggle('active', n.dataset.panel === name));
}

navItems.forEach(n => n.addEventListener('click', () => {
  const p = n.dataset.panel;
  if (p === 'dashboard' && !allPredictions.length) return;
  showPanel(p);
}));

showPanel('upload');

// ── State ────────────────────────────────────────────────
let currentFile    = null;
let allPredictions = [];
let filteredPreds  = [];
let currentPage    = 1;
const PAGE_SIZE    = 25;
let rulChartInst   = null;
let activeEngineId = null;

// ── DOM ──────────────────────────────────────────────────
const dropRegion  = document.getElementById('dropRegion');
const dropInner   = document.getElementById('dropInner');
const fileIn      = document.getElementById('fileIn');
const fileReady   = document.getElementById('fileReady');
const fName       = document.getElementById('fName');
const fSize       = document.getElementById('fSize');
const execBtn     = document.getElementById('execBtn');
const searchIn    = document.getElementById('searchIn');
const pgInfo      = document.getElementById('pgInfo');
const pgPrev      = document.getElementById('pgPrev');
const pgNext      = document.getElementById('pgNext');
const tblBody     = document.getElementById('tblBody');
const dlBtn       = document.getElementById('dlBtn');
const gaugeRow    = document.getElementById('gaugeRow');
const engineFilter= document.getElementById('engineFilter');
const summItems   = document.getElementById('summItems');
const sbInfer     = document.getElementById('sbInfer');
const sbInferLabel= document.getElementById('sbInferLabel');
const navBadge    = document.getElementById('navBadge');
const navDash     = document.getElementById('navDashboard');
const healthArc   = document.getElementById('healthArc');
const healthPct   = document.getElementById('healthPct');
const ehStatus    = document.getElementById('ehStatus');

// ── Drag/drop ─────────────────────────────────────────────
dropRegion.addEventListener('dragover', e => { e.preventDefault(); dropRegion.classList.add('drag-over'); });
dropRegion.addEventListener('dragleave', () => dropRegion.classList.remove('drag-over'));
dropRegion.addEventListener('drop', e => {
  e.preventDefault(); dropRegion.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
dropInner.addEventListener('click', e => { if (e.target.tagName !== 'LABEL') fileIn.click(); });
fileIn.addEventListener('change', () => { if (fileIn.files[0]) setFile(fileIn.files[0]); });

function setFile(f) {
  if (!f.name.endsWith('.csv')) { showToast('Only CSV files supported'); return; }
  currentFile = f;
  fName.textContent = f.name;
  fSize.textContent = fmtBytes(f.size);
  fileReady.style.display = 'flex';
  // Animate preflight checks
  runChecks();
}

function runChecks() {
  const checks = ['chk1','chk2','chk3','chk4'];
  checks.forEach((id,i) => {
    setTimeout(() => {
      const el = document.getElementById(id);
      el.classList.add('checked');
      el.querySelector('.chk-box').textContent = '■';
      if (i === checks.length - 1) execBtn.disabled = false;
    }, 300 * (i + 1));
  });
}

function fmtBytes(b) {
  if (b < 1024) return b + 'B';
  if (b < 1048576) return (b/1024).toFixed(1) + 'KB';
  return (b/1048576).toFixed(2) + 'MB';
}

// ── Execute ───────────────────────────────────────────────
execBtn.addEventListener('click', runInference);

async function runInference() {
  if (!currentFile) return;
  showPanel('processing');
  sbInfer.className = 'sb-dot busy';
  sbInferLabel.textContent = 'INFERENCE ACTIVE';

  const steps = ['ps0','ps1','ps2','ps3','ps4','ps5','ps6'];
  let si = 0;
  function advStep() {
    if (si > 0) { const prev = document.getElementById(steps[si-1]); prev.className = 'ps done'; prev.textContent = prev.textContent.replace('[ ]',''); }
    if (si < steps.length) { const cur = document.getElementById(steps[si]); cur.className = 'ps active'; }
    const pct = Math.round((si / steps.length) * 100);
    const circ = document.getElementById('procCircle');
    circ.style.strokeDashoffset = 565 - (565 * pct / 100);
    document.getElementById('procPct').textContent = pct + '%';
    si++;
  }
  const stepTimer = setInterval(advStep, 700);

  const fd = new FormData();
  fd.append('file', currentFile);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    clearInterval(stepTimer);
    si = steps.length;
    advStep();

    // Final ring to 100%
    document.getElementById('procCircle').style.strokeDashoffset = '0';
    document.getElementById('procPct').textContent = '100%';
    steps.forEach(id => {
      const el = document.getElementById(id);
      el.className = 'ps done';
    });

    setTimeout(() => {
      if (!res.ok || data.error) { showPanel('upload'); showToast('Error: ' + (data.error || 'Unknown server error')); return; }
      sbInfer.className = 'sb-dot ok';
      sbInferLabel.textContent = 'INFERENCE COMPLETE';
      renderDashboard(data);
    }, 600);

  } catch (err) {
    clearInterval(stepTimer);
    showPanel('upload');
    sbInfer.className = 'sb-dot warn';
    sbInferLabel.textContent = 'ERROR';
    showToast('Network error: ' + err.message);
  }
}

// ── Dashboard rendering ───────────────────────────────────
function renderDashboard(data) {
  allPredictions = data.predictions;
  filteredPreds  = [...allPredictions];
  currentPage    = 1;

  // Dynamic thresholds — relative to this dataset's max RUL
  // so CRITICAL/WARNING labels are always meaningful regardless of file size
  const maxRULInData = Math.max(...allPredictions.map(r => r.predicted_RUL));
  window._critThresh = maxRULInData * 0.15;   // bottom 15% = CRITICAL
  window._warnThresh = maxRULInData * 0.40;   // bottom 40% = WARNING

  const m = data.overall_metrics;

  // Gauge cards
  const gauges = [
    { label:'MSE',  val: m.MSE,  cls:'gc-g', max: 20 },
    { label:'MAE',  val: m.MAE,  cls:'gc-a', max: 10 },
    { label:'RMSE', val: m.RMSE, cls:'gc-b', max: 15 },
    { label:'R²',   val: m.R2,   cls:'gc-p', max: 1  },
  ];
  gaugeRow.innerHTML = gauges.map((g,i) => `
    <div class="gauge-card ${g.cls}">
      <div class="gc-label">${g.label}</div>
      <div class="gauge-ring">
        ${makeGaugeSvg(g, i)}
      </div>
      <div class="gc-value" id="gv_${g.label.replace('²','2')}">${g.val}</div>
    </div>
  `).join('');

  // Animate gauge arcs
  setTimeout(() => {
    gauges.forEach(g => {
      const arc = document.getElementById('garc_' + g.label.replace('²','2'));
      if (!arc) return;
      const pct  = g.label === 'R²'
        ? Math.max(0, g.val)          // R2 0-1
        : Math.min(1, g.val / g.max); // others
      const circumference = 2 * Math.PI * 38;
      arc.style.strokeDashoffset = circumference * (1 - pct);
    });
  }, 100);

  // Engine health (based on R2)
  const healthVal = Math.max(0, Math.round(m.R2 * 100));
  const arcLen    = 126;
  healthArc.style.strokeDashoffset = arcLen - (arcLen * healthVal / 100);
  healthPct.textContent = healthVal + '%';
  ehStatus.textContent  = healthVal > 80 ? 'HEALTHY' : healthVal > 50 ? 'DEGRADED' : 'CRITICAL';
  ehStatus.style.color  = healthVal > 80 ? 'var(--g)' : healthVal > 50 ? 'var(--amber)' : 'var(--red)';

  // Summary box
  const engines = [...new Set(allPredictions.map(r => r.engine_id))];
  const minRUL  = Math.min(...allPredictions.map(r => r.predicted_RUL));
  const maxRUL  = Math.max(...allPredictions.map(r => r.predicted_RUL));
  const critCount = allPredictions.filter(r => r.predicted_RUL < window._critThresh).length;

  summItems.innerHTML = [
    { l: 'TOTAL RECORDS',  v: allPredictions.length,  warn: false },
    { l: 'ENGINES',        v: engines.length,          warn: false },
    { l: 'MIN PRED RUL',   v: minRUL.toFixed(1),       warn: minRUL < 30 },
    { l: 'MAX PRED RUL',   v: maxRUL.toFixed(1),       warn: false },
    { l: 'CRITICAL CYCLES',v: critCount,               warn: critCount > 0 },
    { l: 'OVERALL R²',     v: m.R2,                    warn: m.R2 < 0.5 },
  ].map(r => `
    <div class="summ-row">
      <span class="sr-label">${r.l}</span>
      <span class="${r.warn ? 'sr-warn' : 'sr-val'}">${r.v}</span>
    </div>
  `).join('');

  // Engine filter pills
  engineFilter.innerHTML = ['ALL', ...engines].map((e,i) => `
    <span class="eng-pill ${i===0?'active':''}" data-eng="${e}">${e}</span>
  `).join('');
  engineFilter.querySelectorAll('.eng-pill').forEach(p => p.addEventListener('click', () => {
    engineFilter.querySelectorAll('.eng-pill').forEach(x => x.classList.remove('active'));
    p.classList.add('active');
    activeEngineId = p.dataset.eng === 'ALL' ? null : +p.dataset.eng;
    renderChart();
  }));

  renderChart();
  renderTable();

  navBadge.style.display = 'inline';
  showPanel('dashboard');
}

// ── Gauge SVG helper ──────────────────────────────────────
function makeGaugeSvg(g, i) {
  const id   = 'garc_' + g.label.replace('²','2');
  const circ = 2 * Math.PI * 38;
  const cols = ['#00ff88','#f0a500','#3399ff','#cc44ff'];
  return `
    <svg viewBox="0 0 90 90" width="80" height="80">
      <circle cx="45" cy="45" r="38" stroke="#1a2e1a" stroke-width="7" fill="none"/>
      <circle cx="45" cy="45" r="38" stroke="${cols[i]}" stroke-width="7" fill="none"
        stroke-linecap="round" stroke-dasharray="${circ}" stroke-dashoffset="${circ}"
        id="${id}" style="transform:rotate(-90deg);transform-origin:45px 45px;transition:stroke-dashoffset 1.4s cubic-bezier(.17,.67,.29,1)"/>
    </svg>`;
}

// ── Chart.js RUL Line Chart ───────────────────────────────
function renderChart() {
  const src = activeEngineId
    ? allPredictions.filter(r => r.engine_id === activeEngineId)
    : allPredictions;

  const labels  = src.map(r => `E${r.engine_id}·C${r.cycle}`);
  const trueRUL = src.map(r => r.true_RUL);
  const predRUL = src.map(r => r.predicted_RUL);

  if (rulChartInst) rulChartInst.destroy();

  rulChartInst = new Chart(document.getElementById('rulChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'True RUL',
          data: trueRUL,
          borderColor: '#00ff88',
          backgroundColor: 'rgba(0,255,136,0.05)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'Predicted RUL',
          data: predRUL,
          borderColor: '#f0a500',
          backgroundColor: 'rgba(240,165,0,0.04)',
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
          fill: false,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: '#4a6a4a', font: { family: 'Share Tech Mono', size: 10 },
            boxWidth: 12, padding: 16,
          },
        },
        tooltip: {
          backgroundColor: '#0f160f',
          borderColor: '#1a2e1a',
          borderWidth: 1,
          titleColor: '#00ff88',
          bodyColor: '#b0d0b0',
          titleFont: { family: 'Share Tech Mono', size: 10 },
          bodyFont: { family: 'Share Tech Mono', size: 10 },
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#3a5a3a', font: { family: 'Share Tech Mono', size: 9 },
            maxTicksLimit: 12, maxRotation: 0,
          },
          grid: { color: '#1a2e1a40' },
        },
        y: {
          ticks: { color: '#3a5a3a', font: { family: 'Share Tech Mono', size: 9 } },
          grid: { color: '#1a2e1a40' },
        },
      },
    },
  });
}

// ── Predictions Table ─────────────────────────────────────
function renderTable() {
  const start = (currentPage - 1) * PAGE_SIZE;
  const page  = filteredPreds.slice(start, start + PAGE_SIZE);

  tblBody.innerHTML = page.map(r => {
    const delta = (r.predicted_RUL - r.true_RUL);
    const dStr  = (delta >= 0 ? '+' : '') + delta.toFixed(2);
    const dCls  = delta >= 0 ? 'td-dp' : 'td-dn';
    const badge = r.predicted_RUL < window._critThresh
      ? '<span class="badge b-crit">CRITICAL</span>'
      : r.predicted_RUL < window._warnThresh
        ? '<span class="badge b-warn">WARNING</span>'
        : '<span class="badge b-ok">NOMINAL</span>';
    return `
      <tr>
        <td class="td-eng">${r.engine_id}</td>
        <td>${r.cycle}</td>
        <td>${r.true_RUL}</td>
        <td class="td-pred">${r.predicted_RUL}</td>
        <td class="${dCls}">${dStr}</td>
        <td>${badge}</td>
      </tr>`;
  }).join('');

  const total = Math.ceil(filteredPreds.length / PAGE_SIZE);
  pgInfo.textContent = `${currentPage}/${total || 1} · ${filteredPreds.length} rows`;
  pgPrev.disabled = currentPage === 1;
  pgNext.disabled = currentPage >= total;
}

pgPrev.addEventListener('click', () => { if (currentPage > 1) { currentPage--; renderTable(); } });
pgNext.addEventListener('click', () => {
  if (currentPage < Math.ceil(filteredPreds.length / PAGE_SIZE)) { currentPage++; renderTable(); }
});
searchIn.addEventListener('input', () => {
  const q = searchIn.value.trim().toLowerCase();
  filteredPreds = q
    ? allPredictions.filter(r => String(r.engine_id).includes(q) || String(r.cycle).includes(q))
    : [...allPredictions];
  currentPage = 1; renderTable();
});

// ── Download ──────────────────────────────────────────────
dlBtn.addEventListener('click', () => {
  if (!allPredictions.length) return;
  const hdr  = 'engine_id,cycle,true_RUL,predicted_RUL\n';
  const rows = allPredictions.map(r => `${r.engine_id},${r.cycle},${r.true_RUL},${r.predicted_RUL}`).join('\n');
  const blob = new Blob([hdr + rows], { type: 'text/csv' });
  const a    = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: `apu_report_${Date.now()}.csv` });
  a.click(); URL.revokeObjectURL(a.href);
});

// ── Toast ─────────────────────────────────────────────────
function showToast(msg) {
  document.querySelectorAll('.toast').forEach(t => t.remove());
  const t = document.createElement('div');
  Object.assign(t.style, {
    position:'fixed', bottom:'50px', right:'24px', zIndex:'2000',
    background:'#0f160f', border:'1px solid #ff3355', borderRadius:'4px',
    padding:'12px 18px', fontFamily:'Share Tech Mono', fontSize:'0.72rem',
    color:'#ff3355', maxWidth:'320px',
    animation:'fadeUp 0.3s ease',
  });
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}
