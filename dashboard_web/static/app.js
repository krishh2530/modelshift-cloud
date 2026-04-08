(() => {
  "use strict";

  // =========================================================
  // ModelShift-Lite Dashboard UI (app.js) — Optimized + Robust
  // =========================================================
  // Added / improved:
  // - Works with newer backend payloads (decision/taxonomy/top_drifted_features)
  // - Root-cause panel (Top-K drifted features)
  // - History rows can open/download archived reports
  // - Backend clear-history integration
  // - Still backward-compatible with older JSON schemas

  // -----------------------------
  // Config
  // -----------------------------
  const FETCH_INTERVAL_MS = 4000;
  const FETCH_INTERVAL_BG_MS = 15000; // when tab hidden
  const FETCH_TIMEOUT_MS = 8000;

  // If you ever host frontend separately, set:
  // window.MODELSHIFT_API_BASE = "http://127.0.0.1:8000" (example)
  const API_BASE = String(window.MODELSHIFT_API_BASE || "").replace(/\/+$/, "");
  const apiUrl = (path) => `${API_BASE}${path}`;

  // --- PRIVACY SHIELD: KICK OUT UNAUTHENTICATED USERS ---
  // --- PRIVACY SHIELD: KICK OUT UNAUTHENTICATED USERS ---
  // Safely grab the key and strip accidental quotes
  const rawKey = localStorage.getItem("api_key") || localStorage.getItem("apiKey") || localStorage.getItem("token") || "";
  const apiKey = rawKey.replace(/^"|"$/g, '').replace(/^Bearer /i, '').trim(); 

  if (!apiKey && window.location.pathname.includes("/dashboard")) {
      console.warn("No API Key found. Dashboard will load but show authentication errors.");
      // FIX: Removed the automatic redirect to stop the infinite loop!
  }

  // -----------------------------
  // Helpers
  // -----------------------------
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const isObj = (v) => v && typeof v === "object" && !Array.isArray(v);
  const el = (id) => document.getElementById(id);

  const safeUpper = (s) => String(s ?? "—").toUpperCase();

  function fmt(n, digits = 2) {
    if (!isNum(n)) return "—";
    return n.toFixed(digits);
  }

  function fmtMaybeInt(n) {
    if (!isNum(n)) return "—";
    const abs = Math.abs(n);
    if (abs >= 100) return n.toFixed(0);
    if (abs >= 10) return n.toFixed(1);
    return n.toFixed(2);
  }

  function nowIsoLocal() {
    const d = new Date();
    const pad = (x) => String(x).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(
      d.getMinutes()
    )}:${pad(d.getSeconds())}`;
  }

  function escapeHtml(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function pickNum(obj, keys) {
    if (!obj || typeof obj !== "object") return null;
    for (const k of keys) {
      if (isNum(obj[k])) return obj[k];
    }
    return null;
  }

  function pickStr(obj, keys) {
    if (!obj || typeof obj !== "object") return null;
    for (const k of keys) {
      if (typeof obj[k] === "string" && obj[k].trim()) return obj[k];
    }
    return null;
  }

  function clampInt(v, lo, hi) {
    v = Number(v);
    if (!Number.isFinite(v)) v = lo;
    return Math.max(lo, Math.min(hi, Math.round(v)));
  }

  // -----------------------------
  // Deep find utilities (schema-robust)
  // -----------------------------
  function deepFind(obj, predicate, maxDepth = 7) {
    const seen = new Set();
    function rec(node, depth) {
      if (!node || depth > maxDepth) return null;
      if (typeof node === "object") {
        if (seen.has(node)) return null;
        seen.add(node);
      }
      if (predicate(node)) return node;

      if (Array.isArray(node)) {
        for (const it of node) {
          const r = rec(it, depth + 1);
          if (r) return r;
        }
        return null;
      }
      if (typeof node === "object") {
        for (const k of Object.keys(node)) {
          const r = rec(node[k], depth + 1);
          if (r) return r;
        }
      }
      return null;
    }
    return rec(obj, 0);
  }

  function deepFindNumberByKeyRegex(obj, keyRegex) {
    let found = null;
    const seen = new Set();

    function rec(node, depth) {
      if (!node || depth > 9 || found !== null) return;
      if (typeof node === "object") {
        if (seen.has(node)) return;
        seen.add(node);
      }
      if (Array.isArray(node)) {
        for (const it of node) rec(it, depth + 1);
        return;
      }
      if (typeof node === "object") {
        for (const [k, v] of Object.entries(node)) {
          if (keyRegex.test(k) && isNum(v)) {
            found = v;
            return;
          }
          rec(v, depth + 1);
          if (found !== null) return;
        }
      }
    }

    rec(obj, 0);
    return found;
  }

  function deepFindArrayByKeyRegex(obj, keyRegex) {
    let found = null;
    const seen = new Set();

    function rec(node, depth) {
      if (!node || depth > 9 || found) return;
      if (typeof node === "object") {
        if (seen.has(node)) return;
        seen.add(node);
      }
      if (Array.isArray(node)) {
        for (const it of node) rec(it, depth + 1);
        return;
      }
      if (typeof node === "object") {
        for (const [k, v] of Object.entries(node)) {
          if (
            keyRegex.test(k) &&
            Array.isArray(v) &&
            v.length >= 2 &&
            v.every((x) => typeof x === "number" && Number.isFinite(x))
          ) {
            found = v;
            return;
          }
          rec(v, depth + 1);
          if (found) return;
        }
      }
    }

    rec(obj, 0);
    return found;
  }

  // -----------------------------
  // DOM refs
  // -----------------------------
  const runIdText = el("runIdText");
  const statusBadge = el("statusBadge");
  const clockText = el("clockText");
  const livePill = el("livePill");

  const latestIdText = el("latestIdText");
  const prevIdText = el("prevIdText");
  const latestGeneratedText = el("latestGeneratedText");
  const prevGeneratedText = el("prevGeneratedText");

  const zoomSlider = el("zoomSlider");
  const zoomMinus = el("zoomMinus");
  const zoomPlus = el("zoomPlus");
  const zoomValue = el("zoomValue");

  const speedSlider = el("speedSlider");
  const speedMinus = el("speedMinus");
  const speedPlus = el("speedPlus");
  const speedValue = el("speedValue");
  const fpsText = el("fpsText");

  const playBtn = el("playBtn");
  const resetBtn = el("resetBtn");

  const syncAxesChk = el("syncAxesChk");
  const loopChk = el("loopChk");
  const themeBtn = el("themeBtn");

  const exportBtn = el("exportBtn");
  const clearHistoryBtn = el("clearHistoryBtn");

  const chartLatest = el("chartLatest");
  const chartPrevious = el("chartPrevious");

  const latestChartNote = el("latestChartNote");
  const prevChartNote = el("prevChartNote");

  const latestCleanHealth = el("latestCleanHealth");
  const latestDriftedHealth = el("latestDriftedHealth");
  const latestPredKs = el("latestPredKs");
  const latestDeltaEntropy = el("latestDeltaEntropy");
  const latestMostDrifted = el("latestMostDrifted");

  const prevCleanHealth = el("prevCleanHealth");
  const prevDriftedHealth = el("prevDriftedHealth");
  const prevPredKs = el("prevPredKs");
  const prevDeltaEntropy = el("prevDeltaEntropy");

  const ledgerBody = el("ledgerBody");
  const analysisJson = el("analysisJson");
  const alertBox = el("alertBox");
  const historyBody = el("historyBody");
  const selfTestRunBtn = el("selfTestRunBtn");
  const selfTestJson = el("selfTestJson");
  // Optional eval containers (injected if missing)
  let evalLatestBox = el("evalLatestBox");
  let evalPrevBox = el("evalPrevBox");

  // Optional insight containers (injected if missing)
  let decisionBox = el("decisionBox");
  let rootCauseBox = el("rootCauseBox");

  const pages = {
    dash: el("page-dash"),
    analysis: el("page-analysis"),
    alerts: el("page-alerts"),
    config: el("page-config"),
    history: el("page-history"),
  };

  // -----------------------------
  // State
  // -----------------------------
  const state = {
    latest: {},
    previous: {},

    latestSeries: { clean: [], drifted: [] },
    prevSeries: { clean: [], drifted: [] },

    latestMeta: {},
    prevMeta: {},

    latestDecision: null,
    latestTopFeatures: [],

    zoom: Number(zoomSlider?.value ?? 3), // 1..10
    speed: Number(speedSlider?.value ?? 4), // 1..12
    playing: false,
    playhead: 0,
    fps: 0,

    syncAxes: true,
    loop: true,

    lastFetchRunId: null,
    serverHistory: [],

    // render optimization
    dirtyCharts: true,
    dirtyEval: true,
    dirtyHistory: true,
    dirtyInsights: true,
    lastChartKey: "",
  };

  // -----------------------------
  // Navigation
  // -----------------------------
  function setPage(name) {
    const safeName = pages[name] ? name : "dash";
    document.body.setAttribute("data-page", safeName);

    Object.keys(pages).forEach((k) => pages[k]?.classList.toggle("active", k === safeName));
    document.querySelectorAll(".navbtn").forEach((b) => b.classList.toggle("active", b.dataset.page === safeName));

    // Force one render after page switch
    if (safeName === "dash") state.dirtyCharts = true;
  }

  // -----------------------------
  // Clock
  // -----------------------------
  function updateClock() {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    if (clockText) clockText.textContent = `${hh}:${mm}:${ss}`;
  }

  // -----------------------------
  // Extractors
  // -----------------------------
  function extractRunMeta(run) {
    const run_id =
      run?.run_id ??
      run?.id ??
      deepFind(run, (x) => x && typeof x === "object" && typeof x.run_id === "string")?.run_id ??
      "—";

    const status =
      run?.status ??
      run?.state ??
      run?.decision?.status ??
      deepFind(run, (x) => x && typeof x === "object" && typeof x.status === "string")?.status ??
      "—";

    const generated_at =
      run?.generated_at ??
      run?.generated ??
      run?.time ??
      deepFind(run, (x) => x && typeof x === "object" && typeof x.generated_at === "string")?.generated_at ??
      "—";

    return { run_id, status, generated_at };
  }

  function extractDecision(run) {
    const d =
      (isObj(run?.decision) && run.decision) ||
      (isObj(run?.monitor_decision) && run.monitor_decision) ||
      deepFind(
        run,
        (x) =>
          isObj(x) &&
          (typeof x.status === "string" || typeof x.severity === "string" || typeof x.taxonomy === "string") &&
          isObj(x.signals)
      );

    if (!isObj(d)) return null;

    return {
      status: pickStr(d, ["status"]),
      severity: pickStr(d, ["severity"]),
      taxonomy: pickStr(d, ["taxonomy"]),
      signals: isObj(d.signals) ? d.signals : {},
      thresholds: isObj(d.thresholds) ? d.thresholds : {},
    };
  }

  function extractTopDriftedFeatures(run) {
    const direct =
      (Array.isArray(run?.top_drifted_features) && run.top_drifted_features) ||
      (Array.isArray(run?.root_causes) && run.root_causes) ||
      deepFind(
        run,
        (x) =>
          Array.isArray(x) &&
          x.length > 0 &&
          x.every(
            (it) =>
              isObj(it) &&
              (typeof it.feature === "string" || typeof it.name === "string") &&
              (isNum(it.ks_statistic) || isNum(it.ks))
          )
      );

    if (Array.isArray(direct)) {
      return direct
        .map((it) => ({
          feature: String(it.feature ?? it.name ?? "—"),
          ks_statistic: isNum(it.ks_statistic) ? it.ks_statistic : (isNum(it.ks) ? it.ks : null),
          p_value: isNum(it.p_value) ? it.p_value : null,
          severity: typeof it.severity === "string" ? it.severity : null,
        }))
        .filter((it) => isNum(it.ks_statistic))
        .sort((a, b) => b.ks_statistic - a.ks_statistic);
    }

    // Fallback: derive from feature_drift dict if present
    const fd =
      (isObj(run?.feature_drift) && run.feature_drift) ||
      (isObj(run?.feature_drift_results) && run.feature_drift_results) ||
      deepFind(run, (x) => {
        if (!isObj(x)) return false;
        const vals = Object.values(x);
        if (!vals.length) return false;
        const sample = vals[0];
        return isObj(sample) && (isNum(sample.ks_statistic) || isNum(sample.ks));
      });

    if (!isObj(fd)) return [];

    const out = [];
    for (const [feature, v] of Object.entries(fd)) {
      if (!isObj(v)) continue;
      const ks = isNum(v.ks_statistic) ? v.ks_statistic : (isNum(v.ks) ? v.ks : null);
      if (!isNum(ks)) continue;
      out.push({
        feature: String(feature),
        ks_statistic: ks,
        p_value: isNum(v.p_value) ? v.p_value : null,
        severity: typeof v.severity === "string" ? v.severity : null,
      });
    }

    out.sort((a, b) => b.ks_statistic - a.ks_statistic);
    return out;
  }

  function extractMetrics(run) {
    const summary = run?.summary && typeof run.summary === "object" ? run.summary : {};
    const decision = extractDecision(run);

    const clean_health =
      summary?.clean_health ??
      run?.clean_health ??
      run?.metrics?.clean_health ??
      decision?.signals?.clean_health ??
      deepFindNumberByKeyRegex(run, /^clean_health$/i) ??
      deepFindNumberByKeyRegex(run, /clean.*health/i);

    const drifted_health =
      summary?.drifted_health ??
      run?.drifted_health ??
      run?.metrics?.drifted_health ??
      decision?.signals?.drifted_health ??
      deepFindNumberByKeyRegex(run, /^drifted_health$/i) ??
      deepFindNumberByKeyRegex(run, /drift.*health/i);

    const pred_ks =
      summary?.drifted_pred_ks ??
      run?.pred_ks ??
      run?.metrics?.pred_ks ??
      decision?.signals?.prediction_ks ??
      deepFindNumberByKeyRegex(run, /pred.*ks/i);

    const delta_entropy =
      summary?.drifted_entropy_change ??
      run?.delta_entropy ??
      run?.metrics?.delta_entropy ??
      decision?.signals?.entropy_change ??
      deepFindNumberByKeyRegex(run, /(delta|d).*entropy/i);

    // Prefer new schema summary fields first
    let feat = null;
    const fName = summary?.drifted_last_window_feature;
    const fKs = summary?.drifted_last_window_ks;
    if (typeof fName === "string" && isNum(fKs)) {
      feat = { feature: fName, ks: fKs, window_size: run?.window_size };
    } else {
      feat =
        run?.most_drifted_feature ??
        run?.metrics?.most_drifted_feature ??
        deepFind(
          run,
          (x) =>
            x &&
            typeof x === "object" &&
            (typeof x.feature === "string" || typeof x.name === "string") &&
            (isNum(x.ks) || isNum(x.ks_statistic))
        );
      if (feat && !isNum(feat.ks) && isNum(feat.ks_statistic)) {
        feat = { ...feat, ks: feat.ks_statistic };
      }
    }

    // Final fallback from top_drifted_features
    if (!feat) {
      const top = extractTopDriftedFeatures(run);
      if (top.length) {
        feat = { feature: top[0].feature, ks: top[0].ks_statistic, window_size: run?.window_size };
      }
    }

    return { clean_health, drifted_health, pred_ks, delta_entropy, feat };
  }

  function extractSeries(run) {
    const clean =
      run?.series?.clean ??
      run?.series?.clean_health ??
      run?.series?.clean_health_series ??
      run?.health_series?.clean ??
      run?.clean_series ??
      run?.clean_health_series ??
      deepFindArrayByKeyRegex(run, /clean.*(series|health)/i);

    const drifted =
      run?.series?.drifted ??
      run?.series?.drift ??
      run?.series?.drifted_health ??
      run?.series?.drifted_health_series ??
      run?.health_series?.drifted ??
      run?.drifted_series ??
      run?.drifted_health_series ??
      deepFindArrayByKeyRegex(run, /(drift|drifted).*(series|health)/i);

    return {
      clean: Array.isArray(clean) ? clean : [],
      drifted: Array.isArray(drifted) ? drifted : [],
    };
  }

  function extractEvaluation(run) {
    const ev = run?.evaluation;
    if (!ev || typeof ev !== "object") return null;

    const clean = ev.clean && typeof ev.clean === "object" ? ev.clean : null;
    const drifted = ev.drifted && typeof ev.drifted === "object" ? ev.drifted : null;
    const baseline = ev.baseline && typeof ev.baseline === "object" ? ev.baseline : null;

    return { baseline, clean, drifted };
  }

  // -----------------------------
  // Status badge
  // -----------------------------
function setLivePill(isLive) {
  if (!livePill) return;

  const dot = livePill.querySelector(".livepill__dot");
  const txt = livePill.querySelector(".livepill__text");

  if (txt) txt.textContent = isLive ? "LIVE" : "DISCONNECTED";

  // Blink only when live
  livePill.classList.toggle("blink", !!isLive);

  livePill.style.borderColor = isLive
    ? "rgba(80, 200, 120, 0.45)"
    : "rgba(209, 31, 31, 0.45)";

  livePill.style.background = isLive
    ? "rgba(80, 200, 120, 0.08)"
    : "rgba(209, 31, 31, 0.10)";

  if (dot) {
    dot.style.background = isLive
      ? "rgba(80, 200, 120, 0.95)"
      : "rgba(209, 31, 31, 0.95)";
    dot.style.boxShadow = isLive
      ? "0 0 10px rgba(80, 200, 120, 0.55)"
      : "0 0 10px rgba(209, 31, 31, 0.45)";
  }
}
  function setStatusBadge(status) {
  if (!statusBadge) return;

  const s = safeUpper(status);
  statusBadge.textContent = s;
  statusBadge.style.borderColor = "rgba(255,255,255,0.12)";
  statusBadge.style.background = "rgba(0,0,0,0.25)";
  statusBadge.style.color = "";

  if (s.includes("CRITICAL")) {
    statusBadge.style.borderColor = "rgba(209,31,31,0.55)";
    statusBadge.style.background = "rgba(209,31,31,0.12)";
  } else if (s.includes("WARN")) {
    statusBadge.style.borderColor = "rgba(255,170,0,0.45)";
    statusBadge.style.background = "rgba(255,170,0,0.10)";
  } else if (s.includes("STABLE") || s.includes("OK")) {
    statusBadge.style.borderColor = "rgba(80,200,120,0.45)";
    statusBadge.style.background = "rgba(80,200,120,0.08)";
  }
}

  // -----------------------------
  // Ledger
  // -----------------------------
  function fillLedger(latestRun, prevRun) {
    if (!ledgerBody) return;
    ledgerBody.innerHTML = "";

    const rows = [latestRun, prevRun].filter(Boolean);
    for (const r of rows) {
      const meta = extractRunMeta(r);
      const m = extractMetrics(r);

      const tr = document.createElement("tr");

      const td1 = document.createElement("td");
      td1.textContent = meta.generated_at || "—";

      const td2 = document.createElement("td");
      td2.textContent = meta.run_id || "—";

      const td3 = document.createElement("td");
      td3.textContent = meta.status || "—";
      if (safeUpper(meta.status).includes("CRITICAL") || safeUpper(meta.status).includes("WARN")) {
        td3.className = "bad";
      }

      const td4 = document.createElement("td");
      td4.textContent = fmt(m.drifted_health, 2);

      const td5 = document.createElement("td");
      td5.textContent = fmt(m.pred_ks, 4);

      tr.append(td1, td2, td3, td4, td5);
      ledgerBody.appendChild(tr);
    }
  }

  // -----------------------------
  // History
  // -----------------------------
  function historyReportLinks(runId) {
    if (!runId || runId === "—") return "";
    const rid = encodeURIComponent(runId);
    const openHref = apiUrl(`/api/report/${rid}`);
    const dlHref = apiUrl(`/api/report/${rid}?download=1`);
    return `
      <span class="ms-inline-actions">
        <a href="${openHref}" target="_blank" rel="noopener">OPEN</a>
        <span>·</span>
        <a href="${dlHref}" target="_blank" rel="noopener">DL</a>
      </span>
    `;
  }

  function renderHistory() {
    if (!historyBody) return;
    historyBody.innerHTML = "";

    const rows = Array.isArray(state.serverHistory) ? state.serverHistory : [];
    for (const r of rows) {
      const tr = document.createElement("tr");

      const tdSaved = document.createElement("td");
      tdSaved.textContent = r.saved_at || "—";

      const tdGen = document.createElement("td");
      tdGen.textContent = r.generated_at || "—";

      const tdRun = document.createElement("td");
      tdRun.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:2px;">
          <span>${escapeHtml(r.run_id || "—")}</span>
          ${historyReportLinks(r.run_id)}
        </div>
      `;
      if (r.drifted_last_window_feature) {
        tdRun.title = `Top feature: ${r.drifted_last_window_feature}${isNum(r.drifted_last_window_ks) ? ` (KS=${r.drifted_last_window_ks.toFixed(4)})` : ""}`;
      }

      const tdStatus = document.createElement("td");
      tdStatus.textContent = r.status || "—";
      if (safeUpper(r.status).includes("CRITICAL") || safeUpper(r.status).includes("WARN")) {
        tdStatus.className = "bad";
      }

      const tdC = document.createElement("td");
      tdC.textContent = isNum(r.clean_health) ? r.clean_health.toFixed(2) : "—";

      const tdD = document.createElement("td");
      tdD.textContent = isNum(r.drifted_health) ? r.drifted_health.toFixed(2) : "—";

      const tdK = document.createElement("td");
      tdK.textContent = isNum(r.drifted_pred_ks) ? r.drifted_pred_ks.toFixed(4) : "—";

      const tdE = document.createElement("td");
      tdE.textContent = isNum(r.drifted_entropy_change) ? r.drifted_entropy_change.toFixed(4) : "—";

      tr.append(tdSaved, tdGen, tdRun, tdStatus, tdC, tdD, tdK, tdE);
      historyBody.appendChild(tr);
    }
  }

  // -----------------------------
  // Decision / Root Cause UI + CSS injection
  // -----------------------------
  function ensureInsightCss() {
    if (document.getElementById("ms-insight-css")) return;
    const st = document.createElement("style");
    st.id = "ms-insight-css";
    st.textContent = `
      .ms-insight-wrap{
        margin-top:12px;
        border:1px solid rgba(255,255,255,0.10);
        background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.06));
        padding:12px;
      }
      .ms-insight-title{
        font-family:ui-monospace, Menlo, Consolas, monospace;
        font-weight:900;
        letter-spacing:1px;
        opacity:0.92;
        margin-bottom:10px;
      }
      .ms-chip-row{
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        margin-bottom:10px;
      }
      .ms-chip{
        border:1px solid rgba(255,255,255,0.12);
        padding:6px 8px;
        font-family:ui-monospace, Menlo, Consolas, monospace;
        font-size:12px;
        line-height:1;
        background:rgba(0,0,0,0.20);
      }
      .ms-chip.bad{
        border-color:rgba(209,31,31,0.45);
        background:rgba(209,31,31,0.10);
        color:rgba(255,220,220,0.95);
      }
      .ms-chip.warn{
        border-color:rgba(255,170,0,0.45);
        background:rgba(255,170,0,0.08);
      }
      .ms-chip.good{
        border-color:rgba(80,200,120,0.40);
        background:rgba(80,200,120,0.06);
      }
      .ms-signal-grid{
        display:grid;
        grid-template-columns:repeat(4,minmax(0,1fr));
        gap:8px;
      }
      .ms-signal{
        border:1px solid rgba(255,255,255,0.08);
        padding:8px;
        background:rgba(0,0,0,0.14);
      }
      .ms-signal .k{
        font-family:ui-monospace, Menlo, Consolas, monospace;
        font-size:11px;
        opacity:0.75;
        letter-spacing:.7px;
      }
      .ms-signal .v{
        font-family:ui-monospace, Menlo, Consolas, monospace;
        font-size:16px;
        font-weight:800;
        margin-top:3px;
      }
      .ms-rc-table-wrap{
        overflow:auto;
        border:1px solid rgba(255,255,255,0.08);
      }
      table.ms-rc-table{
        width:100%;
        border-collapse:collapse;
        min-width:600px;
      }
      table.ms-rc-table th, table.ms-rc-table td{
        padding:9px 10px;
        border-bottom:1px solid rgba(255,255,255,0.08);
        font-family:ui-monospace, Menlo, Consolas, monospace;
        font-size:12px;
        text-align:left;
      }
      table.ms-rc-table th{
        opacity:0.85;
        letter-spacing:.8px;
      }
      .ms-inline-actions{
        display:inline-flex;
        gap:4px;
        opacity:0.85;
        font-size:11px;
        font-family:ui-monospace, Menlo, Consolas, monospace;
      }
      .ms-inline-actions a{
        color:inherit;
        text-decoration:none;
        border-bottom:1px dotted rgba(255,255,255,0.3);
      }
      .ms-muted{
        font-family:ui-monospace, Menlo, Consolas, monospace;
        font-size:12px;
        opacity:0.68;
      }
      @media (max-width: 980px){
        .ms-signal-grid{ grid-template-columns:repeat(2,minmax(0,1fr)); }
      }
    `;
    document.head.appendChild(st);
  }

  function ensureInsightBoxes() {
    ensureInsightCss();

    if (!decisionBox) {
      decisionBox = document.createElement("div");
      decisionBox.id = "decisionBox";
      if (latestMostDrifted && latestMostDrifted.parentNode) {
        latestMostDrifted.insertAdjacentElement("afterend", decisionBox);
      } else if (pages.dash) {
        pages.dash.appendChild(decisionBox);
      }
    }

    if (!rootCauseBox) {
      rootCauseBox = document.createElement("div");
      rootCauseBox.id = "rootCauseBox";
      if (decisionBox && decisionBox.parentNode) {
        decisionBox.insertAdjacentElement("afterend", rootCauseBox);
      } else if (pages.dash) {
        pages.dash.appendChild(rootCauseBox);
      }
    }
  }

  function severityClass(sev) {
    const s = safeUpper(sev);
    if (s.includes("CRITICAL") || s.includes("HIGH")) return "bad";
    if (s.includes("MEDIUM") || s.includes("WARN")) return "warn";
    if (s.includes("LOW") || s.includes("STABLE")) return "good";
    return "";
  }

  function renderInsights(latestRun) {
    ensureInsightBoxes();

    const decision = state.latestDecision || extractDecision(latestRun);
    const topFeatures = Array.isArray(state.latestTopFeatures) ? state.latestTopFeatures : extractTopDriftedFeatures(latestRun);

    // Decision panel
    if (decisionBox) {
      const chips = [];

      if (decision?.status) {
        chips.push(`<span class="ms-chip ${severityClass(decision.status)}">STATUS: ${escapeHtml(decision.status)}</span>`);
      }
      if (decision?.severity) {
        chips.push(`<span class="ms-chip ${severityClass(decision.severity)}">SEVERITY: ${escapeHtml(decision.severity)}</span>`);
      }
      if (decision?.taxonomy) {
        chips.push(`<span class="ms-chip">TAXONOMY: ${escapeHtml(decision.taxonomy)}</span>`);
      }

      const signals = isObj(decision?.signals) ? decision.signals : {};
      const sigHtml = `
        <div class="ms-signal-grid">
          <div class="ms-signal"><div class="k">AVG_FEATURE_KS</div><div class="v">${fmt(signals.avg_feature_ks, 4)}</div></div>
          <div class="ms-signal"><div class="k">MAX_FEATURE_KS</div><div class="v">${fmt(signals.max_feature_ks, 4)}</div></div>
          <div class="ms-signal"><div class="k">PREDICTION_KS</div><div class="v">${fmt(signals.prediction_ks, 4)}</div></div>
          <div class="ms-signal"><div class="k">COMPOSITE_SCORE</div><div class="v">${fmt(signals.composite_score, 4)}</div></div>
        </div>
      `;

      if (decision) {
        decisionBox.innerHTML = `
          <div class="ms-insight-wrap">
            <div class="ms-insight-title">DECISION ENGINE</div>
            <div class="ms-chip-row">${chips.join("")}</div>
            ${sigHtml}
            <div class="ms-muted" style="margin-top:8px;">
              ${signals.max_feature_name ? `Top signal feature: ${escapeHtml(signals.max_feature_name)} • ` : ""}
              Feature count: ${isNum(signals.feature_count) ? String(signals.feature_count) : "—"}
            </div>
          </div>
        `;
      } else {
        decisionBox.innerHTML = `
          <div class="ms-insight-wrap">
            <div class="ms-insight-title">DECISION ENGINE</div>
            <div class="ms-muted">No composite decision block found in payload (backward-compatible mode).</div>
          </div>
        `;
      }
    }

    // Root-cause panel
    if (rootCauseBox) {
      const rows = (topFeatures || []).slice(0, 8);
      if (!rows.length) {
        rootCauseBox.innerHTML = `
          <div class="ms-insight-wrap">
            <div class="ms-insight-title">ROOT CAUSE CANDIDATES</div>
            <div class="ms-muted">No feature-level drift list found in payload.</div>
          </div>
        `;
      } else {
        const trs = rows
          .map((r, i) => {
            const sev = r.severity || (r.ks_statistic >= 0.35 ? "CRITICAL" : r.ks_statistic >= 0.2 ? "HIGH" : r.ks_statistic >= 0.1 ? "MEDIUM" : "LOW");
            return `
              <tr>
                <td>${i + 1}</td>
                <td>${escapeHtml(String(r.feature).toUpperCase())}</td>
                <td>${fmt(r.ks_statistic, 4)}</td>
                <td>${isNum(r.p_value) ? r.p_value.toExponential(2) : "—"}</td>
                <td class="${severityClass(sev)}">${escapeHtml(sev)}</td>
              </tr>
            `;
          })
          .join("");

        rootCauseBox.innerHTML = `
          <div class="ms-insight-wrap">
            <div class="ms-insight-title">ROOT CAUSE CANDIDATES (TOP DRIFTED FEATURES)</div>
            <div class="ms-rc-table-wrap">
              <table class="ms-rc-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>FEATURE</th>
                    <th>KS</th>
                    <th>P_VALUE</th>
                    <th>SEVERITY</th>
                  </tr>
                </thead>
                <tbody>${trs}</tbody>
              </table>
            </div>
          </div>
        `;
      }
    }
  }

  // -----------------------------
  // Evaluation UI (Premium) + CSS injection
  // -----------------------------
  function ensureEvalCss() {
    if (document.getElementById("ms-eval-css")) return;
    const st = document.createElement("style");
    st.id = "ms-eval-css";
    st.textContent = `
      .evalwrap{
        margin-top:12px;
        border:1px solid rgba(255,255,255,0.10);
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.06));
        padding:12px;
      }
      .evalhead{
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        gap:10px;
        margin-bottom:10px;
      }
      .evaltitle{
        font-family: ui-monospace, Menlo, Consolas, monospace;
        font-weight:900;
        letter-spacing:1px;
        opacity:0.92;
      }
      .evalnote{
        font-family: ui-monospace, Menlo, Consolas, monospace;
        font-size:12px;
        opacity:0.65;
        text-align:right;
      }
      .evaltablewrap{ overflow:auto; border:1px solid rgba(255,255,255,0.08); }
      table.evaltable{
        width:100%;
        border-collapse:collapse;
        min-width:680px;
      }
      table.evaltable th, table.evaltable td{
        padding:10px 12px;
        border-bottom:1px solid rgba(255,255,255,0.08);
        font-family: ui-monospace, Menlo, Consolas, monospace;
        font-size:12.5px;
        text-align:left;
      }
      table.evaltable th{
        opacity:0.85;
        letter-spacing:0.8px;
      }
      .hint{
        font-family: ui-monospace, Menlo, Consolas, monospace;
        font-size:12px;
        opacity:0.70;
      }
      .bad{ color: rgba(209,31,31,0.95); font-weight:900; }
    `;
    document.head.appendChild(st);
  }

  function ensureEvalBoxes() {
    ensureEvalCss();

    // LATEST
    if (!evalLatestBox) {
      evalLatestBox = document.createElement("div");
      evalLatestBox.id = "evalLatestBox";
      if (rootCauseBox && rootCauseBox.parentNode) {
        rootCauseBox.insertAdjacentElement("afterend", evalLatestBox);
      } else if (latestMostDrifted && latestMostDrifted.parentNode) {
        latestMostDrifted.insertAdjacentElement("afterend", evalLatestBox);
      } else {
        pages.dash?.appendChild(evalLatestBox);
      }
    }

    // PREVIOUS
    if (!evalPrevBox) {
      evalPrevBox = document.createElement("div");
      evalPrevBox.id = "evalPrevBox";
      const prevBlock = prevDeltaEntropy?.closest?.(".block") || chartPrevious?.closest?.(".block") || pages.dash;
      prevBlock?.appendChild(evalPrevBox);
    }
  }

  function extractConfusion(evPart) {
    if (!evPart || typeof evPart !== "object") return null;
    const cm = evPart.confusion_matrix ?? evPart.cm ?? evPart.confusionMatrix ?? null;

    if (Array.isArray(cm) && cm.length === 2 && Array.isArray(cm[0]) && Array.isArray(cm[1])) {
      const tn = cm[0][0],
        fp = cm[0][1],
        fn = cm[1][0],
        tp = cm[1][1];
      if ([tn, fp, fn, tp].every(isNum)) return { tn, fp, fn, tp };
    }

    if (cm && typeof cm === "object") {
      const tn = cm.tn,
        fp = cm.fp,
        fn = cm.fn,
        tp = cm.tp;
      if ([tn, fp, fn, tp].every(isNum)) return { tn, fp, fn, tp };
    }
    return null;
  }

  function evalSectionHtml(title, ev) {
    if (!ev || typeof ev !== "object") {
      return `
        <div class="evalwrap">
          <div class="evalhead">
            <div class="evaltitle">${escapeHtml(title)}</div>
            <div class="evalnote">No evaluation metrics (labels unavailable / invalid)</div>
          </div>
        </div>
      `;
    }

    const clean = ev.clean || null;
    const drifted = ev.drifted || null;

    const rows = [
      ["ACCURACY", ["accuracy", "acc"], 4],
      ["PRECISION", ["precision", "prec"], 4],
      ["RECALL", ["recall", "rec"], 4],
      ["F1 SCORE", ["f1_score", "f1", "f1score"], 4],
      ["ROC AUC", ["roc_auc", "auc", "rocAuc"], 4],
      ["LOG LOSS", ["log_loss", "logloss"], 6],
      ["MSE (BRIER)", ["mse", "brier", "brier_score_loss"], 6],
      ["RMSE", ["rmse"], 6],
      ["R²", ["r2", "r_squared"], 6],
    ];

    const cleanCM = extractConfusion(clean);
    const driftedCM = extractConfusion(drifted);

    const trs = rows
      .map(([label, keys, digs]) => {
        const c = pickNum(clean, keys);
        const d = pickNum(drifted, keys);

        let driftBad = false;
        const higherBetter = ["ACCURACY", "PRECISION", "RECALL", "F1 SCORE", "ROC AUC", "R²"].includes(label);
        if (isNum(c) && isNum(d)) driftBad = higherBetter ? d < c : d > c;

        return `
          <tr>
            <td>${escapeHtml(label)}</td>
            <td>${isNum(c) ? c.toFixed(digs) : "—"}</td>
            <td class="${driftBad ? "bad" : ""}">${isNum(d) ? d.toFixed(digs) : "—"}</td>
          </tr>
        `;
      })
      .join("");

    const cmLine = (() => {
      if (!cleanCM && !driftedCM) return "";
      const c = cleanCM
        ? `CLEAN[tn=${cleanCM.tn}, fp=${cleanCM.fp}, fn=${cleanCM.fn}, tp=${cleanCM.tp}]`
        : "CLEAN[—]";
      const d = driftedCM
        ? `DRIFTED[tn=${driftedCM.tn}, fp=${driftedCM.fp}, fn=${driftedCM.fn}, tp=${driftedCM.tp}]`
        : "DRIFTED[—]";
      return `<div class="hint" style="margin-top:10px;">CONFUSION MATRIX: ${escapeHtml(c)} • ${escapeHtml(d)}</div>`;
    })();

    return `
      <div class="evalwrap">
        <div class="evalhead">
          <div class="evaltitle">${escapeHtml(title)}</div>
          <div class="evalnote">Clean vs Drifted (requires labels)</div>
        </div>

        <div class="evaltablewrap">
          <table class="evaltable">
            <thead>
              <tr>
                <th>METRIC</th>
                <th>CLEAN</th>
                <th>DRIFTED</th>
              </tr>
            </thead>
            <tbody>${trs}</tbody>
          </table>
        </div>

        <div class="hint" style="margin-top:10px;">
          Note: MSE/RMSE are computed on probabilities vs labels (Brier-style).
        </div>
        ${cmLine}
      </div>
    `;
  }

  function renderEvaluation(latestRun, prevRun) {
    ensureEvalBoxes();
    const evL = extractEvaluation(latestRun);
    const evP = extractEvaluation(prevRun);

    if (evalLatestBox) evalLatestBox.innerHTML = evalSectionHtml("EVALUATION METRICS — LATEST RUN", evL);
    if (evalPrevBox) evalPrevBox.innerHTML = evalSectionHtml("EVALUATION METRICS — PREVIOUS RUN", evP);
  }

  // -----------------------------
  // Charts (Canvas)
  // -----------------------------
  const canvasCache = new WeakMap();

  function setupCanvas(canvas) {
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const rect = canvas.getBoundingClientRect();
    const cssW = rect.width;
    const cssH = rect.height;

    const pxW = Math.max(1, Math.floor(cssW * dpr));
    const pxH = Math.max(1, Math.floor(cssH * dpr));

    const cached = canvasCache.get(canvas);
    const same =
      cached &&
      cached.dpr === dpr &&
      cached.pxW === pxW &&
      cached.pxH === pxH &&
      cached.cssW === cssW &&
      cached.cssH === cssH;

    if (!same) {
      canvas.width = pxW;
      canvas.height = pxH;
      canvasCache.set(canvas, { dpr, pxW, pxH, cssW, cssH });
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    return { ctx, w: cssW, h: cssH };
  }

  function pointsPerSecond() {
    return 3 + state.speed * 5; // 1..12 => 8..63 pts/sec
  }

  function visibleCount(total) {
    const z = Math.max(1, Math.min(10, state.zoom));
    const frac = 1 / z;
    const count = Math.max(12, Math.floor(total * frac));
    return Math.min(total, count);
  }

  function drawChart(canvas, seriesClean, seriesDrifted, opts) {
    const { ctx, w, h } = setupCanvas(canvas);

    const padL = 60,
      padR = 24,
      padT = 14,
      padB = 40;

    const total = Math.max(seriesClean.length, seriesDrifted.length);
    ctx.clearRect(0, 0, w, h);

    if (total < 2) {
      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.font = "14px ui-monospace, Menlo, Consolas, monospace";
      ctx.fillText("No data available. Run the Python pipeline to beam data here.", 14, 24);
      return;
    }

    const vis = visibleCount(total);
    const center = Math.max(0, Math.min(total - 1, opts.centerIndex));
    let start = Math.floor(center - vis / 2);
    start = Math.max(0, Math.min(start, total - vis));
    const end = start + vis;

    const getVal = (arr, i) => (i < arr.length ? arr[i] : null);

    let ymin = Infinity,
      ymax = -Infinity;
    for (let i = start; i < end; i++) {
      const a = getVal(seriesClean, i);
      const b = getVal(seriesDrifted, i);
      if (isNum(a)) {
        ymin = Math.min(ymin, a);
        ymax = Math.max(ymax, a);
      }
      if (isNum(b)) {
        ymin = Math.min(ymin, b);
        ymax = Math.max(ymax, b);
      }
    }

    if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) {
      ymin = 0;
      ymax = 1;
    }

    let yr = ymax - ymin;
    if (yr < 1e-9) yr = 1;
    const pad = yr * 0.12;
    ymin -= pad;
    ymax += pad;

    const gx0 = padL,
      gy0 = padT,
      gx1 = w - padR,
      gy1 = h - padB;

    // major grid
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    const xTicks = 6,
      yTicks = 5;
    for (let t = 0; t <= xTicks; t++) {
      const x = gx0 + (t / xTicks) * (gx1 - gx0);
      ctx.beginPath();
      ctx.moveTo(x, gy0);
      ctx.lineTo(x, gy1);
      ctx.stroke();
    }
    for (let t = 0; t <= yTicks; t++) {
      const y = gy0 + (t / yTicks) * (gy1 - gy0);
      ctx.beginPath();
      ctx.moveTo(gx0, y);
      ctx.lineTo(gx1, y);
      ctx.stroke();
    }

    // minor dotted grid
    ctx.strokeStyle = "rgba(255,255,255,0.025)";
    ctx.setLineDash([2, 6]);
    const minor = 18;
    for (let x = gx0; x <= gx1; x += minor) {
      ctx.beginPath();
      ctx.moveTo(x, gy0);
      ctx.lineTo(x, gy1);
      ctx.stroke();
    }
    for (let y = gy0; y <= gy1; y += minor) {
      ctx.beginPath();
      ctx.moveTo(gx0, y);
      ctx.lineTo(gx1, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // axis labels
    // axis labels
    ctx.fillStyle = "rgba(255,255,255,0.72)";
    ctx.font = "12px ui-monospace, Menlo, Consolas, monospace";
    
    ctx.textAlign = "right"; // Aligns Y-axis numbers cleanly to the left of the line
    for (let t = 0; t <= yTicks; t++) {
      const v = ymax - (t / yTicks) * (ymax - ymin);
      const y = gy0 + (t / yTicks) * (gy1 - gy0);
      ctx.fillText(fmtMaybeInt(v), gx0 - 8, y + 4); 
    }
    ctx.textAlign = "left"; // Reset for X-axis
    for (let t = 0; t <= xTicks; t++) {
      const idx = Math.round(start + (t / xTicks) * (vis - 1));
      const x = gx0 + (t / xTicks) * (gx1 - gx0);
      ctx.fillText(String(idx), x - 6, gy1 + 22);
    }

    ctx.save();
    ctx.translate(16, (gy0 + gy1) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.fillText("Health", 0, 0);
    ctx.restore();

    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.fillText("window", (gx0 + gx1) / 2 - 20, h - 8);

    const denom = Math.max(1, vis - 1);
    const xScale = (gx1 - gx0) / denom;

    const xMap = (i) => gx0 + (i - start) * xScale;
    const yMap = (v) => gy0 + (1 - (v - ymin) / (ymax - ymin)) * (gy1 - gy0);

    function drawLine(arr, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      for (let i = start; i < end; i++) {
        const v = getVal(arr, i);
        if (!isNum(v)) {
          started = false;
          continue;
        }
        const x = xMap(i);
        const y = yMap(v);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    drawLine(seriesClean, "rgba(255,255,255,0.88)");
    drawLine(seriesDrifted, "rgba(209,31,31,0.88)");

    // legend
    ctx.fillStyle = "rgba(255,255,255,0.82)";
    ctx.font = "13px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText("CLEAN", gx1 - 120, gy0 + 20);
    ctx.fillStyle = "rgba(209,31,31,0.90)";
    ctx.fillText("DRIFTED", gx1 - 120, gy0 + 40);
  }

  // -----------------------------
  // Render from API payload
  // -----------------------------
  function updateFromData(payload) {
    // --- RE-ENABLE EXPORT BUTTON ---
    const exportBtn = document.getElementById("exportBtn");
    if (exportBtn) {
      exportBtn.disabled = false;
      exportBtn.innerText = "EXPORT REPORT";
    }

// --- SMART VIEW DATASET BUTTON ---
    const viewDatasetBtn = document.getElementById("viewDatasetBtn");
    const statusUpper = String(state.latestMeta.status || "").toUpperCase();
    
    // Only show if the status is a warning or critical drift
    const hasDrift = statusUpper.includes("DRIFT") || statusUpper.includes("WARN") || statusUpper.includes("CRITICAL");

    if (viewDatasetBtn && state.latestMeta.run_id && state.latestMeta.run_id !== "—" && hasDrift) {
      viewDatasetBtn.style.display = "inline-block";
      viewDatasetBtn.onclick = () => {
        // Pass API key in the URL so the new tab is authenticated!
        window.open(apiUrl(`/dataset/${state.latestMeta.run_id}?api_key=${apiKey}`), "_blank");
      };
    } else if (viewDatasetBtn) {
      viewDatasetBtn.style.display = "none";
    }
state.latest = isObj(payload?.latest) ? payload.latest : {};
state.previous = isObj(payload?.previous) ? payload.previous : {};

const liveConnected =
  payload?.live_connected === true &&
  Object.keys(state.latest).length > 0;

setLivePill(liveConnected);

    state.latestMeta = extractRunMeta(state.latest);
    state.prevMeta = extractRunMeta(state.previous);

    state.latestDecision = extractDecision(state.latest);
    state.latestTopFeatures = extractTopDriftedFeatures(state.latest);

    if (runIdText) runIdText.textContent = state.latestMeta.run_id || "—";
    if (latestIdText) latestIdText.textContent = state.latestMeta.run_id || "—";
    if (prevIdText) prevIdText.textContent = state.prevMeta.run_id || "—";
    if (latestGeneratedText) latestGeneratedText.textContent = state.latestMeta.generated_at || "—";
    if (prevGeneratedText) prevGeneratedText.textContent = state.prevMeta.generated_at || "—";
    setStatusBadge(state.latestMeta.status || "—");

    const lm = extractMetrics(state.latest);
    const pm = extractMetrics(state.previous);

    if (latestCleanHealth) latestCleanHealth.textContent = fmt(lm.clean_health, 2);
    if (latestDriftedHealth) latestDriftedHealth.textContent = fmt(lm.drifted_health, 2);
    if (latestPredKs) latestPredKs.textContent = fmt(lm.pred_ks, 4);
    if (latestDeltaEntropy) latestDeltaEntropy.textContent = fmt(lm.delta_entropy, 4);

    if (latestMostDrifted) {
      if (lm.feat) {
        const feature = lm.feat.feature || lm.feat.name || "—";
        const ks = isNum(lm.feat.ks) ? lm.feat.ks.toFixed(4) : "—";
        const ws = lm.feat.window_size ?? lm.feat.window ?? lm.feat.n ?? state.latest?.window_size ?? "—";
        latestMostDrifted.textContent = `MOST DRIFTED FEATURE (LAST WINDOW): ${String(feature).toUpperCase()} // KS: ${ks} // WINDOW_SIZE: ${ws}`;
      } else {
        latestMostDrifted.textContent = "MOST DRIFTED FEATURE (LAST WINDOW): —";
      }
    }

    if (prevCleanHealth) prevCleanHealth.textContent = fmt(pm.clean_health, 2);
    if (prevDriftedHealth) prevDriftedHealth.textContent = fmt(pm.drifted_health, 2);
    if (prevPredKs) prevPredKs.textContent = fmt(pm.pred_ks, 4);
    if (prevDeltaEntropy) prevDeltaEntropy.textContent = fmt(pm.delta_entropy, 4);

    state.latestSeries = extractSeries(state.latest);
    state.prevSeries = extractSeries(state.previous);

    const latestStatus = safeUpper(state.latestMeta.status);
    if (latestChartNote) {
      if ((latestStatus.includes("CRITICAL") || latestStatus.includes("WARN")) && isNum(lm.pred_ks)) {
        latestChartNote.textContent = `! DRIFT DETECTED // PRED_KS=${lm.pred_ks.toFixed(4)}`;
      } else if (isNum(lm.pred_ks)) {
        latestChartNote.textContent = `OK // PRED_KS=${lm.pred_ks.toFixed(4)}`;
      } else {
        latestChartNote.textContent = "";
      }
    }

    if (prevChartNote) {
      prevChartNote.textContent = `Prev Run: ${state.prevMeta.run_id || "—"} • Generated: ${state.prevMeta.generated_at || "—"}`;
    }

    // Analysis JSON (proof)
    if (analysisJson) {
      analysisJson.textContent = JSON.stringify({ latest: state.latest, previous: state.previous }, null, 2);
    }

    // Alerts (now can include taxonomy/severity if present)
    if (alertBox) {
      const decision = state.latestDecision;
      const decisionPrefix = decision
        ? `${decision.severity ? `${safeUpper(decision.severity)} • ` : ""}${decision.taxonomy ? `${safeUpper(decision.taxonomy)} • ` : ""}`
        : "";

      if (latestStatus.includes("CRITICAL")) {
        const feature = lm.feat ? lm.feat.feature || lm.feat.name || "—" : "—";
        const ksFeat = lm.feat && isNum(lm.feat.ks) ? lm.feat.ks.toFixed(4) : "—";
        alertBox.textContent = `CRITICAL: ${decisionPrefix}PREDICTION DRIFT HIGH (PRED_KS=${fmt(lm.pred_ks, 4)}). FEATURE=${String(
          feature
        ).toLowerCase()} (KS=${ksFeat}). ΔENTROPY=${fmt(lm.delta_entropy, 4)}. INVESTIGATE IMMEDIATELY.`;
      } else if (latestStatus.includes("WARN")) {
        alertBox.textContent = `WARNING: ${decisionPrefix}DRIFT DETECTED (PRED_KS=${fmt(lm.pred_ks, 4)}). Monitor closely.`;
      } else {
        alertBox.textContent = decisionPrefix ? `OK: ${decisionPrefix}No critical drift detected.` : "OK: No critical drift detected.";
      }
    }

    fillLedger(state.latest, state.previous);

    state.dirtyEval = true;
    state.dirtyCharts = true;
    state.dirtyInsights = true;

    // If run changed, refresh history
    if (state.latestMeta.run_id && state.latestMeta.run_id !== state.lastFetchRunId) {
      state.lastFetchRunId = state.latestMeta.run_id;
      state.playhead = 0;
      fetchHistory();
    }
  }

  // -----------------------------
  // Fetch (abort + timeout)
  // -----------------------------
  let resultsAbort = null;
  let historyAbort = null;

  function abortIfActive(ctrl) {
    try {
      if (ctrl && !ctrl.signal.aborted) ctrl.abort();
    } catch {
      // ignore
    }
  }

  async function fetchJson(url, kind = "results") {
    if (kind === "results") abortIfActive(resultsAbort);
    if (kind === "history") abortIfActive(historyAbort);

    const controller = new AbortController();
    if (kind === "results") resultsAbort = controller;
    if (kind === "history") historyAbort = controller;

    const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

    try {
      const res = await fetch(url, {
        cache: "no-store",
        signal: controller.signal,
        headers: { 
            "Accept": "application/json",
            "X-API-Key": apiKey 
        },
      });

      // Safely handle rejection without an infinite loop
      // Safely handle rejection without an infinite loop
      if (res.status === 401 || res.status === 403) {
          console.warn("API Key rejected by server.");
          const alertBox = document.getElementById("alertBox");
          if (alertBox) {
              alertBox.innerHTML = `⚠️ Authentication Error: Your session is invalid or missing. <a href="/login" style="color:#ffcc00; text-decoration:underline;">Click here to log in</a>.`;
              alertBox.style.borderColor = "rgba(209,31,31,0.55)";
              alertBox.style.background = "rgba(209,31,31,0.12)";
          }
          stopFetchLoop(); // Stop hammering the server
          return null; // Return null to prevent crashes, but DO NOT redirect
      }

      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status}${txt ? ` - ${txt.slice(0, 200)}` : ""}`);
      }

      return await res.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

async function fetchResults() {
  try {
    const data = await fetchJson(apiUrl(`/api/results?t=${Date.now()}`), "results");
    
    // Stop processing if auth failed
    if (!data) return; 

    // ✅ ADD THIS BLOCK HERE
    if (!data.latest || Object.keys(data.latest).length === 0) {
      const exportBtn = document.getElementById("exportBtn");
      if (exportBtn) {
        exportBtn.disabled = true;
        exportBtn.innerText = "No data to export";
      }
      return; 
    }

    // ✅ only runs if data exists
    updateFromData(data);

    // Restore Analysis JSON if needed
    if (analysisJson && /failed to fetch|http\s\d+/i.test(analysisJson.textContent || "")) {
      analysisJson.textContent = JSON.stringify(
        { latest: state.latest, previous: state.previous },
        null,
        2
      );
    }

  } catch (e) {
    // Ignore aborts (normal during polling overlap)
    if (e?.name === "AbortError") return;

    setLivePill(false);

    if (alertBox) {
      alertBox.textContent =
        "WARNING: /api/results fetch failed. Showing last known values.";
    }

    const hasData = state.latest && Object.keys(state.latest).length > 0;

    if (analysisJson && !hasData) {
      analysisJson.textContent =
        `Fetch error: ${String(e)}\n\n` +
        `If you opened the HTML directly, run it from the backend server so /api/results is available.\n` +
        `You can also set window.MODELSHIFT_API_BASE if frontend and backend are on different ports.`;
    }
  }
}

  async function fetchHistory() {
    try {
      const data = await fetchJson(apiUrl(`/api/history?n=50&t=${Date.now()}`), "history");
      state.serverHistory = Array.isArray(data?.runs) ? data.runs : [];
      state.dirtyHistory = true;
    } catch (e) {
      if (e?.name === "AbortError") return;
      state.serverHistory = state.serverHistory || [];
      state.dirtyHistory = true;
    }
  }

  // -----------------------------
  // Animation loop (dirty rendering)
  // -----------------------------
  let lastT = performance.now();
  let fpsAccT = 0;
  let fpsFrames = 0;

  function computeChartKey() {
    const l1 = state.latestSeries.clean.length;
    const l2 = state.latestSeries.drifted.length;
    const p1 = state.prevSeries.clean.length;
    const p2 = state.prevSeries.drifted.length;
    return [
      state.latestMeta.run_id,
      state.prevMeta.run_id,
      l1,
      l2,
      p1,
      p2,
      state.zoom,
      state.speed,
      state.syncAxes ? 1 : 0,
      state.loop ? 1 : 0,
      Math.floor(state.playhead * 10) / 10,
      document.body.getAttribute("data-page"),
    ].join("|");
  }

  function tick(now) {
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;

    fpsAccT += dt;
    fpsFrames += 1;
    if (fpsAccT >= 1.0) {
      state.fps = Math.round(fpsFrames / fpsAccT);
      if (fpsText) fpsText.textContent = `${state.fps} FPS`;
      fpsAccT = 0;
      fpsFrames = 0;
    }

    const onDash = document.body.getAttribute("data-page") === "dash";

    if (onDash && state.playing) {
      const total = Math.max(state.latestSeries.clean.length, state.latestSeries.drifted.length);
      if (total >= 2) {
        state.playhead += pointsPerSecond() * dt;
        if (state.playhead >= total - 1) {
          if (state.loop) state.playhead = 0;
          else state.playing = false;
        }
        state.dirtyCharts = true;
      }
    }

    if (state.dirtyHistory) {
      renderHistory();
      state.dirtyHistory = false;
    }

    if (state.dirtyInsights) {
      renderInsights(state.latest);
      state.dirtyInsights = false;
    }

    if (state.dirtyEval) {
      renderEvaluation(state.latest, state.previous);
      state.dirtyEval = false;
    }

    if (onDash && chartLatest && chartPrevious) {
      const key = computeChartKey();
      if (state.dirtyCharts || key !== state.lastChartKey) {
        state.lastChartKey = key;
        state.dirtyCharts = false;

        const latestTotal = Math.max(state.latestSeries.clean.length, state.latestSeries.drifted.length);
        const prevTotal = Math.max(state.prevSeries.clean.length, state.prevSeries.drifted.length);

        const centerLatest = latestTotal ? Math.min(latestTotal - 1, Math.max(0, state.playhead)) : 0;

        let centerPrev = centerLatest;
        if (state.syncAxes && latestTotal >= 2 && prevTotal >= 2) {
          const ratio = centerLatest / (latestTotal - 1);
          centerPrev = ratio * (prevTotal - 1);
        } else {
          centerPrev = Math.min(Math.max(0, prevTotal - 1), Math.max(0, state.playhead));
        }

        drawChart(chartLatest, state.latestSeries.clean, state.latestSeries.drifted, { centerIndex: centerLatest });
        drawChart(chartPrevious, state.prevSeries.clean, state.prevSeries.drifted, { centerIndex: centerPrev });
      }
    }

    if (playBtn) playBtn.textContent = state.playing ? "PAUSE" : "PLAY";
    requestAnimationFrame(tick);
  }

  // -----------------------------
  // Export Report (server preferred, client fallback)
  // -----------------------------
  function downloadBlob(filename, blob) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1500);
  }

  function buildRootCauseHtmlForReport(run) {
    const rows = extractTopDriftedFeatures(run).slice(0, 10);
    if (!rows.length) {
      return `<div class="ms-muted">No feature-level drift list found in payload.</div>`;
    }

    const trs = rows
      .map((r, i) => {
        const sev = r.severity || (r.ks_statistic >= 0.35 ? "CRITICAL" : r.ks_statistic >= 0.2 ? "HIGH" : r.ks_statistic >= 0.1 ? "MEDIUM" : "LOW");
        return `
          <tr>
            <td>${i + 1}</td>
            <td>${escapeHtml(String(r.feature).toUpperCase())}</td>
            <td>${fmt(r.ks_statistic, 4)}</td>
            <td>${isNum(r.p_value) ? r.p_value.toExponential(2) : "—"}</td>
            <td class="${severityClass(sev)}">${escapeHtml(sev)}</td>
          </tr>
        `;
      })
      .join("");

    return `
      <div class="ms-rc-table-wrap">
        <table class="ms-rc-table">
          <thead>
            <tr><th>#</th><th>FEATURE</th><th>KS</th><th>P_VALUE</th><th>SEVERITY</th></tr>
          </thead>
          <tbody>${trs}</tbody>
        </table>
      </div>
    `;
  }

  function buildDecisionHtmlForReport(run) {
    const decision = extractDecision(run);
    if (!decision) return `<div class="ms-muted">No composite decision block found in payload.</div>`;

    const s = decision.signals || {};
    const chips = [
      decision.status ? `<span class="ms-chip ${severityClass(decision.status)}">STATUS: ${escapeHtml(decision.status)}</span>` : "",
      decision.severity ? `<span class="ms-chip ${severityClass(decision.severity)}">SEVERITY: ${escapeHtml(decision.severity)}</span>` : "",
      decision.taxonomy ? `<span class="ms-chip">TAXONOMY: ${escapeHtml(decision.taxonomy)}</span>` : "",
    ].filter(Boolean).join("");

    return `
      <div class="ms-chip-row">${chips}</div>
      <div class="ms-signal-grid">
        <div class="ms-signal"><div class="k">AVG_FEATURE_KS</div><div class="v">${fmt(s.avg_feature_ks, 4)}</div></div>
        <div class="ms-signal"><div class="k">MAX_FEATURE_KS</div><div class="v">${fmt(s.max_feature_ks, 4)}</div></div>
        <div class="ms-signal"><div class="k">PREDICTION_KS</div><div class="v">${fmt(s.prediction_ks, 4)}</div></div>
        <div class="ms-signal"><div class="k">COMPOSITE_SCORE</div><div class="v">${fmt(s.composite_score, 4)}</div></div>
      </div>
      <div class="ms-muted" style="margin-top:8px;">
        ${s.max_feature_name ? `Top signal feature: ${escapeHtml(s.max_feature_name)} • ` : ""}
        Feature count: ${isNum(s.feature_count) ? String(s.feature_count) : "—"}
      </div>
    `;
  }

  function buildReportHtmlClient() {
    ensureEvalCss();
    ensureInsightCss();

    const latestMeta = state.latestMeta;
    const prevMeta = state.prevMeta;
    const lm = extractMetrics(state.latest);
    const pm = extractMetrics(state.previous);

    // Try to ensure charts are drawn before export
    if (chartLatest && chartPrevious) {
      try {
        const latestTotal = Math.max(state.latestSeries.clean.length, state.latestSeries.drifted.length);
        const prevTotal = Math.max(state.prevSeries.clean.length, state.prevSeries.drifted.length);
        drawChart(chartLatest, state.latestSeries.clean, state.latestSeries.drifted, {
          centerIndex: latestTotal ? Math.min(latestTotal - 1, Math.max(0, state.playhead)) : 0,
        });
        drawChart(chartPrevious, state.prevSeries.clean, state.prevSeries.drifted, {
          centerIndex: prevTotal ? Math.min(prevTotal - 1, Math.max(0, state.playhead)) : 0,
        });
      } catch {
        // no-op
      }
    }

    const latestImg = chartLatest?.toDataURL("image/png", 1.0) || "";
    const prevImg = chartPrevious?.toDataURL("image/png", 1.0) || "";

    const historyRows = (Array.isArray(state.serverHistory) ? state.serverHistory : [])
      .slice(0, 25)
      .map(
        (r) => `
      <tr>
        <td>${escapeHtml(r.saved_at || "—")}</td>
        <td>${escapeHtml(r.generated_at || "—")}</td>
        <td>${escapeHtml(r.run_id || "—")}</td>
        <td>${escapeHtml(r.status || "—")}</td>
        <td>${isNum(r.clean_health) ? r.clean_health.toFixed(2) : "—"}</td>
        <td>${isNum(r.drifted_health) ? r.drifted_health.toFixed(2) : "—"}</td>
        <td>${isNum(r.drifted_pred_ks) ? r.drifted_pred_ks.toFixed(4) : "—"}</td>
        <td>${isNum(r.drifted_entropy_change) ? r.drifted_entropy_change.toFixed(4) : "—"}</td>
      </tr>
    `
      )
      .join("");

    const mostDriftedLine = escapeHtml(latestMostDrifted?.textContent || "");
    const jsonProof = escapeHtml(JSON.stringify({ latest: state.latest, previous: state.previous }, null, 2));

    const evL = extractEvaluation(state.latest);
    const evP = extractEvaluation(state.previous);

    const evalL = evalSectionHtml("EVALUATION METRICS — LATEST RUN", evL);
    const evalP = evalSectionHtml("EVALUATION METRICS — PREVIOUS RUN", evP);

    const decisionHtml = buildDecisionHtmlForReport(state.latest);
    const rootCauseHtml = buildRootCauseHtmlForReport(state.latest);

    return `<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>ModelShift-Lite Report - ${escapeHtml(latestMeta.run_id)}</title>
  <style>
    :root{ --bg:#0b0c0d; --line:rgba(255,255,255,0.10); --text:#e9eef2; --muted:#9aa4ad; --red:#d11f1f; }
    *{box-sizing:border-box}
    body{ margin:0; background:var(--bg); color:var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; padding:26px; }
    .header{ display:flex; justify-content:space-between; align-items:center; border:1px solid var(--line); background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding:16px 18px; }
    .brand{ font-family: ui-monospace, Menlo, Consolas, monospace; font-weight:900; letter-spacing:2px; display:flex; gap:10px; align-items:center; }
    .tag{ border:1px solid var(--line); padding:6px 10px; font-family: ui-monospace, Menlo, Consolas, monospace; font-weight:900; }
    .grid{ display:grid; grid-template-columns:1fr; gap:14px; margin-top:14px; max-width:1200px; }
    .card{ border:1px solid var(--line); background:rgba(255,255,255,0.02); padding:14px; }
    .title{ font-family: ui-monospace, Menlo, Consolas, monospace; font-weight:900; letter-spacing:1px; opacity:0.92; margin-bottom:10px; }
    .meta{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12.5px; line-height:1.4; color:var(--muted); }
    .kpis{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:12px; }
    .kpi{ border:1px solid var(--line); padding:12px; background:rgba(0,0,0,0.18); }
    .kpi .k{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12px; letter-spacing:1px; opacity:0.85; }
    .kpi .v{ font-family: ui-monospace, Menlo, Consolas, monospace; font-weight:900; font-size:44px; margin-top:6px; }
    .warn{ box-shadow: inset 0 0 0 1px rgba(209,31,31,0.35); }
    img{ width:100%; border:1px solid var(--line); background:#0a0b0c; }
    table{ width:100%; border-collapse:collapse; min-width:900px; }
    th,td{ border-bottom:1px solid rgba(255,255,255,0.08); padding:10px 12px; font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12.5px; text-align:left; }
    th{ opacity:0.85; }
    .bad{ color:var(--red); font-weight:900; }
    pre{ margin:0; white-space:pre; overflow:auto; max-height:520px; border:1px solid var(--line); padding:12px; background:rgba(0,0,0,0.22); font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12px; line-height:1.35; }
    .foot{ color:var(--muted); font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12px; margin-top:12px; }
    @media (max-width: 1000px){ .kpis{grid-template-columns:repeat(2,minmax(0,1fr));} }

    .evalwrap{
      margin-top:12px;
      border:1px solid rgba(255,255,255,0.10);
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.06));
      padding:12px;
    }
    .evalhead{ display:flex; justify-content:space-between; align-items:flex-end; gap:10px; margin-bottom:10px; }
    .evaltitle{ font-family: ui-monospace, Menlo, Consolas, monospace; font-weight:900; letter-spacing:1px; opacity:0.92; }
    .evalnote{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12px; opacity:0.65; text-align:right; }
    .evaltablewrap{ overflow:auto; border:1px solid rgba(255,255,255,0.08); }
    table.evaltable{ width:100%; border-collapse:collapse; min-width:680px; }
    table.evaltable th, table.evaltable td{ padding:10px 12px; border-bottom:1px solid rgba(255,255,255,0.08); font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12.5px; text-align:left; }
    table.evaltable th{ opacity:0.85; letter-spacing:0.8px; }
    .hint{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12px; opacity:0.70; }

    .ms-insight-wrap{
      margin-top:12px;
      border:1px solid rgba(255,255,255,0.10);
      background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.06));
      padding:12px;
    }
    .ms-insight-title{
      font-family:ui-monospace, Menlo, Consolas, monospace;
      font-weight:900;
      letter-spacing:1px;
      opacity:0.92;
      margin-bottom:10px;
    }
    .ms-chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }
    .ms-chip{
      border:1px solid rgba(255,255,255,0.12);
      padding:6px 8px;
      font-family:ui-monospace, Menlo, Consolas, monospace;
      font-size:12px;
      line-height:1;
      background:rgba(0,0,0,0.20);
    }
    .ms-chip.bad{ border-color:rgba(209,31,31,0.45); background:rgba(209,31,31,0.10); color:rgba(255,220,220,0.95); }
    .ms-chip.warn{ border-color:rgba(255,170,0,0.45); background:rgba(255,170,0,0.08); }
    .ms-chip.good{ border-color:rgba(80,200,120,0.40); background:rgba(80,200,120,0.06); }
    .ms-signal-grid{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }
    .ms-signal{ border:1px solid rgba(255,255,255,0.08); padding:8px; background:rgba(0,0,0,0.14); }
    .ms-signal .k{ font-family:ui-monospace, Menlo, Consolas, monospace; font-size:11px; opacity:0.75; letter-spacing:.7px; }
    .ms-signal .v{ font-family:ui-monospace, Menlo, Consolas, monospace; font-size:16px; font-weight:800; margin-top:3px; }
    .ms-rc-table-wrap{ overflow:auto; border:1px solid rgba(255,255,255,0.08); }
    table.ms-rc-table{ width:100%; border-collapse:collapse; min-width:600px; }
    table.ms-rc-table th, table.ms-rc-table td{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,0.08); font-family:ui-monospace, Menlo, Consolas, monospace; font-size:12px; text-align:left; }
    table.ms-rc-table th{ opacity:0.85; letter-spacing:.8px; }
    .ms-muted{ font-family:ui-monospace, Menlo, Consolas, monospace; font-size:12px; opacity:0.68; }
  </style>
</head>
<body>
  <div class="header">
    <div class="brand">▸ MODELSHIFT-LITE <span style="opacity:.65;">REPORT</span></div>
    <div class="tag">RUN_ID: ${escapeHtml(latestMeta.run_id)} • ${escapeHtml(nowIsoLocal())}</div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="title">RUN META</div>
      <div class="meta">
        <div><b style="color:var(--text);">STATUS:</b> <span class="${safeUpper(latestMeta.status).includes("CRITICAL") ? "bad" : ""}">${escapeHtml(latestMeta.status)}</span></div>
        <div>Latest Generated: ${escapeHtml(latestMeta.generated_at)} • Previous Generated: ${escapeHtml(prevMeta.generated_at)}</div>
        <div>Latest Run: ${escapeHtml(latestMeta.run_id)} • Previous Run: ${escapeHtml(prevMeta.run_id)}</div>
      </div>
    </div>

    <div class="card">
      <div class="title">LATEST RUN • DASHBOARD SNAPSHOT</div>
      ${latestImg ? `<img src="${latestImg}" alt="Latest chart"/>` : `<div class="meta">Chart image not available</div>`}
      <div class="kpis">
        <div class="kpi"><div class="k">CLEAN HEALTH</div><div class="v">${fmt(lm.clean_health,2)}</div></div>
        <div class="kpi warn"><div class="k">DRIFTED HEALTH</div><div class="v">${fmt(lm.drifted_health,2)}</div></div>
        <div class="kpi"><div class="k">PRED_KS (DRIFT)</div><div class="v">${fmt(lm.pred_ks,4)}</div></div>
        <div class="kpi"><div class="k">Δ ENTROPY</div><div class="v">${fmt(lm.delta_entropy,4)}</div></div>
      </div>
      <div class="meta" style="margin-top:10px;"><b style="color:var(--text);">Most Drifted:</b> ${mostDriftedLine}</div>

      <div class="ms-insight-wrap">
        <div class="ms-insight-title">DECISION ENGINE</div>
        ${decisionHtml}
      </div>

      <div class="ms-insight-wrap">
        <div class="ms-insight-title">ROOT CAUSE CANDIDATES (TOP DRIFTED FEATURES)</div>
        ${rootCauseHtml}
      </div>

      <div style="margin-top:12px;">${evalL}</div>
    </div>

    <div class="card">
      <div class="title">PREVIOUS RUN • SNAPSHOT</div>
      ${prevImg ? `<img src="${prevImg}" alt="Previous chart"/>` : `<div class="meta">Chart image not available</div>`}
      <div class="kpis">
        <div class="kpi"><div class="k">CLEAN HEALTH</div><div class="v">${fmt(pm.clean_health,2)}</div></div>
        <div class="kpi"><div class="k">DRIFTED HEALTH</div><div class="v">${fmt(pm.drifted_health,2)}</div></div>
        <div class="kpi"><div class="k">PRED_KS</div><div class="v">${fmt(pm.pred_ks,4)}</div></div>
        <div class="kpi"><div class="k">Δ ENTROPY</div><div class="v">${fmt(pm.delta_entropy,4)}</div></div>
      </div>
      <div style="margin-top:12px;">${evalP}</div>
    </div>

    <div class="card">
      <div class="title">HISTORY (SAVED RUNS)</div>
      <div style="overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>SAVED_AT</th><th>GENERATED_AT</th><th>RUN_ID</th><th>STATUS</th>
              <th>CLEAN</th><th>DRIFTED</th><th>PRED_KS</th><th>ΔENTROPY</th>
            </tr>
          </thead>
          <tbody>${historyRows || ""}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="title">INPUTS (PROOF JSON)</div>
      <pre>${jsonProof}</pre>
      <div class="foot">Generated by ModelShift-Lite UI export.</div>
    </div>
  </div>
</body>
</html>`;
  }

  async function exportReport() {
    const runId = state.latestMeta?.run_id || "UNKNOWN_RUN";
    
    if (runId === "—" || runId === "UNKNOWN_RUN") {
        alert("No data available to export! Please run the Python pipeline first.");
        return;
    }

    try {
      const res = await fetch(apiUrl(`/api/report/latest?download=1&t=${Date.now()}`), { cache: "no-store" });
      if (res.ok) {
        const blob = await res.blob();
        downloadBlob(`ModelShift-Lite_Report_${runId}.html`, blob);
        return;
      }
    } catch {
      // fall back to client export
    }

    const html = buildReportHtmlClient();
    downloadBlob(`ModelShift-Lite_Report_${runId}.html`, new Blob([html], { type: "text/html" }));
  }

  // -----------------------------
  // Controls
  // -----------------------------
  function setZoom(v) {
    state.zoom = clampInt(v, 1, 10);
    if (zoomSlider) zoomSlider.value = String(state.zoom);
    if (zoomValue) zoomValue.textContent = String(state.zoom);
    state.dirtyCharts = true;
  }

  function setSpeed(v) {
    state.speed = clampInt(v, 1, 12);
    if (speedSlider) speedSlider.value = String(state.speed);
    if (speedValue) speedValue.textContent = String(state.speed);
  }

  function applyTheme() {
    const theme = localStorage.getItem("ms_theme") || "dark";
    // This makes the CSS you added work perfectly!
    if (theme === "light") {
      document.body.classList.add("light-theme");
    } else {
      document.body.classList.remove("light-theme");
    }
  }

  // -----------------------------
  // Background-aware fetch interval
  // -----------------------------
  let fetchTimer = null;

  function startFetchLoop() {
    stopFetchLoop();
    const ms = document.visibilityState === "hidden" ? FETCH_INTERVAL_BG_MS : FETCH_INTERVAL_MS;
    fetchTimer = setInterval(fetchResults, ms);
  }

  function stopFetchLoop() {
    if (fetchTimer) clearInterval(fetchTimer);
    fetchTimer = null;
  }

  // -----------------------------
  // Init
  // -----------------------------
  function init() {
    // nav clicks
    // nav clicks
    document.querySelectorAll(".navbtn").forEach((b) => {
      b.addEventListener("click", () => setPage(b.dataset.page));
    }); // <-- MOVED THE BRACKET HERE!
    
    selfTestRunBtn?.addEventListener("click", () => fetchSelfTest(true));
    fetchSelfTest(false);

    // controls
    zoomSlider?.addEventListener("input", () => setZoom(zoomSlider.value));
    zoomMinus?.addEventListener("click", () => setZoom(state.zoom - 1));
    zoomPlus?.addEventListener("click", () => setZoom(state.zoom + 1));

    speedSlider?.addEventListener("input", () => setSpeed(speedSlider.value));
    speedMinus?.addEventListener("click", () => setSpeed(state.speed - 1));
    speedPlus?.addEventListener("click", () => setSpeed(state.speed + 1));

    playBtn?.addEventListener("click", () => {
      state.playing = !state.playing;
      state.dirtyCharts = true;
    });

    resetBtn?.addEventListener("click", () => {
      state.playhead = 0;
      state.playing = false;
      state.dirtyCharts = true;
    });

    syncAxesChk?.addEventListener("change", () => {
      state.syncAxes = !!syncAxesChk.checked;
      state.dirtyCharts = true;
    });

    loopChk?.addEventListener("change", () => {
      state.loop = !!loopChk.checked;
    });

    // theme toggle (FIXED: single listener only)
    themeBtn?.addEventListener("click", () => {
      const current = localStorage.getItem("ms_theme") || "dark";
      const next = current === "dark" ? "light" : "dark";
      localStorage.setItem("ms_theme", next);
      applyTheme();
      state.dirtyCharts = true;
    });

    exportBtn?.addEventListener("click", exportReport);

    // History button behavior changed:
    // DO NOT delete archived history. Just refresh the History table from server.
clearHistoryBtn?.addEventListener("click", async () => {
  const ok = window.confirm("Clear archived run history? (Latest run will be kept)");
  if (!ok) return;

  try {
    const res = await fetch(apiUrl(`/api/history/clear?t=${Date.now()}`), {
      method: "POST",
      cache: "no-store",
      headers: { 
          "Accept": "application/json",
          "X-API-Key": apiKey || "" 
      },
    });

    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}${txt ? ` - ${txt.slice(0, 200)}` : ""}`);
    }

    const out = await res.json().catch(() => ({}));

    // Refresh views after clear
    await Promise.allSettled([fetchHistory(), fetchResults()]);
    state.dirtyHistory = true;
    state.dirtyEval = true;
    state.dirtyInsights = true;
    state.dirtyCharts = true;

    if (alertBox) {
      alertBox.textContent = out?.ok
        ? "INFO: Archived history cleared."
        : `WARNING: History cleared with issues (${(out?.errors || []).length || 0}).`;
    }
  } catch (e) {
    if (alertBox) {
      alertBox.textContent = `WARNING: Failed to clear history (${String(e)})`;
    }
  }
});

    // initial state
    setZoom(state.zoom);
    setSpeed(state.speed);
    state.syncAxes = !!syncAxesChk?.checked;
    state.loop = !!loopChk?.checked;

    applyTheme();

    // Ensure a valid page is selected on load
    const initialPage =
      document.body.getAttribute("data-page") ||
      document.querySelector(".navbtn.active")?.dataset.page ||
      "dash";
    setPage(initialPage);

    // initial analysis text (until first fetch)
    if (analysisJson && !analysisJson.textContent.trim()) {
      analysisJson.textContent = "Waiting for /api/results ...";
    }

    // clock
    updateClock();
    setInterval(updateClock, 1000);

    // live pill blink
    setLivePill(false);

    // resize => redraw charts once
    window.addEventListener("resize", () => {
      state.dirtyCharts = true;
    });

    // tab visibility => slower fetch in background
    document.addEventListener("visibilitychange", startFetchLoop);

    // initial fetches
    fetchResults();
    fetchHistory();

    // fetch loop
    startFetchLoop();

    // animation loop
    requestAnimationFrame(tick);

    // cleanup on unload
    window.addEventListener("beforeunload", () => {
      stopFetchLoop();
      abortIfActive(resultsAbort);
      abortIfActive(historyAbort);
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
function renderSelfTest(data) {
  if (!selfTestJson) return;
  selfTestJson.textContent = JSON.stringify(data || {}, null, 2);
}

async function fetchSelfTest(run = false) {
  try {
    const url = run ? apiUrl(`/api/selftest/run?t=${Date.now()}`) : apiUrl(`/api/selftest?t=${Date.now()}`);
    const res = await fetch(url, { 
        method: run ? "POST" : "GET", 
        cache: "no-store",
        headers: {
            "Accept": "application/json",
            "X-API-Key": apiKey // Fix: Now it properly sends the key!
        }
    });
    const data = await res.json();
    renderSelfTest(data);

    if (alertBox && run) {
      alertBox.textContent = data?.ok ? "INFO: Self-test passed ✅" : "WARNING: Self-test failed ❌";
    }
  } catch (e) {
    renderSelfTest({ ok: false, message: `Self-test fetch failed: ${String(e)}` });
  }
}
function logoutUser() {
    // Clear the stored credentials
    localStorage.removeItem("api_key"); 
    localStorage.removeItem("email");
    // Kick them back to the login page
    window.location.href = "/login";
}


function toggleTheme() {
    document.body.classList.toggle("light-theme");
}