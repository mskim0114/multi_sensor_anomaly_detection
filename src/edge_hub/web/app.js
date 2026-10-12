"use strict";
// Edge hub monitoring page. All server data is inserted as text (never as HTML).

const $ = (id) => document.getElementById(id);
const state = { view: "overview", overview: null, timer: null, unit: null, role: "viewer", commands: {} };

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  const body = await response.json().catch(() => ({}));
  if (response.status === 401 && !path.endsWith("/login")) { showLogin(); throw new Error("login required"); }
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(child ?? ""));
  return node;
}

const fmtBytes = (n) => { if (n == null) return "—"; const u = ["B","KB","MB","GB","TB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; } return `${n.toFixed(i ? 1 : 0)} ${u[i]}`; };
const fmtAge = (s) => s == null ? "기록 없음" : s < 90 ? `${Math.round(s)}초 전` : s < 5400 ? `${Math.round(s/60)}분 전`
  : s < 172800 ? `${Math.round(s/3600)}시간 전` : `${Math.round(s/86400)}일 전`;
const fmtNum = (v) => typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toPrecision(4)) : String(v ?? "—");

// ------------------------------------------------------------------ login
function showLogin() {
  clearInterval(state.timer);
  for (const id of ["overview", "live", "units"]) $(id).hidden = true;
  $("nav").hidden = true; $("who").hidden = true; $("login").hidden = false;
}

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  $("login-error").textContent = "";
  try {
    const me = await api("/api/v1/login", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user: form.get("user"), password: form.get("password") }) });
    event.target.reset();
    start(me);
  } catch (error) { $("login-error").textContent = error.message; }
});

$("logout").addEventListener("click", async () => { await api("/api/v1/logout", { method: "POST" }).catch(() => {}); showLogin(); });

for (const button of document.querySelectorAll("nav button")) {
  button.addEventListener("click", () => show(button.dataset.view));
}

function start(me) {
  state.role = me.role || "viewer";
  $("login").hidden = true; $("nav").hidden = false; $("who").hidden = false;
  $("user").textContent = `${me.user}${state.role === "operator" ? " (운영자)" : ""}`;
  show(state.view);
}

function show(view) {
  state.view = view;
  for (const b of document.querySelectorAll("nav button")) b.classList.toggle("active", b.dataset.view === view);
  for (const id of ["overview", "live", "units"]) $(id).hidden = id !== view;
  clearInterval(state.timer);
  const tick = { overview: refreshOverview, live: refreshLive, units: refreshUnits }[view];
  tick();
  if (view !== "units") state.timer = setInterval(tick, 5000);
}

// --------------------------------------------------------------- overview
async function refreshOverview() {
  const data = await api("/api/v1/overview"); state.overview = data;
  await Promise.all(data.devices.filter((d) => d.registered).map(async (d) => {
    state.commands[d.device] = (await api(`/api/v1/commands?device=${encodeURIComponent(d.device)}`).catch(() => ({ commands: [] }))).commands;
  }));
  $("server-time").textContent = `서버 시각 ${data.server_utc} · 5초마다 갱신`;
  const cards = $("device-cards"); cards.replaceChildren();
  if (!data.devices.length) cards.append(el("p", { class: "muted" }, "등록된 장치가 없습니다. 서버에서 add-device 로 등록하세요."));
  for (const d of data.devices) cards.append(deviceCard(d));
  fillDeviceSelects(data.devices.map((d) => d.device));
}

function deviceCard(d) {
  const s = d.status || {};
  const up = s.uploads || {}; const by = up.by_state || {};
  const pending = ["pending", "uploading", "error"].reduce((n, k) => n + ((by[k] || {}).units || 0), 0);
  const disk = Object.values(s.disks || {}).find(Boolean);
  const collector = s.collector || {};
  const kv = el("dl", { class: "kv" },
    el("dt", {}, "마지막 신호"), el("dd", {}, `${fmtAge(d.age_s)} (${d.last_seen_utc || "—"})`),
    el("dt", {}, "접속 주소"), el("dd", {}, `${d.remote_addr || "—"} · 현장 IP ${s.local_ip || "—"}`),
    el("dt", {}, "수집기"), el("dd", {}, `${collector.status || "알 수 없음"} ${typeof collector.run_dir === "string" ? "· " + collector.run_dir.split("/").pop() : ""}`),
    el("dt", {}, "수신 완료"), el("dd", {}, `trial ${d.units.dataset.count}개 (${fmtBytes(d.units.dataset.bytes)}) · run ${d.units.runs.count}개 (${fmtBytes(d.units.runs.bytes)})`),
    el("dt", {}, "전송 대기/오류"), el("dd", {}, `${pending}개` + (s.current_upload ? ` · 전송 중 ${s.current_upload.unit} ${fmtBytes(s.current_upload.sent)}/${fmtBytes(s.current_upload.bytes)}` : "")),
    el("dt", {}, "마지막 오류"), el("dd", {}, up.last_error ? `${up.last_error.unit}: ${up.last_error.error}` : "없음"),
    el("dt", {}, "디스크 여유"), el("dd", {}, disk ? `${fmtBytes(disk.free)} / ${fmtBytes(disk.total)}` : "—"),
    el("dt", {}, "SoC 온도"), el("dd", {}, Object.entries(s.temperatures_c || {}).filter(([, v]) => typeof v === "number").slice(0, 4).map(([k, v]) => `${k} ${v.toFixed(1)}°C`).join(" · ") || "—"),
    el("dt", {}, "코드"), el("dd", {}, s.git ? `${String(s.git.commit || "").slice(0, 10)}${s.git.dirty ? " (dirty)" : ""}` : "—"));
  const values = d.latest_values || {};
  const pick = ["ntc.temperature_c", "ct1.current_a_nominal", "sps30.pm2_5_ug_m3", "scd30.co2_ppm", "flir.max_c"];
  const latest = el("p", { class: "muted" }, pick.filter((k) => k in values).map((k) => `${k} ${fmtNum(values[k])}`).join(" · ") || "실시간 값 없음 (수집기 미실행)");
  const status = !d.registered ? ["warn", "미등록"] : d.online ? ["ok", "온라인"] : ["bad", "오프라인"];
  return el("div", { class: "card" },
    el("h3", {}, el("span", { class: `dot ${status[0]}` }), `${d.device} `, el("span", { class: "badge" }, status[1])),
    kv, latest, controlBlock(d),
    el("button", { onclick: () => { $("live-device").value = d.device; show("live"); } }, "실시간 보기"));
}

const COMMAND_STATE = { pending: "전달 대기", delivered: "장치 실행 중", done: "완료", failed: "실패", expired: "만료(장치 미수신)", no_result: "결과 없음" };
function controlBlock(d) {
  const s = d.status || {}; const history = state.commands[d.device] || [];
  const open = history.find((c) => c.state === "pending" || c.state === "delivered");
  const box = el("div", { class: "control" });
  const allowed = s.remote_control === true;
  box.append(el("p", { class: "muted" }, `원격 수집 제어: ${allowed ? "장치에서 허용됨" : "장치에서 비활성 (uplink.json allow_remote_control)"}`));
  if (state.role === "operator" && d.registered) {
    const collecting = (s.collector || {}).status === "running";
    const send = async (action) => {
      const label = { start: "수집을 시작", stop: "수집을 정지", reset_sps30: "SPS30 센서를 리셋" }[action];
      if (!confirm(`${d.device} 에서 ${label}할까요? (자유 수집 run, 기록이 남습니다)`)) return;
      try {
        await api("/api/v1/control", { method: "POST", headers: { "Content-Type": "application/json", "X-Hub-Request": "1" },
          body: JSON.stringify({ device: d.device, action }) });
      } catch (error) { alert(error.message); }
      refreshOverview();
    };
    const startButton = el("button", { onclick: () => send("start") }, "수집 시작");
    const stopButton = el("button", { onclick: () => send("stop") }, "수집 정지");
    const resetButton = el("button", { onclick: () => send("reset_sps30"),
      title: "수집기가 멈춘 상태에서만. SPS30 에 device reset 을 보내고 응답·측정을 확인합니다" }, "SPS30 리셋");
    startButton.disabled = Boolean(open) || !d.online || !allowed || collecting;
    stopButton.disabled = Boolean(open) || !d.online || !allowed || !collecting;
    resetButton.disabled = Boolean(open) || !d.online || !allowed || collecting;
    box.append(el("div", { class: "toolbar" }, startButton, stopButton, resetButton));
  }
  const last = history[0];
  if (last) {
    const r = last.result || {};
    const sps = r.sps30 || null;
    const detail = sps ? `SPS30 ${sps.ok ? "정상 응답" + (sps.measurement ? ` (PM2.5 ${fmtNum(sps.measurement.pm2_5)})` : "") : "실패: " + String(sps.hint || sps.error || "")}`
      : String(r.error || (r.collector || {}).status || "");
    box.append(el("p", { class: "muted" }, `최근 명령: ${last.action} · ${COMMAND_STATE[last.state] || last.state} · ${last.requested_by} · ${(last.requested_utc || "").slice(0, 19)}${detail ? " · " + detail : ""}`));
  }
  return box;
}

function fillDeviceSelects(devices) {
  for (const [id, keepAll] of [["live-device", false], ["units-device", true]]) {
    const select = $(id); const current = select.value;
    const options = (keepAll ? [""] : []).concat(devices);
    if ([...select.options].map((o) => o.value).join() === options.join()) continue;
    select.replaceChildren(...options.map((d) => el("option", { value: d }, d || "전체")));
    if (options.includes(current)) select.value = current;
  }
}

// ------------------------------------------------------------------- live
const PALETTE = [[0,0,4],[59,15,112],[140,41,129],[222,73,104],[254,159,109],[252,253,191]];
function color(t) {
  t = Math.min(1, Math.max(0, t)) * (PALETTE.length - 1); const i = Math.min(PALETTE.length - 2, Math.floor(t)); const f = t - i;
  return PALETTE[i].map((c, k) => Math.round(c + (PALETTE[i + 1][k] - c) * f));
}

function drawThermal(canvas, b64, minId, maxId) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const raw = new Uint16Array(bytes.buffer, 0, 120 * 160);  // little-endian (all supported clients)
  let lo = Infinity, hi = -Infinity;
  for (const v of raw) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const ctx = canvas.getContext("2d"); const image = ctx.createImageData(160, 120);
  for (let i = 0; i < raw.length; i++) {
    const [r, g, b] = color(hi > lo ? (raw[i] - lo) / (hi - lo) : 0);
    image.data.set([r, g, b, 255], i * 4);
  }
  ctx.putImageData(image, 0, 0);
  $(minId).textContent = `${(lo / 100 - 273.15).toFixed(1)}°C`; $(maxId).textContent = `${(hi / 100 - 273.15).toFixed(1)}°C`;
}

async function refreshLive() {
  if (!state.overview) await refreshOverview().catch(() => {});
  const device = $("live-device").value; if (!device) return;
  const [live, trend] = await Promise.all([api(`/api/v1/live?device=${encodeURIComponent(device)}`),
    api(`/api/v1/telemetry?device=${encodeURIComponent(device)}&since_s=${$("live-range").value}`)]);
  const s = live.status || {}; const snap = (live.live || {}).snapshot || {};
  const stale = (live.live || {}).stale;
  $("live-age").textContent = s.received_utc ? `마지막 heartbeat ${s.received_utc} · 스냅샷 ${snap.timestamp_utc || "없음"}` : "heartbeat 없음";
  $("live-age").classList.toggle("error", Boolean(stale));
  if (stale) $("live-age").textContent += " · 수집기 미실행: 아래는 마지막으로 받은 값입니다";
  const thermal = (live.live || {}).thermal;
  if (thermal && thermal.data) { drawThermal($("thermal"), thermal.data, "t-min", "t-max");
    $("thermal-info").textContent = `seq ${thermal.snapshot_sequence}${live.live.thermal_reused ? " (이전 프레임)" : ""}`; }
  else $("thermal-info").textContent = "프레임 없음";
  const table = $("sensor-table"); table.replaceChildren(el("tr", {}, el("th", {}, "센서"), el("th", {}, "상태"), el("th", {}, "값")));
  for (const [name, rec] of Object.entries(snap.sensors || {})) {
    const values = Object.entries(rec.values || {}).filter(([, v]) => typeof v === "number").map(([k, v]) => `${k} ${fmtNum(v)}`).join(" · ");
    table.append(el("tr", {}, el("td", {}, name), el("td", {}, `${rec.status ?? "—"}${rec.fresh === false ? " (stale)" : ""}`), el("td", {}, values || "—")));
  }
  const keys = [...new Set(trend.points.flatMap((p) => Object.keys(p).filter((k) => k !== "t")))].sort();
  const select = $("trend-keys");
  if ([...select.options].map((o) => o.value).join() !== keys.join()) {
    const chosen = new Set([...select.selectedOptions].map((o) => o.value));
    if (!chosen.size) ["ntc.temperature_c", "flir.max_c"].forEach((k) => chosen.add(k));
    select.replaceChildren(...keys.map((k) => { const o = el("option", { value: k }, k); o.selected = chosen.has(k); return o; }));
  }
  const selected = [...select.selectedOptions].map((o) => o.value);
  drawChart($("trend"), trend.points.map((p) => p.t * 1000), Object.fromEntries(selected.map((k) => [k, trend.points.map((p) => p[k] ?? null)])), true);
}
$("live-device").addEventListener("change", refreshLive);
$("live-range").addEventListener("change", refreshLive);
$("trend-keys").addEventListener("change", refreshLive);

const LINE_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d"];
function drawChart(canvas, xs, series, timeAxis) {
  const ctx = canvas.getContext("2d"); const W = canvas.width, H = canvas.height, L = 60, R = 10, T = 10, B = 40;
  const css = getComputedStyle(document.body);
  ctx.clearRect(0, 0, W, H); ctx.font = "12px system-ui"; ctx.fillStyle = css.color; ctx.strokeStyle = "#8884";
  const names = Object.keys(series);
  if (!xs.length || !names.length) { ctx.fillText("데이터 없음 — 항목을 고르세요", L, H / 2); return; }
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || x0 + 1;
  names.forEach((name, n) => {
    const ys = series[name].filter((v) => v != null); if (!ys.length) return;
    // each series gets its own y range (units differ); range shown in the legend
    let lo = Math.min(...ys), hi = Math.max(...ys); if (hi === lo) { hi += 1; lo -= 1; }
    ctx.strokeStyle = LINE_COLORS[n % LINE_COLORS.length]; ctx.beginPath(); let pen = false;
    series[name].forEach((v, i) => {
      if (v == null) { pen = false; return; }
      const x = L + (xs[i] - x0) / (x1 - x0 || 1) * (W - L - R), y = T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true;
    });
    ctx.stroke(); ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(`${name} [${fmtNum(lo)} … ${fmtNum(hi)}]`, L + 4, T + 14 + 14 * n);
  });
  ctx.fillStyle = css.color;
  const label = (x) => timeAxis ? new Date(x).toLocaleTimeString() : String(Math.round(x));
  ctx.fillText(label(x0), L, H - 12); const right = label(x1); ctx.fillText(right, W - R - ctx.measureText(right).width, H - 12);
}

// ------------------------------------------------------------------ units
async function refreshUnits() {
  if (!state.overview) await refreshOverview().catch(() => {});
  const device = $("units-device").value;
  const data = await api(`/api/v1/units${device ? "?device=" + encodeURIComponent(device) : ""}`);
  $("units-count").textContent = `${data.units.length}개 · 클릭하면 상세`;
  const table = $("units-table");
  table.replaceChildren(el("tr", {}, ...["장치", "unit", "상태", "시나리오", "시작(UTC)", "파일", "크기", "유효 window", "수신(UTC)"].map((h) => el("th", {}, h))));
  for (const u of data.units) {
    const s = u.summary || {};
    table.append(el("tr", { class: "clickable", onclick: () => openUnit(u.device, u.unit) },
      el("td", {}, u.device), el("td", {}, u.unit), el("td", {}, `${s.status ?? "—"}${s.test_mode ? " · test" : ""}`),
      el("td", {}, s.scenario_id ?? "—"), el("td", {}, (s.started_utc || "—").slice(0, 19)), el("td", {}, String(u.files)),
      el("td", {}, fmtBytes(u.bytes)), el("td", {}, s.windows_evaluated != null ? `${s.windows_valid}/${s.windows_evaluated}` : "—"),
      el("td", {}, u.committed_utc)));
  }
}
$("units-device").addEventListener("change", refreshUnits);

async function openUnit(device, unit) {
  const d = await api(`/api/v1/unit?device=${encodeURIComponent(device)}&unit=${encodeURIComponent(unit)}`);
  state.unit = { device, unit };
  $("unit-detail").hidden = false; $("unit-title").textContent = `${device} · ${unit}`;
  const meta = $("unit-meta"); meta.replaceChildren();
  const r = d.receipt; const kv = el("dl", { class: "kv" });
  const rows = [["수신", r.committed_utc], ["파일/크기", `${r.files}개 · ${fmtBytes(r.bytes)}`], ["manifest sha256", r.manifest_sha256],
    ["서버 경로", r.raw_path], ...Object.entries(d.summary).map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v) : String(v)])];
  for (const [k, v] of rows) kv.append(el("dt", {}, k), el("dd", {}, v));
  meta.append(kv);
  const numeric = d.scalar_columns.filter((c) => !["timestamp_utc", "quality_invalid_reasons"].includes(c) && !c.endsWith("_status"));
  $("unit-cols").replaceChildren(...numeric.map((c) => { const o = el("option", { value: c }, c); o.selected = ["ntc_temperature_c", "flir_max_c"].includes(c); return o; }));
  $("unit-chunk").replaceChildren(...d.thermal_chunks.map((c) => el("option", { value: c }, c)));
  $("unit-detail").scrollIntoView({ behavior: "smooth" });
  plotUnit(); showFrame();
}

async function plotUnit() {
  if (!state.unit) return;
  const cols = [...$("unit-cols").selectedOptions].map((o) => o.value);
  const { device, unit } = state.unit;
  const d = await api(`/api/v1/unit/scalars?device=${encodeURIComponent(device)}&unit=${encodeURIComponent(unit)}&columns=${encodeURIComponent(cols.join(","))}`);
  drawChart($("unit-chart"), d.sequence, d.series, false);
}
$("unit-plot").addEventListener("click", plotUnit);

async function showFrame() {
  if (!state.unit || !$("unit-chunk").value) { $("unit-frame-info").textContent = "열화상 없음"; return; }
  const { device, unit } = state.unit;
  try {
    const f = await api(`/api/v1/unit/thermal?device=${encodeURIComponent(device)}&unit=${encodeURIComponent(unit)}&chunk=${encodeURIComponent($("unit-chunk").value)}&index=${$("unit-frame").value}`);
    $("unit-frame").max = String(f.frames_in_chunk - 1);
    drawThermal($("unit-thermal"), f.data, "u-min", "u-max");
    $("unit-frame-info").textContent = `sequence ${f.sequence} (${Number(f.index) + 1}/${f.frames_in_chunk})`;
  } catch (error) { $("unit-frame-info").textContent = error.message; }
}
$("unit-chunk").addEventListener("change", () => { $("unit-frame").value = "0"; showFrame(); });
$("unit-frame").addEventListener("input", showFrame);

api("/api/v1/me").then((me) => start(me)).catch(() => showLogin());
