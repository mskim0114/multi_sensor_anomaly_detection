"use strict";
// Edge hub monitoring page. All server data is inserted as text (never as HTML). Strings: i18n.js.

const $ = (id) => document.getElementById(id);
const state = { view: "overview", overview: null, timer: null, unit: null, role: "viewer", commands: {}, user: null };

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
const fmtAge = (s) => s == null ? t("age.none") : s < 90 ? t("age.s", { n: Math.round(s) }) : s < 5400 ? t("age.m", { n: Math.round(s / 60) })
  : s < 172800 ? t("age.h", { n: Math.round(s / 3600) }) : t("age.d", { n: Math.round(s / 86400) });
const fmtNum = (v) => typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toPrecision(4)) : String(v ?? "—");
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Hub-Request": "1" }, body: JSON.stringify(body) });

// ------------------------------------------------------------- language
for (const button of document.querySelectorAll(".lang button")) button.addEventListener("click", () => setLang(button.dataset.lang));
document.addEventListener("langchange", () => {
  if (state.user) { renderUser(); show(state.view); }
  if (state.unit && state.unitDetail) { renderUnitMeta(); renderAnnotation(); redrawUnit(); showZed(); }
});
applyStatic();

// ------------------------------------------------------------------ login
function showLogin() {
  clearInterval(state.timer); state.user = null;
  for (const id of ["overview", "live", "scenarios", "units"]) $(id).hidden = true;
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

function renderUser() { $("user").textContent = `${state.user}${state.role === "operator" ? ` (${t("operator")})` : ""}`; }

function start(me) {
  state.role = me.role || "viewer"; state.user = me.user;
  $("login").hidden = true; $("nav").hidden = false; $("who").hidden = false;
  renderUser();
  show(state.view);
}

function show(view) {
  state.view = view;
  for (const b of document.querySelectorAll("nav button")) b.classList.toggle("active", b.dataset.view === view);
  for (const id of ["overview", "live", "scenarios", "units"]) $(id).hidden = id !== view;
  clearInterval(state.timer);
  const refresh = { overview: refreshOverview, live: refreshLive, scenarios: refreshScenarios, units: refreshUnits }[view];
  refresh();
  if (view === "overview" || view === "live") state.timer = setInterval(refresh, 5000);
}

// --------------------------------------------------------------- overview
async function refreshOverview() {
  const data = await api("/api/v1/overview"); state.overview = data;
  await Promise.all(data.devices.filter((d) => d.registered).map(async (d) => {
    state.commands[d.device] = (await api(`/api/v1/commands?device=${encodeURIComponent(d.device)}`).catch(() => ({ commands: [] }))).commands;
  }));
  $("server-time").textContent = t("overview.server_time", { utc: data.server_utc });
  const b = data.backup;
  const age = b && b.finished_utc ? (Date.parse(data.server_utc) - Date.parse(b.finished_utc)) / 3600e3 : null;
  $("backup-status").textContent = !b ? t("backup.none") : !b.ok ? t("backup.failed", { utc: (b.finished_utc || "").slice(0, 16), error: b.error || "" })
    : t("backup.ok", { utc: (b.finished_utc || "").slice(0, 16), n: b.units ?? 0, size: fmtBytes(b.bytes) }) + (age > 36 ? t("backup.stale") : "");
  $("backup-status").classList.toggle("error", !b || !b.ok || age > 36);
  const w = data.weather;
  const wAge = w && w.finished_utc ? (Date.parse(data.server_utc) - Date.parse(w.finished_utc)) / 3600e3 : null;
  $("weather-status").textContent = !w ? t("weather.none") : !w.ok ? t("weather.failed", { utc: (w.finished_utc || "").slice(0, 16), error: (w.errors || [])[0] || "" })
    : t("weather.ok", { utc: (w.finished_utc || "").slice(0, 16), n: w.sites ?? 0 }) + (wAge > 3 ? t("weather.stale") : "");
  $("weather-status").classList.toggle("error", Boolean(w) && (!w.ok || wAge > 3));
  const cards = $("device-cards"); cards.replaceChildren();
  if (!data.devices.length) cards.append(el("p", { class: "muted" }, t("overview.no_devices")));
  for (const d of data.devices) cards.append(deviceCard(d));
  fillDeviceSelects(data.devices.map((d) => d.device));
}

function zedText(zed) {
  if (!zed) return "—";
  const parts = [zed.enabled ? t("zed.on") : t("zed.off")];
  if (!zed.recorder_running) parts.push(t("zed.no_recorder"));
  else if (zed.state) parts.push(t(`zed.state.${zed.state}`));
  if (zed.state === "recording") parts.push(`${zed.run_id || ""} · ${t("zed.frames", { n: zed.frames ?? 0 })}${zed.missed_ticks ? ` (missed ${zed.missed_ticks})` : ""}`);
  if (zed.error) parts.push(String(zed.error));
  if (zed.config_error) parts.push(String(zed.config_error));
  return parts.join(" · ");
}

function outdoorText(o) {
  if (!o) return t("site.no_weather");
  return `${fmtNum(o.temperature_c)} °C · ${fmtNum(o.humidity_pct)} % · ${fmtNum(o.pressure_hpa)} hPa (${String(o.utc || "").slice(0, 16)} UTC)`;
}

function siteText(d) {
  if (!d.site) return t("site.unset");
  return `${d.site.label} (${d.site.latitude}, ${d.site.longitude}) · ${t("site.outdoor")} ${outdoorText(d.outdoor)}`;
}

function deviceCard(d) {
  const s = d.status || {};
  const up = s.uploads || {}; const by = up.by_state || {};
  const pending = ["pending", "uploading", "error"].reduce((n, k) => n + ((by[k] || {}).units || 0), 0);
  const disk = Object.values(s.disks || {}).find(Boolean);
  const collector = s.collector || {};
  const kv = el("dl", { class: "kv" },
    el("dt", {}, t("card.last_signal")), el("dd", {}, `${fmtAge(d.age_s)} (${d.last_seen_utc || "—"})`),
    el("dt", {}, t("card.address")), el("dd", {}, `${d.remote_addr || "—"} · ${t("card.site_ip")} ${s.local_ip || "—"}`),
    el("dt", {}, t("card.collector")), el("dd", {}, `${collector.status || t("card.unknown")} ${typeof collector.run_dir === "string" ? "· " + collector.run_dir.split("/").pop() : ""}`),
    el("dt", {}, t("card.zed")), el("dd", {}, zedText(s.zed)),
    el("dt", {}, t("card.received")), el("dd", {}, `${t("card.trials", { n: d.units.dataset.count })} (${fmtBytes(d.units.dataset.bytes)}) · ${t("card.runs", { n: d.units.runs.count })} (${fmtBytes(d.units.runs.bytes)})`),
    el("dt", {}, t("card.pending")), el("dd", {}, t("card.count", { n: pending }) + (s.current_upload ? ` · ${t("card.uploading")} ${s.current_upload.unit} ${fmtBytes(s.current_upload.sent)}/${fmtBytes(s.current_upload.bytes)}` : "")),
    el("dt", {}, t("card.local_cleanup")), el("dd", {}, `${s.delete_after_upload ? t("card.cleanup_on", { days: (s.local_retention || {}).keep_days ?? "?", gb: (s.local_retention || {}).min_free_gb ?? "?" }) : t("card.cleanup_off")} · `
      + t("card.cleanup_done", { n: (by.deleted_local || {}).units || 0, size: fmtBytes((by.deleted_local || {}).bytes || 0) })),
    el("dt", {}, t("card.last_error")), el("dd", {}, up.last_error ? `${up.last_error.unit}: ${up.last_error.error}` : t("card.none")),
    el("dt", {}, t("card.disk")), el("dd", {}, disk ? `${fmtBytes(disk.free)} / ${fmtBytes(disk.total)}` : "—"),
    el("dt", {}, t("card.soc_temp")), el("dd", {}, Object.entries(s.temperatures_c || {}).filter(([, v]) => typeof v === "number").slice(0, 4).map(([k, v]) => `${k} ${v.toFixed(1)}°C`).join(" · ") || "—"),
    el("dt", {}, t("card.code")), el("dd", {}, s.git ? `${String(s.git.commit || "").slice(0, 10)}${s.git.dirty ? " (dirty)" : ""}` : "—"),
    el("dt", {}, t("card.site")), el("dd", { class: d.site ? "" : "error" }, siteText(d)));
  const values = d.latest_values || {};
  const pick = ["ntc.temperature_c", "ct1.current_a_nominal", "sps30.pm2_5_ug_m3", "scd30.co2_ppm", "flir.max_c"];
  const latest = el("p", { class: "muted" }, pick.filter((k) => k in values).map((k) => `${k} ${fmtNum(values[k])}`).join(" · ") || t("card.no_live"));
  const status = !d.registered ? ["warn", t("card.unregistered")] : d.online ? ["ok", t("card.online")] : ["bad", t("card.offline")];
  return el("div", { class: "card" },
    el("h3", {}, el("span", { class: `dot ${status[0]}` }), `${d.device} `, el("span", { class: "badge" }, status[1])),
    kv, latest, controlBlock(d),
    el("div", { class: "toolbar" },
      el("button", { onclick: () => { $("live-device").value = d.device; show("live"); } }, t("card.live_button")),
      el("button", { onclick: () => openSiteEditor(d.device) }, state.role === "operator" ? t("site.edit") : t("site.view"))));
}

// ------------------------------------------------------------- site editor
const pad = (n) => String(n).padStart(2, "0");
const localInput = (date) => `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;

function updateMapLink() {
  const lat = parseFloat($("site-lat").value), lon = parseFloat($("site-lon").value);
  const ok = Number.isFinite(lat) && Number.isFinite(lon);
  $("site-map").hidden = !ok;
  if (ok) $("site-map").href = `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=13/${lat}/${lon}`;
}

async function openSiteEditor(device) {
  state.siteDevice = device;
  const data = await api(`/api/v1/site?device=${encodeURIComponent(device)}`);
  $("site-editor").hidden = false;
  $("site-title").textContent = t("site.title", { device });
  const current = data.site || {};
  $("site-label").value = current.label || ""; $("site-lat").value = current.latitude ?? ""; $("site-lon").value = current.longitude ?? "";
  $("site-from").value = localInput(new Date()); $("site-note").value = ""; $("site-msg").textContent = "";
  for (const id of ["site-label", "site-lat", "site-lon", "site-from", "site-note", "site-save"]) $(id).disabled = state.role !== "operator";
  updateMapLink();
  const table = $("site-history");
  table.replaceChildren(el("tr", {}, ...["site.th_from", "site.th_until", "site.label", "site.th_coord", "site.th_by", "site.note"].map((k) => el("th", {}, t(k)))));
  for (const s of [...data.history].reverse()) {
    table.append(el("tr", { class: s.superseded ? "muted" : "" },
      el("td", {}, s.valid_from_utc), el("td", {}, s.superseded ? t("site.superseded") : (s.valid_until_utc || t("site.now"))),
      el("td", {}, s.label), el("td", {}, `${s.latitude}, ${s.longitude}`), el("td", {}, `${s.set_by} · ${String(s.set_utc || "").slice(0, 16)}`),
      el("td", {}, s.note || "")));
  }
  $("site-editor").scrollIntoView({ behavior: "smooth" });
}

$("site-lat").addEventListener("input", updateMapLink);
$("site-lon").addEventListener("input", updateMapLink);
$("site-close").addEventListener("click", () => { $("site-editor").hidden = true; state.siteDevice = null; });
$("site-save").addEventListener("click", async () => {
  const device = state.siteDevice; if (!device) return;
  const from = $("site-from").value ? new Date($("site-from").value) : new Date();
  const body = { device, label: $("site-label").value.trim(), latitude: parseFloat($("site-lat").value),
    longitude: parseFloat($("site-lon").value), valid_from_utc: from.toISOString(), note: $("site-note").value.trim() };
  if (!confirm(t("site.confirm", { device, label: body.label, lat: body.latitude, lon: body.longitude, from: body.valid_from_utc }))) return;
  try {
    await post("/api/v1/site", body);
    $("site-msg").textContent = t("site.saved");
    await openSiteEditor(device); refreshOverview();
  } catch (error) { $("site-msg").textContent = error.message; }
});

function controlBlock(d) {
  const s = d.status || {}; const history = state.commands[d.device] || [];
  const open = history.find((c) => c.state === "pending" || c.state === "delivered");
  const box = el("div", { class: "control" });
  const allowed = s.remote_control === true;
  box.append(el("p", { class: "muted" }, t("control.remote", { state: allowed ? t("control.allowed") : t("control.disabled") })));
  if (state.role === "operator" && d.registered) {
    const collecting = (s.collector || {}).status === "running";
    const zedOn = Boolean((s.zed || {}).enabled);
    const send = async (action) => {
      if (!confirm(t(`control.confirm.${action}`, { device: d.device }))) return;
      try { await post("/api/v1/control", { device: d.device, action }); } catch (error) { alert(error.message); }
      refreshOverview();
    };
    const busy = Boolean(open) || !d.online || !allowed;
    const button = (action, label, disabled, title) => {
      const node = el("button", { onclick: () => send(action), ...(title ? { title } : {}) }, label);
      node.disabled = disabled; return node;
    };
    box.append(el("div", { class: "toolbar" },
      button("start", t("control.start"), busy || collecting),
      button("stop", t("control.stop"), busy || !collecting),
      button("reset_sps30", t("control.reset"), busy || collecting, t("control.reset_title")),
      button(zedOn ? "zed_off" : "zed_on", zedOn ? t("control.zed_off") : t("control.zed_on"), busy || !s.zed, t("control.zed_title"))));
  }
  const last = history[0];
  if (last) {
    const r = last.result || {};
    const sps = r.sps30 || null;
    const detail = sps ? (sps.ok ? t("sps.ok") + (sps.measurement ? ` (PM2.5 ${fmtNum(sps.measurement.pm2_5)})` : "") : t("sps.fail", { msg: String(sps.hint || sps.error || "") }))
      : r.zed ? `ZED ${r.zed.enabled ? t("zed.on") : t("zed.off")}` : String(r.error || (r.collector || {}).status || "");
    box.append(el("p", { class: "muted" }, `${t("control.last")}: ${last.action} · ${t(`cmd.${last.state}`)} · ${last.requested_by} · ${(last.requested_utc || "").slice(0, 19)}${detail ? " · " + detail : ""}`));
  }
  return box;
}

function fillDeviceSelects(devices) {
  for (const [id, keepAll] of [["live-device", false], ["units-device", true]]) {
    const select = $(id); const current = select.value;
    const options = (keepAll ? [""] : []).concat(devices);
    if ([...select.options].map((o) => o.value).join() === options.join() && select.dataset.lang === LANG) continue;
    select.replaceChildren(...options.map((d) => el("option", { value: d }, d || t("units.all"))));
    select.dataset.lang = LANG;
    if (options.includes(current)) select.value = current;
  }
}

// ------------------------------------------------------------------- live
const PALETTE = [[0,0,4],[59,15,112],[140,41,129],[222,73,104],[254,159,109],[252,253,191]];
function color(v) {
  v = Math.min(1, Math.max(0, v)) * (PALETTE.length - 1); const i = Math.min(PALETTE.length - 2, Math.floor(v)); const f = v - i;
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

async function refreshEvents(device) {
  const data = await api(`/api/v1/events?device=${encodeURIComponent(device)}&since_s=7200`);
  const open = data.open;
  $("event-open").textContent = open ? t("event.open", { scenario: open.scenario || "", s: Math.round(Date.now() / 1000 - open.hub_epoch) }) : t("event.waiting");
  $("event-open").classList.toggle("error", Boolean(open));
  $("event-controls").hidden = state.role !== "operator";
  $("event-start").disabled = Boolean(open); $("event-end").disabled = !open;
  const table = $("event-table");
  table.replaceChildren(el("tr", {}, ...["event.th.time", "event.th.kind", "event.th.scenario", "event.th.note", "event.th.user"].map((h) => el("th", {}, t(h))), el("th", {}, "")));
  for (const e of data.events.slice(-30).reverse()) {
    const voidButton = el("button", { onclick: async () => {
      if (!confirm(t("event.confirm_void"))) return;
      await post("/api/v1/events/void", { id: e.id });
      refreshEvents(device);
    } }, t("event.void"));
    voidButton.hidden = e.voided || state.role !== "operator";
    table.append(el("tr", {}, el("td", {}, new Date(e.hub_epoch * 1000).toLocaleTimeString(LANG)),
      el("td", {}, `${t(`event.kind.${e.kind}`)}${e.voided ? t("event.voided") : ""}`),
      el("td", {}, e.scenario || "—"), el("td", {}, e.note || ""), el("td", {}, e.user), el("td", {}, voidButton)));
  }
}
async function sendEvent(kind) {
  const device = $("live-device").value; if (!device) return;
  try {
    await post("/api/v1/events", { device, kind, scenario: $("event-scenario").value, note: $("event-note").value });
    $("event-note").value = "";
  } catch (error) { alert(error.message); }
  refreshEvents(device);
}
$("event-start").addEventListener("click", () => sendEvent("intervention_start"));
$("event-end").addEventListener("click", () => sendEvent("intervention_end"));
$("event-memo").addEventListener("click", () => sendEvent("note"));

function showLiveZed(live) {
  const preview = live.zed_preview; const zed = (live.status || {}).zed;
  const img = $("zed-live");
  if (preview && preview.jpeg_b64) {
    img.src = `data:image/jpeg;base64,${preview.jpeg_b64}`; img.hidden = false;
    $("zed-info").textContent = t("live.zed_info", { tick: preview.tick ?? "—", utc: preview.utc || "" });
    $("zed-live-note").textContent = preview.stale ? t("live.zed_stale") : `${zedText(zed)} · ${t("live.zed_every")}`;
  } else {
    img.hidden = true; $("zed-info").textContent = "";
    $("zed-live-note").textContent = `${t("live.zed_none")}${zed ? " · " + zedText(zed) : ""}`;
  }
}

async function refreshLive() {
  if (!state.overview) await refreshOverview().catch(() => {});
  const device = $("live-device").value; if (!device) return;
  refreshEvents(device).catch(() => {});
  const [live, trend] = await Promise.all([api(`/api/v1/live?device=${encodeURIComponent(device)}`),
    api(`/api/v1/telemetry?device=${encodeURIComponent(device)}&since_s=${$("live-range").value}`)]);
  const s = live.status || {}; const snap = (live.live || {}).snapshot || {};
  const stale = (live.live || {}).stale;
  $("live-age").textContent = s.received_utc ? t("live.heartbeat", { hb: s.received_utc, snap: snap.timestamp_utc || t("live.no_snapshot") }) : t("live.no_heartbeat");
  $("live-age").classList.toggle("error", Boolean(stale));
  if (stale) $("live-age").textContent += t("live.stale");
  const thermal = (live.live || {}).thermal;
  if (thermal && thermal.data) { drawThermal($("thermal"), thermal.data, "t-min", "t-max");
    $("thermal-info").textContent = `seq ${thermal.snapshot_sequence}${live.live.thermal_reused ? t("live.prev_frame") : ""}`; }
  else $("thermal-info").textContent = t("live.no_frame");
  showLiveZed(live);
  const table = $("sensor-table"); table.replaceChildren(el("tr", {}, el("th", {}, t("th.sensor")), el("th", {}, t("th.status")), el("th", {}, t("th.value"))));
  for (const [name, rec] of Object.entries(snap.sensors || {})) {
    const values = Object.entries(rec.values || {}).filter(([, v]) => typeof v === "number").map(([k, v]) => `${k} ${fmtNum(v)}`).join(" · ");
    table.append(el("tr", {}, el("td", {}, name), el("td", {}, `${rec.status ?? "—"}${rec.fresh === false ? " (stale)" : ""}`), el("td", {}, values || "—")));
  }
  const dev = ((state.overview || {}).devices || []).find((x) => x.device === device);
  if (dev) table.append(el("tr", {}, el("td", {}, t("site.outdoor_row")), el("td", {}, dev.site ? dev.site.label : t("site.unset")),
    el("td", {}, dev.site ? outdoorText(dev.outdoor) : "—")));
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
const LABEL_COLORS = { normal: "#16a34a", combustion: "#dc2626", vapor_release: "#7c3aed", dust: "#a16207", transition: "#eab308", invalid: "#334155" };
const labelName = (label) => t(`label.${label}`);
function drawChart(canvas, xs, series, timeAxis, markers = [], extras = {}) {
  const ctx = canvas.getContext("2d"); const W = canvas.width, H = canvas.height, L = 60, R = 10, T = extras.intervals ? 24 : 10, B = 40;
  const css = getComputedStyle(document.body);
  ctx.clearRect(0, 0, W, H); ctx.font = "12px system-ui"; ctx.fillStyle = css.color; ctx.strokeStyle = "#8884";
  const names = Object.keys(series);
  if (!xs.length || !names.length) { ctx.fillText(t("chart.empty"), L, H / 2); return; }
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || x0 + 1;
  const px = (v) => L + (v - x0) / (x1 - x0 || 1) * (W - L - R);
  canvas._geom = { x0, x1, L, R, W };
  for (const s of extras.spans || []) {      // recorded interventions (facts, not labels)
    const a = px(Math.max(x0, s.start)), b = px(Math.min(x1, s.end));
    if (b >= a) { ctx.fillStyle = "#64748b33"; ctx.fillRect(a, T, Math.max(2, b - a), H - T - B); }
  }
  for (const iv of extras.intervals || []) {  // human labels: strip on top + light fill
    const a = px(Math.max(x0, iv.start_tick)), b = px(Math.min(x1, iv.end_tick + 1)); if (b < a) continue;
    const c = LABEL_COLORS[iv.label] || "#888";
    ctx.fillStyle = c; ctx.fillRect(a, 4, b - a, 12);
    ctx.fillStyle = c + "14"; ctx.fillRect(a, T, b - a, H - T - B);
  }
  if (extras.pending != null) { ctx.fillStyle = "#2563eb"; ctx.fillRect(px(extras.pending) - 1, T, 2, H - T - B); }
  if (extras.cursor != null && extras.cursor >= x0 && extras.cursor <= x1) {   // tick shown in the ZED panel
    ctx.fillStyle = "#0891b2"; ctx.fillRect(px(extras.cursor), T, 1, H - T - B);
  }
  names.forEach((name, n) => {
    const ys = series[name].filter((v) => v != null); if (!ys.length) return;
    const band = (extras.bands || {})[name];
    // each series gets its own y range (units differ); range shown in the legend
    let lo = Math.min(...ys, ...(band ? [band.low] : [])), hi = Math.max(...ys, ...(band ? [band.high] : [])); if (hi === lo) { hi += 1; lo -= 1; }
    const lineColor = LINE_COLORS[n % LINE_COLORS.length];
    if (band) {   // baseline mean ± k·std: a visual reference, not a threshold
      const yb = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
      ctx.fillStyle = lineColor + "22"; ctx.fillRect(L, yb(band.high), W - L - R, Math.max(1, yb(band.low) - yb(band.high)));
    }
    ctx.strokeStyle = lineColor; ctx.beginPath(); let pen = false;
    series[name].forEach((v, i) => {
      if (v == null) { pen = false; return; }
      const x = L + (xs[i] - x0) / (x1 - x0 || 1) * (W - L - R), y = T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true;
    });
    ctx.stroke(); ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(`${name} [${fmtNum(lo)} … ${fmtNum(hi)}]`, L + 4, T + 14 + 14 * n);
  });
  ctx.fillStyle = css.color;
  for (const m of markers) {   // experiment phase boundaries (procedure, not labels)
    if (m.x < x0 || m.x > x1) continue;
    const x = L + (m.x - x0) / (x1 - x0 || 1) * (W - L - R);
    ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = "#8888"; ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, H - B); ctx.stroke(); ctx.restore();
    ctx.fillText(m.label, x + 3, H - B - 4);
  }
  for (const [modality, tick] of Object.entries(extras.onsets || {})) {
    if (tick == null || tick < x0 || tick > x1) continue;
    const x = px(tick); ctx.fillStyle = "#0891b2";
    ctx.beginPath(); ctx.moveTo(x, H - B); ctx.lineTo(x - 5, H - B + 9); ctx.lineTo(x + 5, H - B + 9); ctx.fill();
    ctx.fillText(modality, x + 6, H - B + 10);
  }
  ctx.fillStyle = css.color;
  const label = (x) => timeAxis ? new Date(x).toLocaleTimeString(LANG) : String(Math.round(x));
  ctx.fillText(label(x0), L, H - 12); const right = label(x1); ctx.fillText(right, W - R - ctx.measureText(right).width, H - 12);
}

// -------------------------------------------------------------- scenarios
async function refreshScenarios() {
  const spec = await api("/api/v1/scenarios");
  state.scenarioSpec = spec.scenarios;
  $("scenario-note").textContent = t("sc.definitions", { version: spec.definitions_version, note: tr(spec, "note") });
  const cards = $("scenario-cards"); cards.replaceChildren();
  for (const s of spec.scenarios) cards.append(scenarioCard(s));
}

function scenarioCard(s) {
  const kv = el("dl", { class: "kv" },
    el("dt", {}, "scenario_id"), el("dd", {}, s.id),
    el("dt", {}, t("sc.factory")), el("dd", {}, tr(s, "factory_situation") || "—"),
    el("dt", {}, t("sc.target")), el("dd", {}, tr(s, "target_state") || "—"),
    el("dt", {}, t("sc.primary")), el("dd", {}, [...(s.primary_sensors || []), ...((s.secondary_sensors || []).map((x) => `(${x})`))].join(", ") || "—"),
    el("dt", {}, t("sc.bench")), el("dd", {}, tr(s, "bench_intervention") || tr(s, "remark") || "—"));
  const title = LANG === "en" ? s.name_en : s.name_ko;
  const card = el("div", { class: "card" },
    el("h3", {}, `${title} `, el("span", { class: "badge" }, t(`sc.status.${s.status}`))),
    el("p", { class: "muted" }, LANG === "en" ? s.name_ko : s.name_en), kv);
  if ((s.expected || []).length) {
    const table = el("table", {}, el("tr", {}, el("th", {}, t("th.sensor")), el("th", {}, t("sc.expected")), el("th", {}, t("sc.timing"))));
    for (const e of s.expected) table.append(el("tr", {}, el("td", {}, tr(e, "sensor")), el("td", {}, tr(e, "change")), el("td", {}, tr(e, "timing"))));
    const mustNot = tr(s, "must_not_change") || [];
    if (mustNot.length) table.append(el("tr", {}, el("td", {}, t("sc.must_not")), el("td", {}, mustNot.join(", ")), el("td", {}, "")));
    card.append(el("h3", {}, t("sc.signature")), table);
  }
  const safety = tr(s, "safety") || [];
  if (safety.length) card.append(el("h3", {}, t("sc.safety")), el("ul", {}, ...safety.map((x) => el("li", {}, x))));
  const r = s.received || { count: 0, recent: [] };
  card.append(el("h3", {}, t("sc.received", { n: r.count }) + (r.bench ? t("sc.bench_count", { n: r.bench }) : "")));
  if (r.recent.length) {
    const list = el("table", {}, el("tr", {}, ...["unit", t("sc.th.status"), "severity", t("sc.th.start"), t("sc.th.note")].map((h) => el("th", {}, h))));
    for (const trial of r.recent) list.append(el("tr", { class: "clickable", onclick: () => { show("units"); openUnit(trial.device, trial.unit); } },
      el("td", {}, trial.unit.split("/").pop()), el("td", {}, String(trial.status ?? "—")), el("td", {}, String(trial.severity_level ?? "—")),
      el("td", {}, (trial.started_utc || "—").slice(0, 19)), el("td", {}, trial.operator_note || "")));
    card.append(el("div", { class: "scroll" }, list));
  } else card.append(el("p", { class: "muted" }, t("sc.none_yet")));
  return card;
}

// ------------------------------------------------------------------ units
async function refreshUnits() {
  if (!state.scenarioSpec) state.scenarioSpec = (await api("/api/v1/scenarios").catch(() => ({ scenarios: [] }))).scenarios;
  if (!state.overview) await refreshOverview().catch(() => {});
  fillDeviceSelects((state.overview || { devices: [] }).devices.map((d) => d.device));
  const device = $("units-device").value;
  const data = await api(`/api/v1/units${device ? "?device=" + encodeURIComponent(device) : ""}`);
  $("units-count").textContent = t("units.count", { n: data.units.length });
  const table = $("units-table");
  table.replaceChildren(el("tr", {}, ...t("units.th").map((h) => el("th", {}, h))));
  for (const u of data.units) {
    const s = u.summary || {};
    table.append(el("tr", { class: "clickable", onclick: () => openUnit(u.device, u.unit) },
      el("td", {}, u.device), el("td", {}, u.unit), el("td", {}, `${s.status === "interrupted" ? t("units.interrupted") : (s.status ?? "—")}${s.status_source === "uplink_inferred" ? " *" : ""}${s.test_mode ? " · test" : ""}`),
      el("td", {}, s.scenario_id ?? "—"), el("td", {}, (s.started_utc || "—").slice(0, 19)), el("td", {}, String(u.files)),
      el("td", {}, fmtBytes(u.bytes)), el("td", {}, s.windows_evaluated != null ? `${s.windows_valid}/${s.windows_evaluated}` : "—"),
      el("td", {}, s.zed_frames != null ? String(s.zed_frames) : "—"), el("td", {}, u.committed_utc)));
  }
}
$("units-device").addEventListener("change", refreshUnits);

function renderUnitMeta() {
  const d = state.unitDetail; const meta = $("unit-meta"); meta.replaceChildren();
  const r = d.receipt; const kv = el("dl", { class: "kv" });
  const rows = [[t("unit.received"), r.committed_utc], [t("unit.files"), t("unit.files_value", { n: r.files, size: fmtBytes(r.bytes) })],
    ["manifest sha256", r.manifest_sha256], [t("unit.path"), r.raw_path]];
  if (d.zed) rows.push([t("unit.zed_meta"), t("unit.zed_meta_value", { frames: d.zed.frames, missed: d.zed.missed_ticks, sessions: d.zed.sessions.length, status: d.zed.status })
    + ` · ${(d.zed.camera || {}).model || ""} ${(d.zed.camera || {}).serial_number || ""}`]);
  const o = d.outdoor;
  rows.push([t("unit.outdoor"), !o || !o.site ? t("site.unset") : o.temperature_c == null ? `${o.site.label} · ${t("site.no_weather")}`
    : t("unit.outdoor_value", { temp: fmtNum(o.temperature_c), rh: fmtNum(o.humidity_pct), p: fmtNum(o.pressure_hpa), n: o.hours.length, label: o.site.label })]);
  rows.push(...Object.entries(d.summary).map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v) : String(v)]));
  for (const [k, v] of rows) kv.append(el("dt", {}, k), el("dd", {}, v));
  meta.append(kv);
  const experiment = (d.documents || {})["experiment.json"] || {};
  state.unitMarkers = (experiment.phase_plan || []).filter((p) => p.start_tick > 0)
    .map((p) => ({ x: p.start_tick, label: ["anomaly", "recovery"].includes(p.phase) ? t(`unit.phase.${p.phase}`) : p.phase }));
}

async function openUnit(device, unit) {
  const d = await api(`/api/v1/unit?device=${encodeURIComponent(device)}&unit=${encodeURIComponent(unit)}`);
  state.unit = { device, unit }; state.unitDetail = d; state.zedTick = null;
  $("unit-detail").hidden = false; $("unit-title").textContent = `${device} · ${unit}`;
  renderUnitMeta();
  const numeric = d.scalar_columns.filter((c) => !["timestamp_utc", "quality_invalid_reasons"].includes(c) && !c.endsWith("_status"));
  const experiment = (d.documents || {})["experiment.json"] || {};
  const scenario = (state.scenarioSpec || []).find((s) => s.id === experiment.scenario_id);
  const preferred = scenario ? scenario.relevant_columns : ["ntc_temperature_c", "flir_max_c"];
  $("unit-cols").replaceChildren(...numeric.map((c) => { const o = el("option", { value: c }, c); o.selected = preferred.includes(c); return o; }));
  $("unit-chunk").replaceChildren(...d.thermal_chunks.map((c) => el("option", { value: c }, c)));
  const sessions = (d.zed || {}).sessions || [];
  const first = Math.min(...sessions.map((s) => s.first_tick ?? Infinity)), last = Math.max(...sessions.map((s) => s.last_tick ?? -Infinity));
  $("unit-zed-box").hidden = !d.zed;
  if (d.zed && Number.isFinite(first)) {
    $("unit-zed-range").min = String(first); $("unit-zed-range").max = String(last); $("unit-zed-range").value = String(first);
    $("unit-zed-tick").value = String(first); state.zedTick = first;
  }
  $("unit-detail").scrollIntoView({ behavior: "smooth" });
  await loadAnnotation();
  plotUnit(); showFrame(); showZed();
}

async function plotUnit() {
  if (!state.unit) return;
  const cols = [...$("unit-cols").selectedOptions].map((o) => o.value);
  const { device, unit } = state.unit;
  const q = `device=${encodeURIComponent(device)}&unit=${encodeURIComponent(unit)}&columns=${encodeURIComponent(cols.join(","))}`;
  const [d, base] = await Promise.all([api(`/api/v1/unit/scalars?${q}`), api(`/api/v1/unit/baseline?${q}`).catch(() => ({ available: false }))]);
  state.plot = { xs: d.sequence, series: d.series, bands: base.available ? base.stats : {} };
  redrawUnit();
}

function redrawUnit() {
  if (!state.plot) return;
  const a = state.anno || {};
  drawChart($("unit-chart"), state.plot.xs, state.plot.series, false, state.unitMarkers || [],
    { bands: state.plot.bands, spans: a.spans || [], intervals: a.intervals || [], onsets: a.onsets || {}, pending: a.pendingStart,
      cursor: state.unitDetail && state.unitDetail.zed ? state.zedTick : null });
}

// ------------------------------------------------------------ annotation
function eventSpans(events) {
  // pair recorded starts and ends into shaded spans (voided clicks ignored)
  const spans = []; let open = null;
  for (const e of events.filter((x) => !x.voided)) {
    if (e.kind === "intervention_start") open = e;
    else if (e.kind === "intervention_end" && open) { spans.push({ start: open.tick, end: e.tick, scenario: open.scenario }); open = null; }
  }
  if (open) spans.push({ start: open.tick, end: open.tick, scenario: open.scenario, unclosed: true });
  return spans;
}

async function loadAnnotation() {
  const { device, unit } = state.unit;
  const q = `device=${encodeURIComponent(device)}&unit=${encodeURIComponent(unit)}`;
  const [ann, ev] = await Promise.all([api(`/api/v1/annotation?${q}`), api(`/api/v1/unit/events?${q}`).catch(() => ({ events: [], ticks: 0 }))]);
  const cur = ann.current || {};
  state.anno = { revision: ann.revision, intervals: (cur.intervals || []).map((x) => ({ ...x })), onsets: { ...(cur.onsets || {}) },
                 note: cur.note || "", pendingStart: null, events: ev.events, spans: eventSpans(ev.events), ticks: ev.ticks, meta: ann };
  $("anno-note").value = state.anno.note;
  const editable = state.role === "operator";
  for (const id of ["anno-controls", "anno-save-row"]) $(id).hidden = !editable;
  renderAnnotation();
}

function fillSelect(id, values, name) {
  const select = $(id); const current = select.value;
  select.replaceChildren(...values.map((v) => el("option", { value: v }, name(v))));
  if (values.includes(current)) select.value = current;
}

function renderAnnotation() {
  const a = state.anno; if (!a) return;
  const ann = a.meta; const cur = ann.current || {};
  fillSelect("anno-label", ann.labels, labelName);
  fillSelect("anno-conf", ann.confidence, (c) => t(`conf.${c}`));
  fillSelect("anno-modality", ann.modalities, (m) => m);
  $("anno-rev").textContent = ann.revision ? t("anno.revision", { rev: ann.revision, who: cur.annotator, utc: (cur.saved_utc || "").slice(0, 19) }) : t("anno.none");
  const table = $("anno-intervals");
  table.replaceChildren(el("tr", {}, ...t("anno.th").map((h) => el("th", {}, h))));
  a.intervals.sort((x, y) => x.start_tick - y.start_tick).forEach((iv, i) => {
    const del = el("button", { onclick: () => { a.intervals.splice(i, 1); renderAnnotation(); redrawUnit(); } }, t("anno.delete"));
    del.hidden = state.role !== "operator";
    table.append(el("tr", {}, el("td", {}, el("span", { class: "swatch" }), labelName(iv.label)), el("td", {}, String(iv.start_tick)),
      el("td", {}, String(iv.end_tick)), el("td", {}, String(iv.end_tick - iv.start_tick + 1)), el("td", {}, t(`conf.${iv.confidence}`)),
      el("td", {}, iv.note || ""), el("td", {}, del)));
    table.lastChild.querySelector(".swatch").style.background = LABEL_COLORS[iv.label] || "#888";
  });
  const spans = a.spans.map((s) => `${s.scenario || "?"} ${s.start}–${s.end}${s.unclosed ? t("anno.unclosed") : ""}`).join(", ");
  $("anno-onsets").textContent = t("anno.onsets", { onsets: Object.entries(a.onsets).filter(([, v]) => v != null).map(([k, v]) => `${k}=${v}`).join(", ") || t("card.none"),
    spans: spans || t("card.none") });
  $("anno-pending").textContent = a.pendingStart != null && a.pendingEnd == null ? t("anno.pending_start", { tick: a.pendingStart })
    : a.pendingStart != null ? t("anno.pending_range", { a: Math.min(a.pendingStart, a.pendingEnd), b: Math.max(a.pendingStart, a.pendingEnd) }) : "";
}

function chartTick(event) {
  const g = $("unit-chart")._geom; if (!g) return null;
  const rect = event.target.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (event.target.width / rect.width);
  return Math.round(g.x0 + (x - g.L) / (g.W - g.L - g.R) * (g.x1 - g.x0));
}

$("unit-chart").addEventListener("click", (event) => {
  const raw = chartTick(event); if (raw == null) return;
  if (state.unitDetail && state.unitDetail.zed) { state.zedTick = Math.max(0, raw); $("unit-zed-tick").value = String(state.zedTick); $("unit-zed-range").value = String(state.zedTick); showZed(); }
  const a = state.anno;
  if (!a || state.role !== "operator") { redrawUnit(); return; }
  const tick = Math.max(0, Math.min(a.ticks - 1, raw));
  if ($("anno-mode").value === "onset") { a.onsets[$("anno-modality").value] = tick; renderAnnotation(); redrawUnit(); return; }
  if (a.pendingStart == null || a.pendingEnd != null) { a.pendingStart = tick; a.pendingEnd = null; }
  else a.pendingEnd = tick;
  renderAnnotation(); redrawUnit();
});

$("anno-add").addEventListener("click", () => {
  const a = state.anno; if (!a || a.pendingStart == null || a.pendingEnd == null) { alert(t("anno.need_range")); return; }
  const start = Math.min(a.pendingStart, a.pendingEnd), end = Math.max(a.pendingStart, a.pendingEnd);
  if (a.intervals.some((iv) => start <= iv.end_tick && iv.start_tick <= end)) { alert(t("anno.overlap")); return; }
  a.intervals.push({ start_tick: start, end_tick: end, label: $("anno-label").value, confidence: $("anno-conf").value, note: "" });
  a.pendingStart = null; a.pendingEnd = null; renderAnnotation(); redrawUnit();
});
const clearPending = () => { if (state.anno) { state.anno.pendingStart = null; state.anno.pendingEnd = null; renderAnnotation(); redrawUnit(); } };
$("anno-clear").addEventListener("click", clearPending);
$("anno-mode").addEventListener("change", clearPending);

$("anno-save").addEventListener("click", async () => {
  const a = state.anno; const { device, unit } = state.unit;
  try {
    const r = await post("/api/v1/annotation", { device, unit, base_revision: a.revision, intervals: a.intervals, onsets: a.onsets, note: $("anno-note").value });
    $("anno-status").textContent = t("anno.saved", { rev: r.revision });
    await loadAnnotation(); redrawUnit();
  } catch (error) { $("anno-status").textContent = t("anno.save_failed", { msg: error.message }); }
});
$("unit-plot").addEventListener("click", plotUnit);

async function showFrame() {
  if (!state.unit || !$("unit-chunk").value) { $("unit-frame-info").textContent = t("unit.no_thermal"); return; }
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

let zedRequest = 0;
async function showZed() {
  const d = state.unitDetail;
  if (!state.unit || !d || !d.zed || state.zedTick == null) { $("unit-zed").hidden = true; $("unit-zed-info").textContent = d && !d.zed ? "" : t("unit.zed_none"); return; }
  const { device, unit } = state.unit; const ticket = ++zedRequest;
  try {
    const f = await api(`/api/v1/unit/zed?device=${encodeURIComponent(device)}&unit=${encodeURIComponent(unit)}&tick=${state.zedTick}`);
    if (ticket !== zedRequest) return;          // a newer request is on its way
    $("unit-zed").src = `data:image/jpeg;base64,${f.jpeg_b64}`; $("unit-zed").hidden = false;
    $("unit-zed-info").textContent = t("unit.zed_info", { tick: f.tick, frame: f.frame_tick, offset: f.offset_ms ?? "—", session: f.session,
      first: f.saved_ticks[0], last: f.saved_ticks[1], n: f.frames });
  } catch (error) { if (ticket === zedRequest) $("unit-zed-info").textContent = error.message; }
  redrawUnit();
}
const zedInput = (value) => { const v = Number(value); if (!Number.isFinite(v)) return; state.zedTick = Math.max(0, Math.round(v));
  $("unit-zed-tick").value = String(state.zedTick); $("unit-zed-range").value = String(state.zedTick); showZed(); };
$("unit-zed-tick").addEventListener("change", (e) => zedInput(e.target.value));
$("unit-zed-range").addEventListener("input", (e) => zedInput(e.target.value));

api("/api/v1/me").then((me) => start(me)).catch(() => showLogin());
