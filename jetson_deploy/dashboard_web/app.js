"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const setText = (id, value) => { byId(id).textContent = String(value); };
  const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
  const format = (value, decimals = 1) => isNumber(value)
    ? value.toLocaleString("ko-KR", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : "—";
  const ageText = (value) => isNumber(value) ? `${format(Math.max(0, value) / 1000, 1)}초` : "—";
  const statusNames = { ok: "정상", ready: "준비", disabled: "비활성", error: "오류", failed: "실패", unavailable: "사용 불가", missing: "누락", stale: "지연", warming_up: "준비 중", not_ready: "준비 중", pending: "대기" };
  const statusTones = { ok: "good", ready: "good", error: "error", failed: "error", unavailable: "warning", missing: "warning", stale: "warning" };
  const sessionNames = { stopped: "수집 대기", starting: "수집 시작 중", running: "센서 수집 중", stopping: "수집 중지 중", completed: "수집 완료", failed: "수집 실패" };
  const charts = [
    { id: "ntc", sensor: "ntc", key: "temperature_c", color: "#71e0bc", decimals: 1, unit: "°C" },
    { id: "ct", sensor: "ct1", key: "current_a_nominal", color: "#76b3fa", decimals: 2, unit: "A" },
    { id: "pm", sensor: "sps30", key: "pm2_5_ug_m3", color: "#e7ba76", decimals: 1, unit: "μg/m³" },
    { id: "co2", sensor: "scd30", key: "co2_ppm", color: "#bca1fa", decimals: 0, unit: "ppm" },
  ];
  let token = "";
  let state = null;
  let connected = false;
  let pollBusy = false;
  let commandBusy = false;
  let commandError = "";
  let runId = null;
  let history = [];
  let lastThermalKey = null;
  let thermalError = "";
  let thermalVisible = false;
  let thermalIsCurrent = false;
  let stateReceivedAt = 0;
  let connectionError = "";
  let controlGeneration = 0;

  function getToken() {
    const parameters = new URLSearchParams(location.hash.slice(1));
    const provided = parameters.get("token");
    try {
      if (provided) sessionStorage.setItem("factory-safety-dashboard-token", provided);
      token = provided || sessionStorage.getItem("factory-safety-dashboard-token") || "";
    } catch (_) {
      token = provided || "";
    }
    if (provided) window.history.replaceState(null, "", location.pathname + location.search);
  }

  function showNotice(message, tone = "warning") {
    const element = byId("notice");
    element.textContent = message;
    element.dataset.tone = tone;
    element.hidden = !message;
  }

  function setTag(id, label, tone = "") {
    const element = byId(id);
    element.textContent = label;
    element.dataset.tone = tone;
  }

  function setSensorStatus(name, sensor) {
    const status = sensor?.status;
    const text = status ? statusNames[status] || String(status) : "대기";
    setTag(`${name}-status`, name === "bme680" ? `BME680 ${text}` : text, statusTones[status] || "");
  }

  function sensorValue(sensors, sensor, key) {
    if (sensors[sensor]?.status === "disabled") return null;
    const value = sensors[sensor]?.values?.[key];
    return isNumber(value) ? value : null;
  }

  function freshness(sensor) {
    if (!sensor) return "수신 전";
    if (sensor.status === "disabled") return "비활성";
    const suffix = isNumber(sensor.age_ms) ? ` · 경과 ${ageText(sensor.age_ms)}` : "";
    if (sensor.fresh === true) return `새 측정값${suffix}`;
    if (sensor.fresh === false) return `이전 측정값 유지${suffix}`;
    return isNumber(sensor.age_ms) ? `경과 ${ageText(sensor.age_ms)}` : "갱신 정보 없음";
  }

  function describeIssues(quality) {
    const parts = [];
    if (quality?.flir_frame_valid === false) parts.push("열화상 프레임 유효성 확인 실패");
    for (const name of ["invalid_reasons", "model_channel_issues"]) {
      const issues = quality?.[name];
      if (Array.isArray(issues)) {
        for (const issue of issues) parts.push(typeof issue === "string" ? issue : JSON.stringify(issue));
      } else if (issues && typeof issues === "object") {
        for (const [key, value] of Object.entries(issues)) {
          if (Array.isArray(value) && value.length === 0) continue;
          if (value === null || value === false || value === "") continue;
          parts.push(`${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`);
        }
      } else if (issues) parts.push(String(issues));
    }
    return [...new Set(parts)];
  }

  function errorRecordText(error) {
    if (!error) return "";
    if (typeof error !== "object") return String(error);
    return [
      `연산: ${error.operation ?? "—"}`,
      `예외: ${error.exception_class ?? "—"}${error.message ? ` · ${error.message}` : ""}`,
      `errno: ${error.errno ?? "—"}`,
      `마지막 성공 연산: ${error.last_successful_operation ?? "—"}`,
      ...(error.timestamp_utc ? [`발생 시각: ${error.timestamp_utc}`] : []),
    ].join(" · ");
  }

  function sensorErrors(sensors) {
    return Object.entries(sensors || {}).flatMap(([name, sensor]) =>
      sensor?.error ? [`${name.toUpperCase()}: ${errorRecordText(sensor.error)}`] : []);
  }

  function sessionFailure(session) {
    return [session?.failure, errorRecordText(session?.first_error)].filter(Boolean).join("\n");
  }

  function resetRun() {
    history = [];
    lastThermalKey = null;
    thermalError = "";
    thermalVisible = false;
    thermalIsCurrent = false;
    byId("thermal-canvas").hidden = true;
    byId("thermal-empty").hidden = false;
    setText("palette-min", "— °C");
    setText("palette-max", "— °C");
  }

  function updateHistory(snapshot) {
    if (!isNumber(snapshot?.sequence)) return;
    if (history.length && snapshot.sequence < history[history.length - 1].sequence) history = [];
    if (history.some((point) => point.sequence === snapshot.sequence)) return;
    const point = { sequence: snapshot.sequence };
    const sensors = snapshot.sensors || {};
    for (const chart of charts) point[chart.id] = sensorValue(sensors, chart.sensor, chart.key);
    history.push(point);
    history = history.slice(-120);
  }

  function drawCharts() {
    for (const chart of charts) {
      const canvas = byId(`${chart.id}-chart`);
      const bounds = canvas.getBoundingClientRect();
      const width = Math.max(1, bounds.width);
      const height = Math.max(1, bounds.height);
      const scale = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      const context = canvas.getContext("2d");
      if (!context) continue;
      context.scale(scale, scale);
      const top = 8;
      const bottom = height - 8;
      context.strokeStyle = "#2b394b";
      context.lineWidth = 0.5;
      context.setLineDash([3, 5]);
      for (const y of [top, (top + bottom) / 2, bottom]) {
        context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
      }
      context.setLineDash([]);
      const values = history.map((point) => point[chart.id]).filter(isNumber);
      if (!values.length) {
        setText(`${chart.id}-range`, "측정값 대기");
        continue;
      }
      const minimum = Math.min(...values);
      const maximum = Math.max(...values);
      setText(`${chart.id}-range`, `${format(minimum, chart.decimals)}–${format(maximum, chart.decimals)} ${chart.unit}`);
      const margin = Math.max((maximum - minimum) * 0.12, Math.abs(maximum) * 0.003, 0.01);
      const low = minimum - margin;
      const high = maximum + margin;
      const lastSequence = history[history.length - 1].sequence;
      const firstSequence = Math.min(history[0].sequence, lastSequence - 119);
      const getX = (point) => 3 + ((point.sequence - firstSequence) / Math.max(1, lastSequence - firstSequence)) * (width - 6);
      const getY = (value) => bottom - ((value - low) / (high - low)) * (bottom - top);
      context.strokeStyle = chart.color;
      context.fillStyle = chart.color;
      context.lineWidth = 1.8;
      context.lineJoin = "round";
      context.lineCap = "round";
      let previous = null;
      let lastPoint = null;
      for (const point of history) {
        if (!isNumber(point[chart.id])) { previous = null; continue; }
        const x = getX(point);
        const y = getY(point[chart.id]);
        if (previous && point.sequence === previous.sequence + 1) {
          context.beginPath(); context.moveTo(getX(previous), getY(previous[chart.id])); context.lineTo(x, y); context.stroke();
        } else {
          context.beginPath(); context.arc(x, y, 1.5, 0, Math.PI * 2); context.fill();
        }
        previous = point;
        lastPoint = point;
      }
      if (lastPoint && lastPoint.sequence === lastSequence) {
        context.beginPath(); context.arc(getX(lastPoint), getY(lastPoint[chart.id]), 3, 0, Math.PI * 2); context.fill();
      }
    }
  }

  // Palette interpolation changes display colors only; sensor values are never corrected.
  const paletteStops = [[0,18,32,64],[0.22,37,75,135],[0.42,57,121,149],[0.62,153,185,144],[0.8,239,180,98],[1,250,229,190]];
  const palette = Array.from({ length: 256 }, (_, index) => {
    const position = index / 255;
    let upper = 1;
    while (upper < paletteStops.length - 1 && position > paletteStops[upper][0]) upper += 1;
    const a = paletteStops[upper - 1];
    const b = paletteStops[upper];
    const weight = (position - a[0]) / (b[0] - a[0]);
    return [1, 2, 3].map((channel) => Math.round(a[channel] + (b[channel] - a[channel]) * weight));
  });

  function renderThermal(preview) {
    const thermal = preview?.thermal;
    const snapshot = preview?.snapshot;
    thermalIsCurrent = false;
    thermalError = "";
    if (!thermal || !snapshot) return;
    if (thermal.snapshot_sequence !== snapshot.sequence) {
      thermalError = "열화상과 센서 스냅샷 순서가 일치하지 않습니다.";
      return;
    }
    const key = `${preview.run_id}:${snapshot.sequence}:${thermal.frame_sequence}`;
    if (key === lastThermalKey) { thermalIsCurrent = thermalVisible; return; }
    try {
      if (thermal.encoding !== "base64" || thermal.dtype !== "<u2" ||
          !Array.isArray(thermal.shape) || thermal.shape[0] !== 120 || thermal.shape[1] !== 160 ||
          thermal.conversion !== "raw / 100 - 273.15") throw new Error("지원하지 않는 열화상 형식입니다.");
      const binary = atob(thermal.data);
      if (binary.length !== 120 * 160 * 2) throw new Error("열화상 데이터 크기가 일치하지 않습니다.");
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
      const view = new DataView(bytes.buffer);
      const values = new Float64Array(120 * 160);
      let minimum = Infinity;
      let maximum = -Infinity;
      for (let index = 0; index < values.length; index += 1) {
        const value = view.getUint16(index * 2, true) / 100 - 273.15;
        values[index] = value;
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
      const context = byId("thermal-canvas").getContext("2d");
      if (!context) throw new Error("브라우저가 열화상 표시를 지원하지 않습니다.");
      const output = context.createImageData(160, 120);
      for (let index = 0; index < values.length; index += 1) {
        const normalized = maximum === minimum ? 0.5 : (values[index] - minimum) / (maximum - minimum);
        const color = palette[Math.round(normalized * 255)];
        output.data[index * 4] = color[0];
        output.data[index * 4 + 1] = color[1];
        output.data[index * 4 + 2] = color[2];
        output.data[index * 4 + 3] = 255;
      }
      context.putImageData(output, 0, 0);
      setText("palette-min", `${format(minimum, 1)} °C`);
      setText("palette-max", `${format(maximum, 1)} °C`);
      byId("thermal-canvas").hidden = false;
      byId("thermal-empty").hidden = true;
      thermalVisible = true;
      thermalIsCurrent = true;
      lastThermalKey = key;
    } catch (error) {
      thermalError = error instanceof Error ? error.message : "열화상 표시 오류";
    }
  }

  function updateMeasurements(preview) {
    const snapshot = preview?.snapshot;
    const sensors = snapshot?.sensors || {};
    for (const name of ["ntc", "ct1", "sps30", "scd30", "bme680", "flir"]) setSensorStatus(name, sensors[name]);
    for (const name of ["sgp30", "ct2", "ct3", "ct4"]) {
      const status = sensors[name]?.status;
      setText(`${name}-status`, status ? statusNames[status] || status : "비활성");
    }
    for (const chart of charts) setText(`${chart.id}-value`, format(sensorValue(sensors, chart.sensor, chart.key), chart.decimals));
    const metrics = [
      ["thermal-min", "flir", "min_c", 1], ["thermal-mean", "flir", "mean_c", 1], ["thermal-max", "flir", "max_c", 1],
      ["bme-temperature", "bme680", "temperature_c", 1], ["bme-humidity", "bme680", "humidity_pct", 1],
      ["bme-pressure", "bme680", "pressure_hpa", 1], ["bme-gas", "bme680", "gas_ohm", 0],
      ["scd-temperature", "scd30", "temperature_c", 1], ["scd-humidity", "scd30", "humidity_pct", 1],
      ["pm1-value", "sps30", "pm1_0_ug_m3", 1], ["pm4-value", "sps30", "pm4_0_ug_m3", 1],
      ["pm10-value", "sps30", "pm10_ug_m3", 1], ["ct-vrms", "ct1", "vrms", 5],
    ];
    for (const [id, sensor, key, decimals] of metrics) setText(id, format(sensorValue(sensors, sensor, key), decimals));
    setText("ct-sampling", `${format(sensorValue(sensors, "ct1", "sample_count"), 0)} / ${format(sensorValue(sensors, "ct1", "actual_sample_rate"), 1)} Hz`);
    const clipping = sensors.ct1?.values?.clipping;
    setText("ct-clipping", typeof clipping === "boolean" ? (clipping ? "감지됨" : "감지 안 됨") : isNumber(clipping) ? String(clipping) : "—");
    setText("sps30-freshness", freshness(sensors.sps30));
    setText("scd30-freshness", freshness(sensors.scd30));
    setText("thermal-freshness", freshness(sensors.flir));
    setText("frame-number", `FRAME ${format(sensors.flir?.frame_sequence, 0)}`);
    setText("sequence", format(snapshot?.sequence, 0));
    setText("jitter", `${format(snapshot?.tick_jitter_ms, 1)} ms`);
    if (snapshot?.timestamp_utc) {
      const timestamp = new Date(snapshot.timestamp_utc);
      setText("last-update", Number.isNaN(timestamp.getTime()) ? "시각 확인 불가" : timestamp.toLocaleTimeString("ko-KR", { hour12: false }));
      byId("last-update").dateTime = String(snapshot.timestamp_utc);
      byId("last-update").title = String(snapshot.timestamp_utc);
    } else {
      setText("last-update", "—");
      byId("last-update").removeAttribute("datetime");
      byId("last-update").removeAttribute("title");
    }
    const issues = [...describeIssues(snapshot?.quality), ...sensorErrors(sensors)];
    setTag("quality-tag", snapshot ? (issues.length ? "확인 필요" : "수신됨") : "확인 대기", issues.length ? "warning" : "");
    setText("quality-detail", issues.length ? issues.join(" · ") : snapshot ? "현재 스냅샷의 유효성 오류 보고 없음. 센서별 상태를 함께 확인하세요." : "센서 상태와 프레임 유효성을 확인합니다.");
    updateHistory(snapshot);
    renderThermal(preview);
    drawCharts();
  }

  function updateControls() {
    const controls = state?.controls;
    const session = state?.session;
    const canControl = Boolean(connected && token && state?.mode === "live" && controls?.enabled && !commandBusy);
    const active = session?.alive || ["starting", "running", "stopping"].includes(session?.status);
    byId("start-button").disabled = !canControl || active || controls?.start_pending || controls?.stop_pending;
    byId("stop-button").disabled = !canControl || (!session?.managed && !controls?.start_pending) || !(active || controls?.start_pending) || session?.status === "stopping" || controls?.stop_pending;
    const outcome = controls?.last_result;
    const result = commandError || (outcome?.error
      ? [outcome.error, outcome.message].filter(Boolean).join("\n")
      : outcome ? `${outcome.action === "start" ? "수집 시작" : outcome.action === "stop" ? "수집 중지" : "제어"} 요청 처리 완료` : "");
    setText("control-result", result);
    byId("control-result").hidden = !result;
  }

  function updateStateIndicators() {
    const replay = state?.mode === "replay";
    const session = state?.session;
    const current = Boolean(connected && state?.preview_current && state?.preview);
    document.body.dataset.current = String(current);
    byId("replay-banner").hidden = !replay;
    setText("connection-label", connected ? "서버 연결됨" : "서버 연결 끊김");
    byId("connection-dot").className = `dot ${connected ? "good" : "error"}`;
    const status = state?.controls?.start_pending && !["running", "stopping"].includes(session?.status) ? "starting" : session?.status;
    setText("session-status", replay ? "저장 기록 표시 중" : sessionNames[status] || "수집 상태 확인 중");
    byId("session-dot").className = `dot ${status === "running" && connected ? "good" : status === "failed" ? "error" : ["starting", "stopping"].includes(status) ? "warning" : ""}`;
    setTag("current-badge", replay ? "기록 재생" : !connected ? "연결 끊김" : current ? "실시간" : "현재 샘플 없음", current && !replay ? "good" : "warning");
    let detail = "수집을 시작하면 실제 센서 측정값을 표시합니다.";
    if (replay) detail = "저장된 스냅샷을 조회합니다. 수집 제어는 사용할 수 없습니다.";
    else if (!connected) detail = state?.preview ? "마지막으로 수신한 값을 유지하고 있습니다. 현재 측정값이 아닙니다." : "서버 연결을 기다리고 있습니다.";
    else if (status === "starting") detail = "센서 초기화와 첫 스냅샷을 기다리고 있습니다. 시작 중에도 중지할 수 있습니다.";
    else if (status === "running") detail = session?.managed ? "센서 스냅샷을 1초마다 확인합니다." : "외부에서 실행 중인 수집을 표시합니다.";
    else if (status === "stopping") detail = "수집 프로세스가 기록을 마치고 종료할 때까지 기다립니다.";
    else if (status === "completed") detail = "수집을 마쳤습니다. 마지막으로 수신한 기록을 표시합니다.";
    else if (status === "failed") detail = sessionFailure(session) || "수집 프로세스가 오류로 종료되었습니다.";
    setText("session-detail", detail);
    const elapsed = connected ? 0 : Math.max(0, Date.now() - stateReceivedAt);
    setText("preview-age", ageText(isNumber(state?.preview_age_ms) ? state.preview_age_ms + elapsed : null));
    if (!token) showNotice("collect.sh dashboard 실행 시 출력된 토큰 포함 주소로 대시보드를 열어 주세요.", "info");
    else if (!connected) showNotice(connectionError || "서버에 연결할 수 없습니다. 마지막 수신값을 표시하며 현재 수집 상태는 확인할 수 없습니다.", "error");
    else if (sessionFailure(session)) showNotice(`수집 오류: ${sessionFailure(session)}`, "error");
    else if (status === "failed") showNotice("수집 프로세스가 오류로 종료되었습니다. 기록 상태와 센서 오류를 확인하세요.", "error");
    else if (state?.preview_error) showNotice(`미리보기 오류: ${state.preview_error}`, "error");
    else if (sensorErrors(state?.preview?.snapshot?.sensors).length) showNotice(`${replay ? "저장 기록의 센서 오류" : "센서 오류"}\n${sensorErrors(state?.preview?.snapshot?.sensors).join("\n")}`, "error");
    else if (thermalError) showNotice(thermalError, "warning");
    else if (!replay && !current) showNotice(state?.preview ? "현재 실시간 샘플이 없습니다. 아래 값은 마지막으로 받은 기록입니다." : "아직 수신한 데이터가 없습니다. 수집을 시작한 후 첫 스냅샷을 기다려 주세요.", "info");
    else showNotice("");
    const flir = state?.preview?.snapshot?.sensors?.flir;
    let overlay = "";
    if (thermalVisible) {
      if (!connected) overlay = "서버 연결 끊김\n마지막 수신 프레임";
      else if (!replay && !current) overlay = "현재 실시간 프레임 없음\n마지막 수신 프레임";
      else if (thermalError) overlay = thermalError;
      else if (!thermalIsCurrent) overlay = "현재 스냅샷에 열화상 없음\n마지막 수신 프레임";
      else if (flir?.fresh === false) overlay = `이전 프레임 유지 · 경과 ${ageText(flir.age_ms)}`;
      else if (flir?.status && flir.status !== "ok") overlay = `열화상 상태: ${statusNames[flir.status] || flir.status}`;
    } else if (thermalError) overlay = thermalError;
    byId("thermal-overlay").textContent = overlay;
    byId("thermal-overlay").hidden = !overlay;
    setText("footer-state", replay ? "기록 재생 · 실시간 수집 아님" : current ? "최근 120개 수신 스냅샷 · 누락값은 연결하지 않습니다." : "현재 실시간 샘플 없음 · 마지막 수신값은 참고용입니다.");
    updateControls();
  }

  function renderState(next) {
    const previousRun = runId;
    runId = next.preview?.run_id ?? null;
    if (previousRun !== runId) resetRun();
    state = next;
    stateReceivedAt = Date.now();
    byId("replay-banner").hidden = next.mode !== "replay";
    updateMeasurements(next.preview);
    setText("snapshot-count", format(next.session?.snapshot_count, 0));
    const duration = next.settings?.duration_s;
    setText("duration", isNumber(duration) ? duration === 0 ? "제한 없음" : `${format(duration, 0)}초` : "—");
    setText("ct-raw", next.settings?.save_ct_raw === true ? "저장" : next.settings?.save_ct_raw === false ? "저장 안 함" : "—");
    setText("thermal-rate", isNumber(next.settings?.thermal_hz) ? `${format(next.settings.thermal_hz, 0)} Hz` : "—");
    setText("output-dir", next.settings?.out_dir || "—");
    setText("run-dir", next.session?.run_dir || next.preview?.run_dir || "—");
    updateStateIndicators();
  }

  async function request(path, options = {}, timeoutMs = 5000) {
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), timeoutMs);
    try {
      const response = await fetch(path, {
        ...options,
        credentials: "same-origin",
        cache: "no-store",
        headers: { "X-Dashboard-Token": token, ...options.headers },
        signal: abort.signal,
      });
      let data;
      try { data = await response.json(); } catch (_) { throw new Error("서버 응답을 읽을 수 없습니다."); }
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) throw new Error("접근 토큰을 확인할 수 없습니다. collect.sh dashboard에서 출력된 주소로 다시 열어 주세요.");
        throw new Error(typeof data.error === "string" ? data.error : typeof data.message === "string" ? data.message : `서버 응답 오류 (${response.status})`);
      }
      return data;
    } finally {
      clearTimeout(timeout);
    }
  }

  async function poll() {
    if (!token || pollBusy) return;
    pollBusy = true;
    const generation = controlGeneration;
    try {
      const next = await request("/api/state");
      // A state request started before a control action can return obsolete flags.
      if (generation !== controlGeneration) return;
      if (!next || !["live", "replay"].includes(next.mode) || !next.session) throw new Error("대시보드 상태 응답 형식이 올바르지 않습니다.");
      connected = true;
      connectionError = "";
      renderState(next);
    } catch (error) {
      connected = false;
      connectionError = error?.name === "AbortError" ? "서버 응답 시간이 초과되었습니다. 마지막 수신값을 표시하며 현재 수집 상태는 확인할 수 없습니다." : `서버 연결 끊김 · ${error instanceof Error ? error.message : "알 수 없는 오류"}`;
      updateStateIndicators();
    } finally {
      pollBusy = false;
    }
  }

  async function control(action) {
    const button = byId(action === "start" ? "start-button" : "stop-button");
    if (button.disabled || commandBusy) return;
    commandBusy = true;
    controlGeneration += 1;
    commandError = "";
    updateControls();
    setText("control-result", action === "start" ? "수집 시작을 요청하고 있습니다." : "수집 중지를 요청하고 있습니다.");
    byId("control-result").hidden = false;
    try {
      await request("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }) }, 15000);
      if (state?.controls) state.controls[action === "start" ? "start_pending" : "stop_pending"] = true;
      commandError = "";
    } catch (error) {
      commandError = error?.name === "AbortError" ? "제어 요청 응답이 지연되었습니다. 수집 상태를 확인한 뒤 다시 시도하세요." : error instanceof Error ? error.message : "제어 요청에 실패했습니다.";
    } finally {
      commandBusy = false;
      updateControls();
      void poll();
    }
  }

  byId("start-button").addEventListener("click", () => { void control("start"); });
  byId("stop-button").addEventListener("click", () => { void control("stop"); });
  let resizeQueued = false;
  window.addEventListener("resize", () => {
    if (resizeQueued) return;
    resizeQueued = true;
    requestAnimationFrame(() => { resizeQueued = false; drawCharts(); });
  });
  getToken();
  drawCharts();
  if (token) {
    void poll();
    window.setInterval(() => { void poll(); }, 1000);
  } else {
    updateStateIndicators();
    setText("connection-label", "접속 주소 필요");
  }
})();
