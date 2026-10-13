"use strict";
// UI strings, Korean and English. t("key", {name: value}) fills {name}; a missing English key falls back to Korean.

const I18N = {
  ko: {
    "nav.overview": "장치 현황", "nav.live": "실시간", "nav.scenarios": "시나리오", "nav.units": "수신 데이터",
    "logout": "로그아웃", "operator": "운영자",
    "login.title": "로그인", "login.user": "사용자", "login.password": "비밀번호", "login.submit": "로그인",
    "age.none": "기록 없음", "age.s": "{n}초 전", "age.m": "{n}분 전", "age.h": "{n}시간 전", "age.d": "{n}일 전",
    "overview.server_time": "서버 시각 {utc} · 5초마다 갱신",
    "overview.no_devices": "등록된 장치가 없습니다. 서버에서 add-device 로 등록하세요.",
    "backup.none": "서버 백업: 기록 없음 (백업이 아직 한 번도 실행되지 않았습니다)",
    "backup.ok": "서버 백업: 마지막 성공 {utc} UTC · unit {n}개 ({size}) 해시 검증 통과",
    "backup.failed": "서버 백업 실패 ({utc} UTC): {error}", "backup.stale": " · 36시간 넘게 새 백업 없음",
    "card.last_signal": "마지막 신호", "card.address": "접속 주소", "card.site_ip": "현장 IP", "card.collector": "수집기",
    "card.unknown": "알 수 없음", "card.received": "수신 완료", "card.trials": "trial {n}개", "card.runs": "run {n}개",
    "card.pending": "전송 대기/오류", "card.count": "{n}개", "card.uploading": "전송 중", "card.last_error": "마지막 오류",
    "card.none": "없음", "card.local_cleanup": "Jetson 정리", "card.cleanup_on": "업로드·검증 후 {days}일 보관 뒤 삭제 (여유 {gb} GB 미만이면 오래된 것부터 먼저)",
    "card.cleanup_off": "자동 삭제 꺼짐", "card.cleanup_done": "삭제됨 {n}개 ({size})", "card.disk": "디스크 여유", "card.soc_temp": "SoC 온도", "card.code": "코드", "card.zed": "ZED 카메라",
    "card.no_live": "실시간 값 없음 (수집기 미실행)", "card.unregistered": "미등록", "card.online": "온라인",
    "card.offline": "오프라인", "card.live_button": "실시간 보기",
    "zed.on": "켜짐", "zed.off": "꺼짐", "zed.no_recorder": "기록기 미실행", "zed.frames": "{n}프레임",
    "zed.state.disabled": "꺼짐", "zed.state.idle": "대기 (수집 중일 때 기록)", "zed.state.opening": "카메라 여는 중",
    "zed.state.recording": "기록 중", "zed.state.error": "오류", "zed.state.stopped": "기록기 정지",
    "zed.state.starting": "시작 중",
    "control.remote": "원격 수집 제어: {state}", "control.allowed": "장치에서 허용됨",
    "control.disabled": "장치에서 비활성 (uplink.json allow_remote_control)",
    "control.start": "수집 시작", "control.stop": "수집 정지", "control.reset": "SPS30 리셋",
    "control.zed_on": "ZED 켜기", "control.zed_off": "ZED 끄기",
    "control.reset_title": "수집기가 멈춘 상태에서만. SPS30 에 device reset 을 보내고 응답·측정을 확인합니다",
    "control.zed_title": "ZED 기록만 켜고 끕니다. 센서 수집은 영향을 받지 않으며, 수집 중에도 바꿀 수 있습니다",
    "control.confirm.start": "{device} 에서 수집을 시작할까요? (자유 수집 run, 기록이 남습니다)",
    "control.confirm.stop": "{device} 에서 수집을 정지할까요? (기록이 남습니다)",
    "control.confirm.reset_sps30": "{device} 에서 SPS30 센서를 리셋할까요? (기록이 남습니다)",
    "control.confirm.zed_on": "{device} 에서 ZED 기록을 켤까요? (수집 중이면 바로 기록을 시작합니다)",
    "control.confirm.zed_off": "{device} 에서 ZED 기록을 끌까요? (센서 수집은 계속됩니다)",
    "control.last": "최근 명령", "sps.ok": "SPS30 정상 응답", "sps.fail": "SPS30 실패: {msg}",
    "cmd.pending": "전달 대기", "cmd.delivered": "장치 실행 중", "cmd.done": "완료", "cmd.failed": "실패",
    "cmd.expired": "만료(장치 미수신)", "cmd.no_result": "결과 없음",
    "units.all": "전체",
    "live.device": "장치", "live.range": "추세 범위", "range.15m": "15분", "range.1h": "1시간", "range.6h": "6시간", "range.24h": "24시간",
    "live.thermal": "열화상", "live.zed": "ZED 카메라", "live.sensors": "센서 최신값", "live.trend": "추세 (heartbeat 기준)",
    "live.items": "항목", "live.heartbeat": "마지막 heartbeat {hb} · 스냅샷 {snap}", "live.no_snapshot": "없음",
    "live.no_heartbeat": "heartbeat 없음", "live.stale": " · 수집기 미실행: 아래는 마지막으로 받은 값입니다",
    "live.no_frame": "프레임 없음", "live.prev_frame": " (이전 프레임)",
    "live.zed_none": "ZED 영상 없음 — ZED 가 꺼져 있거나 수집이 실행 중이 아닙니다",
    "live.zed_stale": "마지막으로 받은 영상입니다 (지금은 기록 중이 아님)",
    "live.zed_info": "tick {tick} · {utc}", "live.zed_every": "썸네일은 약 30초마다 갱신됩니다",
    "th.sensor": "센서", "th.status": "상태", "th.value": "값",
    "event.title": "개입 기록", "event.waiting": "대기", "event.open": "개입 중 · {scenario} · {s}초",
    "event.help": "실험 중 실제로 개입한 순간에 누릅니다(예: 라이터를 켤 때 [개입 시작], 끌 때 [개입 종료]). 한 trial 에서 여러 번 눌러도 됩니다. 라벨이 아니라 \"언제 무엇을 했는가\"의 기록입니다.",
    "event.scenario": "시나리오", "event.note": "메모", "event.note_placeholder": "예: 30cm, 5초",
    "event.start": "개입 시작", "event.end": "개입 종료", "event.memo": "메모만 기록", "event.void": "무효", "event.voided": " (무효)",
    "event.confirm_void": "이 기록을 무효 처리할까요? (삭제되지 않고 무효로 표시됩니다)",
    "event.th.time": "시각", "event.th.kind": "종류", "event.th.scenario": "시나리오", "event.th.note": "메모", "event.th.user": "기록자",
    "event.kind.intervention_start": "개입 시작", "event.kind.intervention_end": "개입 종료", "event.kind.note": "메모",
    "scenario.combustion": "연소 이상", "scenario.vapor_release": "유해 증기 누출", "scenario.dust": "분진 발생", "scenario.other": "기타",
    "chart.empty": "데이터 없음 — 항목을 고르세요",
    "label.normal": "정상", "label.combustion": "연소 이상", "label.vapor_release": "유해 증기", "label.dust": "분진",
    "label.transition": "전이(애매)", "label.invalid": "무효",
    "sc.status.bench": "사무실 bench 시나리오", "sc.status.field": "현장 시나리오 (유도 방법 TBD)", "sc.status.blocked": "현재 관측 불가",
    "sc.status.baseline": "기준선", "sc.definitions": "정의 {version} · {note}", "sc.factory": "공장 상황", "sc.target": "대상 상태",
    "sc.primary": "주 센서", "sc.bench": "bench 개입", "sc.signature": "멀티센서 서명", "sc.expected": "예상 반응 (가설)",
    "sc.timing": "시점", "sc.must_not": "변하지 않아야 함", "sc.safety": "안전 수칙", "sc.received": "수신된 trial {n}개",
    "sc.bench_count": " (bench {n})", "sc.none_yet": "아직 없음", "sc.th.status": "상태", "sc.th.start": "시작(UTC)", "sc.th.note": "메모",
    "units.count": "{n}개 · 클릭하면 상세",
    "units.th": ["장치", "unit", "상태", "시나리오", "시작(UTC)", "파일", "크기", "유효 window", "ZED", "수신(UTC)"],
    "units.interrupted": "중단됨(재부팅 등)",
    "unit.received": "수신", "unit.files": "파일/크기", "unit.files_value": "{n}개 · {size}", "unit.path": "서버 경로",
    "unit.columns": "열", "unit.plot": "그리기",
    "unit.legend": "띠: 기준선 평균±3σ(참고) · 회색 음영: 실제 개입 기록 · 점선: 계획상 신호음 · 상단 색 막대: 라벨 구간",
    "unit.phase.anomaly": "계획상 개입(신호음)", "unit.phase.recovery": "계획상 회복",
    "unit.thermal_chunk": "열화상 chunk", "unit.no_thermal": "열화상 없음",
    "unit.zed_tick": "ZED tick", "unit.zed_help": "그래프를 클릭하면 그 tick 의 ZED 영상이 표시됩니다.",
    "unit.zed_none": "이 unit 에는 ZED 영상이 없습니다",
    "unit.zed_info": "요청 tick {tick} → 저장된 tick {frame} ({offset} ms) · 세션 {session} · 저장 {first}–{last}, {n}프레임",
    "unit.zed_meta": "ZED", "unit.zed_meta_value": "{frames}프레임 · 누락 tick {missed} · 세션 {sessions}개 · {status}",
    "anno.title": "라벨링", "anno.mode": "모드", "anno.mode_interval": "구간", "anno.mode_onset": "센서 반응 시작점",
    "anno.label": "라벨", "anno.confidence": "확신도", "anno.sensor": "센서", "anno.add": "구간 추가", "anno.clear": "선택 취소",
    "anno.note": "메모", "anno.save": "저장 (새 revision)",
    "anno.help": "그래프를 두 번 클릭해 구간의 시작과 끝을 찍고, 라벨과 확신도를 고른 뒤 [구간 추가]. 센서 반응 시작점은 모드를 바꿔 한 번 클릭합니다. 라벨을 붙이지 않은 tick 은 학습에서 제외됩니다. 기준: docs/LABELING_GUIDE.md",
    "conf.high": "높음", "conf.medium": "보통", "conf.low": "낮음",
    "anno.revision": "revision {rev} · {who} · {utc}", "anno.none": "아직 라벨 없음",
    "anno.th": ["라벨", "시작 tick", "끝 tick", "길이(s)", "확신도", "메모", ""], "anno.delete": "삭제",
    "anno.unclosed": "(종료 없음)", "anno.onsets": "센서 반응 시작점: {onsets} · 기록된 개입: {spans}",
    "anno.pending_start": "시작 {tick} 선택됨 — 끝을 클릭", "anno.pending_range": "구간 {a}–{b} — [구간 추가]",
    "anno.need_range": "그래프에서 시작과 끝을 먼저 클릭하세요", "anno.overlap": "기존 구간과 겹칩니다. 먼저 삭제하거나 다른 범위를 고르세요",
    "anno.saved": "저장됨: revision {rev}", "anno.save_failed": "저장 실패: {msg}",
  },
  en: {
    "nav.overview": "Devices", "nav.live": "Live", "nav.scenarios": "Scenarios", "nav.units": "Received data",
    "logout": "Log out", "operator": "operator",
    "login.title": "Log in", "login.user": "User", "login.password": "Password", "login.submit": "Log in",
    "age.none": "never", "age.s": "{n} s ago", "age.m": "{n} min ago", "age.h": "{n} h ago", "age.d": "{n} days ago",
    "overview.server_time": "Server time {utc} · refreshes every 5 s",
    "overview.no_devices": "No devices registered. Register one on the server with add-device.",
    "backup.none": "Server backup: no record (no backup has run yet)",
    "backup.ok": "Server backup: last success {utc} UTC · {n} units ({size}) hash-verified",
    "backup.failed": "Server backup failed ({utc} UTC): {error}", "backup.stale": " · no new backup for over 36 h",
    "card.last_signal": "Last signal", "card.address": "Address", "card.site_ip": "site IP", "card.collector": "Collector",
    "card.unknown": "unknown", "card.received": "Received", "card.trials": "{n} trials", "card.runs": "{n} runs",
    "card.pending": "Pending / errors", "card.count": "{n}", "card.uploading": "uploading", "card.last_error": "Last error",
    "card.none": "none", "card.local_cleanup": "Jetson cleanup", "card.cleanup_on": "deleted {days} days after upload + verification (oldest first if free space < {gb} GB)",
    "card.cleanup_off": "auto delete off", "card.cleanup_done": "{n} deleted ({size})", "card.disk": "Free disk", "card.soc_temp": "SoC temperature", "card.code": "Code", "card.zed": "ZED camera",
    "card.no_live": "No live values (collector not running)", "card.unregistered": "unregistered", "card.online": "online",
    "card.offline": "offline", "card.live_button": "Live view",
    "zed.on": "on", "zed.off": "off", "zed.no_recorder": "recorder not running", "zed.frames": "{n} frames",
    "zed.state.disabled": "off", "zed.state.idle": "idle (records while collecting)", "zed.state.opening": "opening camera",
    "zed.state.recording": "recording", "zed.state.error": "error", "zed.state.stopped": "recorder stopped",
    "zed.state.starting": "starting",
    "control.remote": "Remote collection control: {state}", "control.allowed": "allowed on the device",
    "control.disabled": "disabled on the device (uplink.json allow_remote_control)",
    "control.start": "Start collection", "control.stop": "Stop collection", "control.reset": "Reset SPS30",
    "control.zed_on": "ZED on", "control.zed_off": "ZED off",
    "control.reset_title": "Only while the collector is stopped. Sends a device reset to the SPS30 and checks its reply and a measurement",
    "control.zed_title": "Switches only the ZED recording. Sensor collection is not affected; this can be changed during a run",
    "control.confirm.start": "Start collection on {device}? (free run; this is logged)",
    "control.confirm.stop": "Stop collection on {device}? (this is logged)",
    "control.confirm.reset_sps30": "Reset the SPS30 sensor on {device}? (this is logged)",
    "control.confirm.zed_on": "Turn ZED recording on for {device}? (starts recording at once if a collection is running)",
    "control.confirm.zed_off": "Turn ZED recording off for {device}? (sensor collection continues)",
    "control.last": "Last command", "sps.ok": "SPS30 responded", "sps.fail": "SPS30 failed: {msg}",
    "cmd.pending": "waiting for device", "cmd.delivered": "running on device", "cmd.done": "done", "cmd.failed": "failed",
    "cmd.expired": "expired (device did not pick it up)", "cmd.no_result": "no result",
    "units.all": "all",
    "live.device": "Device", "live.range": "Trend range", "range.15m": "15 min", "range.1h": "1 hour", "range.6h": "6 hours", "range.24h": "24 hours",
    "live.thermal": "Thermal", "live.zed": "ZED camera", "live.sensors": "Latest sensor values", "live.trend": "Trend (from heartbeats)",
    "live.items": "Series", "live.heartbeat": "Last heartbeat {hb} · snapshot {snap}", "live.no_snapshot": "none",
    "live.no_heartbeat": "no heartbeat", "live.stale": " · collector not running: values below are the last ones received",
    "live.no_frame": "no frame", "live.prev_frame": " (previous frame)",
    "live.zed_none": "No ZED image — ZED is off or no collection is running",
    "live.zed_stale": "Last image received (not recording now)",
    "live.zed_info": "tick {tick} · {utc}", "live.zed_every": "The thumbnail refreshes about every 30 s",
    "th.sensor": "Sensor", "th.status": "Status", "th.value": "Values",
    "event.title": "Intervention log", "event.waiting": "idle", "event.open": "intervening · {scenario} · {s} s",
    "event.help": "Press at the moment you actually intervene (e.g. [Start] when the lighter goes on, [End] when it goes off). You can press several times per trial. This records what was done and when; it is not a label.",
    "event.scenario": "Scenario", "event.note": "Note", "event.note_placeholder": "e.g. 30 cm, 5 s",
    "event.start": "Start intervention", "event.end": "End intervention", "event.memo": "Note only", "event.void": "Void", "event.voided": " (void)",
    "event.confirm_void": "Void this entry? (it is kept and marked void, not deleted)",
    "event.th.time": "Time", "event.th.kind": "Kind", "event.th.scenario": "Scenario", "event.th.note": "Note", "event.th.user": "By",
    "event.kind.intervention_start": "start", "event.kind.intervention_end": "end", "event.kind.note": "note",
    "scenario.combustion": "Combustion", "scenario.vapor_release": "Chemical vapor release", "scenario.dust": "Dust", "scenario.other": "Other",
    "chart.empty": "No data — choose series",
    "label.normal": "normal", "label.combustion": "combustion", "label.vapor_release": "vapor release", "label.dust": "dust",
    "label.transition": "transition (ambiguous)", "label.invalid": "invalid",
    "sc.status.bench": "office bench scenario", "sc.status.field": "field scenario (method TBD)", "sc.status.blocked": "not observable now",
    "sc.status.baseline": "baseline", "sc.definitions": "Definitions {version} · {note}", "sc.factory": "Factory situation", "sc.target": "Target state",
    "sc.primary": "Primary sensors", "sc.bench": "Bench intervention", "sc.signature": "Multi-sensor signature", "sc.expected": "Expected response (hypothesis)",
    "sc.timing": "Timing", "sc.must_not": "Must not change", "sc.safety": "Safety rules", "sc.received": "{n} trials received",
    "sc.bench_count": " ({n} bench)", "sc.none_yet": "none yet", "sc.th.status": "Status", "sc.th.start": "Start (UTC)", "sc.th.note": "Note",
    "units.count": "{n} · click for details",
    "units.th": ["Device", "Unit", "Status", "Scenario", "Start (UTC)", "Files", "Size", "Valid windows", "ZED", "Received (UTC)"],
    "units.interrupted": "interrupted (reboot etc.)",
    "unit.received": "Received", "unit.files": "Files / size", "unit.files_value": "{n} · {size}", "unit.path": "Server path",
    "unit.columns": "Columns", "unit.plot": "Plot",
    "unit.legend": "Band: baseline mean ±3σ (reference) · grey: recorded interventions · dashed: planned beeps · top strip: labels",
    "unit.phase.anomaly": "planned intervention (beep)", "unit.phase.recovery": "planned recovery",
    "unit.thermal_chunk": "Thermal chunk", "unit.no_thermal": "no thermal frames",
    "unit.zed_tick": "ZED tick", "unit.zed_help": "Click the chart to show the ZED image at that tick.",
    "unit.zed_none": "This unit has no ZED images",
    "unit.zed_info": "requested tick {tick} → saved tick {frame} ({offset} ms) · session {session} · saved {first}–{last}, {n} frames",
    "unit.zed_meta": "ZED", "unit.zed_meta_value": "{frames} frames · {missed} missed ticks · {sessions} sessions · {status}",
    "anno.title": "Labeling", "anno.mode": "Mode", "anno.mode_interval": "interval", "anno.mode_onset": "sensor onset",
    "anno.label": "Label", "anno.confidence": "Confidence", "anno.sensor": "Sensor", "anno.add": "Add interval", "anno.clear": "Clear selection",
    "anno.note": "Note", "anno.save": "Save (new revision)",
    "anno.help": "Click the chart twice to mark the start and end of an interval, choose a label and confidence, then [Add interval]. For a sensor onset, switch the mode and click once. Ticks without a label are excluded from training. Rules: docs/LABELING_GUIDE.md",
    "conf.high": "high", "conf.medium": "medium", "conf.low": "low",
    "anno.revision": "revision {rev} · {who} · {utc}", "anno.none": "no labels yet",
    "anno.th": ["Label", "Start tick", "End tick", "Length (s)", "Confidence", "Note", ""], "anno.delete": "Delete",
    "anno.unclosed": "(no end)", "anno.onsets": "Sensor onsets: {onsets} · recorded interventions: {spans}",
    "anno.pending_start": "start {tick} selected — click the end", "anno.pending_range": "interval {a}–{b} — [Add interval]",
    "anno.need_range": "Click the start and the end on the chart first", "anno.overlap": "Overlaps an existing interval. Delete it first or choose another range",
    "anno.saved": "Saved: revision {rev}", "anno.save_failed": "Save failed: {msg}",
  },
};

function loadLang() {
  try { const saved = localStorage.getItem("hub-lang"); if (saved === "ko" || saved === "en") return saved; } catch (e) { /* storage unavailable */ }
  return "ko";
}
let LANG = loadLang();

function t(key, vars = {}) {
  let value = (I18N[LANG] || {})[key];
  if (value === undefined) value = I18N.ko[key];
  if (value === undefined) return key;
  if (Array.isArray(value)) return value;
  return value.replace(/\{(\w+)\}/g, (m, name) => (name in vars ? String(vars[name] ?? "") : m));
}

// Scenario definitions carry English variants as <field>_en; Korean is the default text.
function tr(obj, field) {
  if (!obj) return "";
  return (LANG === "en" && obj[`${field}_en`] != null) ? obj[`${field}_en`] : obj[field];
}

function applyStatic() {
  document.documentElement.lang = LANG;
  for (const node of document.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of document.querySelectorAll("[data-i18n-placeholder]")) node.placeholder = t(node.dataset.i18nPlaceholder);
  for (const button of document.querySelectorAll(".lang button")) button.classList.toggle("active", button.dataset.lang === LANG);
}

function setLang(lang) {
  LANG = lang === "en" ? "en" : "ko";
  try { localStorage.setItem("hub-lang", LANG); } catch (e) { /* storage unavailable */ }
  applyStatic();
  document.dispatchEvent(new Event("langchange"));
}
