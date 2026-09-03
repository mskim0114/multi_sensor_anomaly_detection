#!/usr/bin/env python3
"""Reproducible anomaly-dataset trial runner.

Usage:
    ./jetson_deploy/run_python.sh jetson_deploy/scripts/12_run_trial.py \
        --scenario dust --severity 2 --operator-note "..."

    ./jetson_deploy/run_python.sh jetson_deploy/scripts/12_run_trial.py \
        --scenario normal --severity 0 --duration 30 --test-mode --yes

    ./jetson_deploy/run_python.sh jetson_deploy/scripts/12_run_trial.py --self-test

SAFETY BOUNDARY
---------------
This program performs DATA ACQUISITION AND EXPERIMENT ORCHESTRATION ONLY.

It never creates or controls an anomaly. There is no mains switching, no
electrical load control, no heater control, no fan control, no dust generator
control, no relay and no actuator anywhere in this file. The physical
intervention is the responsibility of a separate safe testbed and of the
operator procedure.

WHAT IT DOES NOT RECORD
-----------------------
It does not invent training labels. `phase` is a description of the experiment
procedure, not a state label. The anomaly phase is when the operator intervenes,
not when the sensors necessarily show anything; recovery is not automatically
normal. `observed_anomaly_onset_tick` and `observed_recovery_tick` start as null
and are filled during a later annotation step.

Sensor drivers are not reimplemented here: the verified SensorCollector from
jetson_deploy/sensors/ is reused, including for preflight, so no sensor is ever
owned by two objects at once.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sensors.collector import (  # noqa: E402
    SENSOR_PROFILE_V1,
    SensorCollector,
    write_run,
)
from sensors.snapshot import (  # noqa: E402
    FLIR_MAX_AGE_MS,
    SCHEMA_VERSION,
    STATUS_OK,
    WINDOW_TICKS,
)

EXPERIMENT_SCHEMA_VERSION = 1

SCENARIOS = ("normal", "overload", "thermal_abnormal", "dust")
SEVERITY_NAMES = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}

# Canonical anomaly trial, docs/JETSON_DATASET_PROTOCOL.md section 4.
CANONICAL_BASELINE_S = 90
CANONICAL_ANOMALY_S = 180
CANONICAL_RECOVERY_S = 90
CANONICAL_TOTAL_S = CANONICAL_BASELINE_S + CANONICAL_ANOMALY_S + CANONICAL_RECOVERY_S

DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
SMOKE_DIRNAME = "_smoke"
PHASE_WARN_S = 5           # stdout warning this many ticks before a boundary

STATUS_CREATED = "created"
STATUS_PREFLIGHT_FAILED = "preflight_failed"
STATUS_READY = "ready"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_ABORTED = "aborted"
STATUS_FAILED = "failed"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, obj: dict) -> None:
    """Write JSON so a crash never leaves a half-written state file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def git_info() -> dict:
    out = {}
    for key, args in (("git_commit", ["rev-parse", "HEAD"]),
                      ("git_branch", ["rev-parse", "--abbrev-ref", "HEAD"])):
        try:
            r = subprocess.run(["git"] + args, cwd=str(REPO_ROOT),
                               capture_output=True, text=True, timeout=5)
            value = r.stdout.strip()
            if value:
                out[key] = value
        except Exception:
            pass
    return out


def host_info() -> dict:
    info = {"host": platform.node(), "kernel": platform.release()}
    try:
        text = Path("/etc/nv_tegra_release").read_text(encoding="utf-8").strip()
        m = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", text)
        info["l4t"] = f"R{m.group(1)}.{m.group(2)}" if m else text.splitlines()[0]
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def validate_severity(scenario: str, severity: int) -> None:
    """Planned intervention severity. Not a model class, not a state label."""
    if scenario == "normal":
        if severity != 0:
            raise ValueError("scenario 'normal' requires --severity 0")
    else:
        if severity not in (1, 2, 3):
            raise ValueError(f"scenario '{scenario}' requires --severity 1, 2 or 3")


def build_plan(scenario: str, baseline_s: int, anomaly_s: int, recovery_s: int,
               normal_duration_s: int | None) -> dict:
    """Phase plan in master ticks. 1 Hz, so tick number == elapsed second.

    A normal control trial gets a single phase. No fake anomaly phase is
    invented for it, and its intervention ticks stay null.
    """
    if scenario == "normal":
        total = normal_duration_s if normal_duration_s is not None else CANONICAL_TOTAL_S
        if total < 1:
            raise ValueError("--duration must be at least 1")
        return {
            "total_ticks": total,
            "phase_plan": [{"phase": "normal_control", "start_tick": 0,
                            "end_tick": total - 1}],
            "intervention_start_tick": None,
            "intervention_end_tick": None,
            "protocol_compliant": total == CANONICAL_TOTAL_S,
            "durations_s": {"total": total},
        }

    for name, value in (("baseline", baseline_s), ("anomaly", anomaly_s),
                        ("recovery", recovery_s)):
        if value < 1:
            raise ValueError(f"--{name}-seconds must be at least 1")
    total = baseline_s + anomaly_s + recovery_s
    a0 = baseline_s
    a1 = baseline_s + anomaly_s - 1
    return {
        "total_ticks": total,
        "phase_plan": [
            {"phase": "baseline", "start_tick": 0, "end_tick": baseline_s - 1},
            {"phase": "anomaly", "start_tick": a0, "end_tick": a1},
            {"phase": "recovery", "start_tick": a1 + 1, "end_tick": total - 1},
        ],
        "intervention_start_tick": a0,
        "intervention_end_tick": a1,
        "protocol_compliant": (baseline_s == CANONICAL_BASELINE_S
                               and anomaly_s == CANONICAL_ANOMALY_S
                               and recovery_s == CANONICAL_RECOVERY_S),
        "durations_s": {"baseline": baseline_s, "anomaly": anomaly_s,
                        "recovery": recovery_s, "total": total},
    }


def allocate_trial_dir(dataset_root: Path, scenario: str, test_mode: bool) -> tuple[Path, str]:
    """Create a fresh trial directory. Never reuses or overwrites a number."""
    if test_mode:
        parent = dataset_root / SMOKE_DIRNAME
        parent.mkdir(parents=True, exist_ok=True)
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        trial_id = f"{scenario}_{run_id}"
        path = parent / trial_id
        suffix = 1
        while True:
            try:
                path.mkdir()
                return path, trial_id
            except FileExistsError:
                suffix += 1
                trial_id = f"{scenario}_{run_id}_{suffix:02d}"
                path = parent / trial_id

    parent = dataset_root / scenario
    parent.mkdir(parents=True, exist_ok=True)
    used = set()
    for entry in parent.iterdir():
        m = re.fullmatch(r"trial_(\d+)", entry.name)
        if m:
            used.add(int(m.group(1)))
    n = (max(used) + 1) if used else 1
    while True:
        trial_id = f"trial_{n:03d}"
        path = parent / trial_id
        try:
            path.mkdir()
            return path, trial_id
        except FileExistsError:
            n += 1


# ---------------------------------------------------------------------------
# preflight, using the collector's own already-initialised devices
# ---------------------------------------------------------------------------
def run_preflight(collector: SensorCollector) -> dict:
    """Check every REQUIRED sensor of the profile through the verified paths.

    The collector has already opened the ADS1115 and started the bus-1 and
    thermal workers, so these checks reuse those objects. No i2cdetect scan and
    no second owner of any device.
    """
    checks: list[dict] = []

    def add(name, ok, detail=""):
        checks.append({"sensor": name, "result": "PASS" if ok else "FAIL",
                       "detail": detail, "required": True})
        return ok

    # ADS1115 - the collector's start() already probed it, confirm the config read
    try:
        cfg = collector.ads.probe()
        add("ADS1115", True, f"/dev/i2c-7 0x48 config=0x{cfg:04x}")
    except Exception as exc:
        add("ADS1115", False, f"{type(exc).__name__}: {exc}")

    # NTC on A2
    try:
        ntc = collector.ads.read_ntc()
        t = float(ntc["temperature_c"])
        ok = -40.0 <= t <= 85.0
        add("NTC", ok, f"A2 {t:.2f} C, R {ntc['resistance_ohm']:.0f} ohm")
    except Exception as exc:
        add("NTC", False, f"{type(exc).__name__}: {exc}")

    # CT1 differential burst
    try:
        ct = collector.ads.ct_burst()
        ct.pop("_codes", None)
        ok = ct["sample_count"] > 0 and ct["actual_sample_rate"] is not None
        add("CT1", ok, f"{ct['sample_count']} samples @ {ct['actual_sample_rate']} SPS, "
                       f"vrms {ct['vrms']:.3e} V")
    except Exception as exc:
        add("CT1", False, f"{type(exc).__name__}: {exc}")

    # SPS30 / SCD30 - bus 1 worker identity plus a live reading
    sps_reading, scd_reading = collector.bus1.latest()
    sps_id = collector.bus1.identity.get("sps30", {})
    scd_id = collector.bus1.identity.get("scd30", {})
    add("SPS30", bool(sps_id.get("serial")) and sps_reading is not None,
        f"/dev/i2c-1 0x69 serial={sps_id.get('serial')} fw={sps_id.get('firmware')}"
        if sps_id else "no identity / no reading")
    add("SCD30", bool(scd_id.get("serial")) and scd_reading is not None,
        f"/dev/i2c-1 0x61 serial={scd_id.get('serial')} fw={scd_id.get('firmware')}"
        if scd_id else "no identity / no reading")

    # BME680 chip id
    try:
        ident = collector.bme680.read_identity()
        ok = ident.get("chip_id") == "0x61"
        add("BME680", ok, f"/dev/i2c-7 0x77 chip_id={ident.get('chip_id')} "
                          f"variant_id={ident.get('variant_id')}")
    except Exception as exc:
        add("BME680", False, f"{type(exc).__name__}: {exc}")

    # FLIR frame and shape
    frame, frame_ns, _ = collector.thermal.latest()
    if frame is None:
        add("FLIR", False, collector.thermal.startup_error or "no frame")
    else:
        shape_ok = tuple(frame.shape) == (collector.thermal.MODEL_HEIGHT,
                                          collector.thermal.MODEL_WIDTH)
        fid = collector.thermal.read_identity()
        add("FLIR", shape_ok, f"{fid.get('device')} shape={tuple(frame.shape)} "
                              f"serial={fid.get('serial')}")

    # SGP30 is disabled by the v1 profile and is deliberately not required.
    checks.append({"sensor": "SGP30", "result": "DISABLED",
                   "detail": "disabled_by_profile: hardware_stability_unresolved",
                   "required": False})

    failed = [c["sensor"] for c in checks if c["required"] and c["result"] != "PASS"]
    return {"checks": checks, "failed_required": failed, "result": "PASS" if not failed else "FAIL"}


def print_preflight(pf: dict, profile_name: str) -> None:
    print()
    print("PREFLIGHT")
    print("-" * 74)
    for c in pf["checks"]:
        print(f"  {c['sensor']:<12} {c['result']:<9} {c['detail']}")
    print("-" * 74)
    print(f"  PROFILE:  {profile_name}")
    print(f"  RESULT:   {pf['result']}")
    if pf["failed_required"]:
        print(f"  FAILED REQUIRED: {', '.join(pf['failed_required'])}")
    print()


# ---------------------------------------------------------------------------
# quality summary
# ---------------------------------------------------------------------------
def quality_summary(report: dict, planned_ticks: int) -> dict:
    q = report["quality"]
    total = report["master"]["snapshot_count"]
    invalid = q["invalid_tick_count"]
    return {
        "policy": (f"thermal-invalid tick when flir status != ok or age_ms > "
                   f"{FLIR_MAX_AGE_MS:.0f}; a {WINDOW_TICKS}-tick window containing one "
                   f"is training-invalid. Raw data is kept; stale frames are never "
                   f"duplicated or interpolated."),
        "total_ticks": total,
        "planned_ticks": planned_ticks,
        "valid_ticks": total - invalid,
        "invalid_ticks": invalid,
        "invalid_reason_counts": q["invalid_tick_reasons"],
        "total_30s_windows": q["windows_evaluated"],
        "valid_30s_windows": q["windows_valid"],
        "invalid_30s_windows": q["windows_invalid"],
        "invalid_window_details": q["invalid_window_details"],
        "missed_master_ticks": report["master"]["missed_ticks"],
        "writer_dropped_chunks": report["storage"]["dropped_chunks"],
        "writer_errors": report["storage"]["write_errors"],
        "sensor_error_counts": {
            "ntc": report["ntc"]["error_count"],
            "ct1": report["ct1"]["error_count"],
            "sps30": report["sps30"]["error_count"],
            "scd30": report["scd30"]["error_count"],
            "bme680": report["bme680"]["error_count"],
            "flir": report["flir"]["error_count"],
            "sgp30": report["sgp30"]["error_count"],
        },
    }


def acceptance(qs: dict, planned_ticks: int) -> dict:
    """Minimum bar for an official completed trial.

    An FFC-invalidated window is NOT fatal - it is recorded, not hidden.
    """
    reasons = []
    if qs["total_ticks"] != planned_ticks:
        reasons.append(f"tick count {qs['total_ticks']} != planned {planned_ticks}")
    if qs["missed_master_ticks"] != 0:
        reasons.append(f"missed master ticks {qs['missed_master_ticks']}")
    if qs["writer_dropped_chunks"] or qs["writer_errors"]:
        reasons.append("writer dropped chunks or write errors")
    for name in ("ntc", "ct1", "sps30", "scd30", "bme680", "flir"):
        if qs["sensor_error_counts"][name] and qs["valid_ticks"] == 0:
            reasons.append(f"required sensor {name} produced no usable data")
    return {"accepted": not reasons, "reasons": reasons}


# ---------------------------------------------------------------------------
# self test - deterministic, no hardware
# ---------------------------------------------------------------------------
def self_test() -> int:
    import tempfile
    failures = []

    def check(label, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            failures.append(label)

    print("SELF TEST (no hardware)")
    print("-" * 74)

    # scenario / severity validation
    for sc in SCENARIOS:
        check(f"scenario '{sc}' accepted", sc in SCENARIOS)
    try:
        validate_severity("normal", 1); ok = False
    except ValueError:
        ok = True
    check("normal + severity 1 rejected", ok)
    try:
        validate_severity("normal", 0); ok = True
    except ValueError:
        ok = False
    check("normal + severity 0 accepted", ok)
    for sev, expect_ok in ((0, False), (1, True), (2, True), (3, True), (4, False)):
        try:
            validate_severity("dust", sev); got = True
        except ValueError:
            got = False
        check(f"dust + severity {sev} -> {'accept' if expect_ok else 'reject'}", got == expect_ok)

    # phase boundaries, canonical
    p = build_plan("dust", CANONICAL_BASELINE_S, CANONICAL_ANOMALY_S, CANONICAL_RECOVERY_S, None)
    check("canonical total ticks 360", p["total_ticks"] == 360)
    check("baseline 0..89", p["phase_plan"][0] == {"phase": "baseline", "start_tick": 0, "end_tick": 89})
    check("anomaly 90..269", p["phase_plan"][1] == {"phase": "anomaly", "start_tick": 90, "end_tick": 269})
    check("recovery 270..359", p["phase_plan"][2] == {"phase": "recovery", "start_tick": 270, "end_tick": 359})
    check("intervention ticks 90/269",
          (p["intervention_start_tick"], p["intervention_end_tick"]) == (90, 269))
    check("canonical protocol_compliant true", p["protocol_compliant"] is True)

    # phase boundaries, override
    p2 = build_plan("dust", 5, 10, 5, None)
    check("override total 20", p2["total_ticks"] == 20)
    check("override anomaly starts at 5", p2["phase_plan"][1]["start_tick"] == 5)
    check("override recovery starts at 15", p2["phase_plan"][2]["start_tick"] == 15)
    check("override end tick 19", p2["phase_plan"][2]["end_tick"] == 19)
    check("override protocol_compliant false", p2["protocol_compliant"] is False)

    # normal control plan
    pn = build_plan("normal", 90, 180, 90, None)
    check("normal single phase", len(pn["phase_plan"]) == 1
          and pn["phase_plan"][0]["phase"] == "normal_control")
    check("normal intervention ticks null",
          pn["intervention_start_tick"] is None and pn["intervention_end_tick"] is None)
    check("normal 360 compliant", pn["protocol_compliant"] is True)
    pn30 = build_plan("normal", 90, 180, 90, 30)
    check("normal duration 30 not compliant", pn30["protocol_compliant"] is False)
    check("normal duration 30 single phase 0..29",
          pn30["phase_plan"][0]["end_tick"] == 29)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # trial numbering
        d1, id1 = allocate_trial_dir(root, "dust", False)
        d2, id2 = allocate_trial_dir(root, "dust", False)
        check("first trial is trial_001", id1 == "trial_001")
        check("second trial is trial_002", id2 == "trial_002")
        (root / "dust" / "trial_007").mkdir()
        d3, id3 = allocate_trial_dir(root, "dust", False)
        check("numbering continues past gap -> trial_008", id3 == "trial_008")
        check("existing dirs untouched", (root / "dust" / "trial_001").is_dir()
              and (root / "dust" / "trial_007").is_dir())
        # test mode isolation
        dt_, idt = allocate_trial_dir(root, "normal", True)
        check("test mode goes under _smoke", SMOKE_DIRNAME in dt_.parts)
        check("test mode not under scenario dir", "normal" not in dt_.parent.name)
        # atomic write and interrupted state
        p = root / "experiment.json"
        atomic_write_json(p, {"status": STATUS_CREATED})
        check("atomic write created file", json.loads(p.read_text())["status"] == STATUS_CREATED)
        atomic_write_json(p, {"status": STATUS_RUNNING})
        check("atomic write replaced state", json.loads(p.read_text())["status"] == STATUS_RUNNING)
        check("no .tmp left behind", not list(root.glob("*.tmp")))

    print("-" * 74)
    print(f"  {'ALL PASS' if not failures else f'{len(failures)} FAILURES'}")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS)
    ap.add_argument("--severity", type=int, choices=(0, 1, 2, 3))
    ap.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    ap.add_argument("--operator-note", default="")
    ap.add_argument("--equipment-condition", default="")
    ap.add_argument("--save-ct-raw", action="store_true")
    ap.add_argument("--no-thermal-save", action="store_true")
    ap.add_argument("--baseline-seconds", type=int, default=CANONICAL_BASELINE_S)
    ap.add_argument("--anomaly-seconds", type=int, default=CANONICAL_ANOMALY_S)
    ap.add_argument("--recovery-seconds", type=int, default=CANONICAL_RECOVERY_S)
    ap.add_argument("--duration", type=int, default=None,
                    help="normal scenario only: total seconds override")
    ap.add_argument("--test-mode", action="store_true",
                    help=f"write under <dataset-root>/{SMOKE_DIRNAME}/ instead of the "
                         "official scenario directories")
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive ENTER confirmation (countdown still runs)")
    ap.add_argument("--self-test", action="store_true",
                    help="run deterministic checks with no hardware and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.scenario or args.severity is None:
        ap.error("--scenario and --severity are required (or use --self-test)")

    try:
        validate_severity(args.scenario, args.severity)
    except ValueError as exc:
        ap.error(str(exc))
    if args.duration is not None and args.scenario != "normal":
        ap.error("--duration is only allowed for --scenario normal; use "
                 "--baseline-seconds/--anomaly-seconds/--recovery-seconds")

    try:
        plan = build_plan(args.scenario, args.baseline_seconds, args.anomaly_seconds,
                          args.recovery_seconds, args.duration)
    except ValueError as exc:
        ap.error(str(exc))
    total_ticks = plan["total_ticks"]

    dataset_root = Path(args.dataset_root)
    trial_dir, trial_id = allocate_trial_dir(dataset_root, args.scenario, args.test_mode)
    exp_path = trial_dir / "experiment.json"

    experiment = {
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "trial_id": trial_id,
        "scenario_id": args.scenario,
        "severity_level": args.severity,
        "severity_name": SEVERITY_NAMES[args.severity],
        "sensor_profile": SENSOR_PROFILE_V1,
        "test_mode": bool(args.test_mode),
        "status": STATUS_CREATED,
        "created_at_utc": utc_now(),
        "started_at_utc": None,
        "completed_at_utc": None,
        "planned_duration_s": total_ticks,
        "planned_durations_s": plan["durations_s"],
        "protocol_compliant": plan["protocol_compliant"],
        "phase_plan": plan["phase_plan"],
        "intervention_start_tick": plan["intervention_start_tick"],
        "intervention_end_tick": plan["intervention_end_tick"],
        "observed_anomaly_onset_tick": None,
        "observed_recovery_tick": None,
        "operator_note": args.operator_note,
        "equipment_condition": args.equipment_condition,
        "note": ("phase is the experiment procedure, not a state label. "
                 "observed_* are filled by a later annotation step. This runner "
                 "never generates training labels."),
    }
    experiment.update(host_info())
    experiment.update(git_info())
    atomic_write_json(exp_path, experiment)

    print("=" * 74)
    print("DATASET TRIAL")
    print("=" * 74)
    print(f"  SCENARIO         {args.scenario}")
    print(f"  SEVERITY         {args.severity} ({SEVERITY_NAMES[args.severity]})")
    print(f"  TRIAL ID         {trial_id}")
    print(f"  DURATION         {total_ticks} s  {plan['durations_s']}")
    print(f"  PROTOCOL         {'compliant (90/180/90)' if plan['protocol_compliant'] else 'NON-CANONICAL (protocol_compliant=false)'}")
    print(f"  SENSOR PROFILE   {SENSOR_PROFILE_V1}")
    print(f"  OUTPUT DIRECTORY {trial_dir}")
    print(f"  MODE             {'TEST (not official dataset)' if args.test_mode else 'OFFICIAL'}")
    print("  NOTE             this program does not create or control the anomaly")

    collector = SensorCollector(
        trial_dir, float(total_ticks),
        save_ct_raw=args.save_ct_raw,
        save_thermal=not args.no_thermal_save,
        enable_sgp30=False,             # v1 profile
    )

    # ---- preflight -------------------------------------------------------
    try:
        collector.start()
    except Exception as exc:
        pf = {"checks": [{"sensor": "ADS1115", "result": "FAIL", "required": True,
                          "detail": f"collector start failed: {type(exc).__name__}: {exc}"}],
              "failed_required": ["ADS1115"], "result": "FAIL"}
        print_preflight(pf, SENSOR_PROFILE_V1)
        experiment.update(status=STATUS_PREFLIGHT_FAILED, preflight=pf)
        atomic_write_json(exp_path, experiment)
        print("PREFLIGHT FAILED - no acquisition was started")
        print(f"  {exp_path}")
        return 2

    try:
        pf = run_preflight(collector)
        print_preflight(pf, SENSOR_PROFILE_V1)
        profile = collector.sensor_profile()
        manifest = collector.sensor_manifest()
        experiment["preflight"] = pf
        experiment["sensor_profile_detail"] = profile
        experiment["sensor_manifest"] = manifest      # frozen at preflight PASS
        if pf["result"] != "PASS":
            experiment["status"] = STATUS_PREFLIGHT_FAILED
            atomic_write_json(exp_path, experiment)
            print("PREFLIGHT FAILED - no acquisition was started")
            print(f"  {exp_path}")
            return 2

        experiment["status"] = STATUS_READY
        atomic_write_json(exp_path, experiment)

        # ---- operator confirmation --------------------------------------
        if not args.yes:
            print("Press ENTER to start trial   /   Ctrl+C to abort")
            try:
                input()
            except EOFError:
                experiment["status"] = STATUS_ABORTED
                experiment["abort_reason"] = "no interactive stdin and --yes not given"
                atomic_write_json(exp_path, experiment)
                print("aborted: stdin is not interactive; pass --yes to run unattended")
                return 130
        for k in (3, 2, 1):
            print(f"  {k}")
            time.sleep(1.0)
        print("  START\a", flush=True)

        # ---- run --------------------------------------------------------
        experiment["status"] = STATUS_RUNNING
        experiment["started_at_utc"] = utc_now()
        atomic_write_json(exp_path, experiment)

        metadata = {
            "run_id": trial_id,
            "schema_version": SCHEMA_VERSION,
            "started_utc": experiment["started_at_utc"],
            "master_tick_hz": 1.0,
            "duration_s": float(total_ticks),
            "save_ct_raw": bool(args.save_ct_raw),
            "save_thermal": not args.no_thermal_save,
            "trial_id": trial_id,
            "scenario_id": args.scenario,
            "severity_level": args.severity,
            "sensor_profile": profile,
            "sensor_manifest": manifest,
            "model_input_channels": ["NTC", "PM1.0", "PM2.5", "PM10",
                                     "CT1", "CT2", "CT3", "CT4"],
            "note": ("Raw acquisition layer. SCD30/BME680 are context sensors and "
                     "are not part of the model input vector. CT2-4 have no physical "
                     "front-end and are reported as disabled, never zero-filled."),
        }
        metadata.update(host_info())
        metadata.update(git_info())
        atomic_write_json(trial_dir / "metadata.json", metadata)

        boundaries = {p["start_tick"]: p["phase"] for p in plan["phase_plan"] if p["start_tick"] > 0}
        warnings = {t - PHASE_WARN_S: name for t, name in boundaries.items()
                    if t - PHASE_WARN_S > 0}

        def on_snapshot(snap: dict) -> None:
            seq = snap["sequence"]
            if seq in warnings:
                print(f"  T-{PHASE_WARN_S} sec: {warnings[seq].upper()} PHASE IN "
                      f"{PHASE_WARN_S} SECONDS\a", flush=True)
            if seq in boundaries:
                print(f"=== {boundaries[seq].upper()} PHASE START ===  tick {seq}\a", flush=True)
            if seq % 30 == 0:
                sen = snap["sensors"]
                bad = [k for k, v in sen.items() if v["status"] not in (STATUS_OK, "disabled")]
                print(f"  [{seq:>4}/{total_ticks}] jit {snap['tick_jitter_ms']:+6.2f} ms  "
                      f"work {snap['tick_work_ms']:6.1f} ms"
                      f"{'  degraded: ' + ','.join(bad) if bad else ''}", flush=True)

        paths = write_run(collector, trial_dir, on_snapshot=on_snapshot)

    except KeyboardInterrupt:
        experiment["status"] = STATUS_ABORTED
        experiment["abort_reason"] = "KeyboardInterrupt"
        experiment["completed_at_utc"] = utc_now()
        atomic_write_json(exp_path, experiment)
        print("\naborted by operator - partial files are kept, directory not deleted")
        print(f"  {trial_dir}")
        return 130
    except Exception as exc:
        experiment["status"] = STATUS_FAILED
        experiment["failure"] = f"{type(exc).__name__}: {exc}"
        experiment["completed_at_utc"] = utc_now()
        atomic_write_json(exp_path, experiment)
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        print(f"  partial files are kept: {trial_dir}")
        return 1
    finally:
        collector.flush_chunks()
        collector.shutdown()

    # ---- finish ----------------------------------------------------------
    report = collector.timing_report()
    atomic_write_json(trial_dir / "timing_report.json", report)

    qs = quality_summary(report, total_ticks)
    acc = acceptance(qs, total_ticks)
    experiment["quality"] = qs
    experiment["acceptance"] = acc
    experiment["snapshots_written"] = paths["snapshots_written"]
    experiment["completed_at_utc"] = utc_now()
    experiment["status"] = STATUS_COMPLETED if acc["accepted"] else STATUS_FAILED
    atomic_write_json(exp_path, experiment)

    print()
    print("=" * 74)
    print(f"TRIAL {experiment['status'].upper()}")
    print("=" * 74)
    m = report["master"]
    print(f"  ticks            {qs['total_ticks']}/{qs['planned_ticks']}   missed {m['missed_ticks']}")
    print(f"  period ms        mean {m['period_ms_mean']}  p95 {m['period_ms_p95']}  max {m['period_ms_max']}")
    print(f"  tick work ms     mean {m['tick_work_ms_mean']}  max {m['tick_work_ms_max']}")
    print(f"  writer           dropped {qs['writer_dropped_chunks']}  errors {qs['writer_errors']}")
    print(f"  thermal ticks    valid {qs['valid_ticks']}  invalid {qs['invalid_ticks']}  {qs['invalid_reason_counts'] or ''}")
    print(f"  30s windows      valid {qs['valid_30s_windows']}  invalid {qs['invalid_30s_windows']}  of {qs['total_30s_windows']}")
    print(f"  sensor errors    {qs['sensor_error_counts']}")
    print(f"  protocol         compliant={experiment['protocol_compliant']}")
    if not acc["accepted"]:
        for r in acc["reasons"]:
            print(f"  NOT ACCEPTED: {r}")
    print(f"  experiment.json  {exp_path}")
    print(f"  trial directory  {trial_dir}")
    return 0 if acc["accepted"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
