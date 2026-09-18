#!/usr/bin/env python3
"""Synchronized multi-sensor collection at the model's 1 Hz time base.

Usage:
    ./jetson_deploy/run_python.sh jetson_deploy/scripts/11_collect_sensors.py --duration 120
    ./jetson_deploy/run_python.sh jetson_deploy/scripts/11_collect_sensors.py --duration 0
    ./jetson_deploy/run_python.sh jetson_deploy/scripts/11_collect_sensors.py --duration 60 --save-ct-raw

One master tick = 1.0 s = one snapshot, which is the tick the trained model
consumes (30 ticks per window). Ticks are scheduled on absolute monotonic
deadlines, so lateness never accumulates.

IMPORTANT: this collector is the sole owner of the ADS1115. Do not run
scripts/08 (NTC) or scripts/09 (CT) at the same time - one ADC cannot serve
both, and concurrent access corrupts both readers.

This is a raw acquisition layer. It does not run inference and does not change
the model input vector [NTC, PM1.0, PM2.5, PM10, CT1, CT2, CT3, CT4].
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import importlib.metadata
import math
import platform
from pathlib import Path
import signal
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sensors.collector import SensorCollector, write_run  # noqa: E402
from sensors.snapshot import (  # noqa: E402
    FLIR_MAX_AGE_MS, SCHEMA_VERSION, WINDOW_TICKS,
)
from sensors.runtime import atomic_json, utc_now  # noqa: E402
from sensors.validation import validate_run  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "results" / "sensor_collection"


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def build_metadata(args, run_id: str) -> dict:
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=5)
    versions = {}
    for package in ("numpy", "adafruit-circuitpython-bme680", "adafruit-circuitpython-sgp30",
                    "sensirion-i2c-sps30", "sensirion-i2c-scd30", "sensirion-i2c-driver"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    l4t_path = Path("/etc/nv_tegra_release")
    return {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit_sha": git_sha(),
        "git_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "timestamp": utc_now(),
        "hostname": platform.node(),
        "jetpack_l4t": l4t_path.read_text().strip() if l4t_path.exists() else None,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "environment_profile": "JETSON-RUNTIME",
        "sensor_driver_versions": versions,
        "acquisition_timing_version": 1,
        "synchronization": "1 Hz software timeline; per-sensor host receipt times, not hardware triggers",
        "thermal_saved_hz": 1.0 if not args.no_thermal_save else 0.0,
        "i2c_clock_hz": {"/dev/i2c-7": 400000, "/dev/i2c-1": 100000},
        "ads1115": {"address": "0x48", "ct_mux": "AIN0-AIN1", "ct_pga_fsr_v": 2.048,
                    "ct_data_rate_sps": 860, "ntc_mux": "A2", "ntc_pga_fsr_v": 4.096,
                    "ntc_data_rate_sps": 128, "ct_burden_ohm": 0.68,
                    "ct_primary_a": 400.0, "ct_secondary_a": 1.0,
                    "calibration": "nominal circuit conversion; field calibration pending"},
        "master_tick_hz": 1.0,
        "duration_s": args.duration,
        "ct_burst_s": args.ct_burst,
        "save_ct_raw": bool(args.save_ct_raw),
        "save_thermal": not args.no_thermal_save,
        "thermal_device": args.thermal_device,
        "buses": {
            "/dev/i2c-7": ["ADS1115 0x48 (CT AIN0-AIN1, NTC A2)",
                           "SGP30 0x58", "BME680 0x77"],
            "/dev/i2c-1": ["SPS30 0x69", "SCD30 0x61"],
            "usb": ["FLIR Lepton via PureThermal (UVC)"],
        },
        "native_rates": {
            "ntc": "1 Hz (single-shot per tick)",
            "ct1": "860 SPS burst per tick",
            "sps30": "~1 Hz",
            "sgp30": "1 Hz strict",
            "scd30": "0.5 Hz (2 s native interval)",
            "bme680": "1 Hz",
            "flir": "continuous ~9 fps, latest frame per tick",
        },
        "model_input_channels": ["NTC", "PM1.0", "PM2.5", "PM10",
                                 "CT1", "CT2", "CT3", "CT4"],
        "quality_policy": {
            "window_ticks": WINDOW_TICKS,
            "flir_max_age_ms": FLIR_MAX_AGE_MS,
            "rule": ("a 30-tick window is training-invalid when any tick has "
                     "flir status != ok or flir age_ms > 500 ms"),
            "raw_data": "kept - never deleted because a window is invalid",
            "repair": ("none - stale source frames retain per-tick raw entries; "
                       "never synthesised or interpolated as fresh"),
            "model_channel_issues": ("recorded for visibility, does not "
                                     "invalidate a window under this policy"),
        },
        "note": ("Raw acquisition layer only. SGP30/SCD30/BME680 are context "
                 "sensors and are not part of the model input vector. CT2-4 have "
                 "no physical front-end and are reported as disabled, never "
                 "zero-filled."),
    }


def print_report(report: dict) -> None:
    def line(label, value):
        print(f"    {label:<28} {value}")

    print()
    print("=" * 74)
    print("Timing report")
    print("=" * 74)
    m = report["master"]
    print("  master tick")
    line("snapshots", m["snapshot_count"])
    line("expected ticks", m["expected_ticks"])
    line("missed ticks", m["missed_ticks"])
    line("period ms mean/p50/p95/max",
         f'{m["period_ms_mean"]} / {m["period_ms_p50"]} / {m["period_ms_p95"]} / {m["period_ms_max"]}')
    line("jitter ms mean", m["jitter_ms_mean"])
    line("abs jitter ms p95/max", f'{m["abs_jitter_ms_p95"]} / {m["abs_jitter_ms_max"]}')
    line("tick work ms mean/max", f'{m["tick_work_ms_mean"]} / {m["tick_work_ms_max"]}')

    s = report["sgp30"]
    print(f'  SGP30 (1 Hz strict){"" if s.get("enabled", True) else "  [DISABLED by profile]"}')
    line("measurements", s["measurement_count"])
    line("interval ms mean/p95/max",
         f'{s["interval_ms_mean"]} / {s["interval_ms_p95"]} / {s["interval_ms_max"]}')
    line("violations >1.1s / >1.5s", f'{s["violations_gt_1100ms"]} / {s["violations_gt_1500ms"]}')
    line("ok / error", f'{s["ok_count"]} / {s["error_count"]}')

    s = report["sps30"]
    print("  SPS30")
    line("fresh snapshots", s["fresh_snapshot_count"])
    line("age ms mean/max", f'{s["age_ms_mean"]} / {s["age_ms_max"]}')
    line("ok / error", f'{s["ok_count"]} / {s["error_count"]}')

    s = report["scd30"]
    print("  SCD30 (0.5 Hz native)")
    line("fresh / non-fresh snapshots",
         f'{s["fresh_snapshot_count"]} / {s["non_fresh_snapshot_count"]}')
    line("fresh interval ms mean/max",
         f'{s["fresh_interval_ms_mean"]} / {s["fresh_interval_ms_max"]}')
    line("age ms max", s["age_ms_max"])
    line("ok / error", f'{s["ok_count"]} / {s["error_count"]}')

    s = report["bme680"]
    print("  BME680")
    line("ok / error", f'{s["ok_count"]} / {s["error_count"]}')

    s = report["ct1"]
    print("  CT1 (ADS1115 AIN0-AIN1)")
    line("bursts", s["burst_count"])
    line("samples/burst mean/min/max",
         f'{s["samples_per_burst_mean"]} / {s["samples_per_burst_min"]} / {s["samples_per_burst_max"]}')
    line("actual SPS mean/min/max",
         f'{s["actual_sps_mean"]} / {s["actual_sps_min"]} / {s["actual_sps_max"]}')
    line("clipping ticks", s["clipping_tick_count"])
    line("errors", s["error_count"])

    s = report["ntc"]
    print("  NTC (ADS1115 A2)")
    line("ok / error", f'{s["ok_count"]} / {s["error_count"]}')

    s = report["storage"]
    print("  storage (async chunk writer)")
    line("chunks written", s["chunks_written"])
    line("queue max depth / size", f'{s["queue_max_depth"]} / {s["queue_maxsize"]}')
    line("dropped chunks", s["dropped_chunks"])
    line("write errors", f'{s["write_errors"]}  {s["last_error"] or ""}')
    for ev in s["degraded_events"]:
        line("  DEGRADED", ev)

    q = report["quality"]
    print("  data quality (training window policy)")
    line("window ticks", q["window_ticks"])
    line("invalid ticks", f'{q["invalid_tick_count"]} {q["invalid_tick_reasons"] or ""}')
    line("windows valid / invalid",
         f'{q["windows_valid"]} / {q["windows_invalid"]}  (of {q["windows_evaluated"]})')
    for w in q["invalid_window_details"]:
        line("  invalid window seqs", f'{w["invalid_sequences"]}  {w["reasons"]}')

    s = report["flir"]
    print("  FLIR Lepton")
    line("ticks with frame", s["ticks_with_frame"])
    line("age ms mean/p95/max",
         f'{s["age_ms_mean"]} / {s["age_ms_p95"]} / {s["age_ms_max"]}')
    line("frames captured", s["frames_captured"])
    line("shape failures / errors", f'{s["shape_failures"]} / {s["error_count"]}')
    print()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0,
                        help="seconds to collect; 0 means until stopped")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--save-ct-raw", action="store_true")
    parser.add_argument("--no-thermal-save", action="store_true")
    parser.add_argument("--ct-burst", type=float, default=0.5)
    parser.add_argument("--thermal-device", default="/dev/video0")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable-sgp30", dest="disable_sgp30", action="store_false",
                       help="opt into SGP30; unresolved hardware stability (custom profile)")
    group.add_argument("--disable-sgp30", dest="disable_sgp30", action="store_true",
                       help="v1 profile default: no access to SGP30")
    parser.set_defaults(disable_sgp30=True)
    return parser


def parse_args(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("duration must be finite and >= 0")
    if 0 < args.duration < 1:
        parser.error("duration must be 0 or at least one second")
    if not math.isfinite(args.ct_burst) or not 0 < args.ct_burst <= 0.5:
        parser.error("ct-burst must be > 0 and <= 0.5 seconds to fit the 1 Hz tick")
    return args


def main(argv=None, *, run_dir=None, on_state=None) -> int:
    args = parse_args(argv)
    if run_dir is None:
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = Path(args.out_dir).expanduser().resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = Path(run_dir)
        run_id = run_dir.name

    state_errors = []

    def state(name, **extra):
        if on_state is not None:
            try:
                on_state(name, **extra)
            except Exception as exc:
                # A full disk must never bypass device shutdown.
                state_errors.append(f"{type(exc).__name__}: {exc}")
                print(f"cannot persist control state: {exc}", file=sys.stderr)

    print(f"run dir: {run_dir}", flush=True)
    print("1 Hz snapshots and thermal storage; continuous camera reader", flush=True)
    print(f"SGP30: {'disabled (v1 profile)' if args.disable_sgp30 else 'enabled (custom)'}", flush=True)
    collector = SensorCollector(
        run_dir, args.duration, save_ct_raw=args.save_ct_raw,
        save_thermal=not args.no_thermal_save, ct_burst_s=args.ct_burst,
        thermal_device=args.thermal_device, enable_sgp30=not args.disable_sgp30)
    interrupted = {"signal": None}
    previous_handlers = {}
    failure = None
    paths = {}
    metadata = {"run_id": run_id, "started_utc": utc_now()}

    def on_stop(signum, frame):
        interrupted["signal"] = signum
        collector.request_stop()

    def progress(snapshot):
        if snapshot["sequence"] == 0:
            state("running", first_snapshot_utc=snapshot["timestamp_utc"])
        if snapshot["sequence"] % 10 == 0:
            sensors = snapshot["sensors"]
            statuses = " ".join(f"{k}={v['status']}" for k, v in sensors.items()
                                if v["status"] != "disabled")
            print(f"[{snapshot['sequence']}] {statuses}", flush=True)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, on_stop)
        state("starting")
        metadata = build_metadata(args, run_id)
        atomic_json(run_dir / "metadata.json", metadata)
        collector.start()
        metadata["sensor_profile"] = collector.sensor_profile()
        metadata["sensor_manifest"] = collector.sensor_manifest()
        atomic_json(run_dir / "metadata.json", metadata)
        paths = write_run(collector, run_dir, on_snapshot=progress)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(f"collection failed: {failure}", file=sys.stderr, flush=True)
    finally:
        state("stopping")
        try:
            collector.flush_chunks()
        except Exception as exc:
            failure = failure or f"chunk flush: {type(exc).__name__}: {exc}"
        try:
            collector.shutdown()
        except Exception as exc:
            failure = failure or f"shutdown: {type(exc).__name__}: {exc}"

    try:
        report = collector.timing_report()
        validation = validate_run(run_dir, save_thermal=not args.no_thermal_save,
                                  expected_snapshots=report["master"]["snapshot_count"])
        storage = report["storage"]
        failed = bool(failure or state_errors or report.get("first_error") or report.get("shutdown_errors")
                      or storage["dropped_chunks"] or storage["write_errors"]
                      or storage["degraded_events"] or not validation["passed"]
                      or report["master"]["missed_ticks"])
        # A successful stop means files were finalized; it does not mean all model
        # windows are training-valid (e.g. the camera can pause during FFC).
        outcome = "failed" if failed else ("stopped" if interrupted["signal"] else "completed")
        report.update(status=outcome, failure=failure, control_state_errors=state_errors, validation=validation,
                      stop_signal=interrupted["signal"], completed_utc=utc_now())
        atomic_json(run_dir / "timing_report.json", report)
        metadata.update(status=outcome, completed_utc=report["completed_utc"],
                        snapshot_count=report["master"]["snapshot_count"])
        atomic_json(run_dir / "metadata.json", metadata)
        print_report(report)
        code = 1 if failed else (128 + interrupted["signal"] if interrupted["signal"] else 0)
        state(outcome, exit_code=code, failure=failure, first_error=report.get("first_error"),
              validation=validation, snapshot_count=report["master"]["snapshot_count"])
        print(f"{outcome}: {run_dir}", flush=True)
        return code
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)



if __name__ == "__main__":
    raise SystemExit(main())
