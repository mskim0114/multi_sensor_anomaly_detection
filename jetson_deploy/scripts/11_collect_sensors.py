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

DEFAULT_OUT_DIR = ROOT / "results" / "sensor_collection"


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def build_metadata(args, run_id: str) -> dict:
    return {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit_sha": git_sha(),
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
            "repair": "none - stale frames are never duplicated or interpolated",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0,
                        help="seconds to collect; 0 means run until Ctrl+C")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--save-ct-raw", action="store_true",
                        help="also store the raw CT waveform codes as NPZ chunks")
    parser.add_argument("--no-thermal-save", action="store_true",
                        help="do not store thermal frames (scalars are still recorded)")
    parser.add_argument("--ct-burst", type=float, default=0.5,
                        help="CT waveform capture seconds per tick")
    parser.add_argument("--thermal-device", default="/dev/video0")
    parser.add_argument("--disable-sgp30", action="store_true",
                        help="do not use SGP30 at all: no I2C access to 0x58, no "
                             "iaq_init, no reconnect retry, status recorded as "
                             "disabled. This is the jetson_factory_v1_2026 profile.")
    args = parser.parse_args()

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)   # startup-fatal if this fails

    print(f"run id:    {run_id}")
    print(f"run dir:   {run_dir}")
    print(f"duration:  {'continuous (Ctrl+C to stop)' if args.duration <= 0 else f'{args.duration:.0f} s'}")
    print(f"ct burst:  {args.ct_burst:.3f} s per tick")
    print(f"thermal:   {'off' if args.no_thermal_save else 'saved in 30-frame NPZ chunks'}")
    print(f"ct raw:    {'saved' if args.save_ct_raw else 'off (scalar RMS always saved)'}")
    print(f"sgp30:     {'DISABLED (no 0x58 access)' if args.disable_sgp30 else 'enabled'}")
    print("NOTE: this process owns the ADS1115. Do not run scripts/08 or 09 concurrently.")

    collector = SensorCollector(
        run_dir, args.duration,
        save_ct_raw=args.save_ct_raw,
        save_thermal=not args.no_thermal_save,
        ct_burst_s=args.ct_burst,
        thermal_device=args.thermal_device,
        enable_sgp30=not args.disable_sgp30,
    )

    interrupted = {"value": False}

    def on_sigint(signum, frame):
        interrupted["value"] = True
        collector.request_stop()
        print("\nstopping after the current tick ...", file=sys.stderr)

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    collector.start()   # startup-fatal: ADS1115 must be reachable

    # metadata is written after start() so it can carry the sensor manifest -
    # the physical identity actually read from each device on this run.
    metadata = build_metadata(args, run_id)
    # sensor_profile = intended configuration of this run.
    # sensor_manifest = physical identity actually read from the devices.
    metadata["sensor_profile"] = collector.sensor_profile()
    metadata["sensor_manifest"] = collector.sensor_manifest()
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    prof = metadata["sensor_profile"]
    print(f'sensor profile: {prof["name"]}'
          f'{"" if prof["matches_profile_v1"] else " (does not match v1)"}')
    print(f'  enabled : {", ".join(prof["enabled_sensors"])}')
    if prof["disabled_sensors"]:
        for d in prof["disabled_sensors"]:
            print(f'  disabled: {d["sensor"]} ({d["reason"]})')
    print("sensor manifest:")
    for name, info in metadata["sensor_manifest"].items():
        ident = ", ".join(f"{k}={v}" for k, v in info.items()
                          if k not in ("bus", "address", "device", "note"))
        loc = info.get("bus") or info.get("device") or "-"
        addr = info.get("address", "")
        print(f"  {name:<8} {loc} {addr:<6} {ident or '(no unique identity available)'}")

    def progress(snapshot: dict) -> None:
        if snapshot["sequence"] % 10:
            return
        sen = snapshot["sensors"]
        print(f'[{snapshot["sequence"]:>5}] '
              f'jit {snapshot["tick_jitter_ms"]:+7.2f} ms  '
              f'work {snapshot["tick_work_ms"]:6.1f} ms  '
              f'ntc {sen["ntc"]["status"]:<11} '
              f'ct1 {sen["ct1"]["status"]:<5} '
              f'sps30 {sen["sps30"]["status"]:<5} '
              f'sgp30 {sen["sgp30"]["status"]:<11} '
              f'scd30 {sen["scd30"]["status"]:<5} '
              f'bme680 {sen["bme680"]["status"]:<5} '
              f'flir {sen["flir"]["status"]}')

    paths = {}
    try:
        paths = write_run(collector, run_dir, on_snapshot=progress)
    finally:
        collector.flush_chunks()
        collector.shutdown()

    report = collector.timing_report()
    (run_dir / "timing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report)

    print(f"metadata:  {run_dir / 'metadata.json'}")
    print(f"scalars:   {paths.get('scalars_path', run_dir / 'scalars.csv')}")
    print(f"snapshots: {paths.get('snapshots_path', run_dir / 'snapshots.jsonl')}")
    print(f"timing:    {run_dir / 'timing_report.json'}")

    if interrupted["value"]:
        return 130
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PermissionError as exc:
        print(f"Permission denied: {exc}", file=sys.stderr)
        print("Add the user to the 'i2c' / 'video' groups and log in again.", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
