#!/usr/bin/env python3
"""Build 30-second analysis windows from raw Jetson acquisition trials.

Usage (run by path on a machine without the training dependencies):

    python3 src/data/scripts/build_windows.py --dry-run
    python3 src/data/scripts/build_windows.py \
        --scan-dir dataset/_smoke --scenario normal \
        --out processed/development_baseline_v1 \
        --expect-trials 5 --expect-snapshots 1800 \
        --expect-windows 60 --expect-valid 49 --expect-invalid 11

Or, on the training server where `src.data` is importable:

    python3 -m src.data.scripts.build_windows --dry-run

The raw trial directories are opened read-only and checksummed; nothing under
the dataset root is written, renamed or deleted. Window counts are cross-checked
against the quality block each trial's own collector wrote, and against the
optional --expect-* values. Any mismatch stops the run before anything is
written.

This tool performs no training, no inference, no CT2-4 imputation, no label
generation and no threshold derivation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_window_builder():
    """Import the builder without importing `src.data` (which needs torch)."""
    name = "_window_builder"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO_ROOT / "src" / "data" / "window_builder.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wb = _load_window_builder()

DEFAULT_SCAN_DIR = REPO_ROOT / "dataset" / "_smoke"
DEFAULT_OUT_DIR = REPO_ROOT / "processed" / "development_baseline_v1"


def _git(*args: str) -> str | None:
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def rel(path: Path) -> str:
    """Repo-relative display path, falling back to the absolute path."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-overlapping 30-second windows from raw trials")
    parser.add_argument("trials", nargs="*", type=Path,
                        help="explicit trial directories (default: scan --scan-dir)")
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR,
                        help=f"directory to scan for trials (default {DEFAULT_SCAN_DIR})")
    parser.add_argument("--scenario", default=None,
                        help="only include trials with this scenario_id")
    parser.add_argument("--trial-status", default="completed",
                        help="only include trials with this status (default completed)")
    parser.add_argument("--operator-note-prefix", default=None,
                        help="only include trials whose operator_note starts with this "
                             "(scenario_id alone does not identify a collection batch)")
    parser.add_argument("--min-planned-duration", type=int, default=None,
                        help="only include trials with at least this planned_duration_s")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"processed output directory (default {DEFAULT_OUT_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing non-empty output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and report counts without writing anything")
    parser.add_argument("--no-checksum", action="store_true",
                        help="skip source SHA-256 (faster; weakens provenance)")
    parser.add_argument("--expect-trials", type=int, default=None)
    parser.add_argument("--expect-snapshots", type=int, default=None)
    parser.add_argument("--expect-windows", type=int, default=None)
    parser.add_argument("--expect-valid", type=int, default=None)
    parser.add_argument("--expect-invalid", type=int, default=None)
    return parser.parse_args(argv)


def resolve_trials(args: argparse.Namespace) -> list[Path]:
    if args.trials:
        return [Path(p) for p in args.trials]
    if not args.scan_dir.is_dir():
        raise SystemExit(f"ERROR: scan directory not found: {args.scan_dir}")
    found = wb.discover_trials(
        args.scan_dir,
        scenario=args.scenario,
        status=args.trial_status,
        operator_note_prefix=args.operator_note_prefix,
        min_snapshot_count=args.min_planned_duration,
    )
    if not found:
        raise SystemExit(
            f"ERROR: no trials in {args.scan_dir} "
            f"(scenario={args.scenario}, status={args.trial_status}, "
            f"note_prefix={args.operator_note_prefix!r}, "
            f"min_planned_duration={args.min_planned_duration})")
    return found


def main(argv=None) -> int:
    args = parse_args(argv)
    trial_paths = resolve_trials(args)

    print("=" * 72)
    print("30-second window builder (offline, read-only on raw)")
    print("=" * 72)
    print(f"  policy source     : jetson_deploy/sensors/snapshot.py")
    print(f"  window ticks      : {wb.WINDOW_TICKS}   stride: {wb.STRIDE_TICKS} "
          f"(non-overlapping)")
    print(f"  flir max age      : {wb.FLIR_MAX_AGE_MS} ms")
    print(f"  legacy channels   : {list(wb.LEGACY_MODEL_EXPECTED_CHANNELS)}")
    print(f"       source       : {wb.LEGACY_CHANNEL_SOURCE}")
    print(f"  observed channels : {list(wb.OBSERVED_SCALAR_CHANNELS)}")
    print(f"  trials            : {len(trial_paths)}")
    print()

    # ---------------- pass 1: load, judge quality, cross-check --------------
    trials: list = []
    per_trial: list[dict] = []
    total_snapshots = 0
    total_windows = 0
    total_valid = 0
    mismatches: list[str] = []

    for path in trial_paths:
        try:
            trial = wb.load_trial(path, checksum=not args.no_checksum)
        except wb.TrialError as exc:
            print(f"STOP: {exc}")
            return 2
        trials.append(trial)

        windows = list(wb.iter_trial_windows(trial))
        valid = sum(
            1 for _, snaps in windows
            if wb.POLICY.window_quality(snaps)["valid"]
        )
        invalid = len(windows) - valid
        reported = wb.trial_expected_window_counts(trial)

        total_snapshots += trial.snapshot_count
        total_windows += len(windows)
        total_valid += valid

        for key, derived in (("windows_evaluated", len(windows)),
                             ("windows_valid", valid),
                             ("windows_invalid", invalid)):
            if reported.get(key) is not None and reported[key] != derived:
                mismatches.append(
                    f"{trial.trial_id}: {key} collector={reported[key]} derived={derived}")

        scd30 = (trial.timing_report.get("scd30") or {})
        per_trial.append({
            "trial_id": trial.trial_id,
            "path": rel(trial.path),
            "scenario_id": trial.experiment.get("scenario_id"),
            "severity_level": trial.experiment.get("severity_level"),
            "status": trial.experiment.get("status"),
            "test_mode": trial.experiment.get("test_mode"),
            "sensor_profile": trial.experiment.get("sensor_profile"),
            "source_git_commit": trial.experiment.get("git_commit"),
            "snapshot_count": trial.snapshot_count,
            "windows": len(windows),
            "windows_valid": valid,
            "windows_invalid": invalid,
            "collector_reported_quality": reported,
            "scd30_polling_error_count": scd30.get("error_count"),
            "scd30_age_ms_max": scd30.get("age_ms_max"),
            "source_checksums": trial.checksums,
        })
        print(f"  {trial.trial_id}  snapshots {trial.snapshot_count:4d}  "
              f"windows {len(windows):2d}  valid {valid:2d}  invalid {invalid:2d}  "
              f"collector {reported['windows_valid']}/{reported['windows_invalid']}")

    total_invalid = total_windows - total_valid
    print()
    print(f"  derived totals: trials {len(trials)}  snapshots {total_snapshots}  "
          f"windows {total_windows}  valid {total_valid}  invalid {total_invalid}")

    expectations = {
        "trials": (args.expect_trials, len(trials)),
        "snapshots": (args.expect_snapshots, total_snapshots),
        "windows": (args.expect_windows, total_windows),
        "valid": (args.expect_valid, total_valid),
        "invalid": (args.expect_invalid, total_invalid),
    }
    for name, (expected, derived) in expectations.items():
        if expected is not None and expected != derived:
            mismatches.append(f"expected {name}={expected} but derived {derived}")

    if mismatches:
        print()
        print("STOP: window accounting does not match. Nothing was written.")
        for line in mismatches:
            print(f"  - {line}")
        return 3
    print("  cross-check: derived counts match every collector quality report"
          + (" and all --expect-* values" if any(e is not None for e, _ in expectations.values())
             else ""))

    if args.dry_run:
        print()
        print("  --dry-run: validation only, no output written.")
        return 0

    # ---------------- pass 2: write processed windows ----------------------
    out_dir: Path = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"\nSTOP: {out_dir} is not empty. Use --force to overwrite.")
        return 4
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob("window_*.npz")):
        stale.unlink()

    print()
    print(f"  writing to {rel(out_dir)}")

    records: list[dict] = []
    global_index = 0
    for trial in trials:
        cache: dict = {}
        for window_index, snaps in wb.iter_trial_windows(trial):
            window_id = f"window_{global_index:06d}"
            frames, thermal_info = wb.gather_thermal(trial, snaps, cache)
            arrays = wb.build_window_arrays(snaps, frames)
            crosscheck = wb.quality_crosscheck(snaps)
            record = wb.build_window_record(
                trial, window_index, window_id, snaps,
                thermal_info, arrays, crosscheck)

            npz_path = out_dir / f"{window_id}.npz"
            np.savez_compressed(npz_path, **arrays)
            record["npz_file"] = npz_path.name
            record["npz_sha256"] = wb.sha256_file(npz_path)
            record["npz_bytes"] = npz_path.stat().st_size
            records.append(record)
            global_index += 1
        cache.clear()

    with (out_dir / "windows.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    written_valid = sum(1 for r in records if r["valid_for_training"])
    written_invalid = len(records) - written_valid
    if (written_valid, written_invalid) != (total_valid, total_invalid):
        print("STOP: written window validity differs from pass-1 accounting")
        print(f"  pass1 {total_valid}/{total_invalid}  written {written_valid}/{written_invalid}")
        structural = [r["window_id"] for r in records if r["structural_errors"]]
        print(f"  windows with structural errors: {structural[:10]}")
        return 5

    policy_text = ((trials[0].timing_report.get("quality") or {}).get("policy")
                   or "window is training-invalid when any tick has flir status "
                      "!= ok or age_ms > 500 ms")

    manifest = {
        "processed_schema_version": wb.PROCESSED_SCHEMA_VERSION,
        "dataset_name": out_dir.name,
        "dataset_stage": "development_pilot",
        "dataset_stage_note": (
            "Korea development/pilot baseline collected in test mode. "
            "This is NOT the official German field dataset."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "src/data/scripts/build_windows.py",
        "builder_git_commit": _git("rev-parse", "HEAD"),
        "builder_git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "window_definition": {
            "seconds": 30,
            "master_tick_hz": 1.0,
            "ticks": wb.WINDOW_TICKS,
            "stride_ticks": wb.STRIDE_TICKS,
            "overlapping": False,
            "note": "overlapping windows are deliberately out of scope for v1",
        },
        "quality_policy": {
            "source_file": "jetson_deploy/sensors/snapshot.py",
            "acquisition_schema_version": wb.ACQUISITION_SCHEMA_VERSION,
            # Taken verbatim from the collector's own report so the
            # processed dataset states the policy it was actually judged by.
            "policy": policy_text,
            "flir_max_age_ms": wb.FLIR_MAX_AGE_MS,
            "new_outlier_thresholds_introduced": False,
            "scd30_polling_errors_used_as_invalidation": False,
            "bme680_warmup_used_as_invalidation": False,
        },
        "channel_boundary": {
            "observed_scalar_channels": list(wb.OBSERVED_SCALAR_CHANNELS),
            "legacy_model_expected_channels": list(wb.LEGACY_MODEL_EXPECTED_CHANNELS),
            "legacy_channel_list_source": wb.LEGACY_CHANNEL_SOURCE,
            "legacy_model_ready": False,
            "legacy_model_not_ready_reason": list(wb.LEGACY_NOT_READY_REASONS),
            "ct_missing_channel_handling": (
                "recorded as available=false; never zero-filled, never "
                "replicated from CT1, never interpolated"
            ),
            "context_only_sensors": ["scd30", "bme680"],
            "disabled_sensors": ["sgp30"],
        },
        "thermal": {
            "frames_per_window": wb.WINDOW_TICKS,
            "frame_shape": list(wb.THERMAL_FRAME_SHAPE),
            "stored_dtype": "uint16",
            "celsius_formula": wb.THERMAL_CELSIUS_FORMULA,
            "reconstruction": (
                "frames addressed by the snapshot's own thermal_chunk/"
                "thermal_index; no nearest-frame substitution, no interpolation"
            ),
        },
        "totals": {
            "trials": len(trials),
            "snapshots": total_snapshots,
            "windows": total_windows,
            "windows_valid_for_training": total_valid,
            "windows_invalid": total_invalid,
            "invalid_rate": round(total_invalid / total_windows, 4) if total_windows else None,
            "scd30_polling_errors": sum(
                (t["scd30_polling_error_count"] or 0) for t in per_trial),
        },
        "cross_check": {
            "matches_collector_quality_reports": True,
            "expectations_supplied": {
                name: expected for name, (expected, _) in expectations.items()
                if expected is not None
            },
            "tick_quality_recomputed_matches_acquisition": all(
                r["quality_policy"]["recomputed_matches_acquisition"] for r in records),
        },
        "trial_selection": {
            "scan_dir": str(args.scan_dir),
            "scenario": args.scenario,
            "trial_status": args.trial_status,
            "operator_note_prefix": args.operator_note_prefix,
            "min_planned_duration_s": args.min_planned_duration,
            "explicit_trial_paths": [str(p) for p in args.trials] or None,
        },
        "source_trials": per_trial,
        "raw_mutation": "none - every raw file opened read-only and checksummed",
    }
    atomic_write_json(out_dir / "manifest.json", manifest)

    stats = wb.baseline_statistics(records, out_dir)
    stats["scd30"] = {
        "polling_errors_total": manifest["totals"]["scd30_polling_errors"],
        "per_trial": {t["trial_id"]: t["scd30_polling_error_count"] for t in per_trial},
        "age_ms_max_observed": max(
            (t["scd30_age_ms_max"] or 0.0) for t in per_trial),
        "used_as_window_invalidation": False,
        "note": (
            "no consecutive errors, no age_ms > 4500, no stale/error snapshot "
            "status and no simultaneous SPS30 instability were observed; "
            "recorded as provenance only"
        ),
    }
    atomic_write_json(out_dir / "baseline_stats.json", stats)

    print_report(manifest, stats, records, out_dir)
    return 0


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def print_report(manifest: dict, stats: dict, records: list[dict], out_dir: Path) -> None:
    print()
    print("=" * 72)
    print("WINDOW SUMMARY")
    print("=" * 72)
    totals = manifest["totals"]
    print(f"  trials {totals['trials']}  snapshots {totals['snapshots']}  "
          f"windows {totals['windows']}  valid {totals['windows_valid_for_training']}  "
          f"invalid {totals['windows_invalid']}  "
          f"invalid rate {totals['invalid_rate'] * 100:.1f} %")
    print(f"  tick-quality recompute matches acquisition: "
          f"{manifest['cross_check']['tick_quality_recomputed_matches_acquisition']}")

    reasons: dict[str, int] = {}
    for record in records:
        if record["valid_for_training"]:
            continue
        for reason in record["invalid_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    print(f"  invalid window reasons: {reasons}")

    print()
    print("=" * 72)
    print("BASELINE STATISTICS - valid normal windows (analysis only, NOT thresholds)")
    print("=" * 72)
    header = f"  {'channel':<22} {'w-mean':>11} {'w-std':>10} {'p05':>11} {'p95':>11} {'min':>11} {'max':>11}"
    print(header)
    for channel, block in stats["scalar_channels"].items():
        pooled = block["pooled_tick_distribution"]
        wmean = block["per_window_mean_distribution"]
        print(f"  {channel:<22} {_fmt(wmean['mean']):>11} {_fmt(wmean['std']):>10} "
              f"{_fmt(pooled['p05']):>11} {_fmt(pooled['p95']):>11} "
              f"{_fmt(pooled['min']):>11} {_fmt(pooled['max']):>11}")
    print("  (w-mean/w-std = distribution of the per-window means; "
          "p05/p95/min/max = pooled ticks)")

    print()
    print("=" * 72)
    print("FLIR - valid windows only")
    print("=" * 72)
    for key, dist in stats["flir"]["per_frame_distribution"].items():
        print(f"  per-frame {key:<7} n {dist['count']:5d}  mean {dist['mean']:7.3f}  "
              f"std {dist['std']:6.3f}  p05 {dist['p05']:7.3f}  p95 {dist['p95']:7.3f}  "
              f"min {dist['min']:7.3f}  max {dist['max']:7.3f}")
    diff = stats["flir"]["recompute_vs_acquisition_max_abs_diff_c"]
    print(f"  recompute vs acquisition max |diff|: "
          + "  ".join(f"{k} {v:.4g} C" for k, v in diff.items()))
    print()
    print("  extreme frames by rank (all windows; reporting aid, no threshold):")
    for row in stats["flir"]["extreme_frames_top5_max_c"]:
        print(f"    max_c {row['max_c']:7.2f}  {row['window_id']}  "
              f"{row['source_trial_id'][-9:]}  seq {row['sequence']:3d}  "
              f"age {row['flir_age_ms']:7.1f} ms  tick_valid {row['tick_thermal_valid']}  "
              f"window_valid {row['window_valid_for_training']}")
    for row in stats["flir"]["extreme_frames_bottom5_min_c"]:
        print(f"    min_c {row['min_c']:7.2f}  {row['window_id']}  "
              f"{row['source_trial_id'][-9:]}  seq {row['sequence']:3d}  "
              f"age {row['flir_age_ms']:7.1f} ms  tick_valid {row['tick_thermal_valid']}  "
              f"window_valid {row['window_valid_for_training']}")

    size = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())
    print()
    print(f"  output {rel(out_dir)}: {len(records)} npz + "
          f"manifest.json + windows.jsonl + baseline_stats.json  "
          f"({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
