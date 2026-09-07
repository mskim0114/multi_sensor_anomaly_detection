"""Offline 30-second window builder for real Jetson trial data.

This module turns a raw acquisition trial (produced by
`jetson_deploy/scripts/12_run_trial.py`) into deterministic 30-second analysis
windows. It runs on the SERVER/TRAINING side: it touches no hardware, opens no
bus, and imports nothing from the acquisition runtime except the frozen
data-quality policy.

Three rules shape everything here.

1. RAW IS IMMUTABLE. Every raw file is opened read-only and checksummed. A
   value that was not measured stays absent - it is never zero-filled,
   replicated from a neighbouring channel, or interpolated. A window that
   cannot be reconstructed is reported as an error, never repaired.

2. THE QUALITY POLICY HAS ONE SOURCE. `jetson_deploy/sensors/snapshot.py`
   defined the FFC-driven window policy before collection started. This module
   loads that file and calls its `tick_quality` / `window_quality` directly, so
   the offline judgement cannot drift from the acquisition-time judgement. As a
   cross-check, the recomputed per-tick quality is compared against the block
   the collector already recorded in every snapshot; a mismatch is a hard error.

3. THIS IS NOT A MODEL ADAPTER. The legacy model expects eight scalar channels
   but only five exist physically (CT2/CT3/CT4 have no front-end). This module
   records that gap as metadata - `legacy_model_ready: false` plus explicit
   reasons - and stops there. Producing the legacy 8-channel tensor is a
   separate, later decision.

Time base: the master tick is 1.0 s and one window is 30 consecutive ticks with
stride 30 (non-overlapping). Overlapping windows are deliberately out of scope.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_SCHEMA_VERSION = 1

# Raw files a trial must carry. Anything missing means the trial is not a
# complete acquisition run and is refused rather than partially processed.
REQUIRED_TRIAL_FILES = (
    "experiment.json",
    "metadata.json",
    "scalars.csv",
    "snapshots.jsonl",
    "timing_report.json",
)

THERMAL_FRAME_SHAPE = (120, 160)

# Lepton radiometric conversion, recorded rather than applied: frames are stored
# exactly as acquired (uint16 centi-kelvin) so a processed window stays
# bit-identical to its source chunk.
THERMAL_CELSIUS_FORMULA = "celsius = raw_uint16 / 100.0 - 273.15"

# What physically exists on this rig, and what the frozen legacy model wants.
# The difference is metadata, never something to fill in.
OBSERVED_SCALAR_CHANNELS = ("NTC", "PM1.0", "PM2.5", "PM10", "CT1")
LEGACY_MODEL_EXPECTED_CHANNELS_FALLBACK = (
    "NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4",
)

# Per-tick scalar series preserved at full 1 Hz resolution. Never reduced to a
# per-window mean: a 30-tick trajectory is the point of a window.
MODEL_CANDIDATE_SERIES = (
    ("ntc_temperature_c", "ntc", "temperature_c"),
    ("pm1_0_ug_m3", "sps30", "pm1_0_ug_m3"),
    ("pm2_5_ug_m3", "sps30", "pm2_5_ug_m3"),
    ("pm10_ug_m3", "sps30", "pm10_ug_m3"),
    ("ct1_vrms", "ct1", "vrms"),
    ("ct1_current_a_nominal", "ct1", "current_a_nominal"),
)

CONTEXT_SERIES = (
    ("pm4_0_ug_m3", "sps30", "pm4_0_ug_m3"),
    ("scd30_co2_ppm", "scd30", "co2_ppm"),
    ("scd30_temperature_c", "scd30", "temperature_c"),
    ("scd30_humidity_pct", "scd30", "humidity_pct"),
    ("bme680_temperature_c", "bme680", "temperature_c"),
    ("bme680_humidity_pct", "bme680", "humidity_pct"),
    ("bme680_pressure_hpa", "bme680", "pressure_hpa"),
    ("bme680_gas_ohm", "bme680", "gas_ohm"),
)

FLIR_SERIES = (("flir_min_c", "min_c"), ("flir_mean_c", "mean_c"), ("flir_max_c", "max_c"))

# Sensors whose native rate is slower than the master tick. Their `fresh` flag
# and `age_ms` are carried through unchanged; a repeated value stays a repeated
# value.
FRESHNESS_SENSORS = ("sps30", "scd30", "flir")

# Statuses are preserved per tick so downstream code can see exactly what the
# acquisition layer saw, including `disabled` and `stale`.
STATUS_SENSORS = ("ntc", "ct1", "ct2", "ct3", "ct4", "sps30", "sgp30", "scd30", "bme680", "flir")

# Channels reported in the baseline statistics (section 11 of the request).
STAT_CHANNELS = (
    "ntc_temperature_c",
    "ct1_vrms",
    "ct1_current_a_nominal",
    "pm1_0_ug_m3",
    "pm2_5_ug_m3",
    "pm10_ug_m3",
)


# --------------------------------------------------------------------------
# Frozen policy, loaded from the acquisition layer by file path
# --------------------------------------------------------------------------
def _load_module_by_path(name: str, path: Path) -> ModuleType:
    """Import a single .py file without importing its package.

    `src/data/__init__.py` pulls in torch and `jetson_deploy` is not a package,
    so neither policy nor channel list can be reached through a normal import
    on a machine that has only the acquisition dependencies installed.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: a module defining dataclasses needs to find
    # itself in sys.modules while its class bodies are being processed.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


_POLICY_PATH = REPO_ROOT / "jetson_deploy" / "sensors" / "snapshot.py"
POLICY = _load_module_by_path("_acquisition_snapshot_policy", _POLICY_PATH)

WINDOW_TICKS: int = POLICY.WINDOW_TICKS
STRIDE_TICKS: int = WINDOW_TICKS  # non-overlapping; overlap is a separate experiment
FLIR_MAX_AGE_MS: float = POLICY.FLIR_MAX_AGE_MS
ACQUISITION_SCHEMA_VERSION: int = POLICY.SCHEMA_VERSION


def legacy_model_channels() -> tuple[tuple[str, ...], str]:
    """Read the frozen model channel list from its own definition, read-only."""
    config_path = REPO_ROOT / "src" / "data" / "config.py"
    try:
        config = _load_module_by_path("_frozen_data_config", config_path)
        return tuple(config.SENSOR_CHANNELS), "src/data/config.py:SENSOR_CHANNELS"
    except Exception as exc:  # pragma: no cover - fallback only
        return LEGACY_MODEL_EXPECTED_CHANNELS_FALLBACK, f"fallback literal ({exc})"


LEGACY_MODEL_EXPECTED_CHANNELS, LEGACY_CHANNEL_SOURCE = legacy_model_channels()

LEGACY_MISSING_CHANNELS = tuple(
    ch for ch in LEGACY_MODEL_EXPECTED_CHANNELS if ch not in OBSERVED_SCALAR_CHANNELS
)
LEGACY_NOT_READY_REASONS = tuple(f"{ch}_missing" for ch in LEGACY_MISSING_CHANNELS)


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """Repo-relative path for provenance records, absolute if outside the repo."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


class TrialError(RuntimeError):
    """A trial cannot be processed. Never recovered from silently."""


@dataclass
class Trial:
    path: Path
    trial_id: str
    experiment: dict
    metadata: dict
    timing_report: dict
    snapshots: list[dict]
    checksums: dict[str, str]

    @property
    def snapshot_count(self) -> int:
        return len(self.snapshots)


def load_trial(path: Path, *, checksum: bool = True) -> Trial:
    """Read one trial directory read-only and validate its internal structure."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise TrialError(f"{path}: not a directory")

    missing = [name for name in REQUIRED_TRIAL_FILES if not (path / name).is_file()]
    if missing:
        raise TrialError(f"{path.name}: missing raw files {missing}")

    experiment = json.loads((path / "experiment.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    timing_report = json.loads((path / "timing_report.json").read_text())

    snapshots = []
    with (path / "snapshots.jsonl").open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                snapshots.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TrialError(f"{path.name}: snapshots.jsonl line {line_no}: {exc}") from exc

    if not snapshots:
        raise TrialError(f"{path.name}: snapshots.jsonl is empty")

    # Sequences must be contiguous from 0. A gap means the master tick loop
    # dropped a tick, and windows would silently span a discontinuity.
    sequences = [s["sequence"] for s in snapshots]
    expected = list(range(len(sequences)))
    if sequences != expected:
        gaps = [seq for seq, exp in zip(sequences, expected) if seq != exp][:5]
        raise TrialError(
            f"{path.name}: snapshot sequences are not contiguous from 0 "
            f"(first divergences at {gaps})"
        )

    for snap in snapshots:
        if snap.get("schema_version") != ACQUISITION_SCHEMA_VERSION:
            raise TrialError(
                f"{path.name}: snapshot {snap.get('sequence')} has schema_version "
                f"{snap.get('schema_version')}, expected {ACQUISITION_SCHEMA_VERSION}"
            )

    checksums: dict[str, str] = {}
    if checksum:
        for name in REQUIRED_TRIAL_FILES:
            checksums[name] = sha256_file(path / name)
        for chunk in sorted(path.glob("thermal_*.npz")):
            checksums[chunk.name] = sha256_file(chunk)

    trial_id = experiment.get("trial_id") or path.name
    return Trial(
        path=path,
        trial_id=trial_id,
        experiment=experiment,
        metadata=metadata,
        timing_report=timing_report,
        snapshots=snapshots,
        checksums=checksums,
    )


def discover_trials(
    scan_dir: Path,
    *,
    scenario: str | None = None,
    status: str | None = "completed",
    operator_note_prefix: str | None = None,
    min_snapshot_count: int | None = None,
) -> list[Path]:
    """Find trial directories under `scan_dir`, sorted by trial id.

    Selection is explicit on purpose. A scenario id alone is not a batch
    identifier: `dataset/_smoke` also holds short orchestration tests that share
    the `normal` scenario id but are not part of any baseline batch. Whatever
    filters are used get recorded in the manifest so the batch is reproducible.
    """
    found = []
    for child in sorted(Path(scan_dir).resolve().iterdir()):
        if not child.is_dir() or not (child / "experiment.json").is_file():
            continue
        try:
            experiment = json.loads((child / "experiment.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if scenario is not None and experiment.get("scenario_id") != scenario:
            continue
        if status is not None and experiment.get("status") != status:
            continue
        if operator_note_prefix is not None and not (
                experiment.get("operator_note") or "").startswith(operator_note_prefix):
            continue
        if min_snapshot_count is not None:
            planned = experiment.get("planned_duration_s") or 0
            if planned < min_snapshot_count:
                continue
        found.append(child)
    return found


# --------------------------------------------------------------------------
# Window construction
# --------------------------------------------------------------------------
def _value(snapshot: dict, sensor: str, key: str):
    obs = snapshot["sensors"].get(sensor) or {}
    return (obs.get("values") or {}).get(key)


def _float_series(snapshots: list[dict], sensor: str, key: str) -> np.ndarray:
    """One channel as float64 with NaN where the sensor reported no value.

    NaN means "not measured". It is a marker, not a number to be filled in
    later by this layer.
    """
    out = np.full(len(snapshots), np.nan, dtype=np.float64)
    for i, snap in enumerate(snapshots):
        val = _value(snap, sensor, key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[i] = float(val)
    return out


def _flir_series(snapshots: list[dict], key: str) -> np.ndarray:
    out = np.full(len(snapshots), np.nan, dtype=np.float64)
    for i, snap in enumerate(snapshots):
        val = (snapshots[i]["sensors"].get("flir") or {}).get("values", {}).get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[i] = float(val)
    return out


def _channel_stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def channel_availability(snapshots: list[dict]) -> dict:
    """Per-channel availability, taken from what the sensors actually reported.

    CT2/CT3/CT4 appear here as `available: false` with the acquisition layer's
    own reason string. They are never zero-filled, never copied from CT1, and
    never interpolated.
    """
    mapping = {
        "NTC": "ntc",
        "PM1.0": "sps30",
        "PM2.5": "sps30",
        "PM10": "sps30",
        "CT1": "ct1",
        "CT2": "ct2",
        "CT3": "ct3",
        "CT4": "ct4",
    }
    out: dict[str, dict] = {}
    for channel, sensor in mapping.items():
        statuses = {(snap["sensors"].get(sensor) or {}).get("status") for snap in snapshots}
        reasons = {
            (snap["sensors"].get(sensor) or {}).get("reason")
            for snap in snapshots
            if (snap["sensors"].get(sensor) or {}).get("reason")
        }
        available = statuses.issubset({POLICY.STATUS_OK, POLICY.STATUS_STALE})
        entry: dict = {
            "available": bool(available),
            "statuses_observed": sorted(s for s in statuses if s is not None),
        }
        if reasons:
            entry["reason"] = sorted(reasons)[0]
        out[channel] = entry
    return out


def _load_thermal_chunk(trial: Trial, chunk_name: str, cache: dict) -> dict:
    if chunk_name not in cache:
        chunk_path = trial.path / chunk_name
        if not chunk_path.is_file():
            raise TrialError(f"{trial.trial_id}: thermal chunk {chunk_name} is missing")
        with np.load(chunk_path) as data:
            cache[chunk_name] = {
                "frames": data["frames"],
                "sequences": data["sequences"],
            }
    return cache[chunk_name]


def gather_thermal(trial: Trial, snapshots: list[dict], cache: dict) -> tuple[np.ndarray, dict]:
    """Reconstruct the window's 30 frames by following thermal_chunk/thermal_index.

    Frames are addressed exactly as the acquisition layer recorded them. There
    is no nearest-frame substitution and no interpolation: if a referenced frame
    is absent or its stored sequence does not match the snapshot, the window is
    reported as a structural error.
    """
    frames = np.zeros((len(snapshots), *THERMAL_FRAME_SHAPE), dtype=np.uint16)
    chunk_sequences = np.full(len(snapshots), -1, dtype=np.int64)
    chunks_used: list[str] = []
    errors: list[str] = []

    for i, snap in enumerate(snapshots):
        flir = snap["sensors"].get("flir") or {}
        chunk_name = flir.get("thermal_chunk")
        index = flir.get("thermal_index")
        if chunk_name is None or index is None:
            errors.append(f"seq{snap['sequence']}_no_thermal_reference")
            continue
        chunk = _load_thermal_chunk(trial, chunk_name, cache)
        stored = chunk["frames"]
        if index < 0 or index >= stored.shape[0]:
            errors.append(f"seq{snap['sequence']}_thermal_index_out_of_range")
            continue
        frame = stored[index]
        if frame.shape != THERMAL_FRAME_SHAPE:
            errors.append(f"seq{snap['sequence']}_frame_shape_{frame.shape}")
            continue
        stored_seq = int(chunk["sequences"][index])
        if stored_seq != snap["sequence"]:
            errors.append(
                f"seq{snap['sequence']}_chunk_sequence_mismatch_{stored_seq}"
            )
            continue
        frames[i] = frame
        chunk_sequences[i] = stored_seq
        if chunk_name not in chunks_used:
            chunks_used.append(chunk_name)

    shape_ok = frames.shape == (WINDOW_TICKS, *THERMAL_FRAME_SHAPE)
    if not shape_ok:
        errors.append(f"window_thermal_shape_{frames.shape}")

    info = {
        "shape": list(frames.shape),
        "dtype": str(frames.dtype),
        "celsius_formula": THERMAL_CELSIUS_FORMULA,
        "chunks": chunks_used,
        "sequence_match": bool(not errors),
        "errors": errors,
    }
    return frames, info


def quality_crosscheck(snapshots: list[dict]) -> list[str]:
    """Recompute per-tick quality and compare with what the collector recorded.

    Agreement proves the offline judgement uses the same policy the acquisition
    layer used. Disagreement is a hard error, not something to reconcile.
    """
    mismatches: list[str] = []
    for snap in snapshots:
        recomputed = POLICY.tick_quality(snap["sensors"])
        recorded = snap.get("quality") or {}
        for key in ("flir_frame_valid", "invalid_reasons", "model_channel_issues"):
            if recomputed.get(key) != recorded.get(key):
                mismatches.append(
                    f"seq{snap['sequence']}_{key}"
                )
                break
    return mismatches


def build_window_arrays(snapshots: list[dict], frames: np.ndarray) -> dict[str, np.ndarray]:
    """Full-resolution arrays for one window. No aggregation, no imputation."""
    arrays: dict[str, np.ndarray] = {
        "sequence": np.array([s["sequence"] for s in snapshots], dtype=np.int64),
        "timestamp_utc": np.array([s["timestamp_utc"] for s in snapshots], dtype="U32"),
        "tick_jitter_ms": np.array(
            [float(s.get("tick_jitter_ms") or 0.0) for s in snapshots], dtype=np.float64
        ),
        "tick_thermal_valid": np.array(
            [bool((s.get("quality") or {}).get("flir_frame_valid", False)) for s in snapshots],
            dtype=bool,
        ),
        "thermal_raw_u16": frames,
    }

    for name, sensor, key in MODEL_CANDIDATE_SERIES + CONTEXT_SERIES:
        arrays[name] = _float_series(snapshots, sensor, key)

    for name, key in FLIR_SERIES:
        arrays[name] = _flir_series(snapshots, key)

    # Freshness and age carried through verbatim: a slower sensor repeating its
    # value is a fact about the sensor, not a defect to smooth away.
    for sensor in FRESHNESS_SENSORS:
        arrays[f"{sensor}_fresh"] = np.array(
            [bool((s["sensors"].get(sensor) or {}).get("fresh", False)) for s in snapshots],
            dtype=bool,
        )
        arrays[f"{sensor}_age_ms"] = np.array(
            [
                float((s["sensors"].get(sensor) or {}).get("age_ms", np.nan))
                if (s["sensors"].get(sensor) or {}).get("age_ms") is not None
                else np.nan
                for s in snapshots
            ],
            dtype=np.float64,
        )

    for sensor in STATUS_SENSORS:
        arrays[f"{sensor}_status"] = np.array(
            [str((s["sensors"].get(sensor) or {}).get("status", "")) for s in snapshots],
            dtype="U12",
        )
    return arrays


def build_window_record(
    trial: Trial,
    window_index: int,
    window_id: str,
    snapshots: list[dict],
    thermal_info: dict,
    arrays: dict[str, np.ndarray],
    quality_mismatches: list[str],
) -> dict:
    """One window's metadata: quality, provenance and the model-schema boundary."""
    quality = POLICY.window_quality(snapshots)

    structural: list[str] = []
    structural.extend(thermal_info.get("errors") or [])
    if quality_mismatches:
        structural.append("tick_quality_mismatch")
    if len(snapshots) != WINDOW_TICKS:
        structural.append("incomplete_window")

    invalid_reasons = sorted(quality["reasons"].keys()) + structural
    valid_for_training = bool(quality["valid"]) and not structural

    experiment = trial.experiment
    gas = arrays["bme680_gas_ohm"]
    first_window = snapshots[0]["sequence"] == 0

    record = {
        "processed_schema_version": PROCESSED_SCHEMA_VERSION,
        "window_id": window_id,
        "window_index_in_trial": window_index,
        "source_trial_id": trial.trial_id,
        "source_trial_path": _rel(trial.path),
        "source_git_commit": experiment.get("git_commit"),
        "source_git_branch": experiment.get("git_branch"),
        "sensor_profile": experiment.get("sensor_profile"),
        "scenario_id": experiment.get("scenario_id"),
        "severity_level": experiment.get("severity_level"),
        "test_mode": experiment.get("test_mode"),
        "dataset_stage": "development_pilot",
        "dataset_stage_note": (
            "Korea development/pilot data collected in test mode; "
            "not the official German field dataset"
        ),

        "start_sequence": snapshots[0]["sequence"],
        "end_sequence": snapshots[-1]["sequence"],
        "start_timestamp_utc": snapshots[0]["timestamp_utc"],
        "end_timestamp_utc": snapshots[-1]["timestamp_utc"],
        "tick_count": len(snapshots),
        "stride_ticks": STRIDE_TICKS,

        "window_quality": quality,
        "valid_for_training": valid_for_training,
        "invalid_reasons": invalid_reasons,
        "structural_errors": structural,
        "quality_policy": {
            "source": "jetson_deploy/sensors/snapshot.py",
            "window_ticks": WINDOW_TICKS,
            "flir_max_age_ms": FLIR_MAX_AGE_MS,
            "recomputed_matches_acquisition": not quality_mismatches,
        },

        "observed_scalar_channels": list(OBSERVED_SCALAR_CHANNELS),
        "legacy_model_expected_channels": list(LEGACY_MODEL_EXPECTED_CHANNELS),
        "legacy_model_ready": False,
        "legacy_model_not_ready_reason": list(LEGACY_NOT_READY_REASONS),
        "legacy_channel_list_source": LEGACY_CHANNEL_SOURCE,
        "channel_availability": channel_availability(snapshots),

        "thermal": {
            key: value for key, value in thermal_info.items() if key != "errors"
        },

        "context": {
            "scd30_fresh_tick_count": int(arrays["scd30_fresh"].sum()),
            "scd30_non_ok_tick_count": int(
                (arrays["scd30_status"] != POLICY.STATUS_OK).sum()
            ),
            "sps30_fresh_tick_count": int(arrays["sps30_fresh"].sum()),
            "sgp30_statuses": sorted(set(arrays["sgp30_status"].tolist())),
            # Flagged as context only. There is no documented BME680 gas warm-up
            # duration for this rig, so no duration is invented here and no
            # window is invalidated for it.
            "bme680_gas_warmup_present": bool(first_window),
            "bme680_gas_warmup_basis": "first_window_of_trial",
            "bme680_gas_warmup_note": (
                "no documented or measured warm-up duration; reported as "
                "first-window context, never used to invalidate a window"
            ),
            "bme680_gas_ohm_min": _channel_stats(gas)["min"],
            "bme680_gas_ohm_max": _channel_stats(gas)["max"],
        },

        "scalar_stats": {
            name: _channel_stats(arrays[name])
            for name, _, _ in MODEL_CANDIDATE_SERIES
        },
        "flir_stats": {
            name: _channel_stats(arrays[name]) for name, _ in FLIR_SERIES
        },

        "source_provenance": {
            "snapshots_jsonl_sha256": trial.checksums.get("snapshots.jsonl"),
            "experiment_json_sha256": trial.checksums.get("experiment.json"),
            "metadata_json_sha256": trial.checksums.get("metadata.json"),
            "thermal_chunk_sha256": {
                name: trial.checksums.get(name)
                for name in (thermal_info.get("chunks") or [])
            },
        },
    }
    if quality_mismatches:
        record["quality_mismatch_detail"] = quality_mismatches[:10]
    return record


def iter_trial_windows(trial: Trial):
    """Yield non-overlapping 30-tick slices. A trailing partial slice is dropped."""
    total = trial.snapshot_count
    for start in range(0, total - WINDOW_TICKS + 1, STRIDE_TICKS):
        yield start // STRIDE_TICKS, trial.snapshots[start:start + WINDOW_TICKS]


def trial_expected_window_counts(trial: Trial) -> dict:
    """What the collector itself reported, for the section-16 cross-check."""
    quality = trial.timing_report.get("quality") or {}
    return {
        "windows_evaluated": quality.get("windows_evaluated"),
        "windows_valid": quality.get("windows_valid"),
        "windows_invalid": quality.get("windows_invalid"),
        "invalid_tick_count": quality.get("invalid_tick_count"),
    }


# --------------------------------------------------------------------------
# Statistics (analysis report only - never an anomaly threshold)
# --------------------------------------------------------------------------
def distribution(values) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def baseline_statistics(records: list[dict], out_dir: Path) -> dict:
    """Descriptive statistics over the valid windows.

    This is an analysis report. It defines no threshold, no anomaly boundary and
    no exclusion rule, and it must not be used as one.
    """
    valid = [r for r in records if r["valid_for_training"]]
    invalid = [r for r in records if not r["valid_for_training"]]

    scalar: dict[str, dict] = {}
    for channel in STAT_CHANNELS:
        per_window = [r["scalar_stats"][channel] for r in valid]
        pooled: list[float] = []
        for record in valid:
            with np.load(out_dir / record["npz_file"]) as data:
                pooled.extend(data[channel].tolist())
        scalar[channel] = {
            "per_window_mean_distribution": distribution(s["mean"] for s in per_window),
            "per_window_std_distribution": distribution(s["std"] for s in per_window),
            "per_window_min_distribution": distribution(s["min"] for s in per_window),
            "per_window_max_distribution": distribution(s["max"] for s in per_window),
            "pooled_tick_distribution": distribution(pooled),
        }

    # FLIR: per-frame statistics inside valid thermal windows, plus a recompute
    # check against the values the acquisition layer stored.
    flir_per_frame = {"min_c": [], "mean_c": [], "max_c": []}
    flir_window_mean = {"min_c": [], "mean_c": [], "max_c": []}
    recompute_max_abs_diff = {"min_c": 0.0, "mean_c": 0.0, "max_c": 0.0}
    for record in valid:
        with np.load(out_dir / record["npz_file"]) as data:
            frames = data["thermal_raw_u16"].astype(np.float64) / 100.0 - 273.15
            derived = {
                "min_c": frames.reshape(frames.shape[0], -1).min(axis=1),
                "mean_c": frames.reshape(frames.shape[0], -1).mean(axis=1),
                "max_c": frames.reshape(frames.shape[0], -1).max(axis=1),
            }
            for key in flir_per_frame:
                stored = data[f"flir_{key}"]
                flir_per_frame[key].extend(derived[key].tolist())
                flir_window_mean[key].append(float(derived[key].mean()))
                finite = np.isfinite(stored)
                if finite.any():
                    diff = float(np.abs(derived[key][finite] - stored[finite]).max())
                    recompute_max_abs_diff[key] = max(recompute_max_abs_diff[key], diff)

    # Extreme frames by rank, across every window. Ranking is a reporting aid;
    # it introduces no threshold and excludes nothing.
    extremes: list[dict] = []
    for record in records:
        with np.load(out_dir / record["npz_file"]) as data:
            for i, seq in enumerate(data["sequence"].tolist()):
                extremes.append({
                    "window_id": record["window_id"],
                    "source_trial_id": record["source_trial_id"],
                    "sequence": int(seq),
                    "min_c": float(data["flir_min_c"][i]),
                    "mean_c": float(data["flir_mean_c"][i]),
                    "max_c": float(data["flir_max_c"][i]),
                    "flir_age_ms": float(data["flir_age_ms"][i]),
                    "tick_thermal_valid": bool(data["tick_thermal_valid"][i]),
                    "window_valid_for_training": record["valid_for_training"],
                })

    top_max = sorted(extremes, key=lambda e: -e["max_c"])[:5]
    bottom_min = sorted(extremes, key=lambda e: e["min_c"])[:5]

    return {
        "note": (
            "Descriptive analysis of development/pilot normal windows. "
            "These numbers are NOT anomaly thresholds and define no exclusion rule."
        ),
        "window_counts": {
            "total": len(records),
            "valid_for_training": len(valid),
            "invalid": len(invalid),
            "invalid_rate": round(len(invalid) / len(records), 4) if records else None,
        },
        "scalar_channels": scalar,
        "flir": {
            "valid_windows": len(valid),
            "per_frame_distribution": {
                key: distribution(values) for key, values in flir_per_frame.items()
            },
            "per_window_mean_distribution": {
                key: distribution(values) for key, values in flir_window_mean.items()
            },
            "recompute_vs_acquisition_max_abs_diff_c": recompute_max_abs_diff,
            "extreme_frames_top5_max_c": top_max,
            "extreme_frames_bottom5_min_c": bottom_min,
        },
    }
