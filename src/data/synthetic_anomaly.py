"""EXP004 P0 data contracts and blind injection; no model/threshold fitting.

This leaf can be tested without importing the torch-dependent src.data package.
Actual dataset preparation belongs to SERVER-TRAINING. Raw files are read only.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
from collections import Counter, defaultdict

import numpy as np

CHANNELS = ("NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4")
THERMAL_SHAPE = (120, 160)
PROTOCOL = "synthetic-modality-v3-proposed"
QUALITY = "joint_hard_validity_and_thermal_soft_QC_v1"
_NAME = re.compile(r"^((agv|oht)\d+)_(\d{4})_(\d{6})$")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def partition_devices(fold_map, fold):
    """Use the existing assign_folds result; never inspect observations here."""
    expected = {f"{kind}{i:02d}" for kind in ("agv", "oht") for i in range(1, 17)}
    if set(fold_map) != expected or fold not in range(4):
        raise ValueError("EXP004 requires agv01..16/oht01..16 and fold 0..3")
    if any(fold_map[d] not in range(4) for d in fold_map):
        raise ValueError("invalid outer fold")
    roles = {d: ("heldout" if f == fold else "fit") for d, f in fold_map.items()}
    for kind in ("agv", "oht"):
        for f in range(4):
            if sum(d.startswith(kind) and v == f for d, v in fold_map.items()) != 4:
                raise ValueError("outer folds must be balanced by equipment type")
        pool = sorted(d for d, f in fold_map.items() if d.startswith(kind) and f == (fold + 1) % 4)
        for i, d in enumerate(pool):
            roles[d] = "dev" if i in (0, 2) else "calibration"
    if Counter(roles.values()) != {"fit": 16, "dev": 4, "calibration": 4, "heldout": 8}:
        raise ValueError("partition count mismatch")
    return dict(sorted(roles.items()))


def file_identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def read_stable(path):
    path = Path(path)
    before = file_identity(path)
    data = path.read_bytes()
    if file_identity(path) != before:
        raise RuntimeError(f"input changed while reading: {path}")
    return data, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "identity": before}


@dataclass
class Tick:
    base_id: str
    device: str
    day: str
    second: int
    label: int
    sensor: np.ndarray
    thermal_mean: float | None
    thermal_variance: float | None
    hard_errors: tuple[str, ...] = ()
    soft_flags: tuple[str, ...] = ()


def read_tick(source, labels, name, split):
    match = _NAME.fullmatch(name)
    if not match:
        raise ValueError(f"unexpected basename: {name}")
    device, _, day, clock = match.groups()
    parsed = datetime.strptime("2000" + day + clock, "%Y%m%d%H%M%S")
    second = parsed.hour * 3600 + parsed.minute * 60 + parsed.second
    payloads, ledger = {}, []
    for suffix, parent in (("csv", source), ("bin", source), ("json", labels)):
        payloads[suffix], item = read_stable(Path(parent) / f"{name}.{suffix}")
        ledger.append(item)
    rows = list(csv.reader(io.StringIO(payloads["csv"].decode("utf-8-sig"))))
    if len(rows) != 2 or tuple(rows[0]) != CHANNELS or len(rows[1]) != 8:
        raise ValueError(f"invalid CSV schema: {name}")
    sensor = np.asarray(rows[1], dtype=np.float64)
    doc = json.loads(payloads["json"])
    raw_label = doc["annotations"][0]["tagging"][0]["state"]
    if str(raw_label) not in {"0", "1", "2", "3"}:
        raise ValueError(f"invalid severity label: {name}")
    label = int(raw_label)
    meta = doc["meta_info"]
    if not isinstance(meta, list) or len(meta) != 1:
        raise ValueError(f"invalid meta_info schema: {name}")
    meta = meta[0]
    errors = []
    if (meta["device_id"] != device or meta["collection_date"] != parsed.strftime("%m-%d")
            or meta["collection_time"] != parsed.strftime("%H:%M:%S")
            or str(meta["duration_time"]) != "1"):
        errors.append("timestamp_or_device_metadata")
    # BIN files are NPY containers, not display images. Never allow pickle.
    thermal = np.load(io.BytesIO(payloads["bin"]), allow_pickle=False)
    if not isinstance(thermal, np.ndarray) or thermal.dtype.kind not in "fiu":
        raise ValueError(f"thermal must be a numeric NPY array: {name}")
    if thermal.shape != THERMAL_SHAPE:
        errors.append("thermal_shape")
    if not np.isfinite(sensor).all() or not np.isfinite(thermal).all():
        errors.append("nonfinite")
    flags, mean, var = [], None, None
    if not errors:
        thermal = thermal.astype(np.float64, copy=False)
        offset = thermal - thermal.flat[0]
        mean = float(thermal.flat[0] + offset.mean())
        var = float(offset.var(ddof=0))
        if (thermal < 0).any():
            flags.append("thermal_negative_C")
        if ((thermal < 30.98) | (thermal > 146.10)).any():
            flags.append("thermal_outside_legacy_range")
    return Tick(f"{split}/{name}", device, day, second, label, sensor, mean, var,
                tuple(errors), tuple(flags)), ledger


def scan_split(root, split, progress=None):
    """Fail on missing/orphan/schema files; preserve per-tick hard QC errors."""
    source, labels = Path(root) / split / "원천데이터", Path(root) / split / "라벨링데이터"
    sets = [{p.stem for p in parent.glob(f"*.{ext}")}
            for parent, ext in ((source, "csv"), (source, "bin"), (labels, "json"))]
    if not sets[0] or not (sets[0] == sets[1] == sets[2]):
        raise ValueError(f"empty or mismatched CSV/BIN/JSON membership: {split}")
    ticks, ledger = [], []
    for i, name in enumerate(sorted(sets[0])):
        tick, files = read_tick(source, labels, name, split)
        ticks.append(tick)
        ledger.extend(files)
        if progress and (i + 1) % 1000 == 0:
            progress(split, i + 1, len(sets[0]))
    return ticks, ledger


def make_windows(ticks):
    """Original 120 s sessions; reject 30-tick candidates containing any gap."""
    seen, groups = set(), defaultdict(list)
    for t in ticks:
        if t.base_id in seen or t.label not in range(4):
            raise ValueError("duplicate base ID or invalid tick label")
        seen.add(t.base_id)
        groups[t.device, t.day].append(t)
    windows, sessions = [], []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda t: t.second)
        if len({t.second for t in ordered}) != len(ordered):
            raise ValueError(f"duplicate timestamp: {key}")
        chunks, chunk = [], []
        for t in ordered:
            if chunk and t.second - chunk[-1].second > 120:
                chunks.append(chunk)
                chunk = []
            chunk.append(t)
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            sid = chunk[0].base_id
            sessions.append({"session_id": sid, "device": key[0], "ticks": len(chunk)})
            for start in range(0, len(chunk) - 29, 10):
                part = chunk[start:start + 30]
                counts = Counter(t.label for t in part)
                reasons = {e for t in part for e in t.hard_errors}
                if any(b.second - a.second != 1 for a, b in zip(part, part[1:])):
                    reasons.add("non_1Hz_window")
                flags = sorted({f for t in part for f in t.soft_flags})
                pure = next(iter(counts)) if len(counts) == 1 else None
                windows.append({"base_window_id": part[0].base_id + ":30",
                                "session_id": sid, "device": key[0], "start_index": start,
                                "raw_base_ids": [t.base_id for t in part],
                                "pure_class": pure, "pure_Normal": pure == 0,
                                "majority_label": min(counts, key=lambda k: (-counts[k], k)),
                                "mixed": pure is None, "hard_valid": not reasons,
                                "hard_errors": sorted(reasons), "base_QC_flags": flags})
    return windows, sessions


def fit_normalizer(ticks, windows, roles):
    selected = {i for w in windows if roles.get(w["device"]) == "fit"
                and w["hard_valid"] and w["pure_Normal"] for i in w["raw_base_ids"]}
    if not selected:
        raise ValueError("no eligible fit-Normal raw ticks")
    by_id = {t.base_id: t for t in ticks}
    ordered = [by_id[i] for i in sorted(selected)]
    if any(t.label != 0 or t.hard_errors or roles.get(t.device) != "fit" for t in ordered):
        raise ValueError("normalizer fit provenance violation")
    sensor = np.stack([t.sensor for t in ordered])
    means = np.array([t.thermal_mean for t in ordered], dtype=np.float64)
    within = np.array([t.thermal_variance for t in ordered], dtype=np.float64)
    if not np.isfinite(sensor).all() or not np.isfinite(means).all() or not np.isfinite(within).all():
        raise ValueError("nonfinite normalizer input")
    offset = sensor - sensor[0]
    sm, ss = sensor[0] + offset.mean(axis=0), offset.std(axis=0, ddof=0)
    constant = np.ptp(sensor, axis=0) == 0
    ss[constant] = 0.0
    if ((ss == 0) & ~constant).any():
        raise ValueError("nonconstant sensor variance underflow")
    thermal_offset = means - means[0]
    tm = float(means[0] + thermal_offset.mean())
    ts = float(np.sqrt(within.mean() + thermal_offset.var(ddof=0)))
    thermal_constant = bool(np.ptp(means) == 0 and (within == 0).all())
    if ts == 0 and not thermal_constant:
        raise ValueError("nonconstant thermal variance underflow")
    result = {"fit_source": "fit_pure_Normal_unique_raw_ticks_only", "ddof": 0,
              "channels": list(CHANNELS), "sensor_mean": sm.tolist(), "sensor_std": ss.tolist(),
              "sensor_scale": np.where(ss == 0, 1.0, ss).tolist(),
              "constant_channels": [CHANNELS[i] for i in np.flatnonzero(ss == 0)],
              "thermal_mean": tm, "thermal_std": ts, "thermal_scale": ts if ts > 0 else 1.0,
              "constant_rule": "exact_ptp_zero_no_tolerance", "thermal_constant": thermal_constant,
              "fit_devices": sorted({t.device for t in ordered}),
              "unique_raw_ticks": len(selected), "fit_raw_ids": sorted(selected),
              "fit_ids_sha256": digest_json(sorted(selected))}
    result["sha256"] = digest_json(result)
    return result


def partition_accounting(windows, sessions, roles):
    result = {}
    for role in sorted(set(roles.values())):
        devices = sorted(d for d, r in roles.items() if r == role)
        ws = [w for w in windows if w["device"] in devices]
        valid = [w for w in ws if w["hard_valid"]]
        normal = [w for w in valid if w["pure_Normal"]]
        no_normal = sorted(set(devices) - {w["device"] for w in normal})
        result[role] = {"devices": devices,
                        "sessions": sum(s["device"] in devices for s in sessions),
                        "raw_ticks": sum(s["ticks"] for s in sessions if s["device"] in devices),
                        "candidate_windows": len(ws), "hard_valid_windows": len(valid),
                        "hard_invalid_windows": len(ws) - len(valid),
                        "hard_invalid_reasons": dict(Counter(e for w in ws for e in w["hard_errors"])),
                        "pure_Normal_windows": len(normal),
                        "pure_class_windows": {str(k): sum(w["pure_class"] == k for w in valid) for k in range(4)},
                        "mixed_windows": sum(w["mixed"] for w in valid),
                        "unique_Normal_ticks": len({i for w in normal for i in w["raw_base_ids"]}),
                        "devices_without_pure_Normal": no_normal,
                        "soft_QC_windows": sum(bool(w["base_QC_flags"]) for w in valid),
                        "soft_QC_Normal_windows": sum(bool(w["base_QC_flags"]) for w in normal),
                        "soft_QC_flags": dict(Counter(f for w in valid for f in w["base_QC_flags"]))}
    return result


def prepare_manifests(ticks, fold_map):
    windows, sessions = make_windows(ticks)
    if {t.device for t in ticks} != set(fold_map):
        raise ValueError("dataset devices differ from outer fold manifest")
    folds, failures = {}, []
    for fold in range(4):
        roles = partition_devices(fold_map, fold)
        accounting = partition_accounting(windows, sessions, roles)
        unavailable = {r: accounting[r]["devices_without_pure_Normal"]
                       for r in ("fit", "dev", "calibration") if accounting[r]["devices_without_pure_Normal"]}
        if unavailable:
            failures.append({"fold": fold, "no_pure_Normal": unavailable})
        normalizer = None if unavailable else fit_normalizer(ticks, windows, roles)
        folds[str(fold)] = {"roles": roles, "accounting": accounting, "normalizer": normalizer}
    result = {"protocol_id": PROTOCOL, "quality_policy_id": QUALITY,
              "schema_version": "exp004-p0-manifest-v1", "window_ticks": 30, "stride_ticks": 10,
              "session_gap_threshold_seconds": 120, "fold_assignment": fold_map,
              "windows": windows, "sessions": sessions, "folds": folds,
              "data_gate": "PASS" if not failures else "FAIL", "data_gate_failures": failures,
              "full_P0_gate": "NOT_EVALUATED", "heldout_model_performance_computed": False}
    result["sha256"] = digest_json(result)
    return result


def sensor_constraint_errors(sensor):
    errors = []
    if (sensor[:, 1:4] < 0).any() or (sensor[:, 1] > sensor[:, 2]).any() or (sensor[:, 2] > sensor[:, 3]).any():
        errors.append("PM_nonnegative_order")
    if (sensor[:, 4:] < 0).any():
        errors.append("negative_current")
    return errors


def inject_blind(sensor, thermal, *, tick_labels, normalizer, protocol, fold, role,
                 base_window_id, family, strength, generator_seed, ct_channel="CT1"):
    """Make one raw-copy event; rejected candidates are returned with reasons.

    Caller must use the declared partition's pure-Normal view. No severity values
    other than the Normal admission mask are accepted. This API never fits scales.
    """
    if protocol != PROTOCOL or fold not in range(4) or role not in {"fit", "dev", "calibration", "heldout-eval"}:
        raise ValueError("invalid injection namespace")
    if family not in {"sensor-only", "thermal-only", "coupled"} or strength not in (1, 2, 4):
        raise ValueError("event outside preregistered grid")
    if ct_channel not in {"CT1", "CT2"}:
        raise ValueError("unknown CT comparison")
    labels = np.asarray(tick_labels)
    sensor, thermal = np.asarray(sensor, dtype=np.float64), np.asarray(thermal, dtype=np.float64)
    if (sensor.shape != (30, 8) or thermal.shape != (30, *THERMAL_SHAPE)
            or labels.shape != (30,) or not (labels == 0).all()
            or not np.isfinite(sensor).all() or not np.isfinite(thermal).all()):
        raise ValueError("event base must be hard-valid, 30-tick pure Normal")
    std = np.asarray(normalizer["sensor_std"], dtype=np.float64)
    ts = float(normalizer["thermal_std"])
    if std.shape != (8,) or not np.isfinite(std).all() or (std < 0).any() or not np.isfinite(ts) or ts < 0:
        raise ValueError("invalid fit-Normal scales")
    event_key = [protocol, fold, role, base_window_id, family, strength, int(generator_seed)]
    seed_hash = digest_json(event_key)
    rng = np.random.Generator(np.random.PCG64(int(seed_hash[:32], 16)))
    s, t = sensor.copy(), thermal.copy()
    meta = {"event_id": seed_hash, "namespace": event_key, "rng": "PCG64",
            "event_id_scope": "paired_CT_sensitivity",
            "event_instance_id": digest_json([event_key, ct_channel]),
            "family": family, "strength": strength, "ct_channel": ct_channel,
            "normalizer_sha256": normalizer.get("sha256"), "fit_ids_sha256": normalizer.get("fit_ids_sha256"),
            "base_constraint_errors": sensor_constraint_errors(sensor), "base_QC_flags": [],
            "sensor_changed": False, "thermal_changed": False, "parameters": {}}
    if (thermal < 0).any():
        meta["base_QC_flags"].append("thermal_negative_C")
    if ((thermal < 30.98) | (thermal > 146.10)).any():
        meta["base_QC_flags"].append("thermal_outside_legacy_range")
    errors = []
    selected_channels = []
    ct_index = CHANNELS.index(ct_channel)
    if family == "sensor-only":
        # Six preregistered subpatterns, including a shared NTC/current change.
        choices = [[0], [1], [2], [3], [ct_index], [0, ct_index]]
        selected_channels = choices[int(rng.integers(len(choices)))]
    elif family == "coupled":
        selected_channels = [0, ct_index]
    meta["parameters"]["sensor_channels"] = [CHANNELS[c] for c in selected_channels]
    if family == "sensor-only":
        operator = str(rng.choice(["ramp", "step", "spike", "variance", "drift"]))
        duration = 30 if operator in {"variance", "drift"} else int(rng.choice([1, 2, 3] if operator == "spike" else [5, 10, 20]))
        lag = 0
    else:
        operator = "ramp"
        duration = int(rng.choice([5, 10, 20]))
        lag = int(rng.choice([0, 2, 5])) if family == "coupled" else 0
    start = int(rng.integers(0, 30 - duration - lag + 1))
    meta["parameters"].update(operator=operator, start=start, duration=duration, lag=lag)
    if selected_channels:
        envelope = np.ones(duration)
        if operator in {"ramp", "drift"}:
            envelope = np.linspace(0, 1, duration)
        if operator == "variance":
            envelope = rng.standard_normal(duration)
            envelope -= envelope.mean()
            envelope /= envelope.std(ddof=0)
        scales = {}
        for channel in selected_channels:
            amplitude = strength * std[channel]
            if amplitude == 0:
                errors.append("zero_sensor_scale:" + CHANNELS[channel])
            s[start:start + duration, channel] += amplitude * envelope
            scales[CHANNELS[channel]] = float(amplitude)
        meta["parameters"]["sensor_delta_scales"] = scales
    if family != "sensor-only":
        amplitude = strength * ts
        if amplitude == 0:
            errors.append("zero_thermal_scale")
        cy, cx = int(rng.integers(120)), int(rng.integers(160))
        sigma = int(rng.choice([3, 6, 12]))
        yy, xx = np.ogrid[:120, :160]
        blob = amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        thermal_envelope = np.linspace(0, 1, duration)
        t[start + lag:start + lag + duration] += thermal_envelope[:, None, None] * blob
        meta["parameters"].update(thermal_delta_peak=float(amplitude), center_yx=[cy, cx], sigma_pixels=sigma,
                                  thermal_envelope="linear_ramp_0_to_1", near_boundary_3sigma=bool(min(cy, 119-cy, cx, 159-cx) < 3*sigma))
    ds, dt = s - sensor, t - thermal
    meta["sensor_changed"], meta["thermal_changed"] = bool(np.any(ds)), bool(np.any(dt))
    meta["sensor_delta_mean"] = ds.mean(axis=0).tolist()
    meta["sensor_delta_std"] = ds.std(axis=0, ddof=0).tolist()
    meta["thermal_delta_mean"] = float(dt.mean())
    meta["thermal_delta_max"] = float(dt.max())
    meta["post_injection_constraint_errors"] = sensor_constraint_errors(s)
    meta["introduced_constraint_errors"] = sorted(set(meta["post_injection_constraint_errors"]) - set(meta["base_constraint_errors"]))
    if not np.isfinite(s).all() or not np.isfinite(t).all():
        errors.append("nonfinite_after_injection")
    if not (meta["sensor_changed"] or meta["thermal_changed"]):
        errors.append("no_op")
    errors.extend("base:" + e for e in meta["base_constraint_errors"])
    errors.extend("post:" + e for e in meta["post_injection_constraint_errors"])
    meta["rejection_reasons"] = sorted(set(errors))
    meta["accepted"] = not errors
    return s, t, meta
