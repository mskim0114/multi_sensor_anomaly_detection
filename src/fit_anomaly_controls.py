#!/usr/bin/env python3
"""Connect EXP004 fit/dev/cal Normal views to four controls and blind calibration.

The core accepts a raw-window reader for fixture validation. Server execution reads
only manifest-admitted Normal windows and verifies their raw content hashes again.
Held-out scoring, model selection, neural training and threshold bootstrap are absent.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def leaf(name, relative):
    key = "_exp004_controls_" + name
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


sa = leaf("data", "src/data/synthetic_anomaly.py")
ab = leaf("baselines", "src/models/anomaly_baselines.py")
# Calibration is loaded when used, so independent manifest tools remain usable.


def normalized_pair(sensor, thermal, normalizer, ct_channel):
    ix = [0, 1, 2, 3, sa.CHANNELS.index(ct_channel)]
    s = (np.asarray(sensor) - np.asarray(normalizer["sensor_mean"])) / np.asarray(normalizer["sensor_scale"])
    t = (np.asarray(thermal) - normalizer["thermal_mean"]) / normalizer["thermal_scale"]
    return s[:, ix], t


def fit_controls(windows, roles, normalizer, load_window, *, fold, seed=42, ct_channel="CT1", progress=None):
    """Fit declared views, retaining every attempted event and rejection reason."""
    ac = leaf("calibration", "src/evaluation/anomaly_calibration.py")
    if ct_channel not in {"CT1", "CT2"}:
        raise ValueError("CT sensitivity must be explicitly CT1 or CT2")
    ncopy = dict(normalizer)
    claimed = ncopy.pop("sha256")
    if sa.digest_json(ncopy) != claimed:
        raise ValueError("normalizer content hash mismatch")
    fit_ids = {i for w in windows if roles[w["device"]] == "fit" and w["hard_valid"] and w["pure_Normal"]
               for i in w["raw_base_ids"]}
    if fit_ids != set(normalizer["fit_raw_ids"]) or sa.digest_json(sorted(fit_ids)) != normalizer["fit_ids_sha256"]:
        raise ValueError("normalizer raw tick provenance differs from fit Normal view")
    views = {role: [w for w in windows if roles[w["device"]] == role and w["hard_valid"] and w["pure_Normal"]]
             for role in ("fit", "dev", "calibration")}
    for role, ws in views.items():
        if {w["device"] for w in ws} != {d for d, r in roles.items() if r == role}:
            raise ValueError(f"missing pure-Normal device in {role}")

    def checked_window(w):
        raw_s, raw_t, tick_labels = load_window(w)
        tick_labels = np.asarray(tick_labels)
        if tick_labels.shape != (30,) or not (tick_labels == 0).all():
            raise ValueError("actual reader labels are not 30-tick pure Normal")
        return raw_s, raw_t, tick_labels

    fit_s = []
    for i, w in enumerate(views["fit"]):
        raw_s, raw_t, _ = checked_window(w)
        s, _ = normalized_pair(raw_s, raw_t, normalizer, ct_channel)
        fit_s.append(s)
        if progress and (i + 1) % 100 == 0:
            progress("fit", i + 1, len(views["fit"]))
    md = ab.MahalanobisBaseline.fit(np.stack(fit_s), [w["device"] for w in views["fit"]], role="fit",
                                    channel_names=["NTC", "PM1.0", "PM2.5", "PM10", ct_channel])

    def base_scores(raw_s, raw_t):
        s, t = normalized_pair(raw_s, raw_t, normalizer, ct_channel)
        return {"B-NTC": float(ab.score_b_ntc(s[None])[0]),
                "B-MD": float(md.score(s[None])[0]),
                "B-THERM": float(ab.score_b_therm(t[None])[0])}

    dev = []
    for w in views["dev"]:
        raw_s, raw_t, _ = checked_window(w)
        dev.append(base_scores(raw_s, raw_t))
    late = ab.LateFusionBaseline.fit([d["B-MD"] for d in dev], [d["B-THERM"] for d in dev],
                                     [w["device"] for w in views["dev"]], role="dev")

    def scores(raw_s, raw_t):
        result = base_scores(raw_s, raw_t)
        result["B-LATE"] = float(late.score([result["B-MD"]], [result["B-THERM"]])[0])
        return result

    normal, events = [], []
    for i, w in enumerate(views["calibration"]):
        raw_s, raw_t, tick_labels = checked_window(w)
        common = {"base_window_id": w["base_window_id"], "device": w["device"],
                  "session_id": w["session_id"], "start_index": w["start_index"],
                  "base_QC_flags": w["base_QC_flags"]}
        normal.append({**common, "scores": scores(raw_s, raw_t)})
        for family in ("sensor-only", "thermal-only", "coupled"):
            for strength in (1, 2, 4):
                s, t, event = sa.inject_blind(raw_s, raw_t, tick_labels=tick_labels,
                                             normalizer=normalizer, protocol=sa.PROTOCOL, fold=fold,
                                             role="calibration", base_window_id=w["base_window_id"],
                                             family=family, strength=strength, generator_seed=seed,
                                             ct_channel=ct_channel)
                if event["base_QC_flags"] != w["base_QC_flags"]:
                    raise ValueError("raw thermal QC no longer matches the prepared base")
                events.append({**common, **event, "scores": scores(s, t) if event["accepted"] else None})
        if progress and (i + 1) % 25 == 0:
            progress("calibration", i + 1, len(views["calibration"]))
    accepted = [e for e in events if e["accepted"]]
    result = {"protocol_id": sa.PROTOCOL, "fold": fold, "seed": seed, "ct_channel": ct_channel,
              "normalizer_sha256": claimed, "B-MD": md.to_dict(), "B-MD_sha256": md.params_sha256(),
              "B-LATE": late.to_dict(), "B-LATE_sha256": late.params_sha256(),
              "fit_window_ids": [w["base_window_id"] for w in views["fit"]],
              "dev_window_ids": [w["base_window_id"] for w in views["dev"]],
              "calibration_Normal": normal, "calibration_events": events,
              "event_accounting": {"attempted": len(events), "accepted": len(accepted),
                                   "rejected": len(events)-len(accepted),
                                   "rejection_reasons": dict(sa.Counter(r for e in events for r in e["rejection_reasons"]))},
              "calibration": {}, "calibration_gate": "PASS", "calibration_diagnostics": {},
              "full_P0_gate": "NOT_EVALUATED",
              "pending": ["threshold_block_bootstrap", "fit_dev_event_bank_validation", "remaining_folds_and_seeds"],
              "heldout_scored": False, "gradient_training_runs": 0}
    for model in ("B-NTC", "B-MD", "B-THERM", "B-LATE"):
        try:
            result["calibration"][model] = ac.calibrate_with_sensitivity(
                [r["scores"][model] for r in normal], [r["device"] for r in normal],
                [r["scores"][model] for r in accepted], [r["device"] for r in accepted],
                [r["family"] for r in accepted], [r["strength"] for r in accepted], role="calibration")
            primary = result["calibration"][model]["primary"]
            flags = {policy: {key: value for key, value in primary[policy].items()
                              if key.startswith("degenerate_") and value is True}
                     for policy in ("NQ", "SYN")}
            degraded = primary["SYN"].get("synthetic", {}).get("weighting_degraded_by_missing_cells", False)
            result["calibration_diagnostics"][model] = {"primary_degeneracy_flags": flags,
                                                       "weighting_degraded": degraded}
            if degraded:
                result["calibration_gate"] = "FAIL"
            elif any(flags.values()) and result["calibration_gate"] != "FAIL":
                result["calibration_gate"] = "DEGENERATE"
        except ValueError as exc:
            result["calibration_gate"] = "FAIL"
            result["calibration"][model] = {"error": str(exc)}
    result["sha256"] = sa.digest_json(result)
    return result


def make_reader(data_root, raw_manifest):
    expected = {item["path"]: item["sha256"] for item in raw_manifest}
    root = Path(data_root).resolve()

    def reader(window):
        if not window["hard_valid"] or not window["pure_Normal"] or len(window["raw_base_ids"]) != 30:
            raise ValueError("reader only admits hard-valid pure-Normal windows")
        sensors, frames, tick_labels = [], [], []
        for identifier in window["raw_base_ids"]:
            split, name = identifier.split("/")
            if split != "Training" or not sa._NAME.fullmatch(name):
                raise ValueError("invalid raw base ID")
            payloads = {}
            for suffix, directory in (("csv", "원천데이터"), ("bin", "원천데이터"), ("json", "라벨링데이터")):
                path = root / split / directory / f"{name}.{suffix}"
                payload, item = sa.read_stable(path)
                if item["sha256"] != expected.get(str(path)):
                    raise ValueError(f"raw content changed: {path}")
                payloads[suffix] = payload
            label = json.loads(payloads["json"])["annotations"][0]["tagging"][0]["state"]
            if str(label) != "0":
                raise ValueError("non-Normal label entered fitting reader")
            tick_labels.append(int(label))
            rows = list(csv.reader(io.StringIO(payloads["csv"].decode("utf-8-sig"))))
            sensors.append(np.asarray(rows[1], dtype=np.float64))
            frames.append(np.load(io.BytesIO(payloads["bin"]), allow_pickle=False))
        s, t = np.stack(sensors), np.stack(frames)
        if s.shape != (30, 8) or t.shape != (30, 120, 160) or not np.isfinite(s).all() or not np.isfinite(t).all():
            raise ValueError("invalid raw arrays at control fitting boundary")
        return s, t, np.asarray(tick_labels, dtype=int)
    return reader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--seed", type=int, choices=[42, 123, 456], default=42)
    parser.add_argument("--ct-channel", choices=["CT1", "CT2"], default="CT1")
    args = parser.parse_args(argv)
    if (platform.machine() != "x86_64" or Path('/etc/nv_tegra_release').exists()
            or Path(sys.prefix).resolve() != (Path.home() / 'venvs/factory_training').resolve()
            or os.environ.get('PYTHONNOUSERSITE') != '1'):
        parser.error('server data fitting requires the audited SERVER-TRAINING venv')
    manifest = json.loads((args.prepared/'split_manifest.json').read_text())
    got = manifest.pop('sha256')
    if sa.digest_json(manifest) != got or manifest['data_gate'] != 'PASS' or manifest['protocol_id'] != sa.PROTOCOL:
        raise ValueError('invalid or failed prepared manifest')
    ledger = json.loads((args.prepared/'raw_manifest.json').read_text())
    if sa.digest_json(ledger) != manifest['source_manifest_sha256']:
        raise ValueError('raw manifest hash mismatch')
    data_root = Path(manifest['provenance']['data_root']).resolve()
    out = args.out.expanduser().resolve()
    if out == data_root or data_root in out.parents or out in data_root.parents or out.exists():
        parser.error('output must be a new directory separate from raw data')
    out.mkdir(parents=True, exist_ok=False)
    provenance = {'timestamp':datetime.now(timezone.utc).isoformat(), 'hostname':platform.node(),
                  'environment_profile':'SERVER-TRAINING', 'python_version':platform.python_version(),
                  'numpy_version':np.__version__, 'prepared_manifest_sha256':got,
                  'base_data_provenance':manifest['provenance'], 'git_dirty':True,
                  'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), ROOT/'src/data/synthetic_anomaly.py',
                                           ROOT/'src/models/anomaly_baselines.py', ROOT/'src/evaluation/anomaly_calibration.py')},
                  'purpose':'P0_control_calibration_diagnostic_only', 'gradient_training_runs':0}
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    try:
        selected = manifest['folds'][str(args.fold)]
        reader = make_reader(data_root, ledger)
        result = fit_controls(manifest['windows'], selected['roles'], selected['normalizer'], reader,
                              fold=args.fold,seed=args.seed,ct_channel=args.ct_channel,
                              progress=lambda role,n,total: print(f'{role}: {n}/{total}',flush=True))
        (out/'controls.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'calibration_gate':result['calibration_gate'], 'event_accounting':result['event_accounting'],
                          'heldout_scored':False,'full_P0_gate':'NOT_EVALUATED'}),flush=True)
        return 0 if result['calibration_gate']=='PASS' else 2
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'exception':type(exc).__name__,'message':str(exc)})+'\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
