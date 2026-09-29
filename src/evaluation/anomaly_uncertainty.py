"""Conditional calibration resampling plans; never refit a model or generator.

Plans index frozen Normal scores and all accepted synthetic derivatives together.
Repeated draws of the same device receive separate identities for equal-device
weighting. This module constructs plans; it does not claim confidence coverage.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np


def grouped_normal_runs(records, *, stride, offset=0):
    """Split a session again wherever pure-Normal window starts are discontinuous."""
    if (not records or not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0
            or not isinstance(offset, int) or isinstance(offset, bool) or not 0 <= offset < stride):
        raise ValueError("nonempty Normal records and positive stride required")
    groups = defaultdict(list)
    seen = set()
    for i, r in enumerate(records):
        key = r["base_window_id"]
        if key in seen:
            raise ValueError("duplicate calibration Normal base")
        seen.add(key)
        if not (r["device"].startswith("agv") or r["device"].startswith("oht")):
            raise ValueError("unknown equipment stratum")
        if not isinstance(r["start_index"], int) or r["start_index"] < 0:
            raise ValueError("invalid window start")
        if r["start_index"] % stride != offset:
            raise ValueError("window start disagrees with declared stride/offset grid")
        groups[r["device"], r["session_id"]].append(i)
    by_device = defaultdict(list)
    for (device, session), indices in sorted(groups.items()):
        ordered = sorted(indices, key=lambda i: records[i]["start_index"])
        starts = [records[i]["start_index"] for i in ordered]
        if len(set(starts)) != len(starts):
            raise ValueError("duplicate start in one device/session")
        run = []
        for i in ordered:
            if run and records[i]["start_index"] - records[run[-1]]["start_index"] != stride:
                by_device[device].append(run)
                run = []
            run.append(i)
        if run:
            by_device[device].append(run)
    return dict(by_device)


def run_resolution(normal_records, *, stride, offset=0, block_starts=12):
    """Describe conditional temporal resolution without selecting a block length."""
    runs = grouped_normal_runs(normal_records, stride=stride, offset=offset)
    lengths = [len(run) for device_runs in runs.values() for run in device_runs]
    fixed = [n for n in lengths if n <= block_starts]
    return {"stride_ticks": stride, "offset_ticks": offset, "block_window_starts": block_starts,
            "runs": len(lengths), "windows": sum(lengths),
            "lengths_by_device": {d: [len(r) for r in rs] for d, rs in runs.items()},
            "run_length_min_median_max": [min(lengths), float(np.median(lengths)), max(lengths)],
            "whole_run_fixed_multiset_runs": len(fixed), "whole_run_fixed_multiset_windows": sum(fixed),
            "whole_run_fixed_multiset_window_fraction": sum(fixed) / sum(lengths),
            "limitations": ["within_run_score_distribution_not_resampled_when_length_le_block",
                            "device_resampling_still_varies_device_composition",
                            "no_guarantee_of_interval_coverage_or_direction_of_bias"]}


def make_resampling_plan(normal_records, events, *, stride, offset=0, block_starts=12, seed=20260929,
                         replicate=0, role="calibration"):
    """One AGV/OHT-stratified device draw with circular blocks inside each run."""
    if role != "calibration":
        raise ValueError("uncertainty may only resample calibration scores")
    if (not isinstance(block_starts, int) or isinstance(block_starts, bool) or block_starts < 1
            or not isinstance(replicate, int) or replicate < 0):
        raise ValueError("invalid block length or replicate")
    runs = grouped_normal_runs(normal_records, stride=stride, offset=offset)
    strata = {p: sorted(d for d in runs if d.startswith(p)) for p in ("agv", "oht")}
    if any(not ds for ds in strata.values()):
        raise ValueError("both AGV and OHT calibration strata are required")
    event_indices = defaultdict(list)
    bases = {r["base_window_id"]: r["device"] for r in normal_records}
    seen_events = set()
    for i, e in enumerate(events):
        if e["base_window_id"] not in bases or bases[e["base_window_id"]] != e["device"]:
            raise ValueError("synthetic event does not belong to a calibration base")
        identity = (e["base_window_id"], e["event_id"])
        if identity in seen_events:
            raise ValueError("duplicate synthetic event")
        seen_events.add(identity)
        # Rejections remain in the input accounting, but have no score to refit.
        if e["accepted"]:
            event_indices[e["base_window_id"]].append(i)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, replicate, block_starts])))
    normal_rows, synthetic_rows, draws, short_runs = [], [], [], []
    for prefix, devices in strata.items():
        chosen = rng.choice(devices, size=len(devices), replace=True)
        for slot, device in enumerate(chosen):
            device = str(device)
            draw_id = f"{prefix}_draw{slot}:{device}"
            draws.append({"source_device": device, "resampled_device": draw_id})
            for run_id, run in enumerate(runs[device]):
                if len(run) < block_starts:
                    sampled = list(run)
                    short_runs.append({"draw": draw_id, "run": run_id, "length": len(run)})
                else:
                    sampled = []
                    while len(sampled) < len(run):
                        start = int(rng.integers(len(run)))
                        sampled.extend(run[(start + j) % len(run)] for j in range(block_starts))
                    sampled = sampled[:len(run)]
                for index in sampled:
                    cluster = len(normal_rows)
                    normal_rows.append({"source_index": index, "device": draw_id, "cluster": cluster})
                    base = normal_records[index]["base_window_id"]
                    for event_index in event_indices[base]:
                        synthetic_rows.append({"source_index": event_index, "device": draw_id, "cluster": cluster})
    result = {"method": "stratified_equipment_then_within_run_circular_score_blocks",
              "role": role, "seed": seed, "replicate": replicate, "block_window_starts": block_starts,
              "stride_ticks": stride, "offset_ticks": offset,
              "device_draws": draws, "short_runs": short_runs,
              "normal_rows": normal_rows, "synthetic_rows": synthetic_rows,
              "refit_model_normalizer_or_generator": False, "retune_primary_threshold": False}
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def materialize_scores(plan, normal_records, events, model):
    """Turn a plan into the existing calibration API's positional score arguments."""
    n, s = plan["normal_rows"], plan["synthetic_rows"]
    return ([normal_records[r["source_index"]]["scores"][model] for r in n], [r["device"] for r in n],
            [events[r["source_index"]]["scores"][model] for r in s], [r["device"] for r in s],
            [events[r["source_index"]]["family"] for r in s], [events[r["source_index"]]["strength"] for r in s])


def fixed_subset_sensitivity(normal_records, events, model, calibration_module, *, stride, alpha=0.01):
    """Predeclared device omission and stride/offset checks; never select among them."""
    grouped_normal_runs(normal_records, stride=stride)
    if stride != 10:
        raise ValueError("v3 stride-30 sensitivity requires the declared stride-10 primary bank")
    scenarios = [(f"leave_out_{d}", [r for r in normal_records if r['device'] != d], stride, 0)
                 for d in sorted({r['device'] for r in normal_records})]
    scenarios.extend((f"stride30_offset{offset}", [r for r in normal_records if r['start_index'] % 30 == offset], 30, offset)
                     for offset in (0, 10, 20))
    results = {}
    for name, normal, subset_stride, offset in scenarios:
        bases = {r['base_window_id'] for r in normal}
        accepted = [e for e in events if e['accepted'] and e['base_window_id'] in bases]
        try:
            resolution = run_resolution(normal, stride=subset_stride, offset=offset)
            args = ([r['scores'][model] for r in normal], [r['device'] for r in normal],
                    [e['scores'][model] for e in accepted], [e['device'] for e in accepted],
                    [e['family'] for e in accepted], [e['strength'] for e in accepted])
            nq = calibration_module.calibrate_nq(args[0], args[1], alpha=alpha, role='calibration')
            syn = calibration_module.calibrate_syn(*args, alpha=alpha, role='calibration')
            if syn['tau'] < nq['tau']:
                raise AssertionError('subset calibration violates NQ/SYN ordering')
            weights, _ = calibration_module.stratified_cell_weights(
                args[3], args[4], args[5], expected_device_ids=sorted(set(args[1])))
            tpr_nq = float(weights[np.asarray(args[2]) > nq['tau']].sum())
            results[name] = {'status':'PASS', 'normal_windows':len(normal),
                             'accepted_events':len(accepted), 'run_resolution':resolution,
                             'NQ':nq, 'SYN':syn, 'synthetic_TPR_at_NQ':tpr_nq}
        except ValueError as exc:
            results[name] = {'status':'FAIL', 'error':str(exc)}
    return {'model':model, 'alpha':alpha, 'scenarios':results,
            'retuned_primary_threshold':False, 'heldout_scored':False}


def conditional_intervals(normal_records, events, model, calibration_module, *,
                          stride, offset=0, alpha=0.01, replicates=2000, block_starts=12, seed=20260929):
    """Describe threshold variability conditional on these frozen scores.

    A short probe is explicitly distinguished from the planned 2000 replicates.
    Missing-cell failures invalidate the interval rather than being silently dropped.
    The primary fitted threshold is never overwritten or reselected here.
    """
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError("positive replicate count required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    resolution = run_resolution(normal_records, stride=stride, offset=offset, block_starts=block_starts)
    thresholds, failures, short_count = [], [], 0
    for rep in range(replicates):
        plan = make_resampling_plan(normal_records, events, stride=stride, offset=offset,
                                    block_starts=block_starts, seed=seed, replicate=rep)
        short_count += bool(plan['short_runs'])
        args = materialize_scores(plan, normal_records, events, model)
        try:
            nq = calibration_module.calibrate_nq(args[0], args[1], alpha=alpha, role='calibration')
            syn = calibration_module.calibrate_syn(*args, alpha=alpha, role='calibration')
            if syn['tau'] < nq['tau']:
                raise AssertionError('resampled calibration violates NQ/SYN ordering')
            thresholds.append({'replicate':rep,'NQ':nq['tau'],'SYN':syn['tau'],
                               'SYN_degenerate':syn['degenerate_always_normal']})
        except ValueError as exc:
            failures.append({'replicate':rep,'error':str(exc)})
    valid_quantiles = ({policy: np.quantile([r[policy] for r in thresholds],[.025,.975],method='linear').tolist()
                       for policy in ['NQ','SYN']} if thresholds else None)
    interval = valid_quantiles if not failures else None
    return {'scope':'conditional_on_frozen_scores_model_normalizer_and_generator',
            'model':model,'replicates_requested':replicates,'replicates_valid':len(thresholds),
            'planned_replicate_count_met':replicates==2000,'block_window_starts':block_starts,
            'seed':seed,'percentile_interval':interval,'failed_replicates':failures,
            'alpha':alpha, 'run_resolution':resolution, 'failure_fraction':len(failures)/replicates,
            'valid_only_quantiles_diagnostic_not_CI':valid_quantiles if failures else None,
            'interval_endpoints':'linear_interpolation_for_description_only_not_selected_thresholds',
            'replicates_with_short_runs':short_count,'threshold_samples':thresholds,
            'retuned_primary_threshold':False,'coverage_guarantee':False}
