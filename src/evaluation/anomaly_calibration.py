#!/usr/bin/env python3
"""NQ and SYN threshold policies for EXP-20260928-004 (v3 §7 / plan `threshold:`).

Both policies pick one scalar `tau`; a window is flagged when `score > tau`. Both see
only the calibration role, and neither can see real severity: no function here accepts a
label argument, and `assert_no_label_arguments()` proves that by introspection so a later
edit fails a test rather than leaking (design §7-1).

    NQ   the weighted empirical (1 - alpha) quantile of calibration-Normal scores.
         Synthetic data plays no part, so NQ is the control that shows what the
         synthetic bank actually buys.
    SYN  minimise `0.5 * FPR_normal + 0.5 * (1 - TPR_synthetic)` subject to
         `FPR_normal <= alpha`, over the observed score values. Ties resolve to the
         lower FPR, then to the higher tau.

Weighting follows the plan: Normal windows are weighted equally per machine, synthetic
windows equally per (machine, family, strength) cell. A missing cell is never filled in
silently; by default it is an error, and `on_missing_cells="report"` makes it an explicit,
flagged degradation instead.

`equal_equipment_weights` is implemented here rather than imported from
`src/models/anomaly_baselines.py`, because importing that would pull in
`src/models/__init__.py` and therefore torch, which the Jetson does not have. The test
suite loads both modules as leaves and asserts the two implementations agree, so the
duplication cannot drift.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Sequence

import numpy as np

CALIBRATION_ROLE = "calibration"
DEFAULT_ALPHA = 0.01
SENSITIVITY_ALPHAS = (0.005, 0.02)
COMPARATOR = "score_gt_tau"
# The design's fixed axes. They must come from the protocol, not from whatever the
# generator happened to emit: building the expected grid out of the observed values
# cannot detect a whole family, a whole strength or a whole machine going missing.
V3_FAMILIES = ("sensor-only", "thermal-only", "coupled")
V3_STRENGTHS = (1, 2, 4)
SYN_OBJECTIVE = "0.5*FPR_normal+0.5*(1-TPR_synthetic)"

_FORBIDDEN_ARG_TOKENS = ("label", "severity", "class", "target", "ground_truth", "y_true")


# --------------------------------------------------------------------------- guards

def assert_no_label_arguments(module=None) -> list[str]:
    """Fail if any public callable here accepts a label-like argument (design §7-1)."""
    import sys

    module = module or sys.modules[__name__]
    checked = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        targets = []
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            targets.append((name, obj))
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            for meth_name, raw in vars(obj).items():
                if meth_name.startswith("_"):
                    continue
                meth = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
                if inspect.isfunction(meth):
                    targets.append((f"{name}.{meth_name}", meth))
        for label, fn in targets:
            checked.append(label)
            for param in inspect.signature(fn).parameters:
                low = param.lower()
                for token in _FORBIDDEN_ARG_TOKENS:
                    if token in low:
                        raise AssertionError(
                            f"{label} accepts {param!r}, which looks like a real label. "
                            "Threshold fitting must not be able to see severity (§7-1).")
    return checked


def _finite(name: str, a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty; a threshold cannot be fitted on nothing")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} holds non-finite values; screen them out rather than repair")
    return arr


def _check_role(role: str) -> None:
    if role != CALIBRATION_ROLE:
        raise ValueError(
            f"thresholds may only be fitted on role {CALIBRATION_ROLE!r}, got {role!r}. "
            "Fitting on fit, dev or heldout would break the protocol's role separation.")


def digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------- weighting

def equal_equipment_weights(device_ids: Sequence[str]) -> np.ndarray:
    """Per-window weights giving every machine the same total weight, summing to 1."""
    ids = list(device_ids)
    if not ids:
        raise ValueError("no windows given; cannot build equal-equipment weights")
    counts: dict[str, int] = {}
    for d in ids:
        counts[d] = counts.get(d, 0) + 1
    n_dev = len(counts)
    w = np.array([1.0 / (n_dev * counts[d]) for d in ids], dtype=np.float64)
    total = w.sum()
    if not np.isclose(total, 1.0, rtol=0, atol=1e-12):
        raise AssertionError(f"weights sum to {total!r}, expected 1")
    return w


def _exact_int(value, what: str) -> int:
    """Accept only a value that is already an integer. `int(1.9)` would hide a bug."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be an integer, got the boolean {value!r}")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be an integer, got {value!r}") from exc
    if not as_float.is_integer():
        raise ValueError(f"{what} must be an integer, got {value!r}; silently truncating it "
                         "would hide a generator error")
    return int(as_float)


def stratified_cell_weights(device_ids: Sequence[str], families: Sequence[str],
                            strengths: Sequence[int], *,
                            expected_device_ids: Sequence[str],
                            expected_families: Sequence[str] = V3_FAMILIES,
                            expected_strengths: Sequence[int] = V3_STRENGTHS,
                            on_missing_cells: str = "raise") -> tuple[np.ndarray, dict]:
    """Weights giving every (machine, family, strength) cell the same total weight.

    The expected grid is supplied, never inferred from the data. Inferring it would make
    the check blind exactly where it matters: if the generator emitted no `thermal-only`
    event at all, or no strength 4, or nothing for one machine, those cells would be
    absent from an inferred grid too and the loss would pass unnoticed. `expected_device_ids`
    therefore comes from the calibration-Normal machines and the other two axes default to
    the protocol's fixed values.

    A missing cell is an error by default; `on_missing_cells="report"` weights the observed
    cells and records the degradation so it is visible rather than silent.
    """
    if on_missing_cells not in {"raise", "report"}:
        raise ValueError("on_missing_cells must be 'raise' or 'report'")
    dev, fam = list(device_ids), list(families)
    stg = [_exact_int(s, "strength") for s in strengths]
    if not (len(dev) == len(fam) == len(stg)):
        raise ValueError(f"synthetic descriptors disagree in length: "
                         f"{len(dev)} devices, {len(fam)} families, {len(stg)} strengths")
    if not dev:
        raise ValueError("no synthetic windows given")

    exp_dev = sorted(set(expected_device_ids))
    exp_fam = list(dict.fromkeys(expected_families))
    exp_stg = [_exact_int(s, "expected strength") for s in expected_strengths]
    if not exp_dev or not exp_fam or not exp_stg:
        raise ValueError("expected axes must each hold at least one value")

    unknown_fam = sorted(set(fam) - set(exp_fam))
    unknown_stg = sorted(set(stg) - set(exp_stg))
    unknown_dev = sorted(set(dev) - set(exp_dev))
    if unknown_fam or unknown_stg or unknown_dev:
        raise ValueError(
            "synthetic windows fall outside the declared grid: "
            f"families {unknown_fam}, strengths {unknown_stg}, machines {unknown_dev}. "
            "The preregistered grid is the authority; an unexpected value is a generator "
            "error, not a new cell.")

    keys = list(zip(dev, fam, stg))
    counts: dict[tuple, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1

    expected = [(d, f, s) for d in exp_dev for f in exp_fam for s in exp_stg]
    missing = [list(k) for k in expected if k not in counts]
    absent_families = sorted(f for f in exp_fam if f not in set(fam))
    absent_strengths = sorted(s for s in exp_stg if s not in set(stg))
    absent_devices = sorted(d for d in exp_dev if d not in set(dev))
    if missing and on_missing_cells == "raise":
        whole = []
        if absent_families:
            whole.append(f"families {absent_families}")
        if absent_strengths:
            whole.append(f"strengths {absent_strengths}")
        if absent_devices:
            whole.append(f"machines {absent_devices}")
        extra = f" Entirely absent: {'; '.join(whole)}." if whole else ""
        raise ValueError(
            f"{len(missing)} of {len(expected)} (machine, family, strength) cells produced no "
            f"synthetic window, e.g. {missing[:3]}.{extra} The design weights cells equally, so "
            "dropping them would change the weighting. Fix the generation or pass "
            "on_missing_cells='report' to accept and record the degradation.")

    n_cells = len(counts)
    w = np.array([1.0 / (n_cells * counts[k]) for k in keys], dtype=np.float64)
    if not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-12):
        raise AssertionError(f"weights sum to {w.sum()!r}, expected 1")
    accounting = {
        "weighting": "equal_equipment_family_strength",
        "expected_devices": exp_dev, "expected_families": exp_fam,
        "expected_strengths": exp_stg, "expected_axes_source": "declared_not_inferred",
        "observed_devices": sorted(set(dev)), "observed_families": sorted(set(fam)),
        "observed_strengths": sorted(set(stg)),
        "cells_expected": len(expected), "cells_observed": n_cells,
        "cells_missing": missing,
        "entirely_absent_families": absent_families,
        "entirely_absent_strengths": absent_strengths,
        "entirely_absent_devices": absent_devices,
        "windows_per_cell": {"min": int(min(counts.values())), "max": int(max(counts.values()))},
        "on_missing_cells": on_missing_cells,
        "weighting_degraded_by_missing_cells": bool(missing),
    }
    return w, accounting


# ----------------------------------------------------------------- rate at a threshold

def decide(scores, tau: float) -> np.ndarray:
    """The comparator the whole protocol uses: flagged when `score > tau`."""
    return np.asarray(scores, dtype=np.float64) > float(tau)


def weighted_rate_above(scores: np.ndarray, weights: np.ndarray, tau: float) -> float:
    """Weighted fraction of scores strictly greater than tau."""
    return float(weights[scores > tau].sum())


# ------------------------------------------------------------------------- NQ

def calibrate_nq(normal_scores, normal_device_ids, *, alpha: float = DEFAULT_ALPHA,
                 role: str = CALIBRATION_ROLE) -> dict:
    """NQ: the smallest observed tau whose weighted Normal exceedance is at most alpha.

    The comparator is strict, so a tau equal to an observed score excludes every window
    at that score. Searching the observed values and taking the smallest feasible one
    means the constraint holds exactly rather than approximately, which is what
    "conservative ties" has to mean when many Normal scores sit on the boundary: a
    plain interpolated quantile could land between tied values and overshoot alpha.
    """
    _check_role(role)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    s = _finite("normal_scores", normal_scores)
    w = equal_equipment_weights(normal_device_ids)
    if len(w) != len(s):
        raise ValueError(f"normal_device_ids has {len(w)} entries for {len(s)} scores")

    uniq = np.unique(s)
    feasible = [v for v in uniq if weighted_rate_above(s, w, v) <= alpha]
    # tau = max score always gives exceedance 0, so `feasible` is never empty
    tau = float(feasible[0])
    fpr = weighted_rate_above(s, w, tau)
    result = {
        "policy": "NQ", "comparator": COMPARATOR, "role": role, "alpha": alpha,
        "rule": "smallest observed score whose weighted exceedance <= alpha",
        "tau": tau, "achieved_normal_fpr": fpr,
        "target_normal_fpr": alpha,
        "normal": {"windows": int(len(s)), "devices": len(set(normal_device_ids)),
                   "weighting": "equal_equipment",
                   "score_min": float(s.min()), "score_max": float(s.max()),
                   "unique_scores": int(len(uniq))},
        "degenerate_all_normal_scores_equal": bool(len(uniq) == 1),
        "tau_is_at_normal_max": bool(tau >= float(s.max())),
        "synthetic_used": False,
    }
    result["sha256"] = digest(result)
    return result


# ------------------------------------------------------------------------- SYN

def calibrate_syn(normal_scores, normal_device_ids, synthetic_scores,
                  synthetic_device_ids, synthetic_families, synthetic_strengths, *,
                  alpha: float = DEFAULT_ALPHA, role: str = CALIBRATION_ROLE,
                  on_missing_cells: str = "raise",
                  candidate_pruning: bool = True) -> dict:
    """SYN: minimise `0.5*FPR + 0.5*(1-TPR)` over observed taus, subject to FPR <= alpha.

    The objective is piecewise constant in tau and changes only at observed score values,
    so evaluating every unique observed score covers every distinct decision rule. Ties
    go to the lower FPR first and then to the higher tau, which is the design's rule and
    also the choice that keeps false alarms down when detection is unaffected.

    `candidate_pruning` drops candidates below tau_NQ before the loop. That is exact, not
    an approximation: the weighted Normal exceedance is constant between consecutive Normal
    scores, so the smallest feasible tau over all reals is exactly tau_NQ, and nothing below
    it can satisfy `FPR <= alpha`. The surviving candidates are evaluated by the identical
    arithmetic in the identical order, so the reported tau and rates are bit-for-bit what
    the exhaustive search returns; `candidate_pruning=False` runs that exhaustive search and
    the test suite pins the two together, including on tie and one-ULP fixtures.
    """
    _check_role(role)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    sn = _finite("normal_scores", normal_scores)
    wn = equal_equipment_weights(normal_device_ids)
    if len(wn) != len(sn):
        raise ValueError(f"normal_device_ids has {len(wn)} entries for {len(sn)} scores")
    sy = _finite("synthetic_scores", synthetic_scores)
    # The expected grid is the protocol's, and the machines are the calibration-Normal
    # machines, so a whole family, strength or machine going missing is detected.
    wy, cells = stratified_cell_weights(
        synthetic_device_ids, synthetic_families, synthetic_strengths,
        expected_device_ids=sorted(set(normal_device_ids)),
        expected_families=V3_FAMILIES, expected_strengths=V3_STRENGTHS,
        on_missing_cells=on_missing_cells)
    if len(wy) != len(sy):
        raise ValueError(f"synthetic descriptors cover {len(wy)} windows for {len(sy)} scores")

    all_candidates = np.unique(np.concatenate([sn, sy]))
    if candidate_pruning:
        floor = calibrate_nq(sn, normal_device_ids, alpha=alpha, role=role)["tau"]
        candidates = all_candidates[all_candidates >= floor]
    else:
        floor, candidates = None, all_candidates
    rows = []
    for tau in candidates:
        fpr = weighted_rate_above(sn, wn, tau)
        if fpr > alpha:
            continue
        tpr = weighted_rate_above(sy, wy, tau)
        rows.append((0.5 * fpr + 0.5 * (1.0 - tpr), fpr, -float(tau), float(tau), tpr))
    if not rows:                                                   # pragma: no cover
        raise AssertionError("no feasible tau, which cannot happen: tau = max score gives FPR 0")
    rows.sort()                       # objective, then FPR, then -tau (so higher tau wins ties)
    objective, fpr, _, tau, tpr = rows[0]

    best_objective = rows[0][0]
    tied = [r for r in rows if r[0] == best_objective]
    per_stratum = _per_stratum_tpr(sy, synthetic_device_ids, synthetic_families,
                                   synthetic_strengths, tau)
    result = {
        "policy": "SYN", "comparator": COMPARATOR, "role": role, "alpha": alpha,
        "objective": SYN_OBJECTIVE, "constraint": "FPR_normal <= alpha",
        "tie_rule": "lower FPR, then higher tau",
        "tau": tau, "objective_value": objective,
        "achieved_normal_fpr": fpr, "achieved_synthetic_tpr": tpr,
        "target_normal_fpr": alpha,
        "normal": {"windows": int(len(sn)), "devices": len(set(normal_device_ids)),
                   "weighting": "equal_equipment"},
        "synthetic": {"windows": int(len(sy)), **cells},
        "candidates_evaluated": int(len(candidates)),
        "candidates_total": int(len(all_candidates)),
        "candidates_pruned_below_tau_nq": int(len(all_candidates) - len(candidates)),
        "candidate_pruning": bool(candidate_pruning),
        "pruning_floor_tau_nq": floor,
        "candidates_feasible": int(len(rows)),
        "tied_at_optimum": int(len(tied)),
        "synthetic_tpr_by_stratum": per_stratum,
        # A tau that flags nothing is a formally optimal answer when no synthetic score
        # clears the FPR budget; it is a degeneracy, not a result, so it is named.
        "degenerate_always_normal": bool(tpr == 0.0),
        "degenerate_all_normal_scores_equal": bool(len(np.unique(sn)) == 1),
        "degenerate_all_synthetic_scores_equal": bool(len(np.unique(sy)) == 1),
        "tau_is_at_normal_max": bool(tau >= float(sn.max())),
        "synthetic_used": True,
    }
    result["sha256"] = digest(result)
    return result


def _per_stratum_tpr(scores: np.ndarray, device_ids, families, strengths,
                     tau: float) -> dict:
    """Detection rate inside each (family, strength) cell, machine-weighted within the cell.

    Reported because a single pooled TPR can hide a bank that is detected entirely
    through one family or one strength.
    """
    out: dict[str, dict] = {}
    keys = list(zip(list(families), [int(s) for s in strengths]))
    devs = list(device_ids)
    for fam, stg in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == (fam, stg)]
        sub_scores = scores[idx]
        sub_w = equal_equipment_weights([devs[i] for i in idx])
        out[f"{fam}|{stg}"] = {
            "windows": len(idx),
            "devices": len(set(devs[i] for i in idx)),
            "tpr": weighted_rate_above(sub_scores, sub_w, tau),
        }
    return out


# ------------------------------------------------------------------- sensitivity

def calibrate_with_sensitivity(normal_scores, normal_device_ids, synthetic_scores,
                               synthetic_device_ids, synthetic_families,
                               synthetic_strengths, *, alpha: float = DEFAULT_ALPHA,
                               sensitivity_alphas: Sequence[float] = SENSITIVITY_ALPHAS,
                               role: str = CALIBRATION_ROLE,
                               on_missing_cells: str = "raise",
                               candidate_pruning: bool = True) -> dict:
    """Both policies at the primary alpha, plus the plan's sensitivity alphas.

    The sensitivity entries exist to show how much the threshold moves with the budget.
    They are not alternatives to choose from after seeing held-out performance.
    """
    primary = {
        "alpha": alpha,
        "NQ": calibrate_nq(normal_scores, normal_device_ids, alpha=alpha, role=role),
        "SYN": calibrate_syn(normal_scores, normal_device_ids, synthetic_scores,
                             synthetic_device_ids, synthetic_families, synthetic_strengths,
                             alpha=alpha, role=role, on_missing_cells=on_missing_cells,
                             candidate_pruning=candidate_pruning),
    }
    sensitivity = []
    for a in sensitivity_alphas:
        sensitivity.append({
            "alpha": a,
            "NQ": calibrate_nq(normal_scores, normal_device_ids, alpha=a, role=role),
            "SYN": calibrate_syn(normal_scores, normal_device_ids, synthetic_scores,
                                 synthetic_device_ids, synthetic_families,
                                 synthetic_strengths, alpha=a, role=role,
                                 on_missing_cells=on_missing_cells,
                                 candidate_pruning=candidate_pruning),
        })
    out = {
        "experiment_id": "EXP-20260928-004", "stage": "threshold_calibration",
        "real_labels_used": False,
        "primary": primary, "sensitivity": sensitivity,
        "nq_equals_syn": primary["NQ"]["tau"] == primary["SYN"]["tau"],
        "selection_used_heldout": False,
    }
    out["sha256"] = digest(out)
    return out
