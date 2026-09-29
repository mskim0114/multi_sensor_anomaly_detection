#!/usr/bin/env python3
"""Statistical baselines for EXP-20260928-004 (v3 §3.4): B-NTC, B-MD, B-THERM, B-LATE.

These exist so the cost of the neural detectors can be read against something. They fit
no weights by gradient descent, but they do estimate statistics from data, so they are
not "rules with no fitting" (v3 §3.4).

    B-NTC    max over the window of |run-normalised NTC|
    B-MD     squared Mahalanobis distance of 15 window features (mean, std ddof=0,
             max|.| for each of the five primary channels), fitted on fit pure-Normal
    B-THERM  max over time and pixels of |run-normalised temperature|
    B-LATE   0.5 * ECDF_dev(B-MD) + 0.5 * ECDF_dev(B-THERM)

Three rules shape the API, and each is enforced rather than documented:

1.  **Real severity never reaches a fitting path.** No function here takes a label,
    severity or class argument. `assert_no_label_arguments()` checks that by
    introspection, and the test suite calls it, so adding such a parameter later fails a
    test rather than silently leaking (design §7-1).
2.  **Roles are separate.** Base score parameters come from `fit` only; the B-LATE ECDF
    comes from `dev` only. `MahalanobisBaseline.fit` and `EcdfScaler.fit` each record
    the role they were fitted on and refuse anything but their own.
3.  **Every fitted object serialises to plain data and hashes.** `to_dict()` /
    `from_dict()` / `params_sha256()` let a run store its parameters and a later reader
    prove the same parameters were used.

Windows arriving here are already run-normalised: the sensor window is
`(T, 5)` z-scored with the run-local fit pure-Normal mean/std, and the thermal window is
`(T, H, W)` z-scored with the run-local fit pure-Normal global mean/std, the same
normaliser T-AE uses. `fit_run_normalizer` and `fit_thermal_normalizer` build those
parameters from raw physical units when a caller needs them; they are plain dicts so a
preparation step elsewhere can produce them instead.

NumPy only, no torch, so this is testable on fixtures without a GPU or the dataset.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# v3 fixes these; they are not tuned.
MD_LAMBDA = 0.1
MD_EPSILON = 1e-6
MD_FEATURES_PER_CHANNEL = ("temporal_mean", "temporal_std_ddof0", "temporal_max_abs")
PRIMARY_CHANNEL_COUNT = 5
MD_FEATURE_COUNT = PRIMARY_CHANNEL_COUNT * len(MD_FEATURES_PER_CHANNEL)
CONSTANT_SCALE_FALLBACK = 1.0
LATE_WEIGHTS = (0.5, 0.5)

FIT_ROLE = "fit"
DEV_ROLE = "dev"

# Anything whose name suggests a real label must never appear in a signature here.
_FORBIDDEN_ARG_TOKENS = ("label", "severity", "class", "target", "ground_truth", "y_true")


# --------------------------------------------------------------------------- guards

def assert_no_label_arguments(module=None) -> list[str]:
    """Fail if any public callable in this module accepts a label-like argument.

    The design's first prohibition is that threshold and parameter fitting must not be
    able to see real severity. A comment cannot enforce that; this can, and the test
    suite calls it, so a later edit that adds such a parameter breaks a test.
    """
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
                # staticmethod/classmethod are descriptors, not functions; `vars` hands
                # back the descriptor, so unwrap it or the fit paths go unchecked.
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
                            "Baselines must not be able to see severity (design §7-1).")
    return checked


def _finite(name: str, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        bad = int(np.count_nonzero(~np.isfinite(a)))
        raise ValueError(f"{name} holds {bad} non-finite value(s); quality screening must "
                         "reject these before scoring, not repair them")
    return a


def _check_role(got: str, want: str, what: str) -> None:
    if got != want:
        raise ValueError(f"{what} may only be fitted on role {want!r}, got {got!r}. "
                         "Role separation is part of the protocol, not a convention.")


# ------------------------------------------------------- equal-equipment weighting

def equal_equipment_weights(device_ids: Sequence[str]) -> np.ndarray:
    """Per-window weights giving every machine the same total weight, summing to 1.

    For D machines and n_d windows on machine d, each of those windows gets
    `1 / (D * n_d)`. Machine count drives the estimate rather than window count, which
    matters because window counts per machine differ by up to a factor of 1.5 here and
    because machine-to-machine spread is the dominant uncertainty (N23).
    """
    ids = list(device_ids)
    if not ids:
        raise ValueError("no windows given; cannot build equal-equipment weights")
    counts: dict[str, int] = {}
    for d in ids:
        counts[d] = counts.get(d, 0) + 1
    n_dev = len(counts)
    w = np.array([1.0 / (n_dev * counts[d]) for d in ids], dtype=np.float64)
    # exact by construction; assert so a future change cannot silently unbalance it
    total = w.sum()
    if not np.isclose(total, 1.0, rtol=0, atol=1e-12):
        raise AssertionError(f"weights sum to {total!r}, expected 1")
    return w


def _weighted_mean_std(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and population std along axis 0, centred on the first row first.

    Subtracting `x[0]` before accumulating removes the rounding residue that makes a
    column of identical values report a variance of ~1e-32 instead of zero. With the
    offset, an exactly constant column gives exactly `std == 0`, so "zero variance" can
    be tested exactly and no tolerance is needed. A tolerance would be worse than
    useless here: it would reclassify a channel with a genuinely small but real spread
    as constant and discard that signal.
    """
    x0 = x[0]
    d = x - x0
    mean_d = w @ d
    var = w @ (d - mean_d) ** 2
    return x0 + mean_d, np.sqrt(np.maximum(var, 0.0))


def constant_columns(x: np.ndarray, std: np.ndarray, *, what: str = "column") -> np.ndarray:
    """Mask of columns whose every value is identical. Exact test, no tolerance.

    A column with `ptp > 0` but `std == 0` cannot be constant; it means the variance
    underflowed or cancelled. Substituting scale 1.0 there would silently erase a real
    spread, so it is an error instead.
    """
    identical = np.ptp(x, axis=0) == 0.0
    underflowed = (~identical) & (np.asarray(std) == 0.0)
    if np.any(underflowed):
        idx = [int(i) for i in np.flatnonzero(underflowed)]
        raise ValueError(
            f"{what}(s) {idx} vary (ptp > 0) yet their variance evaluated to zero. This is "
            "numerical underflow, not a constant channel; it is not silently replaced by "
            "scale 1.0 because that would erase a real spread.")
    return identical


def _scale_from_std(std: np.ndarray, constant: np.ndarray) -> np.ndarray:
    scale = np.asarray(std, dtype=np.float64).copy()
    scale[constant] = CONSTANT_SCALE_FALLBACK
    return scale


# --------------------------------------------------------------- run normalisers

def fit_run_normalizer(raw_ticks: np.ndarray, device_ids: Sequence[str]) -> dict:
    """Convenience only. **Not the protocol normaliser** -- do not wire this into a run.

    v3 fits the run normaliser on *unique raw ticks* with each tick counting once
    (`normalizer: run_local_fit_pure_Normal_unique_raw_ticks_only`), and the preparation
    step produces it. This helper weights by machine instead, which is the rule for the
    B-MD features and the ECDF, not for the raw normaliser. The two differ whenever
    machines contribute unequal tick counts, so using this one would silently change the
    protocol. It exists to build fixtures and to let a caller check an injected
    normaliser's shape.

    `raw_ticks` is `(N, 5)`; passing overlapping windows' ticks would double-count the
    shared ones.
    """
    x = _finite("raw_ticks", raw_ticks)
    if x.ndim != 2 or x.shape[1] != PRIMARY_CHANNEL_COUNT:
        raise ValueError(f"raw_ticks must be (N, {PRIMARY_CHANNEL_COUNT}), got {x.shape}")
    w = equal_equipment_weights(device_ids)
    if len(w) != len(x):
        raise ValueError(f"device_ids has {len(w)} entries for {len(x)} ticks")
    mean, std = _weighted_mean_std(x, w)
    const_mask = constant_columns(x, std, what="channel")
    std = np.where(const_mask, 0.0, std)
    constant = [int(i) for i in np.flatnonzero(const_mask)]
    scale = _scale_from_std(std, const_mask)
    return {"kind": "sensor_zscore", "role": FIT_ROLE, "mean": mean.tolist(),
            "std": std.tolist(),
            "scale": scale.tolist(), "constant_channels": constant,
            "constant_scale_fallback": CONSTANT_SCALE_FALLBACK,
            "constant_rule": "exact_ptp_zero_no_tolerance",
            "n_unique_ticks": int(len(x)), "n_devices": len(set(device_ids))}


def fit_thermal_normalizer(raw_frames: np.ndarray, device_ids: Sequence[str]) -> dict:
    """Global mean/std of temperature, the same normaliser T-AE uses.

    One scalar pair for the whole array, not per pixel: a per-pixel normaliser would
    absorb a stationary hotspot into the background. `raw_frames` is `(N, H, W)` of
    unique frames; each frame contributes its own pixels, weighted by its machine.
    """
    x = _finite("raw_frames", raw_frames)
    if x.ndim != 3:
        raise ValueError(f"raw_frames must be (N, H, W), got {x.shape}")
    w = equal_equipment_weights(device_ids)
    if len(w) != len(x):
        raise ValueError(f"device_ids has {len(w)} entries for {len(x)} frames")
    per_frame = x.reshape(len(x), -1)
    # Centre on the first pixel before accumulating, for the same reason as the sensor
    # path: E[x^2] - mean^2 on raw magnitudes loses the low bits to cancellation, and an
    # exactly constant field must come out at exactly zero variance.
    x0 = float(per_frame[0, 0])
    d = per_frame - x0
    mean_d = float(w @ d.mean(axis=1))
    second_d = float(w @ (d ** 2).mean(axis=1))
    mean = x0 + mean_d
    std = float(np.sqrt(max(second_d - mean_d ** 2, 0.0)))
    constant = bool(np.ptp(x) == 0.0)
    if not constant and std == 0.0:
        raise ValueError("thermal frames vary (ptp > 0) yet their variance evaluated to "
                         "zero; this is numerical underflow, not a constant field")
    return {"kind": "thermal_global_zscore", "role": FIT_ROLE, "mean": mean,
            "std": 0.0 if constant else std,
            "scale": CONSTANT_SCALE_FALLBACK if constant else std,
            "constant": constant,
            "constant_scale_fallback": CONSTANT_SCALE_FALLBACK,
            "constant_rule": "exact_ptp_zero_no_tolerance",
            "n_unique_frames": int(len(x)), "n_devices": len(set(device_ids))}


def select_primary_channels(params: dict, primary_indices: Sequence[int]) -> dict:
    """Narrow an 8-channel normaliser to the five primary channels, in the given order.

    The preparation step fits its normaliser over the dataset's own eight channels in
    `src/data/config.py:SENSOR_CHANNELS` order; the baselines read five. Selecting here
    keeps the index mapping in one place instead of at every call site.
    """
    idx = list(primary_indices)
    if len(idx) != PRIMARY_CHANNEL_COUNT:
        raise ValueError(f"expected {PRIMARY_CHANNEL_COUNT} indices, got {len(idx)}")
    mean = np.asarray(params["mean"], dtype=np.float64)
    scale = np.asarray(params["scale"], dtype=np.float64)
    if any(not 0 <= i < len(mean) for i in idx):
        raise ValueError(f"indices {idx} out of range for a {len(mean)}-channel normalizer")
    if len(set(idx)) != len(idx):
        raise ValueError(f"indices {idx} repeat a channel")
    const = set(int(c) for c in params.get("constant_channels", []))
    out = dict(params)
    out.update({"mean": mean[idx].tolist(), "scale": scale[idx].tolist(),
                "constant_channels": [j for j, i in enumerate(idx) if i in const],
                "selected_from_indices": idx,
                "source_channel_count": int(len(mean))})
    return out


def apply_sensor_normalizer(raw_window: np.ndarray, params: dict) -> np.ndarray:
    x = _finite("raw_window", raw_window)
    mean = np.asarray(params["mean"], dtype=np.float64)
    scale = np.asarray(params["scale"], dtype=np.float64)
    if x.shape[-1] != len(mean):
        raise ValueError(f"window has {x.shape[-1]} channels, normalizer has {len(mean)}")
    return (x - mean) / scale


def apply_thermal_normalizer(raw_window: np.ndarray, params: dict) -> np.ndarray:
    x = _finite("raw_window", raw_window)
    return (x - float(params["mean"])) / float(params["scale"])


# ------------------------------------------------------------------ B-NTC, B-THERM

def score_b_ntc(sensor_windows: np.ndarray, ntc_index: int = 0) -> np.ndarray:
    """max over the window of |normalised NTC|. Input `(B, T, 5)` already normalised."""
    x = _finite("sensor_windows", sensor_windows)
    if x.ndim != 3 or x.shape[2] != PRIMARY_CHANNEL_COUNT:
        raise ValueError(f"sensor_windows must be (B, T, {PRIMARY_CHANNEL_COUNT}), got {x.shape}")
    if not 0 <= ntc_index < PRIMARY_CHANNEL_COUNT:
        raise ValueError(f"ntc_index {ntc_index} out of range")
    return np.abs(x[:, :, ntc_index]).max(axis=1)


def score_b_therm(thermal_windows: np.ndarray) -> np.ndarray:
    """max over time and pixels of |normalised temperature|. Input `(B, T, H, W)`."""
    x = _finite("thermal_windows", thermal_windows)
    if x.ndim != 4:
        raise ValueError(f"thermal_windows must be (B, T, H, W), got {x.shape}")
    return np.abs(x.reshape(len(x), -1)).max(axis=1)


# ------------------------------------------------------------------------- B-MD

def md_feature_names(channel_names: Sequence[str] | None = None) -> list[str]:
    chans = list(channel_names) if channel_names else [f"ch{i}" for i in range(PRIMARY_CHANNEL_COUNT)]
    if len(chans) != PRIMARY_CHANNEL_COUNT:
        raise ValueError(f"expected {PRIMARY_CHANNEL_COUNT} channel names, got {len(chans)}")
    return [f"{c}.{f}" for c in chans for f in MD_FEATURES_PER_CHANNEL]


def md_features(sensor_windows: np.ndarray) -> np.ndarray:
    """15 features from `(B, T, 5)` normalised windows: mean, std(ddof=0), max|.| per channel.

    Channel-major order, matching `md_feature_names`.
    """
    x = _finite("sensor_windows", sensor_windows)
    if x.ndim != 3 or x.shape[2] != PRIMARY_CHANNEL_COUNT:
        raise ValueError(f"sensor_windows must be (B, T, {PRIMARY_CHANNEL_COUNT}), got {x.shape}")
    if x.shape[1] < 1:
        raise ValueError("windows must have at least one tick")
    mean = x.mean(axis=1)
    std = x.std(axis=1, ddof=0)
    maxabs = np.abs(x).max(axis=1)
    # interleave per channel so the order matches md_feature_names
    out = np.empty((len(x), MD_FEATURE_COUNT), dtype=np.float64)
    out[:, 0::3], out[:, 1::3], out[:, 2::3] = mean, std, maxabs
    return out


@dataclass
class MahalanobisBaseline:
    """B-MD. Squared Mahalanobis distance of the 15 window features.

    Features are standardised with the fit weighted mean/scale, and the distance is taken
    from the standardised-space centroid `mu`, exactly as v3 writes it:
    `(z - mu)' C_reg^-1 (z - mu)`.

    `mu` is zero in exact arithmetic, but it is stored and subtracted rather than assumed,
    because a near-constant feature makes it materially non-zero. The stored mean is a
    float64 rounding of the true weighted mean, so recomputing `z` leaves a residue of
    order one ULP of the feature; divided by a scale of order 1e-16 that residue becomes
    order one. Measured on a feature whose spread is a single ULP, `mu` reached 0.845 and
    the distribution's own centroid scored 9.26 instead of 0. Since near-constant channels
    are deliberately preserved rather than flattened, the centroid has to be carried
    explicitly or the covariance and the distance would be centred on different points.

    The covariance is shrunk by the fixed v3 rule; the two constants are not tuned and not
    selected on any performance.
    """

    feature_names: list[str]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_centroid: np.ndarray
    constant_features: list[int]
    covariance: np.ndarray            # standardised-space weighted covariance, before shrinkage
    covariance_regularised: np.ndarray
    lam: float = MD_LAMBDA
    epsilon: float = MD_EPSILON
    role: str = FIT_ROLE
    n_windows: int = 0
    n_devices: int = 0
    _chol: np.ndarray | None = field(default=None, repr=False, compare=False)

    @staticmethod
    def fit(sensor_windows: np.ndarray, device_ids: Sequence[str], *,
            role: str = FIT_ROLE, channel_names: Sequence[str] | None = None,
            lam: float = MD_LAMBDA, epsilon: float = MD_EPSILON) -> "MahalanobisBaseline":
        _check_role(role, FIT_ROLE, "B-MD")
        feats = md_features(sensor_windows)
        w = equal_equipment_weights(device_ids)
        if len(w) != len(feats):
            raise ValueError(f"device_ids has {len(w)} entries for {len(feats)} windows")
        mean, std = _weighted_mean_std(feats, w)
        const_mask = constant_columns(feats, std, what="feature")
        constant = [int(i) for i in np.flatnonzero(const_mask)]
        scale = _scale_from_std(np.where(const_mask, 0.0, std), const_mask)
        z = (feats - mean) / scale
        centroid = w @ z
        centred = z - centroid
        cov = (centred * w[:, None]).T @ centred
        cov = 0.5 * (cov + cov.T)                     # symmetrise against float drift
        p = cov.shape[0]
        cov_reg = (1.0 - lam) * cov + lam * (np.trace(cov) / p) * np.eye(p) + epsilon * np.eye(p)
        cov_reg = 0.5 * (cov_reg + cov_reg.T)
        return MahalanobisBaseline(
            feature_names=md_feature_names(channel_names), feature_mean=mean,
            feature_scale=scale, feature_centroid=centroid,
            constant_features=constant, covariance=cov,
            covariance_regularised=cov_reg, lam=lam, epsilon=epsilon, role=role,
            n_windows=int(len(feats)), n_devices=len(set(device_ids)))

    def _factor(self) -> np.ndarray:
        if self._chol is None:
            try:
                self._chol = np.linalg.cholesky(self.covariance_regularised)
            except np.linalg.LinAlgError as exc:                      # pragma: no cover
                raise ValueError("regularised covariance is not positive definite; the "
                                 "shrinkage constants should have prevented this") from exc
        return self._chol

    def score(self, sensor_windows: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance, solved as a linear system rather than inverted.

        With the Cholesky factor L of the regularised covariance, solving `L y = z` gives
        `y'y = z' C^-1 z` directly, which is better conditioned than forming `C^-1`.
        """
        z = (md_features(sensor_windows) - self.feature_mean) / self.feature_scale
        y = np.linalg.solve(self._factor(), (z - self.feature_centroid).T)
        return (y ** 2).sum(axis=0)

    def to_dict(self) -> dict:
        return {"id": "B-MD", "role": self.role, "feature_names": list(self.feature_names),
                "feature_mean": self.feature_mean.tolist(),
                "feature_scale": self.feature_scale.tolist(),
                "feature_centroid": self.feature_centroid.tolist(),
                "centroid_note": "standardised-space mu; zero in exact arithmetic, kept "
                                 "because a near-constant feature makes it material",
                "constant_features": list(self.constant_features),
                "constant_scale_fallback": CONSTANT_SCALE_FALLBACK,
                "covariance": self.covariance.tolist(),
                "covariance_regularised": self.covariance_regularised.tolist(),
                "covariance_rule": "(1-lambda)*C+lambda*trace(C)/p*I+epsilon*I",
                "lambda": self.lam, "epsilon": self.epsilon,
                "score": "squared_Mahalanobis_distance",
                "n_windows": self.n_windows, "n_devices": self.n_devices}

    @staticmethod
    def from_dict(d: dict) -> "MahalanobisBaseline":
        return MahalanobisBaseline(
            feature_names=list(d["feature_names"]),
            feature_mean=np.asarray(d["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(d["feature_scale"], dtype=np.float64),
            feature_centroid=np.asarray(d["feature_centroid"], dtype=np.float64),
            constant_features=list(d["constant_features"]),
            covariance=np.asarray(d["covariance"], dtype=np.float64),
            covariance_regularised=np.asarray(d["covariance_regularised"], dtype=np.float64),
            lam=float(d["lambda"]), epsilon=float(d["epsilon"]), role=d["role"],
            n_windows=int(d.get("n_windows", 0)), n_devices=int(d.get("n_devices", 0)))

    def params_sha256(self) -> str:
        return params_sha256(self.to_dict())


# ------------------------------------------------------------------------ B-LATE

@dataclass
class EcdfScaler:
    """Equal-equipment weighted ECDF of one component's dev-Normal scores.

    `ECDF(s) = sum_i w_i * 1[dev_score_i <= s]` with `w_i = 1/(D*n_d)`. Ties count, so a
    query equal to a dev score is included, and the result is bounded to [0, 1] with no
    extrapolation: below the dev minimum it is 0, at or above the maximum it is 1.
    """

    component: str
    sorted_scores: np.ndarray
    cumulative_weights: np.ndarray
    role: str = DEV_ROLE
    n_windows: int = 0
    n_devices: int = 0
    degenerate: bool = False

    @staticmethod
    def fit(scores: np.ndarray, device_ids: Sequence[str], *, component: str,
            role: str = DEV_ROLE) -> "EcdfScaler":
        _check_role(role, DEV_ROLE, "B-LATE ECDF")
        s = _finite("scores", scores)
        if s.ndim != 1:
            raise ValueError(f"scores must be one-dimensional, got {s.shape}")
        w = equal_equipment_weights(device_ids)
        if len(w) != len(s):
            raise ValueError(f"device_ids has {len(w)} entries for {len(s)} scores")
        order = np.argsort(s, kind="mergesort")       # stable, so equal scores keep order
        ss, ws = s[order], w[order]
        return EcdfScaler(component=component, sorted_scores=ss,
                          cumulative_weights=np.cumsum(ws), role=role,
                          n_windows=int(len(s)), n_devices=len(set(device_ids)),
                          degenerate=bool(np.ptp(ss) == 0.0))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = _finite("scores", scores)
        # rightmost insertion point gives the weight of all dev scores <= s (ties counted)
        idx = np.searchsorted(self.sorted_scores, s, side="right")
        out = np.where(idx > 0, self.cumulative_weights[np.maximum(idx - 1, 0)], 0.0)
        return np.clip(out, 0.0, 1.0)

    def saturation(self, scores: np.ndarray) -> dict:
        """How often a query falls outside the dev range, where the ECDF carries no detail."""
        u = self.transform(scores)
        s = np.asarray(scores, dtype=np.float64)
        n = len(s) or 1
        return {"component": self.component,
                "below_dev_min_fraction": float(np.count_nonzero(s < self.sorted_scores[0]) / n),
                "at_or_above_dev_max_fraction": float(
                    np.count_nonzero(s >= self.sorted_scores[-1]) / n),
                "u_eq_0_fraction": float(np.count_nonzero(u <= 0.0) / n),
                "u_eq_1_fraction": float(np.count_nonzero(u >= 1.0) / n),
                "degenerate_dev_scores": self.degenerate}

    def to_dict(self) -> dict:
        return {"id": f"ECDF[{self.component}]", "role": self.role,
                "component": self.component,
                "sorted_scores": self.sorted_scores.tolist(),
                "cumulative_weights": self.cumulative_weights.tolist(),
                "weighting": "equal_equipment", "tie_rule": "weighted_fraction_le_query",
                "out_of_range": "bounded_0_to_1_no_extrapolation",
                "n_windows": self.n_windows, "n_devices": self.n_devices,
                "degenerate": self.degenerate}

    @staticmethod
    def from_dict(d: dict) -> "EcdfScaler":
        return EcdfScaler(component=d["component"],
                          sorted_scores=np.asarray(d["sorted_scores"], dtype=np.float64),
                          cumulative_weights=np.asarray(d["cumulative_weights"], dtype=np.float64),
                          role=d["role"], n_windows=int(d.get("n_windows", 0)),
                          n_devices=int(d.get("n_devices", 0)),
                          degenerate=bool(d.get("degenerate", False)))

    def params_sha256(self) -> str:
        return params_sha256(self.to_dict())


@dataclass
class LateFusionBaseline:
    """B-LATE. `0.5 * ECDF_dev(B-MD) + 0.5 * ECDF_dev(B-THERM)`.

    The same rule applies to the neural late-fusion control L (v3 §3.3), so this class
    takes two already-computed component scores rather than owning either detector.
    """

    ecdf_a: EcdfScaler
    ecdf_b: EcdfScaler
    weights: tuple[float, float] = LATE_WEIGHTS

    def __post_init__(self) -> None:
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError(f"late weights must sum to 1, got {self.weights}")

    @staticmethod
    def fit(scores_a: np.ndarray, scores_b: np.ndarray, device_ids: Sequence[str], *,
            component_a: str = "B-MD", component_b: str = "B-THERM",
            role: str = DEV_ROLE) -> "LateFusionBaseline":
        return LateFusionBaseline(
            ecdf_a=EcdfScaler.fit(scores_a, device_ids, component=component_a, role=role),
            ecdf_b=EcdfScaler.fit(scores_b, device_ids, component=component_b, role=role))

    def score(self, scores_a: np.ndarray, scores_b: np.ndarray) -> np.ndarray:
        a, b = np.asarray(scores_a), np.asarray(scores_b)
        if a.shape != b.shape:
            raise ValueError(f"component scores must align, got {a.shape} and {b.shape}")
        wa, wb = self.weights
        return wa * self.ecdf_a.transform(a) + wb * self.ecdf_b.transform(b)

    def saturation(self, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
        return {"weights": list(self.weights),
                "components": [self.ecdf_a.saturation(scores_a),
                               self.ecdf_b.saturation(scores_b)]}

    def to_dict(self) -> dict:
        return {"id": "B-LATE", "weights": list(self.weights),
                "ecdf": [self.ecdf_a.to_dict(), self.ecdf_b.to_dict()]}

    @staticmethod
    def from_dict(d: dict) -> "LateFusionBaseline":
        a, b = d["ecdf"]
        return LateFusionBaseline(ecdf_a=EcdfScaler.from_dict(a),
                                  ecdf_b=EcdfScaler.from_dict(b),
                                  weights=tuple(d["weights"]))

    def params_sha256(self) -> str:
        return params_sha256(self.to_dict())


# ------------------------------------------------------------------------ hashing

def params_sha256(params: dict) -> str:
    """Hash of a parameter set, stable across processes and key order.

    `json.dumps` with sorted keys and a fixed float repr, so a later run can prove it
    scored with the same parameters. Python's `hash()` would not do: it is salted.
    """
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def baseline_manifest(*, fold: int, sensor_normalizer: dict, thermal_normalizer: dict,
                      md: MahalanobisBaseline, late: LateFusionBaseline | None = None) -> dict:
    """Everything a later reader needs to reproduce these baselines on one fold."""
    out = {
        "experiment_id": "EXP-20260928-004",
        "component_set": ["B-NTC", "B-MD", "B-THERM"] + (["B-LATE"] if late else []),
        "fold": fold,
        "gradient_training_runs": 0,
        "sensor_normalizer": sensor_normalizer,
        "sensor_normalizer_sha256": params_sha256(sensor_normalizer),
        "thermal_normalizer": thermal_normalizer,
        "thermal_normalizer_sha256": params_sha256(thermal_normalizer),
        "b_ntc": {"id": "B-NTC", "score": "max_over_time_abs_run_normalized_NTC",
                  "fitted_parameters": "none_beyond_sensor_normalizer"},
        "b_therm": {"id": "B-THERM",
                    "score": "max_over_time_and_pixels_abs_run_global_normalized_temperature",
                    "fitted_parameters": "none_beyond_thermal_normalizer"},
        "b_md": md.to_dict(),
        "b_md_sha256": md.params_sha256(),
    }
    if late is not None:
        out["b_late"] = late.to_dict()
        out["b_late_sha256"] = late.params_sha256()
    return out
