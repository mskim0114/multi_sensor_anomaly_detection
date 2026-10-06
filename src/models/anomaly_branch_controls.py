"""EXP-004 read-only branch controls for S/T/F BIN detectors.

Given the native logits and the per-branch bottleneck features that an observer has already
captured from one eval forward, this module replays only the head (and, for F, the fusion
layer) to ask how much each branch drives the logit. It never re-runs an encoder or raw input.

Per present branch it records, as continuous measurements:
* feature population std across examples (mean and max over the 128 features);
* paired change: per Normal/positive pair, the max |z_positive - z_normal|;
* permutation control: the branch's features shifted by one position (fixed cyclic order over
  the given examples, no RNG), other branch native, logit delta summary;
* zero control: the branch's features set to 0, logit delta summary. Zero also moves the
  logit through the head/fusion bias when the native features are a nonzero constant, so it
  is recorded but never used as evidence that a branch learned an input signal.

Categories are separate from measurements and use frozen thresholds:
* collapse_like_representation: feature population std (mean over features) <= 1e-6;
* negligible_head_contribution: permutation max |logit delta| <= 1e-6.
They are labels for the recorded values, not conclusions about learning or field use.

Guarantees: eval + no_grad; every module.training flag restored, also on failure; parameters
and gradients untouched; CPU and (if already initialised) CUDA RNG compared, never rewritten
(rewriting an identical CUDA RNG state re-initialises cuDNN LSTM dropout, review (44)).
The replay must reproduce the native logits within ``replay_atol`` (default exact) in the
same micro-batch partition, or the call fails.
"""
from __future__ import annotations

import math

import torch

SCHEMA = "exp004-bin-branch-controls-v1"
BRANCHES = ("sensor", "thermal")
ARM_BRANCHES = {"S": ("sensor",), "T": ("thermal",), "F": ("sensor", "thermal")}
COLLAPSE_STD = 1e-6
NEGLIGIBLE_DELTA = 1e-6
SHUFFLE_POLICY = "fixed_cyclic_shift_by_one_over_given_example_order_no_rng"


def _check_tensor(name, value, shape_tail):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError(f"{name} must be a float32 tensor")
    if value.dim() != 1 + len(shape_tail) or tuple(value.shape[1:]) != tuple(shape_tail):
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))})")
    if not torch.isfinite(value).all():
        raise ValueError(f"non-finite {name}")


def _replay(model, features, micro_batch):
    """head(+fusion) on given branch features, in the same micro-batch partition."""
    n = next(iter(features.values())).shape[0]
    out = []
    for start in range(0, n, micro_batch):
        part = {k: v[start:start + micro_batch] for k, v in features.items()}
        if model.arm == "F":
            z = model.fusion(torch.cat([part["sensor"], part["thermal"]], dim=-1))
        else:
            z = part[ARM_BRANCHES[model.arm][0]]
        out.append(model.head(z).squeeze(-1))
    logits = torch.cat(out)
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite replayed logits")
    return logits


def _delta(a, b):
    d = (a.double() - b.double()).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "count": int(d.numel()),
            "exact_zero": bool((d == 0).all())}


def _branch_stats(z, paired):
    x = z.detach().double()
    std = x.std(dim=0, unbiased=False)
    stats = {"feature_population_std_mean": float(std.mean()), "feature_population_std_max": float(std.max()),
             "examples": int(x.shape[0]), "features": int(x.shape[1])}
    if paired:
        pair = (x[1::2] - x[0::2]).abs().amax(dim=1)
        stats.update(pair_count=int(pair.numel()), pair_max_abs_feature_change_max=float(pair.max()),
                     pair_max_abs_feature_change_mean=float(pair.mean()),
                     pairs_exactly_equal=int((pair == 0).sum()))
    return stats


def _rng_states():
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    return torch.get_rng_state(), cuda


def _check_rng_unchanged(states):
    cpu, cuda = states
    if not torch.equal(torch.get_rng_state(), cpu):
        raise RuntimeError("branch controls consumed CPU RNG")
    if cuda is not None:
        now = torch.cuda.get_rng_state_all()
        if len(now) != len(cuda) or any(not torch.equal(a, b) for a, b in zip(now, cuda)):
            raise RuntimeError("branch controls consumed CUDA RNG")


def analyze_branch_controls(model, native_logits, *, sensor_z=None, thermal_z=None, micro_batch=2,
                            paired=True, replay_atol=0.0):
    """Measure per-branch logit dependence from captured native features (no encoder re-run).

    ``native_logits`` (N,) and each present branch's z (N, hidden) must come from the same
    eval forward. With ``paired`` the examples are Normal, positive, Normal, positive, ...
    """
    if getattr(model, "objective", None) != "BIN" or getattr(model, "arm", None) not in ARM_BRANCHES:
        raise ValueError("model must be an S, T or F BIN detector")
    if not isinstance(micro_batch, int) or isinstance(micro_batch, bool) or micro_batch < 1:
        raise ValueError("micro_batch must be a positive int")
    if (isinstance(replay_atol, bool) or not isinstance(replay_atol, (int, float))
            or not math.isfinite(replay_atol) or replay_atol < 0):
        raise ValueError("replay_atol must be a finite number >= 0")
    present = ARM_BRANCHES[model.arm]
    given = {"sensor": sensor_z, "thermal": thermal_z}
    for branch in BRANCHES:
        if branch in present and given[branch] is None:
            raise ValueError(f"arm {model.arm} needs {branch}_z")
        if branch not in present and given[branch] is not None:
            raise ValueError(f"arm {model.arm} has no {branch} branch; do not pass {branch}_z")
    if not isinstance(native_logits, torch.Tensor) or native_logits.dtype != torch.float32 \
            or native_logits.dim() != 1 or not torch.isfinite(native_logits).all():
        raise ValueError("native_logits must be a finite float32 (N,) tensor")
    n = native_logits.shape[0]
    hidden = model.head.in_features
    features = {}
    for branch in present:
        _check_tensor(f"{branch}_z", given[branch], (hidden,))
        if given[branch].shape[0] != n:
            raise ValueError(f"{branch}_z and native_logits differ in length")
        features[branch] = given[branch]
    if n < 2:
        raise ValueError("at least two examples are needed for a permutation control")
    if paired and n % 2:
        raise ValueError("paired examples must alternate Normal, positive")

    flags = {name: module.training for name, module in model.named_modules()}
    grads = {name: p.grad for name, p in model.named_parameters()}
    rng = _rng_states()
    try:
        model.eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise ValueError(f"non-finite parameter {name}")
            device = model.head.weight.device
            features = {k: v.to(device) for k, v in features.items()}
            native = native_logits.to(device)
            replayed = _replay(model, features, micro_batch)
            replay = _delta(replayed, native)
            replay.update(atol=float(replay_atol), passed=replay["max_abs"] <= replay_atol)
            if not replay["passed"]:
                raise ValueError(f"head replay differs from native logits (max_abs {replay['max_abs']!r})")
            branches = {}
            for branch in BRANCHES:
                if branch not in present:
                    branches[branch] = {"status": "not_applicable"}
                    continue
                z = features[branch]
                permuted = dict(features, **{branch: torch.roll(z, shifts=1, dims=0)})
                zeroed = dict(features, **{branch: torch.zeros_like(z)})
                perm = _delta(_replay(model, permuted, micro_batch), native)
                zero = _delta(_replay(model, zeroed, micro_batch), native)
                stats = _branch_stats(z, paired)
                branches[branch] = {
                    "status": "measured", **stats, "permutation_logit_delta": perm, "zero_logit_delta": zero,
                    "zero_control_note": "includes head/fusion bias response; not evidence of a learned input signal",
                    "categories": {
                        "collapse_like_representation": stats["feature_population_std_mean"] <= COLLAPSE_STD,
                        "negligible_head_contribution": perm["max_abs"] <= NEGLIGIBLE_DELTA},
                }
            record = {"schema": SCHEMA, "arm": model.arm, "examples": int(n), "paired": bool(paired),
                      "micro_batch": micro_batch, "shuffle_policy": SHUFFLE_POLICY,
                      "category_thresholds": {"collapse_like_representation_std_mean_max": COLLAPSE_STD,
                                              "negligible_head_contribution_permutation_max_abs": NEGLIGIBLE_DELTA},
                      "categories_are_labels_not_conclusions": True,
                      "head_replay": replay, "branches": branches}
    finally:
        for name, module in model.named_modules():
            module.training = flags[name]
        rng_error = None
        try:
            _check_rng_unchanged(rng)
        except RuntimeError as error:
            rng_error = error
    if rng_error is not None:
        raise rng_error
    for name, parameter in model.named_parameters():
        if parameter.grad is not grads[name]:
            raise RuntimeError(f"branch controls changed the gradient of {name}")
    return record
