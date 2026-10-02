"""EXP-004 read-only diagnostics of the thermal branch of a BIN detector.

In P1b, on 8 dev bases (Normal and thermal-only s4 pairs) at the selected checkpoint, the
T-BIN and F-BIN thermal representation was nearly constant across inputs and the pair logits
were equal (review updates (40)-(41); one checkpoint, one fold, one seed, training trajectory
not inspected). This module observes that branch without changing it: it
recomputes every LSTM gate from the module's own weights, checks the recomputation against
the real ``nn.LSTM`` output element by element, and reports per-layer gate statistics.

Guarantees
* Observation only: ``eval()`` + ``no_grad()``; every ``module.training`` flag is restored
  afterwards, even if the call fails. RNG states are compared, never rewritten: an eval
  forward consumes no RNG, and calling ``torch.cuda.set_rng_state`` - even with an identical
  state - makes the next cuDNN LSTM training forward re-initialize its dropout state, which
  changes every later update (measured on the training server, review update (44)). If an
  RNG state did change, the call fails instead of restoring it. CUDA RNG is only read when
  CUDA is already initialized, so a CPU-only call never initializes CUDA. No backward pass, no
  ``.grad`` is created, no parameter is touched.
* A non-finite input, weight or intermediate value is a failure (``ValueError``), never a
  statistic.
* Results are plain JSON values. Statistics are returned as sums, sums of squares, counts
  and maxima, so microbatch results combine exactly with ``combine_diagnostics``. A gate
  value counts as saturated when its pre-activation exceeds 6 in absolute value
  (``abs_gt6_count`` of ``total_count``); sigmoid(6) = 0.9975.
* The gate recomputation is only trusted when its parity with ``nn.LSTM`` passes; the parity
  record is part of every result.

Input is the model's thermal tensor ``(N, 30, 120, 160)`` float32, already normalized.
PyTorch gate order is (input, forget, cell, output).
"""
from __future__ import annotations

import math

import torch
from torch import nn

SCHEMA = "exp004-bin-thermal-lstm-diagnostics-v1"
GATES = ("input", "forget", "cell", "output")
SATURATION = 6.0
DEFAULT_RTOL, DEFAULT_ATOL = 1e-5, 1e-6


def _finite(tensor, what):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"non-finite {what}")


def _check_lstm(lstm):
    if not isinstance(lstm, nn.LSTM):
        raise ValueError("expected an nn.LSTM")
    if not lstm.batch_first or lstm.bidirectional or getattr(lstm, "proj_size", 0) or not lstm.bias:
        raise ValueError("only batch_first, unidirectional LSTMs with bias and no projection are supported")


def _moments(tensor):
    """Exact, combinable summary of the absolute value of a tensor."""
    flat = tensor.detach().double().reshape(-1)
    count, total = int(flat.numel()), float(flat.abs().sum())
    return {"count": count, "abs_sum": total, "abs_mean": total / count if count else None,
            "abs_max": float(flat.abs().max()) if count else 0.0}


def _unit_moments(tensor):
    """Per-unit sum and sum of squares over the example axis, for an exact across-example std."""
    x = tensor.detach().double()
    return {"count": int(x.shape[0]), "sum": x.sum(0).tolist(), "sum_sq": (x * x).sum(0).tolist()}


def lstm_recurrence(lstm, x):
    """Recompute ``lstm(x)`` gate by gate. Returns (last-layer output sequence, per-layer stats).

    ``x`` is (N, T, input_size). Inter-layer dropout is not applied, which matches ``eval()``.
    """
    _check_lstm(lstm)
    if x.dim() != 3 or x.shape[-1] != lstm.input_size:
        raise ValueError("LSTM input must be (N, T, input_size)")
    _finite(x, "LSTM input")
    layers, inp = [], x
    for layer in range(lstm.num_layers):
        w_ih = getattr(lstm, f"weight_ih_l{layer}")
        w_hh = getattr(lstm, f"weight_hh_l{layer}")
        bias = getattr(lstm, f"bias_ih_l{layer}") + getattr(lstm, f"bias_hh_l{layer}")
        for name, value in (("weight_ih", w_ih), ("weight_hh", w_hh), ("bias", bias)):
            _finite(value, f"LSTM layer {layer} {name}")
        n, steps = inp.shape[:2]
        h = inp.new_zeros(n, lstm.hidden_size)
        c = inp.new_zeros(n, lstm.hidden_size)
        pre_steps, outputs, cells = [], [], []
        for t in range(steps):
            pre = inp[:, t] @ w_ih.T + h @ w_hh.T + bias
            i, f, g, o = pre.chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            pre_steps.append(pre)
            outputs.append(h)
            cells.append(c)
        pre_all = torch.stack(pre_steps, 1)
        out = torch.stack(outputs, 1)
        cell = torch.stack(cells, 1)
        _finite(pre_all, f"LSTM layer {layer} gate pre-activation")
        _finite(out, f"LSTM layer {layer} output")
        gates = {}
        for gate, chunk in zip(GATES, pre_all.chunk(4, dim=-1)):
            stats = _moments(chunk)
            gates[gate] = {"total_count": stats["count"], "abs_sum": stats["abs_sum"],
                           "abs_mean": stats["abs_mean"], "abs_max": stats["abs_max"],
                           "abs_gt6_count": int((chunk.abs() > SATURATION).sum())}
        layers.append({"layer": layer, "gates": gates, "cell": _moments(cell),
                       "last_hidden": _unit_moments(out[:, -1])})
        inp = out
    return inp, layers


def _parity(manual, reference, rtol, atol):
    diff = (manual.double() - reference.double()).abs()
    scale = reference.double().abs()
    max_rel = float((diff / scale.clamp_min(1e-30)).max()) if diff.numel() else 0.0
    return {"max_abs_error": float(diff.max()) if diff.numel() else 0.0, "max_rel_error": max_rel,
            "rtol": rtol, "atol": atol, "count": int(diff.numel()),
            "violations": int((diff > atol + rtol * scale).sum()),
            "passed": bool((diff <= atol + rtol * scale).all())}


def _rng_states():
    # get_rng_state_all() would lazily initialize CUDA; only touch it if it is already up.
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    return torch.get_rng_state(), cuda


def _check_rng_unchanged(states):
    """Compare only. Rewriting the CUDA state would itself perturb cuDNN RNN dropout."""
    cpu, cuda = states
    if not torch.equal(torch.get_rng_state(), cpu):
        raise RuntimeError("diagnostics consumed CPU RNG; the observation is not invariant")
    if cuda is not None:
        now = torch.cuda.get_rng_state_all()
        if len(now) != len(cuda) or any(not torch.equal(a, b) for a, b in zip(now, cuda)):
            raise RuntimeError("diagnostics consumed CUDA RNG; the observation is not invariant")


def thermal_lstm_diagnostics(model, thermal, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Observe the thermal branch of a T or F BIN detector on one (micro)batch.

    Parity compares the manual recurrence with ``nn.LSTM`` on the whole last-layer output
    sequence, and the manual last step with the encoder's own output ``z``.
    """
    for name, value in (("rtol", rtol), ("atol", atol)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite number >= 0")
    if getattr(model, "objective", None) != "BIN" or getattr(model, "arm", None) not in ("T", "F"):
        raise ValueError("model must be a T or F BIN detector")
    encoder = getattr(model, "thermal_encoder", None)
    if encoder is None:
        raise ValueError("model has no thermal encoder")
    if not isinstance(thermal, torch.Tensor) or thermal.dtype != torch.float32:
        raise ValueError("thermal must be a float32 tensor")
    if thermal.dim() != 4 or thermal.shape[0] < 1:
        raise ValueError("thermal must be (N, 30, 120, 160)")
    _finite(thermal, "thermal input")
    flags = {name: module.training for name, module in model.named_modules()}
    grads = {name: p.grad for name, p in model.named_parameters()}
    rng = _rng_states()
    try:
        model.eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                _finite(parameter, f"parameter {name}")
            n, steps = thermal.shape[:2]
            z = encoder(thermal)  # also enforces the (N, 30, 120, 160) contract
            frames = encoder.thermal_conv(thermal.reshape(n * steps, 1, *thermal.shape[2:]))
            fc = encoder.fc_thermal(frames.reshape(n * steps, -1)).reshape(n, steps, -1)
            _finite(fc, "fc_thermal output")
            reference, _ = encoder.lstm(fc)
            manual, layers = lstm_recurrence(encoder.lstm, fc)
            _finite(z, "thermal encoder output")
            record = {
                "schema": SCHEMA, "arm": model.arm, "examples": int(n), "ticks": int(steps),
                "saturation_abs_pre_activation": SATURATION,
                "fc_thermal": _moments(fc),
                "thermal_z": _unit_moments(z),
                "thermal_z_values": z.detach().double().tolist(),
                "layers": layers,
                "parity_sequence": _parity(manual, reference, rtol, atol),
                "parity_encoder_output": _parity(manual[:, -1], z, rtol, atol),
            }
    finally:
        for name, module in model.named_modules():
            module.training = flags[name]
        rng_error = None
        try:
            _check_rng_unchanged(rng)
        except RuntimeError as error:  # do not mask an exception raised by the observation itself
            rng_error = error
    if rng_error is not None:
        raise rng_error
    for name, parameter in model.named_parameters():
        if parameter.grad is not grads[name]:
            raise RuntimeError(f"diagnostics changed the gradient of {name}")
    return record


def _merge_moments(a, b):
    count, total = a["count"] + b["count"], a["abs_sum"] + b["abs_sum"]
    return {"count": count, "abs_sum": total, "abs_mean": total / count,
            "abs_max": max(a["abs_max"], b["abs_max"])}


def _merge_units(a, b):
    return {"count": a["count"] + b["count"],
            "sum": [x + y for x, y in zip(a["sum"], b["sum"])],
            "sum_sq": [x + y for x, y in zip(a["sum_sq"], b["sum_sq"])]}


def _merge_parity(a, b):
    if (a["rtol"], a["atol"]) != (b["rtol"], b["atol"]):
        raise ValueError("cannot combine parity records with different tolerances")
    return {"max_abs_error": max(a["max_abs_error"], b["max_abs_error"]),
            "max_rel_error": max(a["max_rel_error"], b["max_rel_error"]),
            "rtol": a["rtol"], "atol": a["atol"], "count": a["count"] + b["count"],
            "violations": a["violations"] + b["violations"], "passed": a["passed"] and b["passed"]}


def combine_diagnostics(parts):
    """Exactly merge microbatch results of the same model, threshold and tolerances."""
    parts = list(parts)
    if not parts:
        raise ValueError("nothing to combine")
    first = parts[0]
    out = first
    for part in parts[1:]:
        for key in ("schema", "arm", "ticks", "saturation_abs_pre_activation"):
            if part[key] != first[key]:
                raise ValueError(f"cannot combine results with different {key}")
        if len(part["layers"]) != len(out["layers"]):
            raise ValueError("cannot combine results with different layer counts")
        layers = []
        for la, lb in zip(out["layers"], part["layers"]):
            gates = {}
            for gate in GATES:
                ga, gb = la["gates"][gate], lb["gates"][gate]
                count, total = ga["total_count"] + gb["total_count"], ga["abs_sum"] + gb["abs_sum"]
                gates[gate] = {"total_count": count, "abs_sum": total, "abs_mean": total / count,
                               "abs_max": max(ga["abs_max"], gb["abs_max"]),
                               "abs_gt6_count": ga["abs_gt6_count"] + gb["abs_gt6_count"]}
            layers.append({"layer": la["layer"], "gates": gates,
                           "cell": _merge_moments(la["cell"], lb["cell"]),
                           "last_hidden": _merge_units(la["last_hidden"], lb["last_hidden"])})
        out = {**out, "examples": out["examples"] + part["examples"],
               "fc_thermal": _merge_moments(out["fc_thermal"], part["fc_thermal"]),
               "thermal_z": _merge_units(out["thermal_z"], part["thermal_z"]),
               "thermal_z_values": out["thermal_z_values"] + part["thermal_z_values"],
               "layers": layers,
               "parity_sequence": _merge_parity(out["parity_sequence"], part["parity_sequence"]),
               "parity_encoder_output": _merge_parity(out["parity_encoder_output"],
                                                      part["parity_encoder_output"])}
    return out


def _unit_std(units):
    """Population std per unit from sums; mean and max over units."""
    n = units["count"]
    if n < 2:
        return {"mean": None, "max": None}
    stds = [math.sqrt(max(sq / n - (s / n) ** 2, 0.0)) for s, sq in zip(units["sum"], units["sum_sq"])]
    return {"mean": sum(stds) / len(stds), "max": max(stds)}


def summarize(record):
    """Human-readable view: means, saturation fractions and across-example std."""
    summary = {"arm": record["arm"], "examples": record["examples"],
               "parity_passed": record["parity_sequence"]["passed"] and record["parity_encoder_output"]["passed"],
               "fc_thermal_abs_mean": record["fc_thermal"]["abs_mean"],
               "fc_thermal_abs_max": record["fc_thermal"]["abs_max"],
               "thermal_z_across_example_std": _unit_std(record["thermal_z"]), "layers": []}
    for layer in record["layers"]:
        summary["layers"].append({
            "layer": layer["layer"],
            "gate_pre_abs_mean": {g: v["abs_mean"] for g, v in layer["gates"].items()},
            "gate_abs_gt6_fraction": {g: v["abs_gt6_count"] / v["total_count"]
                                      for g, v in layer["gates"].items()},
            "last_hidden_across_example_std": _unit_std(layer["last_hidden"])})
    return summary
