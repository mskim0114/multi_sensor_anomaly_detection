#!/usr/bin/env python3
"""Fit-only, first-epoch BIN diagnostics, separate from the preserved P1b runner.

The real CLI runs one T-BIN initial LR, never a sweep. A separately frozen plan
and exact-source fixtures are required before cache attachment or torch import.
The first epoch uses P1b's original 30-epoch sampler/scheduler and stops after
261 updates. No dev forward, checkpoint selection, alarm threshold or retry.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_bin_onset_p1b", ROOT / "src/train_anomaly_p1b.py")
p1b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p1b
spec.loader.exec_module(p1b)
bs, sa = p1b.bs, p1b.sa
SCHEMA = "exp004-bin-onset-diagnostic-v1"
PLAN_SCHEMA = "exp004-bin-onset-plan-v1"
RECIPE = "exp004-bin-onset-v1"
FIXTURE_SCHEMA = "exp004-bin-onset-fixtures-v1"
NEW_FILES = ("src/diagnose_anomaly_bin_onset.py", "tests/test_anomaly_bin_onset.py",
             "src/models/anomaly_bin_diagnostics.py", "tests/test_anomaly_bin_diagnostics.py")
SOURCE_FILES = (*p1b.SOURCE_FILES, *NEW_FILES)
PROBE_STEPS = (0, 1, 2, 5, 10, 20, 50, 100, 261)
LR_VALUES = (1e-3, 1e-4)
REAL_STEPS = 261
RUNTIME_FIELDS = ("hostname", "python_version", "numpy_version", "torch_version", "cuda_runtime_version",
                  "device", "cuda_device_index", "gpu_name", "gpu_uuid", "nvidia_smi_index",
                  "cuda_visible_devices", "cublas_workspace_config", "cudnn_version", "driver_version")
RULES = {
    "collapse_like": {"visible_thermal_z_mean_feature_std_max": 1e-6,
                      "visible_pair_max_abs_logit_gap_max": 1e-6},
    "responsive_at_probe": {"visible_thermal_z_mean_feature_std_min_exclusive": 1e-4,
                            "visible_pair_max_abs_logit_gap_min_exclusive": 1e-6},
}


def source_hashes():
    return {p: p1b.file_sha(ROOT / p) for p in SOURCE_FILES}


def selected_probes(inputs):
    """Deterministic fit metadata only; independent of training RNG and results."""
    index = inputs["indexes"]["fit"]
    stored = {w["base_window_id"]: (i, w) for i, w in
              enumerate(inputs["cache_manifest"]["windows"])}
    rows = []
    for device in sorted({b.device for b in index.bases}):
        for family in ("sensor-only", "thermal-only"):
            candidates = sorted((b.base_window_id, e.event_id, b, e)
                                for b in index.bases if b.device == device
                                for e in b.events if e.family == family and e.strength == 4)
            if not candidates:
                raise ValueError("fit device lacks an accepted strength4 probe cell")
            _, _, base, event = candidates[0]
            cache_index, window = stored[base.base_window_id]
            rows.append({"role": "fit", "device": device, "family": family, "strength": 4,
                         "cache_index": cache_index, "base_window_id": base.base_window_id,
                         "event_id": event.event_id, "event_instance_id": event.event_instance_id,
                         "window_record_sha256": sa.digest_json(window),
                         "normal_then_positive_labels": [0, 1]})
    return rows


def resolve_probe_pairs(inputs, rows):
    if rows != selected_probes(inputs):
        raise ValueError("probe selection differs from deterministic fit-only metadata")
    bases = {b.base_window_id: b for b in inputs["indexes"]["fit"].bases}
    pairs = []
    for row in rows:
        base = bases[row["base_window_id"]]
        event = next(e for e in base.events if e.event_id == row["event_id"])
        pairs.append(bs.BinPair(base.base_window_id, base.device, base.window_index,
                                event.event_id, event.event_instance_id, event.family, event.strength,
                                "CT1", event.sensor_changed, event.thermal_changed))
    return tuple(pairs)


def validate_inputs(args):
    """All gates remain metadata-only; baseline source/evidence is unchanged."""
    inputs = p1b.validate_inputs(args)
    plan = p1b._pinned(args.plan, args.plan_sha256, "onset plan")
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("recipe_id") != RECIPE
            or plan.get("status") != "FROZEN_DIAGNOSTIC"
            or plan.get("synthetic_generator_protocol_id") != sa.PROTOCOL
            or plan.get("fold") != 0 or plan.get("ct_channel") != "CT1"
            or plan.get("generator_seed") != 42 or plan.get("training_seed") != 42
            or plan.get("arm") != "T" or plan.get("objective") != "BIN"
            or plan.get("initial_lrs") != list(LR_VALUES)
            or plan.get("epochs") != 1 or plan.get("steps_per_run") != 261
            or plan.get("effective_batch") != 16 or plan.get("micro_batch") != 2
            or plan.get("probe_steps") != list(PROBE_STEPS)
            or plan.get("optimizer") != dict(name="AdamW", betas=[0.9, 0.999], eps=1e-8,
                                              weight_decay=0.01, amsgrad=False)
            or plan.get("scheduler") != dict(name="CosineAnnealingLR", T_max=30,
                                              eta_min=0.0, step_at_epoch_end=True)
            or any(plan.get(k) is not False for k in
                   ("dev_window_read", "calibration_window_read", "heldout_window_read",
                    "threshold_generation", "paper_result_eligible"))
            or plan.get("actual_roles_read") != ["fit"]):
        raise ValueError("onset recipe is not separately frozen with the exact fit-only budget")
    if any(plan.get("diagnostic_rules", {}).get(k) != v for k, v in RULES.items()):
        raise ValueError("diagnostic rules differ from the recorded implementation")
    live = source_hashes()
    if (plan.get("baseline_source_files") != inputs["source_files"]
            or plan.get("candidate_source_files") != live
            or plan.get("input_file_sha256") != {
                "split_manifest.json": p1b.file_sha(args.prepared / "split_manifest.json"),
                "events.jsonl": p1b.file_sha(args.bank_result / "events.jsonl"),
                "cache_manifest.json": p1b.file_sha(args.cache_run / "cache/manifest.json"),
                "cache_result.json": p1b.file_sha(args.cache_run / "result.json")}):
        raise ValueError("onset plan source/input file pins differ")
    refs = resolve_probe_pairs(inputs, plan.get("health_probes"))
    if len(refs) != 32 or len(inputs["indexes"]["fit"].bases) != 2088:
        raise ValueError("real onset diagnostic requires 16 fit devices and 2088 bases")
    baseline = p1b._pinned(args.baseline_result, args.baseline_result_sha256, "original T-BIN result")
    if (baseline.get("schema") != p1b.RESULT_SCHEMA or baseline.get("gate") != "PASS"
            or baseline.get("arm") != "T" or baseline.get("objective") != "BIN"
            or baseline.get("scope") != "P1b_real_data"
            or baseline.get("source_files") != inputs["source_files"]
            or baseline.get("input_evidence") != inputs["evidence"]
            or baseline.get("real_data_optimizer_steps") != 7830):
        raise ValueError("original T-BIN result differs from the preserved recipe/input evidence")
    first = baseline["epochs"][0]
    reference = {"result_file_sha256": args.baseline_result_sha256, "result_payload_sha256": baseline["sha256"],
                 "initial_state_sha256": baseline["initial_state_sha256"],
                 "epoch": first["epoch"], "train_loss": first["train_loss"],
                 "optimizer_steps": first["optimizer_steps"], "lr_used": first["lr_used_this_epoch"],
                 "fit_order_manifest_sha256": first["fit_order_manifest_sha256"]}
    if reference != plan.get("baseline_reference") or first["epoch"] != 1 or first["optimizer_steps"] != 261:
        raise ValueError("original first-epoch reference is not pinned in the plan")
    provenance_path = args.baseline_result.with_name("provenance.json")
    if p1b.file_sha(provenance_path) != baseline["provenance_file_sha256"]:
        raise ValueError("original T-BIN provenance file pin differs")
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get("scope") != "P1b_real_data" or provenance.get("arm") != "T"
            or provenance.get("objective") != "BIN" or provenance.get("environment_profile") != "SERVER-TRAINING"):
        raise ValueError("original T-BIN provenance scope differs")
    original_runtime = {k: provenance[k] for k in
                        ("hostname", "python_version", "numpy_version", "torch_version", "cuda_runtime_version")}
    original_runtime.update(device=baseline["device"], gpu_name=baseline["gpu_name"],
                            cuda_visible_devices=baseline["cuda_visible_devices"],
                            cublas_workspace_config=baseline["determinism"]["cublas_workspace_config"])
    expected_runtime = plan.get("runtime_reference", {})
    if (set(expected_runtime) != set(RUNTIME_FIELDS)
            or any(expected_runtime[k] != v for k, v in original_runtime.items())
            or expected_runtime["cuda_device_index"] != 0 or expected_runtime["device"] != "cuda:0"):
        raise ValueError("runtime reference differs from the preserved original environment")
    fixture = p1b._pinned(args.diagnostic_fixtures, args.diagnostic_fixtures_sha256,
                          "diagnostic fixtures")
    expected = sum(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                   for p in NEW_FILES if p.startswith("tests/")
                   for n in ast.walk(ast.parse((ROOT / p).read_text())))
    if (fixture.get("schema") != FIXTURE_SCHEMA or fixture.get("gate") != "PASS"
            or fixture.get("source_files") != live
            or fixture.get("unit_tests") != dict(tests_run=expected, failures=0, errors=0, skipped=0)
            or type(fixture.get("fixture_optimizer_steps")) is not int
            or fixture["fixture_optimizer_steps"] < 1
            or type(fixture.get("real_data_optimizer_steps")) is not int
            or fixture["real_data_optimizer_steps"] != 0
            or fixture.get("runtime_identity") != expected_runtime
            or not isinstance(fixture.get("git_commit_sha"), str)
            or len(fixture["git_commit_sha"]) != 40
            or any(c not in "0123456789abcdef" for c in fixture["git_commit_sha"])):
        raise ValueError("diagnostic source-bound fixtures are incomplete")
    inputs.update(plan=plan, probe_pairs=refs, candidate_source_files=live,
                  diagnostic_fixture_payload_sha256=fixture["sha256"],
                  candidate_git_commit_sha=fixture["git_commit_sha"], baseline_reference=reference,
                  baseline_provenance_file_sha256=baseline["provenance_file_sha256"])
    return inputs


def _driver_identity(stdout, gpu_uuid):
    rows = [row for row in csv.reader(stdout.splitlines())
            if len(row) == 4 and row[1].strip() == gpu_uuid]
    if len(rows) != 1:
        raise ValueError("nvidia-smi does not uniquely identify the selected CUDA GPU UUID")
    index, _, name, driver = (value.strip() for value in rows[0])
    if not index.isdigit() or not name or not driver:
        raise ValueError("invalid selected GPU driver identity")
    return {"nvidia_smi_index": int(index), "gpu_name": name, "driver_version": driver}


def runtime_identity(device):
    """Record actual CUDA identity before cache reads; no training or RNG setter."""
    th, _, _ = p1b._training()
    if device.type != "cuda" or type(device.index) is not int:
        raise ValueError("explicit CUDA device index required for original replay")
    properties = th.cuda.get_device_properties(device)
    raw_uuid = str(properties.uuid)
    gpu_uuid = raw_uuid if raw_uuid.startswith("GPU-") else "GPU-" + raw_uuid
    output = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True,
                            text=True, timeout=10)
    driver = _driver_identity(output.stdout, gpu_uuid)
    if driver["gpu_name"] != properties.name:
        raise ValueError("selected CUDA and nvidia-smi GPU names differ")
    return {"hostname": platform.node(), "python_version": platform.python_version(),
            "numpy_version": np.__version__, "torch_version": str(th.__version__),
            "cuda_runtime_version": th.version.cuda, "device": str(device), "cuda_device_index": device.index,
            "gpu_uuid": gpu_uuid, **driver, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_version": th.backends.cudnn.version()}


def check_runtime_identity(actual, expected):
    """Exact prospective identity; missing historical driver/UUID are not inferred."""
    if set(actual) != set(RUNTIME_FIELDS) or set(expected) != set(RUNTIME_FIELDS):
        raise ValueError("runtime identity fields differ")
    mismatches = {k: {"actual": actual[k], "expected": expected[k]} for k in RUNTIME_FIELDS
                  if actual[k] != expected[k]}
    if mismatches:
        raise p1b.P1bFailure("original replay runtime identity differs", {"runtime_mismatches": mismatches})
    return {"gate": "PASS_EXACT_RUNTIME_IDENTITY", "fields_compared": list(RUNTIME_FIELDS),
            "historical_limit": "Original persisted provenance lacks cuDNN, driver and physical UUID; those pins describe the prospectively inspected environment"}


@contextmanager
def observation_mode(model, th):
    """Restore mixed module modes and RNG even if an observation raises."""
    modes = [(module, module.training) for module in model.modules()]
    cpu = th.get_rng_state().clone()
    cuda = th.cuda.get_rng_state_all() if th.cuda.is_initialized() else None
    try:
        model.eval()
        with th.no_grad():
            yield
    finally:
        cpu_changed = not th.equal(cpu, th.get_rng_state())
        changed = cpu_changed
        # A CUDA setter calls set_current_seed even for equal bytes. In torch
        # 2.6 this resets cuDNN's RNN dropout-state flag and changes the NEXT
        # training update. Healthy eval observations must not invoke a setter.
        if cuda is not None:
            for i, (before, after) in enumerate(zip(cuda, th.cuda.get_rng_state_all())):
                if not th.equal(before, after):
                    th.cuda.set_rng_state(before, i)
                    changed = True
        elif th.cuda.is_initialized():
            changed = True
        if cpu_changed:
            th.set_rng_state(cpu)
        for module, training in modes:
            module.training = training
        if changed and sys.exc_info()[0] is None:
            raise ValueError("diagnostic observer consumed model RNG or initialized CUDA")


def gradient_norms(model, th):
    """Read the real accumulated backward gradients; never mutate or backward."""
    squares, counts = Counter(), Counter()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not th.isfinite(parameter.grad).all():
            raise ValueError(f"missing/nonfinite gradient: {name}")
        if name.startswith("thermal_encoder.fc_thermal."):
            group = "fc_thermal"
        elif name.startswith("thermal_encoder.lstm."):
            group = "thermal_lstm_" + name.rsplit("_l", 1)[-1]
        elif name.startswith("head."):
            group = "head"
        else:
            group = name.split(".", 1)[0]
        squares[group] += float(parameter.grad.detach().to(th.float64).square().sum())
        counts[group] += parameter.numel()
    result = {k: {"l2": math.sqrt(v), "elements": counts[k]} for k, v in squares.items()}
    if any(not math.isfinite(v["l2"]) for v in result.values()):
        raise ValueError("nonfinite gradient norm")
    return result


def _tensors(loader, refs, arm, device, th, helpers, timing):
    pairs = [loader.load("fit", ref) for ref in refs]
    start = time.perf_counter()
    arrays = {name: np.stack([pair[i] for pair in pairs])
              for i, name in enumerate(("sensor", "thermal"))
              if name in p1b.detectors.ARM_MODALITIES[arm]}
    timing["stack"] += time.perf_counter() - start
    start = time.perf_counter()
    tensors = {name: th.from_numpy(array).to(device=device, dtype=th.float32)
               for name, array in arrays.items()}
    helpers._sync(device)
    timing["transfer"] += time.perf_counter() - start
    return tensors


def observe_fit(model, loader, pairs, *, arm, device, micro_batch=2):
    """Collect native logits and parity-checked gate statistics on fixed fit pairs."""
    th, helpers, _ = p1b._training()
    diagnostics = p1b._leaf("_bin_onset_diagnostics", "src/models/anomaly_bin_diagnostics.py")
    examples = p1b.paired_examples(pairs)
    logits, features, layers, parts = [], [], {}, []
    timing = Counter()
    loader.begin_phase()
    with observation_mode(model, th):
        for start in range(0, len(examples), micro_batch):
            tensors = _tensors(loader, examples[start:start + micro_batch], arm,
                               device, th, helpers, timing)
            output = model(**tensors)["logit"]
            if output.shape != (len(examples[start:start + micro_batch]),) or not th.isfinite(output).all():
                raise ValueError("probe logit shape/nonfinite mismatch")
            logits.extend(output.cpu().tolist())
            if "thermal" in tensors:
                measured = diagnostics.thermal_lstm_diagnostics(model, tensors["thermal"])
                if any(measured[p].get("passed") is not True
                       for p in ("parity_sequence", "parity_encoder_output")):
                    raise ValueError("manual LSTM diagnostic failed native elementwise parity")
                parts.append(measured)
                features.extend(measured["thermal_z_values"])
                for layer in measured["layers"]:
                    accumulated = layers.setdefault(layer["layer"], {g: Counter() for g in ("i", "f", "g", "o")})
                    for gate, values in layer["gates"].items():
                        name = {"input": "i", "forget": "f", "cell": "g", "output": "o"}[gate]
                        accumulated[name]["abs_gt6_count"] += values["abs_gt6_count"]
                        accumulated[name]["total_count"] += values["total_count"]
    gates = {str(layer): {g: {**dict(v), "abs_gt6_fraction": v["abs_gt6_count"] / v["total_count"]}
                         for g, v in values.items()} for layer, values in layers.items()}
    z = np.asarray(features, dtype=np.float64) if features else None
    if z is not None and (z.ndim != 2 or len(z) != len(examples) or not np.isfinite(z).all()):
        raise ValueError("probe thermal z shape/nonfinite mismatch")
    groups = {}
    for family in ("sensor-only", "thermal-only"):
        indices = [i for i, pair in enumerate(pairs) if pair.family == family]
        gaps = [abs(logits[2 * i + 1] - logits[2 * i]) for i in indices]
        group = {"pair_count": len(indices), "max_abs_logit_gap": max(gaps),
                 "mean_abs_logit_gap": float(np.mean(gaps)),
                 "exact_equal_logit_pairs": sum(g == 0 for g in gaps),
                 "mean_signed_positive_minus_normal_logit": float(np.mean([logits[2 * i + 1] - logits[2 * i] for i in indices])),
                 "positive_higher_logit_pairs": sum(logits[2 * i + 1] > logits[2 * i] for i in indices),
                 "balanced_probe_BCE": float(np.mean([0.5 * (np.logaddexp(0.0, logits[2 * i]) +
                                                              np.logaddexp(0.0, -logits[2 * i + 1])) for i in indices]))}
        if z is not None:
            rows = [p for i in indices for p in (2 * i, 2 * i + 1)]
            group["thermal_z_mean_feature_std"] = float(z[rows].std(axis=0, ddof=0).mean())
            group["max_pair_abs_z_gap"] = float(max(np.abs(z[2 * i + 1] - z[2 * i]).max() for i in indices))
        groups[family] = group
    combined = diagnostics.combine_diagnostics(parts) if parts else None
    return {"pair_count": len(pairs), "logits_normal_then_positive": logits,
            "thermal_z": features, "groups": groups, "layers": gates,
            "parity": {p: combined[p] for p in ("parity_sequence", "parity_encoder_output")} if combined else None,
            "fc_thermal": combined["fc_thermal"] if combined else None,
            "diagnostic_scope": "fit_only_strength4_pathway_probe"}


def classify_snapshot(measured):
    """Output sensitivity, never a declaration that an untrained model learned."""
    visible = measured["groups"]["thermal-only"]
    std = visible["thermal_z_mean_feature_std"]
    gap = visible["max_abs_logit_gap"]
    if not all(math.isfinite(v) for v in (std, gap)):
        raise ValueError("nonfinite diagnostic criterion")
    if std < 0 or gap < 0:
        raise ValueError("invalid diagnostic criterion range")
    if std <= 1e-6 and gap <= 1e-6:
        return "collapse_like"
    if std > 1e-4 and gap > 1e-6:
        return "responsive_at_probe"
    return "intermediate"


def check_baseline_replay(record, reference):
    """Aggregate consistency with P1b, not proof of every raw-data update."""
    if record["initial_lr"] != 1e-3:
        return {"gate": "NOT_APPLICABLE_DIFFERENT_LR"}
    comparisons = {"initial_state_sha256": record["initial_state_tensor_sha256"],
                   "train_loss": record["train_loss"], "optimizer_steps": record["optimizer_steps_total"],
                   "fit_order_manifest_sha256": record["fit_order_manifest_sha256"]}
    mismatches = {k: {"actual": v, "expected": reference[k]} for k, v in comparisons.items() if v != reference[k]}
    if mismatches:
        raise p1b.P1bFailure("original first-epoch aggregate consistency failed", {"mismatches": mismatches})
    return {"gate": "PASS_EXACT_AGGREGATES", "comparisons": comparisons,
            "scope": "Initial state/order/exact mean train loss/step count; no original per-step states exist for comparison"}


def onset_interval(probes):
    previous = None
    for probe in probes:
        if probe["classification"] == "collapse_like":
            return {"lower_exclusive": previous, "upper_inclusive": probe["step"],
                    "baseline_already_collapse_like": previous is None}
        previous = probe["step"]
    return {"not_observed_through_step": previous, "prevention_established": False}


def train_first_epoch(inputs, loader, *, initial_lr, device, scope, arm="T", seed=42,
                      nominal_batch=16, micro_batch=2, probe_steps=PROBE_STEPS,
                      observer=None, on_probe=None, on_step=None, progress=None,
                      model_factory=None):
    """Original first-epoch update formula with optional, observational callbacks.

    CPU/tiny batch callers must explicitly declare synthetic_fixture. Real budget
    is fixed by main's gates, and no dev data is referenced by this loop.
    """
    if initial_lr not in LR_VALUES or arm not in ("S", "T", "F"):
        raise ValueError("unknown onset LR/arm")
    if scope not in ("synthetic_fixture", "P1c_fit_diagnostic"):
        raise ValueError("explicit diagnostic/fixture scope required")
    for v, name in ((nominal_batch, "nominal_batch"), (micro_batch, "micro_batch")):
        p1b._positive_int(v, name)
    if nominal_batch % 2 or type(seed) is not int or seed < 0:
        raise ValueError("complete pairs and a nonnegative integer seed required")
    if tuple(sorted(set(probe_steps))) != tuple(probe_steps) or any(type(s) is not int or s < 0 for s in probe_steps):
        raise ValueError("probe steps must be sorted unique nonnegative integers")
    th, helpers, models_module = p1b._training()
    if th.get_default_dtype() != th.float32:
        raise ValueError("FP32 default dtype required")
    determinism = helpers.enable_determinism()
    th.manual_seed(seed)
    if model_factory is None:
        models, initial_manifest = models_module.build_arm_set("BIN", seed=seed)
        model = models[arm].to(device)
        del models
    else:
        model, initial_manifest = model_factory(seed)
        model = model.to(device)
    initial_sha = models_module.state_tensor_sha256(model)
    optimizer = th.optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-8,
                              weight_decay=0.01, amsgrad=False)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    sampler = bs.BasePairedSampler(inputs["indexes"]["fit"], seed, nominal_batch // 2, 30)
    batches = [p1b.paired_examples(p) for p in sampler.epoch_batches(1)]
    if not batches or any(s > len(batches) for s in probe_steps):
        raise ValueError("probe step outside the first epoch")
    progress = {} if progress is None else progress
    progress.update(optimizer_steps_completed=0, phase="model_setup")
    timing, exposures, batch_sizes = Counter(), Counter(), Counter()
    stream, probes, train_terms = hashlib.sha256(), [], []
    started = time.perf_counter()
    model.train()
    loader.begin_phase()

    baseline_probe_groups = {}
    def measure(step):
        if observer is None or step not in probe_steps:
            return
        progress.update(phase="probe", probe_step=step)
        tick = time.perf_counter()
        with observation_mode(model, th):
            measured = observer(model, step)
        helpers._sync(device)
        entry = {"step": step, "measurement": measured, "wall_seconds": time.perf_counter() - tick}
        if "0" in measured.get("layers", {}) and "thermal-only" in measured.get("groups", {}):
            entry["classification"] = classify_snapshot(measured)
        for family, group in measured.get("groups", {}).items():
            if "balanced_probe_BCE" not in group:
                continue
            if step == 0:
                baseline_probe_groups[family] = dict(group)
            original = baseline_probe_groups.get(family)
            if original is not None:
                group["BCE_change_from_step0"] = group["balanced_probe_BCE"] - original["balanced_probe_BCE"]
                group["mean_abs_logit_gap_change_from_step0"] = group["mean_abs_logit_gap"] - original["mean_abs_logit_gap"]
        timing["probe"] += entry["wall_seconds"]
        probes.append(entry)
        if on_probe:
            tick = time.perf_counter()
            on_probe(entry)
            timing["record_io"] += time.perf_counter() - tick

    measure(0)
    for number, refs in enumerate(batches, 1):
        progress.update(phase="train", batch=number, epoch=1)
        loader.context = dict(progress)
        optimizer.zero_grad(set_to_none=True)
        actual, total = len(refs), 0.0
        batch_sizes[actual] += 1
        projection = {"count": 0, "abs_sum": 0.0, "abs_max": 0.0}
        def read_projection(module, args, output):
            tick = time.perf_counter()
            detached = output.detach()
            if not th.isfinite(detached).all():
                raise ValueError("nonfinite actual train fc_thermal output")
            projection["count"] += detached.numel()
            projection["abs_sum"] += float(detached.to(th.float64).abs().sum())
            projection["abs_max"] = max(projection["abs_max"], float(detached.abs().max()))
            timing["train_projection_observation"] += time.perf_counter() - tick
        fc = getattr(getattr(model, "thermal_encoder", None), "fc_thermal", None)
        handle = fc.register_forward_hook(read_projection) if fc is not None else None
        try:
            total, projection_seconds = _train_batch(model, loader, refs, arm, device, th, helpers,
                                                       micro_batch, timing, exposures, stream, progress)
        finally:
            if handle is not None:
                handle.remove()
        projection["abs_mean"] = projection["abs_sum"] / projection["count"] if projection["count"] else None
        timing["training_compute"] -= projection_seconds
        progress["phase"] = "gradient_observation"
        tick = time.perf_counter()
        norms = gradient_norms(model, th)
        helpers._sync(device)
        timing["gradient_observation"] += time.perf_counter() - tick
        progress["phase"] = "optimizer_step"
        tick = time.perf_counter()
        optimizer.step()
        progress["optimizer_steps_completed"] = number
        helpers._sync(device)
        timing["training_compute"] += time.perf_counter() - tick
        train_terms.append(total * actual)
        if on_step:
            progress["phase"] = "step_record"
            tick = time.perf_counter()
            on_step({"step": number, "train_loss": total, "actual_batch": actual,
                     "lr_used": scheduler.get_last_lr()[0], "gradient_norms": norms,
                     "train_fc_thermal": projection,
                     "train_batch_input_scope": "Changes by step; auxiliary trace is not a fixed-probe onset measurement"})
            timing["record_io"] += time.perf_counter() - tick
        measure(number)
    lr_used = scheduler.get_last_lr()[0]
    scheduler.step()
    helpers._sync(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    record = _epoch_record(inputs, loader, initial_lr, arm, seed, scope, nominal_batch, micro_batch,
                           sampler, batches, scheduler, lr_used, exposures, batch_sizes, stream,
                           initial_manifest, initial_sha, determinism, timing, started, models_module, state, probes, train_terms)
    return record, state


def _train_batch(model, loader, refs, arm, device, th, helpers, micro_batch,
                 timing, exposures, stream, progress):
    total = 0.0
    observation_before = timing["train_projection_observation"]
    actual = len(refs)
    for start, stop, weight in helpers.partition_microbatches(actual, micro_batch):
        progress["micro"] = [start, stop]
        tensors = _tensors(loader, refs[start:stop], arm, device, th, helpers, timing)
        tick = time.perf_counter()
        logit = model(**tensors)["logit"]
        labels = th.tensor([r.label for r in refs[start:stop]], dtype=th.float32, device=device)
        if logit.shape != labels.shape:
            raise ValueError("BIN logit shape differs from labels")
        loss = th.nn.functional.binary_cross_entropy_with_logits(logit, labels) * weight
        value = float(loss.detach())
        if not math.isfinite(value):
            raise ValueError("nonfinite train loss")
        loss.backward()
        helpers._sync(device)
        timing["training_compute"] += time.perf_counter() - tick
        total += value
        for ref in refs[start:stop]:
            exposures["fit_positive" if ref.label else "fit_Normal"] += 1
            stream.update((json.dumps({"epoch": 1, "base": ref.base_window_id,
                                      "event": ref.event_id, "label": ref.label},
                                     sort_keys=True, separators=(",", ":")) + "\n").encode())
    return total, timing["train_projection_observation"] - observation_before


def _epoch_record(inputs, loader, initial_lr, arm, seed, scope, nominal_batch, micro_batch,
                  sampler, batches, scheduler, lr_used, exposures, batch_sizes, stream,
                  initial_manifest, initial_sha, determinism, timing, started, models_module, state, probes, train_terms):
    record = {"schema": SCHEMA, "recipe_id": RECIPE, "gate": "PASS", "scope": scope,
              "arm": arm, "objective": "BIN", "initial_lr": initial_lr,
              "lr_used_this_epoch": lr_used, "next_epoch_lr": scheduler.get_last_lr()[0],
              "cosine_T_max": 30, "seed": seed, "epochs_completed": 1,
              "effective_batch": nominal_batch, "micro_batch": micro_batch,
              "optimizer_steps_total": len(batches),
              "real_data_optimizer_steps": len(batches) if scope == "P1c_fit_diagnostic" else 0,
              "fixture_optimizer_steps": len(batches) if scope == "synthetic_fixture" else 0,
              "train_loss": math.fsum(train_terms) / sum(exposures.values()),
              "exposures": dict(exposures),
              "actual_batch_sizes": {str(k): v for k, v in sorted(batch_sizes.items())},
              "fit_order_manifest_sha256": sampler.epoch_manifest(1)["sha256"],
              "fit_example_stream_sha256": stream.hexdigest(), "initial_manifest": initial_manifest,
              "initial_state_tensor_sha256": initial_sha,
              "final_state_tensor_sha256": models_module.state_tensor_sha256(state),
              "determinism": determinism, "phase_seconds": dict(timing),
              "loader_phase_seconds": dict(loader.seconds), "loader_counts": dict(loader.counts),
              "wall_seconds": time.perf_counter() - started,
              "probe_steps_completed": [p["step"] for p in probes],
              "checkpoint_selection": "NONE_FINAL_STATE_ONLY", "dev_window_read": False,
              "calibration_window_read": False, "heldout_window_read": False,
              "threshold_generation": False, "paper_result_eligible": False}
    if probes and all("classification" in p for p in probes):
        record["onset"] = onset_interval(probes)
    return record


def _append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _record_failure(out, exception, progress):
    record = {"schema": SCHEMA, "gate": "FAILED", "run_usable": False,
              "timestamp": datetime.now(timezone.utc).isoformat(),
              "exception": type(exception).__name__, "message": str(exception),
              "context": {**progress, **getattr(exception, "context", {})},
              "retried": False, "learning_rate_adjusted": False}
    try:
        record["partial_files_preserved"] = sorted(str(p.relative_to(out))
                                                  for p in out.rglob("*") if p.is_file())
        p1b._write_json(out / "failure.json", record)
    except BaseException as error:
        try:
            print(json.dumps({"original_failure": record, "failure_record_write_error": repr(error)},
                             default=repr, allow_nan=False), file=sys.stderr, flush=True)
        except BaseException:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("prepared", "bank-result", "cache-run", "sampler-fixtures", "runner-fixtures",
                 "diagnostic-fixtures", "plan", "baseline-result", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("cache-result-sha256", "sampler-fixtures-sha256", "runner-fixtures-sha256",
                 "diagnostic-fixtures-sha256", "plan-sha256", "baseline-result-sha256"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--initial-lr", type=float, choices=LR_VALUES, required=True)
    args = parser.parse_args(argv)
    out = p1b._check_output(args)
    if any(out == p.resolve() or p.resolve() in out.parents or out in p.resolve().parents
           for p in (args.plan, args.diagnostic_fixtures, args.baseline_result)):
        raise ValueError("output overlaps diagnostic evidence")
    out.mkdir(parents=True, exist_ok=False)
    progress, cache = dict(stage="inputs", optimizer_steps_completed=0), None
    started = time.perf_counter()
    try:
        inputs = validate_inputs(args)
        progress["stage"] = "environment"
        device, environment, _ = p1b.runtime_environment()
        identity = runtime_identity(device)
        progress["runtime_identity"] = identity
        runtime_gate = check_runtime_identity(identity, inputs["plan"]["runtime_reference"])
        provenance = {"schema": SCHEMA, "timestamp": datetime.now(timezone.utc).isoformat(),
                      "hostname": platform.node(), "python_version": platform.python_version(),
                      "git_commit_sha": inputs["candidate_git_commit_sha"], "git_dirty": True,
                      "environment_profile": "SERVER-TRAINING", "environment": environment,
                      "jetpack_l4t": environment.get("jetpack_l4t"),
                      "sensor_driver_versions": environment.get("sensor_driver_versions", {}),
                      "runtime_identity": identity, "runtime_gate": runtime_gate,
                      "source_files": inputs["candidate_source_files"],
                      "plan_file_sha256": args.plan_sha256, "plan_payload_sha256": inputs["plan"]["sha256"],
                      "diagnostic_fixture_payload_sha256": inputs["diagnostic_fixture_payload_sha256"],
                      "command": list(argv) if argv is not None else sys.argv,
                      "actual_roles_read": ["fit"], "paper_result_eligible": False}
        p1b._write_json(out / "provenance.json", provenance)
        progress["stage"] = "cache_attach"
        cache = p1b.rc.RawNormalCache.attach(inputs["cache_path"]).open()
        if cache.manifest["sha256"] != inputs["cache_manifest"]["sha256"]:
            raise ValueError("cache manifest changed after metadata gate")
        loader = p1b.PairLoader(inputs, cache.reader(verify_each_read=False))
        probe_loader = p1b.PairLoader(inputs, cache.reader(verify_each_read=False))
        progress["stage"] = "first_epoch"
        record, state = train_first_epoch(
            inputs, loader, initial_lr=args.initial_lr, device=device, scope="P1c_fit_diagnostic",
            observer=lambda model, step: observe_fit(model, probe_loader, inputs["probe_pairs"],
                                                      arm="T", device=device),
            on_probe=lambda e: _append_json(out / "probes.jsonl", e),
            on_step=lambda e: _append_json(out / "steps.jsonl", e), progress=progress)
        if record["optimizer_steps_total"] != REAL_STEPS or record["probe_steps_completed"] != list(PROBE_STEPS):
            raise ValueError("incomplete diagnostic budget or probe coverage")
        progress["stage"] = "final_state"
        th, _, models_module = p1b._training()
        th.save({"model_state_dict": state, "epoch": 1, "selection": "NONE_FINAL_STATE_ONLY"},
                out / "final-state.pt")
        recovered = th.load(out / "final-state.pt", map_location="cpu", weights_only=True)
        if models_module.state_tensor_sha256(recovered["model_state_dict"]) != record["final_state_tensor_sha256"]:
            raise ValueError("final-state serialization parity failed")
        progress["stage"] = "baseline_replay"
        record["baseline_replay"] = check_baseline_replay(record, inputs["baseline_reference"])
        progress["stage"] = "final_inputs"
        final = validate_inputs(args)
        if any(final[k] != inputs[k] for k in ("evidence", "candidate_source_files", "plan", "candidate_git_commit_sha",
                                               "diagnostic_fixture_payload_sha256", "baseline_reference",
                                               "baseline_provenance_file_sha256")):
            raise ValueError("sources, plan or evidence changed during diagnostics")
        record.update(source_files=inputs["candidate_source_files"],
                      plan_payload_sha256=inputs["plan"]["sha256"],
                      input_evidence=inputs["evidence"],
                      diagnostic_fixture_payload_sha256=inputs["diagnostic_fixture_payload_sha256"],
                      baseline_reference=inputs["baseline_reference"],
                      baseline_provenance_file_sha256=inputs["baseline_provenance_file_sha256"],
                      runtime_identity=identity, runtime_gate=runtime_gate,
                      provenance_file_sha256=p1b.file_sha(out / "provenance.json"),
                      final_state_file_sha256=p1b.file_sha(out / "final-state.pt"),
                      probes_file_sha256=p1b.file_sha(out / "probes.jsonl"),
                      steps_file_sha256=p1b.file_sha(out / "steps.jsonl"),
                      probe_loader_counts=dict(probe_loader.counts),
                      probe_loader_phase_seconds=dict(probe_loader.seconds),
                      wall_until_final_record_seconds=time.perf_counter() - started)
        record["sha256"] = sa.digest_json(record)
        progress["stage"] = "final_record"
        p1b._write_json(out / "result.json", record)
        print(json.dumps({"gate": "PASS", "out": str(out), "initial_lr": args.initial_lr,
                          "optimizer_steps": record["optimizer_steps_total"]}), flush=True)
        return 0
    except BaseException as exception:
        _record_failure(out, exception, progress)
        raise
    finally:
        if cache is not None:
            cache._data = None


if __name__ == "__main__":
    raise SystemExit(main())
