#!/usr/bin/env python3
"""One frozen T-BIN LR1e-4 stability run, separate from P1b and onset sources.

The CLI has no sweep, resume, CPU, short-budget or retry option. Metadata/source
gates precede cache attachment. Epoch1 must reproduce the preserved onset run
before any dev forward. Fit probe categories never stop the planned30 epochs.

r3 (review (52)-(53)): the probe's manual-LSTM parity check uses the frozen
PROBE_PARITY tolerance instead of the diagnostics module's 1e-5/1e-6 defaults,
which float32 accumulation exceeded on trained weights at epoch 10 of the first
attempt. The onset observer and diagnostics module stay byte-identical (they are
bound by the preserved LR1e-4 replay); only the tolerance passed to the gate
recomputation changes, and only while a probe runs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import importlib.util
spec = importlib.util.spec_from_file_location("_bin_stability_onset", ROOT / "src/diagnose_anomaly_bin_onset.py")
onset = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = onset
spec.loader.exec_module(onset)
p1b, bs, sa = onset.p1b, onset.bs, onset.sa
SCHEMA = "exp004-bin-stability-result-v1"
PLAN_SCHEMA = "exp004-bin-stability-plan-v1"
FIXTURE_SCHEMA = "exp004-bin-stability-fixtures-v1"
RECIPE = "exp004-tbin-lr1e4-stability-v1"
NEW_FILES = ("src/diagnose_anomaly_bin_stability.py", "tests/test_anomaly_bin_stability.py")
SOURCE_FILES = (*onset.SOURCE_FILES, *NEW_FILES)
TEST_FILES = tuple(p for p in (*onset.NEW_FILES, *NEW_FILES) if p.startswith("tests/"))
EPOCHS, REAL_STEPS = 30, 7830
PROBE_STEPS = (0, 1, 2, 5, 10, 20, 50, 100, 261, 262, 263, 266, 271, 281, 311, 361,
               522, 523, 524, 527, 532, 542, 572, 622, 783, 1305, 2610, 3915, 5220, 6525, 7830)
PROBE_PARITY = {"rtol": 1e-4, "atol": 1e-4}
DIAGNOSTICS_MODULE = ("_bin_onset_diagnostics", "src/models/anomaly_bin_diagnostics.py")
SCREEN = {"evaluation_epochs": [25, 30], "balanced_thermal_only_probe_BCE_max": math.log(2) - 0.02,
          "contrast_BCE_gain_min": 0.02}
FROZEN_VALUES = {"schema": PLAN_SCHEMA, "recipe_id": RECIPE, "status": "FROZEN_STABILITY",
                 "fold": 0, "ct_channel": "CT1", "generator_seed": 42, "training_seed": 42,
                 "arm": "T", "objective": "BIN", "initial_lr": 1e-4, "epochs": 30,
                 "effective_batch": 16, "micro_batch": 2, "steps_per_epoch": 261,
                 "proposed_optimizer_steps": 7830, "probe_steps": list(PROBE_STEPS),
                 "optimizer": dict(name="AdamW", betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01, amsgrad=False),
                 "scheduler": dict(name="CosineAnnealingLR", T_max=30, eta_min=0.0, step_at_epoch_end=True),
                 "actual_roles_read": ["fit", "dev"], "negative_control_required": True,
                 "first_epoch_gate_before_dev": True, "collapse_early_stop": False,
                 "resume": False, "retry": False, "calibration_window_read": False,
                 "heldout_window_read": False, "threshold_generation": False, "paper_result_eligible": False,
                 "probe_parity_tolerance": dict(PROBE_PARITY)}


def source_hashes():
    return {p: p1b.file_sha(ROOT / p) for p in SOURCE_FILES}


def check_recipe(plan):
    for key, value in FROZEN_VALUES.items():
        if plan.get(key) != value or type(plan.get(key)) is not type(value):
            raise ValueError(f"stability frozen recipe differs: {key}")
    if any(plan.get("diagnostic_rules", {}).get(k) != v for k, v in onset.RULES.items()):
        raise ValueError("stability output category rules differ")
    if any(plan.get("proposed_fit_signal_screen", {}).get(k) != v for k, v in SCREEN.items()):
        raise ValueError("stability exploratory fit screen differs")


def replay_reference(result, file_sha):
    return {"result_payload_sha256": result["sha256"], "result_file_sha256": file_sha,
            **{k: result[k] for k in ("initial_state_tensor_sha256", "final_state_tensor_sha256",
                                     "train_loss", "fit_order_manifest_sha256", "fit_example_stream_sha256")},
            "optimizer_steps": result["optimizer_steps_total"]}


def validate_inputs(args):
    original_args = SimpleNamespace(**vars(args))
    original_args.plan, original_args.plan_sha256 = args.onset_plan, args.onset_plan_sha256
    inputs = onset.validate_inputs(original_args)
    onset_plan = inputs["plan"]
    plan = p1b._pinned(args.plan, args.plan_sha256, "stability plan")
    check_recipe(plan)
    live = source_hashes()
    if (plan.get("candidate_source_files") != live
            or plan.get("baseline_source_files") != inputs["source_files"]
            or plan.get("input_file_sha256") != onset_plan["input_file_sha256"]
            or plan.get("health_probes") != onset_plan["health_probes"]
            or plan.get("runtime_reference") != onset_plan["runtime_reference"]
            or plan.get("onset_plan_file_sha256") != args.onset_plan_sha256
            or plan.get("onset_plan_payload_sha256") != onset_plan["sha256"]):
        raise ValueError("stability source/input/onset pins differ")
    replay = p1b._pinned(args.replay_result, args.replay_result_sha256, "LR1e-4 onset replay")
    if (replay.get("schema") != onset.SCHEMA or replay.get("gate") != "PASS"
            or replay.get("scope") != "P1c_fit_diagnostic" or replay.get("initial_lr") != 1e-4
            or replay.get("arm") != "T" or replay.get("objective") != "BIN"
            or replay.get("real_data_optimizer_steps") != 261
            or replay.get("source_files") != onset.source_hashes()
            or replay.get("input_evidence") != inputs["evidence"]
            or replay.get("plan_payload_sha256") != onset_plan["sha256"]
            or replay.get("runtime_identity") != plan["runtime_reference"]
            or not math.isfinite(replay["train_loss"])):
        raise ValueError("preserved LR1e-4 replay does not match the onset recipe")
    reference = replay_reference(replay, args.replay_result_sha256)
    if reference != plan.get("first_epoch_replay_reference"):
        raise ValueError("stability replay reference differs")
    # File verification only, never loading reference weights into the new run.
    for filename, key in (("final-state.pt", "final_state_file_sha256"),
                          ("provenance.json", "provenance_file_sha256")):
        if p1b.file_sha(args.replay_result.with_name(filename)) != replay[key]:
            raise ValueError("preserved replay artifact changed: " + filename)
    fixture = p1b._pinned(args.stability_fixtures, args.stability_fixtures_sha256, "stability fixtures")
    expected = sum(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                   for p in TEST_FILES for n in ast.walk(ast.parse((ROOT / p).read_text())))
    if (fixture.get("schema") != FIXTURE_SCHEMA or fixture.get("gate") != "PASS"
            or fixture.get("source_files") != live
            or fixture.get("unit_tests") != dict(tests_run=expected, failures=0, errors=0, skipped=0)
            or fixture.get("test_files") != list(TEST_FILES)
            or type(fixture.get("fixture_optimizer_steps")) is not int or fixture["fixture_optimizer_steps"] < 1
            or type(fixture.get("real_data_optimizer_steps")) is not int or fixture["real_data_optimizer_steps"] != 0
            or fixture.get("runtime_identity") != plan["runtime_reference"]
            or len(fixture.get("git_commit_sha", "")) != 40
            or any(c not in "0123456789abcdef" for c in fixture["git_commit_sha"])
            or plan.get("candidate_fixture_evidence") != {
                "file_sha256": args.stability_fixtures_sha256, "payload_sha256": fixture["sha256"]}):
        raise ValueError("stability source-bound fixture evidence is incomplete")
    inputs.update(plan=plan, candidate_source_files=live, first_epoch_reference=reference,
                  candidate_git_commit_sha=fixture["git_commit_sha"],
                  stability_fixture_payload_sha256=fixture["sha256"])
    return inputs


def contrast_metrics(normal, positive):
    n, p = np.asarray(normal, np.float64), np.asarray(positive, np.float64)
    if n.shape != p.shape or n.ndim != 1 or not n.size or not np.isfinite(n).all() or not np.isfinite(p).all():
        raise ValueError("invalid paired logits for contrast")
    # Match onset's per-pair arithmetic, including exact zero for equal logits.
    actual = float(np.mean([0.5 * (np.logaddexp(0.0, a) + np.logaddexp(0.0, -b)) for a, b in zip(n, p)]))
    midpoint = (n + p) / 2
    equal = float(np.mean([0.5 * (np.logaddexp(0.0, m) + np.logaddexp(0.0, -m)) for m in midpoint]))
    if not all(math.isfinite(x) for x in (actual, equal)):
        raise ValueError("nonfinite contrast metric")
    return {"balanced_probe_BCE": actual, "paired_midpoint_BCE": equal, "contrast_BCE_gain": equal - actual}


class NegativeControlLoader:
    """Read the already normalized probe inputs; no additional data/model call."""
    def __init__(self, loader):
        self.loader, self.previous, self.rows = loader, None, []

    def begin_phase(self):
        self.previous = None
        self.loader.begin_phase()

    def load(self, role, ref):
        pair = self.loader.load(role, ref)
        array = np.asarray(pair[1])
        if role != "fit" or array.dtype != np.dtype("float32"):
            raise ValueError("negative control requires normalized FP32 fit inputs")
        if not ref.label:
            self.previous = (ref.base_window_id, array.shape, array.tobytes(order="C"))
        elif ref.family == "sensor-only":
            raw = array.tobytes(order="C")
            if self.previous != (ref.base_window_id, array.shape, raw):
                raise p1b.P1bFailure("sensor-only negative control thermal bytes differ",
                                     {"negative_control_base": ref.base_window_id, "event": ref.event_id})
            self.rows.append({"base_window_id": ref.base_window_id, "event_id": ref.event_id,
                              "dtype": str(array.dtype), "shape": list(array.shape),
                              "normal_thermal_sha256": hashlib.sha256(self.previous[2]).hexdigest(),
                              "positive_thermal_sha256": hashlib.sha256(raw).hexdigest()})
        return pair


class _ProbeParityTolerance:
    """While a probe runs, call the gate recomputation with PROBE_PARITY.

    onset.observe_fit looks thermal_lstm_diagnostics up on the cached diagnostics
    module at call time, so a temporary attribute swap changes the tolerance and
    nothing else. The original function is always put back, also on failure.
    The last parity records are kept so a failure can report the numbers.
    """

    def __enter__(self):
        self.module = self.original = self.last = None
        name, _ = DIAGNOSTICS_MODULE
        if name not in sys.modules and importlib.util.find_spec("torch") is None:
            # Without torch the real observer cannot recompute gates either; loading the
            # module here would only leave a half-registered entry in sys.modules.
            return self
        self.module = p1b._leaf(*DIAGNOSTICS_MODULE)
        self.original = self.module.thermal_lstm_diagnostics

        def tolerant(model, thermal):
            measured = self.original(model, thermal, **PROBE_PARITY)
            self.last = {p: {k: measured[p][k] for k in
                             ("max_abs_error", "max_rel_error", "rtol", "atol", "count", "violations", "passed")}
                         for p in ("parity_sequence", "parity_encoder_output")}
            return measured
        self.module.thermal_lstm_diagnostics = tolerant
        return self

    def __exit__(self, *exc):
        if self.module is not None:
            self.module.thermal_lstm_diagnostics = self.original
        return False


def observe_fit(model, loader, pairs, *, device):
    checked = NegativeControlLoader(loader)
    with _ProbeParityTolerance() as parity:
        try:
            measured = onset.observe_fit(model, checked, pairs, arm="T", device=device, micro_batch=2)
        except ValueError as error:
            if "parity" not in str(error):
                raise
            raise p1b.P1bFailure("probe manual-LSTM parity exceeded the frozen tolerance",
                                 {"probe_parity_tolerance": dict(PROBE_PARITY),
                                  "last_microbatch_parity": parity.last}) from error
    measured["probe_parity_tolerance"] = dict(PROBE_PARITY)
    values = measured["logits_normal_then_positive"]
    for family, group in measured["groups"].items():
        ids = [i for i, pair in enumerate(pairs) if pair.family == family]
        metrics = contrast_metrics([values[2*i] for i in ids], [values[2*i+1] for i in ids])
        if metrics["balanced_probe_BCE"] != group["balanced_probe_BCE"]:
            raise ValueError("native probe BCE and contrast arithmetic differ")
        group.update(metrics)
    control = measured["groups"]["sensor-only"]
    if (len(checked.rows) != control["pair_count"]
            or control["exact_equal_logit_pairs"] != control["pair_count"]
            or control["max_abs_logit_gap"] != 0 or control["contrast_BCE_gain"] != 0):
        raise p1b.P1bFailure("sensor-only negative control logits/contrast differ", {"negative_control": control})
    measured["negative_control"] = {"gate": "PASS_EXACT_INPUT_LOGIT_CONTRAST", "pairs": checked.rows,
                                    "pair_count": control["pair_count"], "contrast_BCE_gain": 0.0}
    return measured


def check_first_epoch(actual, reference):
    keys = ("initial_state_tensor_sha256", "final_state_tensor_sha256", "train_loss", "optimizer_steps",
            "fit_order_manifest_sha256", "fit_example_stream_sha256")
    mismatches = {k: {"actual": actual.get(k), "expected": reference[k]} for k in keys if actual.get(k) != reference[k]}
    if mismatches:
        raise p1b.P1bFailure("LR1e-4 first-epoch replay failed before dev", {"first_epoch_mismatches": mismatches})
    return {"gate": "PASS_EXACT_FIRST_EPOCH", "comparisons": {k: actual[k] for k in keys},
            "before_dev_forward": True, "scope": "Exact tensor endpoints, train mean and recorded order/stream; no claim of every intermediate state"}


def parse_gpu_activity(gpu_text, process_text, gpu_uuid, own_pid):
    selected = [r for r in csv.reader(gpu_text.splitlines()) if len(r) == 4 and r[0].strip() == gpu_uuid]
    if len(selected) != 1:
        raise ValueError("GPU activity UUID is missing/ambiguous")
    def amount(value):
        value = value.strip()
        if value in ("N/A", "[N/A]"):
            return None
        number = int(value)
        if number < 0:
            raise ValueError("invalid GPU activity amount")
        return number
    processes = []
    for row in csv.reader(process_text.splitlines()):
        if not row:
            continue
        if len(row) != 4:
            raise ValueError("invalid compute process inventory")
        uuid, pid, name, memory = (v.strip() for v in row)
        if uuid == gpu_uuid:
            processes.append({"pid": int(pid), "process_name": name, "memory_used_MiB": amount(memory),
                              "own_process": int(pid) == own_pid})
    return {"gpu_uuid": gpu_uuid, "own_pid": own_pid, "processes": processes,
            "other_compute_process_count": sum(not p["own_process"] for p in processes),
            "own_process_present": any(p["own_process"] for p in processes),
            "utilization_percent": amount(selected[0][1]), "memory_used_MiB": amount(selected[0][2]),
            "memory_total_MiB": amount(selected[0][3]), "exclusive_access_enforced": False,
            "scope": "Point-in-time inventory, not proof of no contention between observations"}


def gpu_activity(gpu_uuid):
    outputs = []
    for query in ("--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
                  "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"):
        outputs.append(subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                      check=True, capture_output=True, text=True, timeout=10).stdout)
    return {"timestamp": datetime.now(timezone.utc).isoformat(),
            **parse_gpu_activity(*outputs, gpu_uuid, os.getpid())}


def trajectory_summary(probes, *, screen=True):
    collapsed = [i for i, p in enumerate(probes) if p.get("classification") == "collapse_like"]
    result = {"sampled_categories": [{"step": p["step"], "classification": p.get("classification")} for p in probes],
              "onset": onset.onset_interval(probes), "sparse_sampling": True, "prevention_established": False}
    result["collapse_persistent_since_first_sample"] = bool(collapsed) and all(
        p["classification"] == "collapse_like" for p in probes[collapsed[0]:])
    result["responsive_recovery_sampled"] = bool(collapsed) and any(
        p["classification"] == "responsive_at_probe" for p in probes[collapsed[0]+1:])
    if not screen:
        result["fit_signal_screen"] = {"gate": "NOT_EVALUATED_ARTIFICIAL_SHORT_FIXTURE"}
        return result
    by_step = {p["step"]: p for p in probes}
    rows = []
    for epoch in SCREEN["evaluation_epochs"]:
        group = by_step[261*epoch]["measurement"]["groups"]["thermal-only"]
        bce, gain = group["balanced_probe_BCE"], group["contrast_BCE_gain"]
        if not math.isfinite(bce) or not math.isfinite(gain):
            raise ValueError("nonfinite fit screening value")
        rows.append({"epoch": epoch, "balanced_probe_BCE": bce, "contrast_BCE_gain": gain,
                     "passed": bce <= SCREEN["balanced_thermal_only_probe_BCE_max"] and gain >= SCREEN["contrast_BCE_gain_min"]})
    result["fit_signal_screen"] = {"gate": "PASS" if all(r["passed"] for r in rows) else "UNMET", "epochs": rows,
                                   "scope": "Exploratory compute-budget screen on selected trained events; may reflect memorization, not generalization. UNMET does not mean zero learning."}
    return result


def _tensors(loader, role, refs, device, th, helpers, timing):
    pairs = [loader.load(role, ref) for ref in refs]
    tick = time.perf_counter()
    array = np.stack([p[1] for p in pairs])
    timing[role + "_stack"] += time.perf_counter() - tick
    tick = time.perf_counter()
    tensor = th.from_numpy(array).to(device=device, dtype=th.float32)
    helpers._sync(device)
    timing[role + "_transfer"] += time.perf_counter() - tick
    return {"thermal": tensor}


def evaluate_dev(model, loader, refs, device, th, helpers, timing, exposures, progress, micro_batch):
    """Original P1b class/device/base/event weighted BCE, once per epoch."""
    terms = []
    loader.begin_phase()
    with onset.observation_mode(model, th):
        for position in range(0, len(refs), micro_batch):
            chunk = refs[position:position+micro_batch]
            progress.update(phase="dev", dev_position=position)
            loader.context = dict(progress)
            tensors = _tensors(loader, "dev", chunk, device, th, helpers, timing)
            tick = time.perf_counter()
            logits = model(**tensors)["logit"]
            labels = th.tensor([r.label for r in chunk], dtype=th.float32, device=device)
            if logits.shape != labels.shape:
                raise ValueError("dev BIN logit shape differs")
            loss = th.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            weights = th.tensor([r.weight for r in chunk], dtype=th.float64, device=device)
            if not th.isfinite(loss).all() or not th.isfinite(weights).all() or (weights <= 0).any():
                raise p1b.P1bFailure("nonfinite dev BCE/invalid weights", dict(progress))
            terms.append(float((loss.to(th.float64) * weights).sum()))
            helpers._sync(device)
            timing["dev_compute"] += time.perf_counter() - tick
            for ref in chunk:
                exposures["dev_positive" if ref.label else "dev_Normal"] += 1
    value = math.fsum(terms)
    p1b._finite(value, "nonfinite dev weighted mean", progress)
    return value


def train_loop(inputs, loader, *, device, scope, epochs=EPOCHS, nominal_batch=16, micro_batch=2,
               seed=42, probe_steps=PROBE_STEPS, observer=None, on_probe=None, on_step=None,
               on_epoch=None, on_terminal=None, on_improvement=None, on_boundary=None,
               progress=None, model_factory=None, replay=None):
    """Artificial fixtures may reduce budget; main always uses the exact frozen budget."""
    for value, name in ((epochs, "epochs"), (nominal_batch, "nominal_batch"), (micro_batch, "micro_batch")):
        p1b._positive_int(value, name)
    if scope not in ("synthetic_fixture", "P1c_fit_dev_stability") or nominal_batch % 2 or seed != 42:
        raise ValueError("explicit stability scope, seed42 and complete pairs required")
    if scope == "P1c_fit_dev_stability" and (epochs != 30 or nominal_batch != 16 or micro_batch != 2
                                            or model_factory is not None or observer is None or replay is None
                                            or tuple(probe_steps) != PROBE_STEPS):
        raise ValueError("real stability budget/observer/replay must match frozen recipe")
    if tuple(sorted(set(probe_steps))) != tuple(probe_steps) or any(type(s) is not int or s < 0 for s in probe_steps):
        raise ValueError("invalid probe steps")
    th, helpers, models_module = p1b._training()
    if th.get_default_dtype() != th.float32:
        raise ValueError("FP32 default dtype required")
    determinism = helpers.enable_determinism()
    th.manual_seed(seed)
    if model_factory is None:
        models, manifest = models_module.build_arm_set("BIN", seed=seed)
        model = models["T"].to(device)
        del models
    else:
        model, manifest = model_factory(seed)
        model = model.to(device)
    initial_sha = models_module.state_tensor_sha256(model)
    optimizer = th.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    sampler = bs.BasePairedSampler(inputs["indexes"]["fit"], seed, nominal_batch//2, 30)
    per_epoch = len(tuple(sampler.epoch_batches(1)))
    if not per_epoch or any(s > per_epoch*epochs for s in probe_steps):
        raise ValueError("probe step outside stability budget")
    dev_refs = bs.dev_examples(inputs["indexes"]["dev"])
    if not dev_refs:
        raise ValueError("dev examples required")
    progress = {} if progress is None else progress
    progress.update(phase="model_setup", optimizer_steps_completed=0)
    timing, exposures, batch_sizes = Counter(), Counter(), Counter()
    stream, history, probes, baseline, best = hashlib.sha256(), [], [], {}, None
    started, peak = time.perf_counter(), 0
    model.train()

    def callback(fn, *args, phase="record_io"):
        if fn:
            tick = time.perf_counter()
            fn(*args)
            timing[phase] += time.perf_counter() - tick

    def measure(step):
        if observer is None or step not in probe_steps:
            return
        progress.update(phase="probe", probe_step=step)
        tick = time.perf_counter()
        with onset.observation_mode(model, th):
            measured = observer(model, step)
        helpers._sync(device)
        entry = {"step": step, "measurement": measured, "classification": onset.classify_snapshot(measured),
                 "wall_seconds": time.perf_counter() - tick}
        timing["probe"] += entry["wall_seconds"]
        for family, group in measured["groups"].items():
            if step == 0:
                baseline[family] = dict(group)
            for key in ("balanced_probe_BCE", "mean_abs_logit_gap", "contrast_BCE_gain"):
                if key in group and key in baseline.get(family, {}):
                    group[key + "_change_from_step0"] = group[key] - baseline[family][key]
        probes.append(entry)
        callback(on_probe, entry)

    measure(0)
    replay_gate = {"gate": "NOT_EVALUATED_ARTIFICIAL_FIXTURE"}
    for epoch in range(1, epochs+1):
        epoch_start = time.perf_counter()
        timing_before, loader_before = timing.copy(), loader.seconds.copy()
        progress.update(epoch=epoch, phase="train")
        if device.type == "cuda":
            th.cuda.reset_peak_memory_stats(device)
        model.train()
        loader.begin_phase()
        terms, train_count = [], 0
        for batch, pairs in enumerate(sampler.epoch_batches(epoch), 1):
            refs = p1b.paired_examples(pairs)
            progress.update(phase="train", epoch=epoch, batch=batch)
            loader.context = dict(progress)
            optimizer.zero_grad(set_to_none=True)
            projection = {"count": 0, "abs_sum": 0.0, "abs_max": 0.0}
            observation_before = timing["train_projection_observation"]
            def read_projection(module, args, output):
                tick = time.perf_counter()
                value = output.detach()
                if not th.isfinite(value).all():
                    raise ValueError("nonfinite actual train fc_thermal output")
                projection["count"] += value.numel()
                projection["abs_sum"] += float(value.to(th.float64).abs().sum())
                projection["abs_max"] = max(projection["abs_max"], float(value.abs().max()))
                timing["train_projection_observation"] += time.perf_counter() - tick
            fc = getattr(getattr(model, "thermal_encoder", None), "fc_thermal", None)
            handle = fc.register_forward_hook(read_projection) if fc is not None else None
            total, actual = 0.0, len(refs)
            batch_sizes[actual] += 1
            try:
                for start, stop, weight in helpers.partition_microbatches(actual, micro_batch):
                    chunk = refs[start:stop]
                    progress["micro"] = [start, stop]
                    tensors = _tensors(loader, "fit", chunk, device, th, helpers, timing)
                    tick = time.perf_counter()
                    logits = model(**tensors)["logit"]
                    labels = th.tensor([r.label for r in chunk], dtype=th.float32, device=device)
                    if logits.shape != labels.shape:
                        raise ValueError("train BIN logit shape differs")
                    loss = th.nn.functional.binary_cross_entropy_with_logits(logits, labels) * weight
                    value = float(loss.detach())
                    p1b._finite(value, "nonfinite train loss", progress)
                    loss.backward()
                    helpers._sync(device)
                    timing["training_compute"] += time.perf_counter() - tick
                    total += value
                    for ref in chunk:
                        exposures["fit_positive" if ref.label else "fit_Normal"] += 1
                        stream.update((json.dumps({"epoch": epoch, "base": ref.base_window_id, "event": ref.event_id,
                                                   "label": ref.label}, sort_keys=True, separators=(",", ":")) + "\n").encode())
            finally:
                if handle is not None:
                    handle.remove()
            timing["training_compute"] -= timing["train_projection_observation"] - observation_before
            projection["abs_mean"] = projection["abs_sum"] / projection["count"] if projection["count"] else None
            progress["phase"] = "gradient_observation"
            tick = time.perf_counter()
            norms = onset.gradient_norms(model, th)
            helpers._sync(device)
            timing["gradient_observation"] += time.perf_counter() - tick
            progress["phase"] = "optimizer_step"
            tick = time.perf_counter()
            optimizer.step()
            progress["optimizer_steps_completed"] += 1
            helpers._sync(device)
            timing["training_compute"] += time.perf_counter() - tick
            terms.append(total * actual)
            train_count += actual
            callback(on_step, {"step": progress["optimizer_steps_completed"], "epoch": epoch, "batch": batch,
                               "train_loss": total, "actual_batch": actual, "lr_used": scheduler.get_last_lr()[0],
                               "gradient_norms": norms, "train_fc_thermal": projection,
                               "measurement_timing": "forward/backward before this update; fixed probes after update",
                               "train_batch_input_scope": "Changes by step; auxiliary trace is not a fixed-probe onset"})
            measure(progress["optimizer_steps_completed"])
        lr_used = scheduler.get_last_lr()[0]
        scheduler.step()
        tick = time.perf_counter()
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        terminal_sha = models_module.state_tensor_sha256(state)
        timing["terminal_state_snapshot"] += time.perf_counter() - tick
        entry = {"epoch": epoch, "optimizer_steps": batch, "train_loss": math.fsum(terms)/train_count,
                 "lr_used_this_epoch": lr_used, "next_epoch_lr": scheduler.get_last_lr()[0],
                 "fit_order_manifest_sha256": sampler.epoch_manifest(epoch)["sha256"],
                 "cumulative_fit_example_stream_sha256": stream.hexdigest(), "terminal_state_tensor_sha256": terminal_sha}
        progress["phase"] = "terminal_persistence"
        callback(on_terminal, state, entry, phase="terminal_persistence")
        if epoch == 1 and replay is not None:
            progress["phase"] = "first_epoch_replay"
            replay_gate = check_first_epoch({"initial_state_tensor_sha256": initial_sha,
                "final_state_tensor_sha256": terminal_sha, "train_loss": entry["train_loss"],
                "optimizer_steps": batch, "fit_order_manifest_sha256": entry["fit_order_manifest_sha256"],
                "fit_example_stream_sha256": stream.hexdigest()}, replay)
        entry["dev_loss"] = evaluate_dev(model, loader, dev_refs, device, th, helpers, timing, exposures, progress, micro_batch)
        if best is None or entry["dev_loss"] < best["dev_loss"]:
            best = {"epoch": epoch, "dev_loss": entry["dev_loss"], "state": state}
            progress["phase"] = "selected_checkpoint_persistence"
            callback(on_improvement, state, epoch, entry["dev_loss"], phase="checkpoint_persistence")
        if device.type == "cuda":
            entry["peak_gpu_allocated_bytes"] = int(th.cuda.max_memory_allocated(device))
            peak = max(peak, entry["peak_gpu_allocated_bytes"])
        entry["wall_seconds"] = time.perf_counter() - epoch_start
        entry["phase_seconds"] = {k: v-timing_before[k] for k,v in timing.items()}
        entry["loader_phase_seconds"] = {k: v-loader_before[k] for k,v in loader.seconds.items()}
        history.append(entry)
        progress["phase"] = "history_persistence"
        callback(on_epoch, entry)
        progress["phase"] = "epoch_boundary"
        callback(on_boundary, epoch, phase="gpu_boundary_record")
    record = {"schema": SCHEMA, "recipe_id": RECIPE, "gate": "PASS", "scope": scope,
              "arm": "T", "objective": "BIN", "initial_lr": 1e-4, "training_seed": seed,
              "epochs_completed": len(history), "planned_epochs": epochs, "epochs": history,
              "optimizer_steps_total": progress["optimizer_steps_completed"],
              "real_data_optimizer_steps": progress["optimizer_steps_completed"] if scope == "P1c_fit_dev_stability" else 0,
              "fixture_optimizer_steps": progress["optimizer_steps_completed"] if scope == "synthetic_fixture" else 0,
              "effective_batch": nominal_batch, "micro_batch": micro_batch, "cosine_T_max": 30,
              "initial_manifest": manifest, "initial_state_tensor_sha256": initial_sha,
              "final_state_tensor_sha256": terminal_sha, "selected_state_tensor_sha256": models_module.state_tensor_sha256(best["state"]),
              "dev_checkpoint_epoch": best["epoch"], "dev_checkpoint_loss": best["dev_loss"],
              "dev_checkpoint_ties": [e["epoch"] for e in history if e["dev_loss"] == best["dev_loss"] and e["epoch"] != best["epoch"]],
              "checkpoint_rule": "strict_less_than; earliest_epoch_on_exact_tie",
              "dev_reduction": "sum(weight * unreduced_BCE); no second mean",
              "dev_weight_manifest": bs.dev_weight_manifest(inputs["indexes"]["dev"]),
              "first_epoch_replay": replay_gate, "actual_fit_event_stream_sha256": stream.hexdigest(),
              "exposures": dict(exposures), "actual_batch_sizes": {str(k):v for k,v in sorted(batch_sizes.items())},
              "determinism": determinism, "probe_steps_completed": [p["step"] for p in probes],
              "phase_seconds": dict(timing), "loader_counts": dict(loader.counts), "loader_phase_seconds": dict(loader.seconds),
              "train_loop_wall_seconds": time.perf_counter()-started, "peak_gpu_allocated_bytes_run_max": peak,
              "trajectory": trajectory_summary(probes, screen=scope == "P1c_fit_dev_stability") if probes else None,
              "actual_roles_read": ["fit", "dev"], "calibration_window_read": False, "heldout_window_read": False,
              "threshold_generation": False, "paper_result_eligible": False, "full_P0_gate": "NOT_EVALUATED"}
    return record, best["state"]


def save_state(path, state, epoch, **metadata):
    th, _, models_module = p1b._training()
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as stream:
        th.save({"model_state_dict": state, "epoch": epoch, **metadata}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    saved = th.load(path, map_location="cpu", weights_only=True)
    if saved["epoch"] != epoch or any(saved[k] != v for k,v in metadata.items()) or models_module.state_tensor_sha256(saved["model_state_dict"]) != models_module.state_tensor_sha256(state):
        raise ValueError("serialized state/metadata parity failed")


def observe_and_preserve(model, step, out, loader, pairs, *, device):
    """Probe; if the probe fails, keep the exact model state of that step, then re-raise.

    r3: the first attempt's epoch-10 state was lost because the probe runs before the
    terminal save. A state-save failure never replaces the original exception.
    """
    try:
        return observe_fit(model, loader, pairs, device=device)
    except Exception as error:
        path = Path(out) / f"failure-probe-state-step-{step:04}.pt"
        try:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_state(path, state, None, selection="PROBE_FAILURE_STATE_NOT_DEV_SELECTED", optimizer_step=step)
            note = str(path.relative_to(out))
        except Exception as save_error:
            note = f"state save failed: {type(save_error).__name__}: {save_error}"
        if isinstance(error, p1b.P1bFailure):
            error.context["probe_failure_state"] = note
        raise


def record_failure(out, exception, progress):
    failure = {"schema": SCHEMA, "gate": "FAILED", "run_usable": False,
               "timestamp": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
               "python_version": platform.python_version(), "git_dirty": True,
               "exception": type(exception).__name__, "message": str(exception),
               "stage": progress.get("stage"), "context": {**progress, **getattr(exception,"context",{})},
               "retried": False, "learning_rate_adjusted": False, "paper_result_eligible": False}
    try:
        failure["partial_files_preserved"] = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        p1b._write_json(out / "failure.json", failure)
    except BaseException as error:
        try:
            print(json.dumps({"original_failure": failure,"failure_record_write_error":repr(error)},
                             default=repr,allow_nan=False),file=sys.stderr,flush=True)
        except BaseException:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    paths = ("prepared", "bank-result", "cache-run", "sampler-fixtures", "runner-fixtures", "diagnostic-fixtures",
             "onset-plan", "baseline-result", "replay-result", "stability-fixtures", "plan", "out")
    pins = ("cache-result", "sampler-fixtures", "runner-fixtures", "diagnostic-fixtures", "onset-plan",
            "baseline-result", "replay-result", "stability-fixtures", "plan")
    for flag in paths:
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in pins:
        parser.add_argument("--" + flag + "-sha256", required=True)
    args = parser.parse_args(argv)
    out = p1b._check_output(args)
    for name in ("plan", "onset_plan", "diagnostic_fixtures", "stability_fixtures", "baseline_result", "replay_result"):
        p = getattr(args, name).resolve()
        if out == p or p in out.parents or out in p.parents:
            raise ValueError("output overlaps stability evidence")
    out.mkdir(parents=True, exist_ok=False)
    progress, cache = dict(stage="inputs", optimizer_steps_completed=0), None
    started = time.perf_counter()
    try:
        inputs = validate_inputs(args)
        progress["stage"] = "environment"
        device, environment, _ = p1b.runtime_environment()
        identity = onset.runtime_identity(device)
        runtime_gate = onset.check_runtime_identity(identity, inputs["plan"]["runtime_reference"])
        progress["runtime_identity"] = identity
        provenance = {"schema": SCHEMA, "timestamp": datetime.now(timezone.utc).isoformat(),
                      "hostname": platform.node(), "python_version": platform.python_version(),
                      "git_commit_sha": inputs["candidate_git_commit_sha"], "git_dirty": True,
                      "environment_profile": "SERVER-TRAINING", "environment": environment,
                      "jetpack_l4t": environment.get("jetpack_l4t"), "sensor_driver_versions": environment.get("sensor_driver_versions", {}),
                      "runtime_identity": identity, "runtime_gate": runtime_gate,
                      "source_files": inputs["candidate_source_files"], "plan_file_sha256": args.plan_sha256,
                      "plan_payload_sha256": inputs["plan"]["sha256"],
                      "stability_fixture_payload_sha256": inputs["stability_fixture_payload_sha256"],
                      "first_epoch_replay_reference": inputs["first_epoch_reference"],
                      "command": list(argv) if argv is not None else sys.argv, "actual_roles_read": ["fit", "dev"], "paper_result_eligible": False}
        p1b._write_json(out / "provenance.json", provenance)
        def boundary(epoch):
            tick = time.perf_counter()
            onset.check_runtime_identity(onset.runtime_identity(device), identity)
            if source_hashes() != inputs["candidate_source_files"]:
                raise ValueError("candidate source changed during stability run")
            value = gpu_activity(identity["gpu_uuid"])
            value.update(epoch=epoch, runtime_identity=identity, source_files_unchanged=True)
            onset._append_json(out / "gpu-activity.jsonl", value)
            return time.perf_counter() - tick
        boundary(0)
        progress["stage"] = "cache_attach"
        cache = p1b.rc.RawNormalCache.attach(inputs["cache_path"]).open()
        if cache.manifest["sha256"] != inputs["cache_manifest"]["sha256"]:
            raise ValueError("cache manifest changed after metadata gate")
        loader = p1b.PairLoader(inputs, cache.reader(verify_each_read=False))
        probe_loader = p1b.PairLoader(inputs, cache.reader(verify_each_read=False))
        terminals = []
        (out / "terminal-states").mkdir()
        def terminal(state, entry):
            path = out / "terminal-states" / f"epoch-{entry['epoch']:02}.pt"
            save_state(path, state, entry["epoch"], selection="EPOCH_TERMINAL_NOT_DEV_SELECTED")
            terminals.append({"epoch": entry["epoch"], "file": str(path.relative_to(out)),
                              "file_sha256": p1b.file_sha(path), "tensor_sha256": entry["terminal_state_tensor_sha256"]})
        progress["stage"] = "training"
        record, selected = train_loop(inputs, loader, device=device, scope="P1c_fit_dev_stability",
            epochs=EPOCHS, probe_steps=PROBE_STEPS, replay=inputs["first_epoch_reference"], progress=progress,
            observer=lambda m,s: observe_and_preserve(m, s, out, probe_loader, inputs["probe_pairs"], device=device),
            on_probe=lambda e: onset._append_json(out / "probes.jsonl", e),
            on_step=lambda e: onset._append_json(out / "steps.jsonl", e),
            on_epoch=lambda e: onset._append_json(out / "epochs.jsonl", e),
            on_terminal=terminal, on_improvement=lambda state,e,loss: save_state(out / "checkpoint.pt", state, e, selection="DEV_EARLIEST_MINIMUM", dev_loss=loss),
            on_boundary=boundary)
        if record["optimizer_steps_total"] != REAL_STEPS or record["epochs_completed"] != EPOCHS or record["probe_steps_completed"] != list(PROBE_STEPS):
            raise ValueError("incomplete frozen stability budget/probes")
        progress["stage"] = "artifact_verification"
        th, _, models_module = p1b._training()
        saved = th.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)
        persisted = [json.loads(line) for line in (out / "epochs.jsonl").read_text().splitlines()]
        if (saved["epoch"] != record["dev_checkpoint_epoch"] or saved["dev_loss"] != record["dev_checkpoint_loss"]
                or models_module.state_tensor_sha256(saved["model_state_dict"]) != models_module.state_tensor_sha256(selected)
                or persisted != record["epochs"] or len(terminals) != EPOCHS):
            raise ValueError("persisted checkpoint/history/terminal coverage differs")
        progress["stage"] = "final_inputs"
        final = validate_inputs(args)
        for key in ("evidence", "candidate_source_files", "plan", "candidate_git_commit_sha", "first_epoch_reference", "stability_fixture_payload_sha256"):
            if final[key] != inputs[key]:
                raise ValueError("source/plan/input evidence changed during run: " + key)
        record.update(source_files=inputs["candidate_source_files"], plan_payload_sha256=inputs["plan"]["sha256"],
                      input_evidence=inputs["evidence"], runtime_identity=identity, runtime_gate=runtime_gate,
                      stability_fixture_payload_sha256=inputs["stability_fixture_payload_sha256"],
                      first_epoch_replay_reference=inputs["first_epoch_reference"], terminal_states=terminals,
                      provenance_file_sha256=p1b.file_sha(out / "provenance.json"),
                      artifact_file_sha256={name:p1b.file_sha(out / name) for name in
                          ("checkpoint.pt", "epochs.jsonl", "probes.jsonl", "steps.jsonl", "gpu-activity.jsonl")},
                      probe_loader_counts=dict(probe_loader.counts), probe_loader_phase_seconds=dict(probe_loader.seconds),
                      wall_until_final_record_seconds=time.perf_counter()-started)
        record["sha256"] = sa.digest_json(record)
        p1b._write_json(out / "result.json", record)
        print(json.dumps({"gate": "PASS", "out": str(out), "optimizer_steps": record["optimizer_steps_total"]}), flush=True)
        return 0
    except BaseException as exception:
        record_failure(out, exception, progress)
        raise
    finally:
        if cache is not None:
            cache._data = None


if __name__ == "__main__":
    raise SystemExit(main())
