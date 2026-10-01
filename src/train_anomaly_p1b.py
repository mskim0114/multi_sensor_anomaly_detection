#!/usr/bin/env python3
"""EXP-004 P1b cost pilots: S/F-AE and S/T/F-BIN, using a verified raw cache.

This entry point does not build a cache or read source dataset windows. Metadata
gates precede cache attachment/reader and optimizer creation. Actual CLI runs are
fixed to CT1/fold0/seed42/30 epochs and require reviewed, exact-source fixture
evidence and the audited CUDA environment. CPU cores are for artificial fixtures.
No calibration/heldout window, threshold fitting or automatic run sweep is exposed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import site
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _leaf(name, relative):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


cache_cli = _leaf("_p1b_cache_cli", "src/build_anomaly_raw_cache.py")
rc, controls, sa = cache_cli.rc, cache_cli.controls, cache_cli.sa
bs = _leaf("_p1b_bin_sampler", "src/data/anomaly_bin_sampler.py")
SOURCE_FILES = (*cache_cli.SOURCE_FILES, "src/data/anomaly_bin_sampler.py",
                "tests/test_anomaly_bin_sampler.py", "src/train_anomaly_p1b.py",
                "tests/test_anomaly_p1b.py", "configs/experiments/synthetic_modality_plan.yaml")
RESULT_SCHEMA = "exp004-p1b-cost-pilot-v1"
FIXTURE_SCHEMA = "exp004-p1b-runner-fixtures-v1"
ARMS = frozenset((("S", "AE"), ("F", "AE"), ("S", "BIN"), ("T", "BIN"), ("F", "BIN")))
REAL_EPOCHS = 30
OUTPUT_PARENT = Path.home() / "review_runs"
torch = pilot = detectors = None


def _training():
    global torch, pilot, detectors
    if torch is None:
        import torch as imported
        torch = imported
        pilot = _leaf("_p1b_p1a_helpers", "src/train_anomaly_pilot.py")
        detectors = pilot.detectors
    return torch, pilot, detectors


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def source_hashes():
    return {p: file_sha(ROOT / p) for p in SOURCE_FILES}


def _signed(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    body = {k: v for k, v in value.items() if k != "sha256"}
    if value.get("sha256") != sa.digest_json(body):
        raise ValueError(f"{label} payload digest mismatch")
    return value


def _pinned(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64 or file_sha(path) != expected:
        raise ValueError(f"{label} file pin mismatch")
    return _signed(json.loads(Path(path).read_text()), label)


def _positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_inputs(args):
    """Metadata-only gates. Never attach a cache, create a reader or import torch."""
    live = source_hashes()
    fixture = _pinned(args.runner_fixtures, args.runner_fixtures_sha256, "runner fixtures")
    expected_tests = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and n.name.startswith("test_") for n in ast.walk(
                             ast.parse((ROOT / "tests/test_anomaly_p1b.py").read_text())))
    summary = fixture.get("unit_tests", {})
    if (fixture.get("schema") != FIXTURE_SCHEMA or fixture.get("gate") != "PASS"
            or fixture.get("source_files") != live
            or type(summary.get("tests_run")) is not int
            or summary["tests_run"] != expected_tests or expected_tests < 1
            or any(type(summary.get(k)) is not int or summary[k] != 0
                   for k in ("failures", "errors", "skipped"))
            or type(fixture.get("real_data_optimizer_steps")) is not int
            or fixture["real_data_optimizer_steps"] != 0
            or type(fixture.get("fixture_optimizer_steps")) is not int
            or fixture["fixture_optimizer_steps"] < 1
            or sorted(fixture.get("main_completed_arms", [])) != sorted(f"{a}-{o}" for a, o in ARMS)):
        raise ValueError("runner fixture evidence/counts/sources do not satisfy P1b")
    revision = fixture.get("git_commit_sha")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("runner evidence requires a git revision")
    sampler = _pinned(args.sampler_fixtures, args.sampler_fixtures_sha256, "sampler fixtures")
    if (sampler.get("status") != "PASS"
            or sampler.get("unit_tests") != dict(tests_run=29, failures=0, errors=0, skipped=0)
            or sampler.get("source_files") != {p: live[p] for p in
                                              ("src/data/anomaly_bin_sampler.py", "tests/test_anomaly_bin_sampler.py")}
            or type(sampler.get("optimizer_steps_actual")) is not int
            or sampler["optimizer_steps_actual"] != 0):
        raise ValueError("sampler evidence does not match the verified candidate")
    split_path = Path(args.prepared) / "split_manifest.json"
    split = _signed(json.loads(split_path.read_text()), "split")
    if split.get("data_gate") != "PASS" or split.get("protocol_id") != sa.PROTOCOL:
        raise ValueError("split is not a PASS for this protocol")
    fold = split["folds"]["0"]
    normalizer = _signed(fold["normalizer"], "normalizer")
    windows = {role: [w for w in split["windows"] if fold["roles"].get(w["device"]) == role
                      and w["hard_valid"] is True and w["pure_Normal"] is True]
               for role in ("fit", "dev")}
    fit_ids = sorted({i for w in windows["fit"] for i in w["raw_base_ids"]})
    if fit_ids != normalizer["fit_raw_ids"] or sa.digest_json(fit_ids) != normalizer["fit_ids_sha256"]:
        raise ValueError("normalizer fit provenance differs from selected windows")
    bank = _signed(json.loads((Path(args.bank_result) / "bank.json").read_text()), "bank")
    event_path = Path(args.bank_result) / "events.jsonl"
    if (bank.get("bank_gate") != "PASS" or bank.get("protocol_id") != sa.PROTOCOL
            or type(bank.get("fold")) is not int or bank["fold"] != 0
            or type(bank.get("seed")) is not int or bank["seed"] != 42 or bank.get("ct_channel") != "CT1"
            or bank.get("normalizer_sha256") != normalizer["sha256"]
            or bank.get("heldout_read") is not False or bank.get("calibration_read") is not False
            or type(bank.get("gradient_training_runs")) is not int or bank["gradient_training_runs"] != 0
            or bank.get("full_P0_gate") != "NOT_EVALUATED"
            or file_sha(event_path) != bank.get("event_manifest_sha256")):
        raise ValueError("bank namespace/provenance/input pin mismatch")
    with event_path.open() as stream:
        events = [json.loads(line) for line in stream]
    if {e.get("role") for e in events} != {"fit", "dev"}:
        raise ValueError("event manifest contains foreign roles")
    indexes = {role: bs.build_event_index(
        windows[role], fold["roles"], [e for e in events if e["role"] == role], role=role,
        fold=0, generator_seed=42, ct_channel="CT1", normalizer_sha256=normalizer["sha256"],
        fit_ids_sha256=normalizer["fit_ids_sha256"]) for role in ("fit", "dev")}
    for role in indexes:
        devices = {b.device for b in indexes[role].bases}
        accepted_cells = {(b.device, e.family, e.strength) for b in indexes[role].bases for e in b.events}
        expected_cells = {(d, family, strength) for d in devices
                          for family in ("sensor-only", "thermal-only", "coupled") for strength in (1, 2, 4)}
        coverage = bank.get("coverage", {}).get(role, {})
        if (accepted_cells != expected_cells or coverage.get("bank_gate") != "PASS"
                or coverage.get("missing_accepted_cells") != []
                or type(coverage.get("expected_cells")) is not int
                or coverage["expected_cells"] != len(expected_cells)):
            raise ValueError("bank accepted-cell coverage is not complete")
        if indexes[role].record() != sampler.get("indexes", {}).get(role):
            raise ValueError("selected bank differs from sampler-verified metadata")
    if sampler.get("input_file_sha256") != {
            "split_manifest.json": file_sha(split_path), "events.jsonl": file_sha(event_path)}:
        raise ValueError("sampler input file pins differ from this run")
    return _validate_cache(args, split, normalizer, windows, indexes, events, live,
                           {"runner_fixture_sha256": fixture["sha256"],
                            "sampler_fixture_sha256": sampler["sha256"],
                            "git_commit_sha": revision})


def _validate_cache(args, split, normalizer, windows, indexes, events, live, evidence):
    run = Path(args.cache_run)
    if (run / "failure.json").exists() or (run / "cache/failure.json").exists():
        raise ValueError("failed cache run must not be reused")
    result = _pinned(run / "result.json", args.cache_result_sha256, "cache result")
    manifest = _signed(json.loads((run / "cache/manifest.json").read_text()), "cache manifest")
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if (result.get("schema") != cache_cli.RESULT_SCHEMA or result.get("gate") != "PASS"
            or type(result.get("fold")) is not int or result["fold"] != 0
            or type(result.get("optimizer_steps")) is not int or result["optimizer_steps"] != 0
            or result.get("cache_manifest_sha256") != manifest["sha256"]
            or result.get("source_files") != cache_cli.source_hashes()
            or file_sha(provenance_path) != result.get("provenance_file_sha256")
            or manifest.get("provenance") != provenance
            or provenance.get("source_files") != result["source_files"]
            or provenance.get("normalizer_sha256") != normalizer["sha256"]
            or provenance.get("binding") != result.get("binding")):
        raise ValueError("cache result/provenance/source binding mismatch")
    combined = windows["fit"] + windows["dev"]
    identities = [rc.window_identity(w, role) for role in ("fit", "dev") for w in windows[role]]
    stored = manifest.get("windows", [])
    if (manifest.get("schema") != rc.SCHEMA or manifest.get("protocol_id") != sa.PROTOCOL
            or manifest.get("allowed_roles") != ["fit", "dev"]
            or manifest.get("verified_after_flush") is not True
            or len(stored) != len(identities)
            or [{k: record.get(k) for k in identity} for record, identity in zip(stored, identities)] != identities):
        raise ValueError("cache does not hold the exact selected fit/dev Normal identities")
    for name in rc.ARRAYS:
        if manifest["arrays"][name]["file"] != f"{name}.npy":
            raise ValueError("cache array file differs from the canonical layout")
    layout = rc.declared_layout({name: manifest["arrays"][name]["dtype"] for name in rc.ARRAYS})
    raw_path = Path(args.prepared) / "raw_manifest.json"
    if sa.digest_json(json.loads(raw_path.read_text())) != split["source_manifest_sha256"]:
        raise ValueError("raw source manifest payload mismatch")
    role_list = [identity["role"] for identity in identities]
    binding = {"fold": 0, "source_manifest_file_sha256": file_sha(raw_path),
               "source_manifest_payload_sha256": split["source_manifest_sha256"],
               "split_manifest_file_sha256": file_sha(Path(args.prepared) / "split_manifest.json"),
               "split_manifest_payload_sha256": split["sha256"],
               "roles_sha256": sa.digest_json(split["folds"]["0"]["roles"]),
               "count": len(combined), "role_counts": {r: len(windows[r]) for r in windows},
               "base_order_sha256": sa.digest_json([w["base_window_id"] for w in combined]),
               "role_order_sha256": sa.digest_json(role_list),
               "window_order_sha256": sa.digest_json(identities),
               "declared_dtypes": {name: layout[name].str for name in rc.ARRAYS},
               "array_payload_bytes": rc.required_bytes(len(combined), layout)}
    if (result["binding"] != binding
            or manifest.get("source_manifest_sha256") != split["source_manifest_sha256"]
            or manifest.get("data_root") != str(Path(split["provenance"]["data_root"]).resolve())
            or manifest.get("windows_count") != binding["count"]
            or manifest.get("role_counts") != binding["role_counts"]
            or any(manifest.get(key) != binding[key] for key in
                   ("base_order_sha256", "role_order_sha256", "window_order_sha256"))):
        raise ValueError("cache manifest/order/data provenance differs from prepared input")
    parity = result.get("parity", {})
    groups = len({(i["role"], i["device"]) for i in identities})
    attempts = groups * 18
    if (parity.get("gate") != "PASS" or parity.get("raw_windows") != len(combined)
            or parity.get("normal_tensor_pairs") != 2 * len(combined)
            or parity.get("synthetic_sample_windows") != groups
            or parity.get("synthetic_paired_attempts") != attempts
            or type(parity.get("synthetic_accepted")) is not int
            or type(parity.get("synthetic_rejected")) is not int
            or parity["synthetic_accepted"] + parity["synthetic_rejected"] != attempts
            or min(parity["synthetic_accepted"], parity["synthetic_rejected"]) < 0
            or parity.get("synthetic_exhaustive") is not False
            or result.get("canonical_window_reads") != 2 * len(combined)
            or result.get("canonical_role_reads") != {r: 2 * len(windows[r]) for r in windows}
            or result.get("raw_roles_read") != ["dev", "fit"]
            or result.get("full_P0_gate") != "NOT_EVALUATED"):
        raise ValueError("cache completion parity/read coverage is incomplete")
    evidence.update(cache_result_sha256=result["sha256"], cache_manifest_sha256=manifest["sha256"],
                    split_payload_sha256=split["sha256"], bank_event_file_sha256=file_sha(
                        Path(args.bank_result) / "events.jsonl"))
    return {"split": split, "normalizer": normalizer, "windows": windows, "indexes": indexes,
            "events": events, "cache_path": run / "cache", "cache_manifest": manifest,
            "source_files": live, "evidence": evidence}


class P1bFailure(RuntimeError):
    def __init__(self, message, context):
        super().__init__(message)
        self.context = dict(context)


class PairLoader:
    """Verified raw cache -> inject -> normalize -> FP32, with one-base reuse.

    Both normalized modalities are prepared for the shared injection contract;
    only the arm's modalities are passed to its model. This is offline pipeline
    cost accounting, not field sensor-only latency. No torch RNG is consumed.
    """

    def __init__(self, inputs, reader):
        self.inputs, self.reader = inputs, reader
        self.counts, self.seconds = Counter(), Counter()
        self.context = {}
        self._last = None
        self._events = {(e["role"], e["event_id"]): e for e in inputs["events"] if e["accepted"]}
        self._stored = {w["base_window_id"]: w for w in inputs["cache_manifest"]["windows"]}

    def begin_phase(self):
        self._last = None

    def _raw(self, role, example):
        if role not in ("fit", "dev"):
            raise ValueError("loader permits fit/dev only")
        window = self.inputs["windows"][role][example.window_index]
        if (window["base_window_id"] != example.base_window_id or window["device"] != example.device
                or example.ct_channel != "CT1"):
            raise ValueError("example differs from its role-filtered window")
        key = role, example.base_window_id
        if self._last is None or self._last[0] != key:
            start = time.perf_counter()
            arrays = self.reader(window)
            self.seconds["cache_io"] += time.perf_counter() - start
            start = time.perf_counter()
            for name, array in zip(rc.ARRAYS, arrays):
                actual = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
                if actual != self._stored[example.base_window_id]["array_sha256"][name]:
                    raise ValueError(f"cached {name} changed for {example.base_window_id}")
            self.seconds["array_hash"] += time.perf_counter() - start
            s, t, labels = arrays
            if (s.shape != (30, 8) or t.shape != (30, 120, 160) or labels.shape != (30,)
                    or not (labels == 0).all() or not np.isfinite(s).all() or not np.isfinite(t).all()):
                raise ValueError("cached base is not a hard-valid pure Normal window")
            for array in arrays:
                array.setflags(write=False)
            self._last = key, arrays
            self.counts[role + "_cache_base_reads"] += 1
        return self._last[1]

    def load(self, role, example):
        s, t, labels = self._raw(role, example)
        if example.label == 1:
            expected = self._events.get((role, example.event_id))
            if expected is None:
                raise ValueError("positive example is not accepted in the frozen bank")
            start = time.perf_counter()
            s, t, meta = sa.inject_blind(
                s, t, tick_labels=labels, normalizer=self.inputs["normalizer"], protocol=sa.PROTOCOL,
                fold=0, role=role, base_window_id=example.base_window_id, family=example.family,
                strength=example.strength, generator_seed=42, ct_channel=example.ct_channel)
            self.seconds["injection"] += time.perf_counter() - start
            for field in ("event_id", "event_instance_id", "family", "strength", "ct_channel",
                          "sensor_changed", "thermal_changed"):
                if meta.get(field) != getattr(example, field):
                    raise P1bFailure("re-injected event identity differs from frozen bank",
                                     {**self.context, "field": field, "event_id": example.event_id})
            if meta.get("accepted") is not True or meta.get("parameters") != expected["parameters"]:
                raise P1bFailure("re-injected event acceptance/parameters differ from frozen bank", self.context)
            self.counts[role + "_injections"] += 1
        elif example.label != 0 or example.event_id is not None:
            raise ValueError("Normal example must have label0 and no event ID")
        start = time.perf_counter()
        pair = tuple(np.ascontiguousarray(x, dtype=np.float32) for x in
                     controls.normalized_pair(s, t, self.inputs["normalizer"], "CT1"))
        self.seconds["normalization"] += time.perf_counter() - start
        if (pair[0].shape != (30, 5) or pair[1].shape != (30, 120, 160)
                or any(not np.isfinite(x).all() for x in pair)):
            raise ValueError("normalized model input shape/nonfinite mismatch")
        self.counts[role + "_examples"] += 1
        return pair


def normal_example(window, position, *, weight=0.0):
    return bs.WeightedExample(window["base_window_id"], window["device"], position,
                              None, None, None, None, "CT1", False, False, 0, weight)


def paired_examples(pairs):
    examples = []
    for pair in pairs:
        examples.append(normal_example({"base_window_id": pair.base_window_id, "device": pair.device},
                                       pair.window_index))
        examples.append(bs.WeightedExample(**asdict(pair), label=1, weight=0.0))
    return tuple(examples)


def _finite(value, message, context):
    if not math.isfinite(value):
        raise P1bFailure(message, {**context, "value": repr(value)})


def _collision_bound(index, arm):
    if arm not in ("S", "T"):
        return None
    examples = bs.dev_examples(index)
    normals = {e.base_window_id: e.weight for e in examples if not e.label}
    positives = {e.event_id: e.weight for e in examples if e.label}
    flag = "sensor_changed" if arm == "S" else "thermal_changed"
    terms = []
    for base in index.bases:
        a = normals[base.base_window_id]
        b = math.fsum(positives[e.event_id] for e in base.events if not getattr(e, flag))
        if b:
            terms.append((a + b) * math.log(a + b) - a * math.log(a) - b * math.log(b))
    return math.fsum(terms)


def train_loop(inputs, loader, *, arm, objective, device, seed, scope, epochs=30,
               nominal_batch=16, micro_batch=2, on_epoch=None, on_improvement=None,
               progress=None):
    """One full-budget model; fixture callers explicitly declare synthetic scope.

    AE uses the P1a Normal order and scalar element-mean loss. BIN uses identical
    pair manifests for all arms. Neither objective sees real abnormal labels.
    """
    _positive_int(epochs, "epochs")
    _positive_int(nominal_batch, "nominal_batch")
    _positive_int(micro_batch, "micro_batch")
    if type(seed) is not int or seed < 0 or arm not in ("S", "T", "F") or objective not in ("AE", "BIN"):
        raise ValueError("invalid training seed/arm/objective")
    if scope not in ("synthetic_fixture", "P1b_real_data"):
        raise ValueError("explicit real-data or synthetic-fixture scope required")
    if nominal_batch % 2 and objective == "BIN":
        raise ValueError("BIN effective batch must preserve complete pairs")
    fit_normal = tuple(normal_example(w, i) for i, w in enumerate(inputs["windows"]["fit"]))
    dev_normal = tuple(normal_example(w, i) for i, w in enumerate(inputs["windows"]["dev"]))
    if not fit_normal or not dev_normal:
        raise ValueError("both fit/dev Normal windows are required")
    progress = {} if progress is None else progress
    progress.update(phase="model_setup", optimizer_steps_completed=0)
    th, helpers, models_module = _training()
    if th.get_default_dtype() != th.float32:
        raise ValueError("P1b requires the FP32 default dtype")
    determinism = helpers.enable_determinism()
    th.manual_seed(seed)
    models, arm_manifest = models_module.build_arm_set(objective, seed=seed)
    model = models[arm].to(device)
    del models
    initial_sha = models_module.state_tensor_sha256(model)
    optimizer = th.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                              weight_decay=0.01, amsgrad=False)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    sampler = (bs.BasePairedSampler(inputs["indexes"]["fit"], seed, nominal_batch // 2, epochs)
               if objective == "BIN" else None)
    dev_refs = bs.dev_examples(inputs["indexes"]["dev"]) if sampler else dev_normal
    timing, batch_sizes, exposures, event_stream = Counter(), Counter(), Counter(), hashlib.sha256()

    def batch_tensors(role, refs):
        loader.context = dict(progress)
        pairs = [loader.load(role, ref) for ref in refs]
        start = time.perf_counter()
        arrays = {name: np.stack([pair[i] for pair in pairs])
                  for i, name in enumerate(("sensor", "thermal"))
                  if name in models_module.ARM_MODALITIES[arm]}
        timing["stack"] += time.perf_counter() - start
        start = time.perf_counter()
        tensors = {name: th.from_numpy(array).to(device=device, dtype=th.float32)
                   for name, array in arrays.items()}
        helpers._sync(device)
        timing["transfer"] += time.perf_counter() - start
        return tensors

    def losses(tensors, refs):
        outputs = model(**tensors)
        if objective == "AE":
            value, _ = models_module.autoencoder_loss(outputs, **tensors, arm=arm)
            return value
        target = th.tensor([ref.label for ref in refs], dtype=th.float32, device=device)
        if outputs["logit"].shape != target.shape:
            raise ValueError("BIN logit must have exactly one scalar per example")
        return th.nn.functional.binary_cross_entropy_with_logits(outputs["logit"], target, reduction="none")

    history, best, peak = [], None, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        helpers._sync(device)
        if device.type == "cuda":
            th.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        before = {**{k: timing[k] for k in ("stack", "transfer", "compute")},
                  **{k: loader.seconds[k] for k in ("cache_io", "array_hash", "injection", "normalization")}}
        progress.update(epoch=epoch, phase="train", batch=0)
        model.train()
        loader.begin_phase()
        if sampler:
            manifest = sampler.epoch_manifest(epoch)
            fit_batches = [paired_examples(pairs) for pairs in sampler.epoch_batches(epoch)]
            order_digest = manifest["sha256"]
        else:
            order = helpers.epoch_permutation(len(fit_normal), seed=seed, epoch=epoch)
            fit_batches = [tuple(fit_normal[int(i)] for i in order[p:p + nominal_batch])
                           for p in range(0, len(order), nominal_batch)]
            order_digest = sa.digest_json([fit_normal[int(i)].base_window_id for i in order])
        train_sum, train_count, steps = 0.0, 0, 0
        for batch_number, refs in enumerate(fit_batches):
            progress.update(batch=batch_number, phase="train")
            actual = len(refs)
            batch_sizes[actual] += 1
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for start, stop, weight in helpers.partition_microbatches(actual, micro_batch):
                chunk = refs[start:stop]
                progress["micro"] = [start, stop]
                tensors = batch_tensors("fit", chunk)
                tick = time.perf_counter()
                loss = losses(tensors, chunk)
                loss = (loss.mean() if objective == "BIN" else loss) * weight
                value = float(loss.detach())
                _finite(value, "nonfinite train loss", progress)
                loss.backward()
                helpers._sync(device)
                timing["compute"] += time.perf_counter() - tick
                batch_loss += value
                for ref in chunk:
                    exposures["fit_positive" if ref.label else "fit_Normal"] += 1
                    event_stream.update((json.dumps({"epoch": epoch, "base": ref.base_window_id,
                                                    "event": ref.event_id, "label": ref.label},
                                                   sort_keys=True, separators=(",", ":")) + "\n").encode())
            tick = time.perf_counter()
            progress["phase"] = "gradient_check"
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not th.isfinite(parameter.grad).all():
                    raise P1bFailure("missing/nonfinite gradient", {**progress, "parameter": name})
            optimizer.step()
            steps += 1
            progress["optimizer_steps_completed"] += 1
            helpers._sync(device)
            timing["compute"] += time.perf_counter() - tick
            train_sum += batch_loss * actual
            train_count += actual
        lr_used = scheduler.get_last_lr()[0]
        scheduler.step()
        train_loss = train_sum / train_count
        _finite(train_loss, "nonfinite train epoch mean", progress)

        model.eval()
        loader.begin_phase()
        dev_terms = []
        with th.no_grad():
            for position in range(0, len(dev_refs), micro_batch):
                chunk = dev_refs[position:position + micro_batch]
                progress.update(phase="dev", dev_position=position)
                tensors = batch_tensors("dev", chunk)
                tick = time.perf_counter()
                loss = losses(tensors, chunk)
                if objective == "BIN":
                    weights = th.tensor([ref.weight for ref in chunk], dtype=th.float64, device=device)
                    value = float((loss.to(th.float64) * weights).sum())
                    # Every unreduced loss must be finite, even if cancellation would hide it.
                    if not th.isfinite(loss).all():
                        raise P1bFailure("nonfinite dev BCE", progress)
                else:
                    value = float(loss) * len(chunk)
                _finite(value, "nonfinite dev loss", progress)
                helpers._sync(device)
                timing["compute"] += time.perf_counter() - tick
                dev_terms.append(value)
                for ref in chunk:
                    exposures["dev_positive" if ref.label else "dev_Normal"] += 1
        dev_loss = math.fsum(dev_terms) if objective == "BIN" else sum(dev_terms) / len(dev_normal)
        _finite(dev_loss, "nonfinite dev epoch mean", progress)
        helpers._sync(device)
        epoch_wall = time.perf_counter() - epoch_start
        parts = {k: timing[k] - before[k] for k in ("stack", "transfer", "compute")}
        parts.update({k: loader.seconds[k] - before[k] for k in
                      ("cache_io", "array_hash", "injection", "normalization")})
        entry = {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
                 "lr_used_this_epoch": lr_used, "next_epoch_lr": scheduler.get_last_lr()[0],
                 "optimizer_steps": steps, "fit_order_manifest_sha256": order_digest,
                 "wall_seconds": epoch_wall, "phase_seconds": parts,
                 "unaccounted_seconds": epoch_wall - sum(parts.values())}
        if device.type == "cuda":
            entry["peak_gpu_allocated_bytes"] = int(th.cuda.max_memory_allocated(device))
            peak = max(peak, entry["peak_gpu_allocated_bytes"])
        history.append(entry)
        if best is None or dev_loss < best["dev_loss"]:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = {"epoch": epoch, "dev_loss": dev_loss, "state": state}
            progress["phase"] = "checkpoint_callback"
            if on_improvement:
                on_improvement(state, epoch, dev_loss)
        progress["phase"] = "history_callback"
        if on_epoch:
            on_epoch(entry)
    return _training_record(inputs, loader, model, arm_manifest, initial_sha, history, best,
                            timing, batch_sizes, exposures, event_stream, determinism,
                            arm=arm, objective=objective, scope=scope, seed=seed, epochs=epochs,
                            device=device, nominal_batch=nominal_batch, micro_batch=micro_batch,
                            peak=peak, loop_wall=time.perf_counter() - started), best["state"]


def _training_record(inputs, loader, model, arm_manifest, initial_sha, history, best,
                     timing, batch_sizes, exposures, event_stream, determinism, *,
                     arm, objective, scope, seed, epochs, device, nominal_batch,
                     micro_batch, peak, loop_wall):
    steps = sum(e["optimizer_steps"] for e in history)
    parameter_counts = Counter()
    for name, parameter in model.named_parameters():
        component = ("encoder" if "encoder" in name else "decoder" if "decoder" in name
                     else "head" if "head" in name else "fusion_or_other")
        parameter_counts[component] += parameter.numel()
    record = {
        "schema": RESULT_SCHEMA, "gate": "PASS", "scope": scope,
        "protocol_id": sa.PROTOCOL, "arm": arm, "objective": objective,
        "fold": 0, "ct_channel": "CT1", "generator_seed": 42, "training_seed": seed,
        "planned_epochs": epochs, "epochs_completed": len(history), "epochs": history,
        "full_epoch_budget_completed": len(history) == epochs,
        "real_30_epoch_budget_completed": scope == "P1b_real_data" and len(history) == 30,
        "optimizer": {"name": "AdamW", "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01, "amsgrad": False},
        "scheduler": {"name": "CosineAnnealingLR", "interval": "epoch", "T_max": epochs},
        "effective_batch": nominal_batch, "micro_batch": micro_batch,
        "effective_batch_counts": {str(k): v for k, v in sorted(batch_sizes.items())},
        "optimizer_steps_total": steps,
        "fixture_optimizer_steps": steps if scope == "synthetic_fixture" else 0,
        "real_data_optimizer_steps": steps if scope == "P1b_real_data" else 0,
        "dtype": "float32", "amp": False, "device": str(device), "determinism": determinism,
        "initial_state_sha256": initial_sha, "shared_initialization_manifest": arm_manifest,
        "dropout_masks_matched_across_arms": False,
        "selected_state_sha256": detectors.state_tensor_sha256(best["state"]),
        "dev_checkpoint_epoch": best["epoch"], "dev_checkpoint_loss": best["dev_loss"],
        "checkpoint_rule": "strict_less_than; earliest_epoch_on_exact_tie",
        "dev_checkpoint_ties": [e["epoch"] for e in history if e["dev_loss"] == best["dev_loss"]
                                and e["epoch"] != best["epoch"]],
        "dev_checkpoint_ties_scope": "other_epochs_tied_with_selected; excludes_selected_epoch",
        "model_selected_on": ("full_dev_class_device_base_event_weighted_BCE" if objective == "BIN"
                              else "dev_Normal_simple_mean_element_MSE"),
        "dev_reduction": ("sum(weight * unreduced_BCE); no second mean" if objective == "BIN"
                          else "sum(chunk_element_mean_MSE * chunk_count) / Normal_count"),
        "fit_Normal_windows": len(inputs["windows"]["fit"]),
        "dev_Normal_windows": len(inputs["windows"]["dev"]),
        "example_exposures": dict(exposures), "actual_fit_event_stream_sha256": event_stream.hexdigest(),
        "loader_counts": dict(loader.counts), "train_loop_wall_seconds": loop_wall,
        "phase_seconds": {**dict(loader.seconds), **dict(timing)},
        "epoch_wall_seconds": [e["wall_seconds"] for e in history],
        "parameter_counts": {**dict(parameter_counts), "total": sum(parameter_counts.values())},
        "peak_gpu_allocated_bytes_run_max": peak,
        "cost_basis": "offline shared pair preparation plus this arm's forward/backward",
        "field_inference_latency_measured": False,
        "calibration_window_read": False, "heldout_window_read": False,
        "source_dataset_windows_read": 0, "threshold_generated": False,
        "heldout_model_performance_computed": False, "paper_result_eligible": False,
        "full_P0_gate": "NOT_EVALUATED",
    }
    if objective == "BIN":
        record["dev_weight_manifest"] = bs.dev_weight_manifest(inputs["indexes"]["dev"])
        record["invisible_same_base_BCE_lower_bound_nats"] = _collision_bound(inputs["indexes"]["dev"], arm)
        record["collision_bound_assumptions"] = (
            "unchanged arm input within each base; no cross-base collisions counted; "
            "metadata-derived bound, not an attained model loss or performance")
    if device.type == "cuda":
        record["gpu_name"] = torch.cuda.get_device_name(device)
        record["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        record["exclusive_gpu_access_enforced"] = False
    return record


def _save_checkpoint(path, state, epoch, dev_loss):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save({"state_dict": state, "epoch": epoch, "dev_loss": dev_loss}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_training(out, inputs, loader, *, verbose=True, progress=None, **options):
    """Persist every completed epoch and each best checkpoint, without retry."""
    history_path, checkpoint = out / "epochs.jsonl", out / "checkpoint.pt"
    persistence = Counter()
    with history_path.open("x", encoding="utf-8") as history:
        def persist_epoch(entry):
            start = time.perf_counter()
            history.write(json.dumps(entry, allow_nan=False) + "\n")
            history.flush()
            os.fsync(history.fileno())
            persistence["history_save"] += time.perf_counter() - start
            if verbose:
                print(json.dumps(entry, allow_nan=False), flush=True)

        def persist_best(state, epoch, dev_loss):
            start = time.perf_counter()
            _save_checkpoint(checkpoint, state, epoch, dev_loss)
            persistence["checkpoint_save"] += time.perf_counter() - start

        record, state = train_loop(inputs, loader, on_epoch=persist_epoch,
                                   on_improvement=persist_best, progress=progress, **options)
    if progress is not None:
        progress.update(stage="checkpoint_verification", phase="checkpoint_verification")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    with history_path.open() as stream:
        persisted = [json.loads(line) for line in stream]
    if (saved["epoch"] != record["dev_checkpoint_epoch"]
            or saved["dev_loss"] != record["dev_checkpoint_loss"]
            or detectors.state_tensor_sha256(saved["state_dict"]) != detectors.state_tensor_sha256(state)
            or persisted != record["epochs"]):
        raise P1bFailure("persisted checkpoint/history differs from selected model", progress or {})
    record.update(checkpoint_file=checkpoint.name, checkpoint_file_sha256=file_sha(checkpoint),
                  epoch_history_file=history_path.name, epoch_history_file_sha256=file_sha(history_path),
                  checkpoint_save="temporary_file_flush_fsync_then_os_replace",
                  persistence_seconds=dict(persistence))
    return record


def runtime_environment():
    environment = cache_cli.server_environment()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization")
    th, _, _ = _training()
    if not th.cuda.is_available():
        raise ValueError("real P1b requires CUDA; no CPU fallback")
    environment.update(torch_version=th.__version__, cuda_runtime_version=th.version.cuda)
    return th.device("cuda:0"), environment, "P1b_real_data"


def _check_output(args):
    out = args.out.resolve()
    if out == OUTPUT_PARENT.resolve() or OUTPUT_PARENT.resolve() not in out.parents or out.exists():
        raise ValueError("output must be a new directory below the review-runs parent")
    guarded = [ROOT, args.prepared.resolve(), args.bank_result.resolve(), args.cache_run.resolve()]
    split_path = args.prepared / "split_manifest.json"
    # This untrusted metadata is used only to protect a path, never to authorize a read.
    if split_path.is_file():
        metadata = json.loads(split_path.read_text())
        raw_root = metadata.get("provenance", {}).get("data_root")
        if raw_root:
            guarded.append(Path(raw_root).resolve())
    if any(out == p or p in out.parents or out in p.parents for p in guarded):
        raise ValueError("output overlaps a source/input directory")
    return out


def _record_failure(out, exception, progress):
    failure = {"schema": RESULT_SCHEMA, "gate": "FAILED", "run_usable": False,
               "timestamp": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
               "python_version": platform.python_version(), "git_dirty": True,
               "exception": type(exception).__name__, "message": str(exception),
               "stage": progress.get("stage"),
               "context": {**progress, **getattr(exception, "context", {})},
               "retried": False, "learning_rate_adjusted": False, "out_of_memory_retry": False,
               "partial_files_preserved": []}
    try:
        failure["partial_files_preserved"] = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    except BaseException as error:
        failure["partial_file_listing_error"] = repr(error)
    try:
        _write_json(out / "failure.json", failure)
    except BaseException as error:
        # The original exception must survive a second interruption or a failed disk write.
        try:
            print(json.dumps({"failure_record_write_error": repr(error), "original_failure": failure},
                             default=repr, allow_nan=False), file=sys.stderr, flush=True)
        except BaseException:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("prepared", "bank-result", "cache-run", "sampler-fixtures", "runner-fixtures", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("cache-result-sha256", "sampler-fixtures-sha256", "runner-fixtures-sha256"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--arm", choices=("S", "T", "F"), required=True)
    parser.add_argument("--objective", choices=("AE", "BIN"), required=True)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if (args.arm, args.objective) not in ARMS:
        parser.error("P1b includes S/F-AE and S/T/F-BIN; T-AE is already recovered")
    if args.micro_batch < 1:
        parser.error("micro-batch must be positive")
    out = _check_output(args)
    out.mkdir(parents=True, exist_ok=False)
    started, timings = time.perf_counter(), {}
    progress = {"stage": "inputs", "optimizer_steps_completed": 0,
                "arm": args.arm, "objective": args.objective, "scope": "not_yet_verified"}
    cache = None
    try:
        tick = time.perf_counter()
        inputs = validate_inputs(args)
        timings["input_validation"] = time.perf_counter() - tick
        progress.update(source_files=inputs["source_files"], input_evidence=inputs["evidence"], stage="environment")
        device, environment, scope = runtime_environment()
        progress.update(scope=scope, environment_profile=environment["environment_profile"], stage="provenance")
        provenance = {**environment, "timestamp": datetime.now(timezone.utc).isoformat(),
                      "git_commit_sha": inputs["evidence"]["git_commit_sha"], "git_dirty": True,
                      "git_dirty_basis": "conservative uncommitted diagnostic; exact source hashes authoritative",
                      "source_files": inputs["source_files"], "input_evidence": inputs["evidence"],
                      "scope": scope, "arm": args.arm, "objective": args.objective,
                      "normalizer_sha256": inputs["normalizer"]["sha256"],
                      "fold": 0, "ct_channel": "CT1", "training_seed": 42, "generator_seed": 42,
                      "jetpack_l4t": environment.get("jetpack_l4t"),
                      "sensor_driver_versions": environment.get("sensor_driver_versions", {}),
                      "calibration_window_read": False, "heldout_window_read": False,
                      "paper_result_eligible": False, "packages_installed": [], "command": argv or sys.argv}
        _write_json(out / "provenance.json", provenance)
        progress["stage"] = "cache_attach"
        tick = time.perf_counter()
        cache = rc.RawNormalCache.attach(inputs["cache_path"]).open()
        timings["cache_attach_full_file_hash_and_open"] = time.perf_counter() - tick
        if cache.manifest["sha256"] != inputs["cache_manifest"]["sha256"]:
            raise ValueError("cache manifest changed after the metadata gate")
        loader = PairLoader(inputs, cache.reader(verify_each_read=False))
        progress["stage"] = "training"
        record = run_training(out, inputs, loader, arm=args.arm, objective=args.objective,
                              device=device, seed=42, scope=scope, epochs=REAL_EPOCHS,
                              micro_batch=args.micro_batch, progress=progress, verbose=not args.quiet)
        progress["stage"] = "final_inputs"
        tick = time.perf_counter()
        final_inputs = validate_inputs(args)
        if final_inputs["source_files"] != inputs["source_files"] or final_inputs["evidence"] != inputs["evidence"]:
            raise ValueError("sources or evidence changed during training")
        timings["final_metadata_validation"] = time.perf_counter() - tick
        record.update(source_files=inputs["source_files"], input_evidence=inputs["evidence"],
                      provenance_file_sha256=file_sha(out / "provenance.json"),
                      preparation_phase_seconds=timings, wall_until_final_record_seconds=time.perf_counter() - started)
        record["sha256"] = sa.digest_json(record)
        progress["stage"] = "final_record"
        _write_json(out / "result.json", record)
        if not args.quiet:
            print(json.dumps({"gate": "PASS", "out": str(out), "arm": args.arm, "objective": args.objective,
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
