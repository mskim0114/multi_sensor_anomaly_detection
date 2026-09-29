#!/usr/bin/env python3
"""EXP-004 P1a pilot: T-AE, fold 0, seed 42, 30 epochs. Uncommitted diagnostic.

Scope is deliberately one arm. S and F are not trained here and no held-out or
calibration window is ever read: the runner refuses any role outside fit/dev, and the
resulting record carries ``calibration_window_read``/``heldout_window_read`` as measured
facts, scoped to raw window reads and to model selection rather than to reading the
controls and manifest JSON that the pre-run gates require.

Raw data is immutable here. Labels, QC flags and the fold normalizer are consumed as
given, never recomputed or corrected, and nothing is installed. Real training requires
CUDA in the audited SERVER-TRAINING venv; there is no silent CPU fallback, because a CPU
run would have different cost numbers and a different determinism record. The tiny
fixture path used by the tests is a separate entry point that touches no dataset.

This module runs an optimiser. It must not be executed on real data before review.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _leaf(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


detectors = _leaf("_pilot_detectors", "src/models/anomaly_detectors.py")
controls = _leaf("_pilot_controls", "src/fit_anomaly_controls.py")
sa = controls.sa

PILOT_ID = "EXP-20260929-P1a"
ARM = "T"
OBJECTIVE = "AE"
FOLD = 0
SEED = 42
EPOCHS = 30
NOMINAL_EFFECTIVE_BATCH = 16
MICRO_BATCH = 2
ALLOWED_ROLES = ("fit", "dev")
THERMAL_OUTPUT_ELEMENTS = detectors.WINDOW_TICKS * detectors.THERMAL_SHAPE[0] * detectors.THERMAL_SHAPE[1]

# Fixture evidence must pin the exact code it was produced from, or a change made after the
# tests passed would slip through a bare "verified: true".
MODEL_CODE_FILES = ("src/models/anomaly_detectors.py", "src/models/v2_plus.py",
                    "tests/test_anomaly_detectors.py")
RUNNER_CODE_FILES = ("src/train_anomaly_pilot.py", "tests/test_anomaly_pilot.py",
                     "src/fit_anomaly_controls.py", "src/data/synthetic_anomaly.py",
                     "src/models/anomaly_baselines.py")
EXPECTED_MODEL_FIXTURE_TESTS = 40


class PilotFailure(RuntimeError):
    """A non-finite loss or gradient. Preserved and re-raised; never retried."""

    def __init__(self, message, context):
        super().__init__(message)
        self.context = context


def enable_determinism():
    """Strictly deterministic FP32 with TF32 off. A nondeterministic op raises, not warns.

    ``warn_only=True`` would let a nondeterministic kernel run and still let the record
    claim determinism, so strict mode is used instead: if any operation in this model has
    no deterministic implementation the run fails immediately. CUBLAS_WORKSPACE_CONFIG has
    to be in the environment before CUDA initialises, so it is checked rather than set.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {"requested": True, "strict": True, "warn_only": False,
            "nondeterministic_op_policy": "raise_immediately",
            "cudnn_deterministic": True, "cudnn_benchmark": False,
            "tf32_matmul": False, "tf32_cudnn": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cublas_config_valid_for_determinism":
                os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def _sync(device):
    """Make a timing boundary real on CUDA; kernels are otherwise still queued."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def partition_microbatches(actual_batch: int, micro_batch: int):
    """Split one nominal batch into weighted micro-batches.

    The weights divide by the ACTUAL batch size, not the nominal 16, so a trailing
    partial batch is averaged over the windows it really contains. Summing
    ``weight * element_mean_mse(chunk)`` then reproduces the element-mean MSE of the
    whole batch exactly, because every window holds the same element count.
    """
    if not isinstance(actual_batch, int) or isinstance(actual_batch, bool) or actual_batch < 1:
        raise ValueError("actual batch size must be a positive int")
    if not isinstance(micro_batch, int) or isinstance(micro_batch, bool) or micro_batch < 1:
        raise ValueError("micro batch size must be a positive int")
    chunks = []
    start = 0
    while start < actual_batch:
        stop = min(start + micro_batch, actual_batch)
        chunks.append((start, stop, (stop - start) / actual_batch))
        start = stop
    return chunks


def epoch_permutation(count: int, *, seed: int, epoch: int):
    """Reproducible Normal-window order; depends only on (seed, epoch), not on wall time."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch])))
    return rng.permutation(count)


def select_windows(manifest, roles, role):
    """Eligible pure-Normal windows of one permitted role, in manifest order."""
    if role not in ALLOWED_ROLES:
        raise ValueError(f"pilot reads {ALLOWED_ROLES} only; refused role {role!r}")
    chosen = [w for w in manifest["windows"]
              if roles.get(w["device"]) == role and w["hard_valid"] and w["pure_Normal"]]
    if not chosen:
        raise ValueError(f"no eligible pure-Normal window in role {role}")
    devices = sorted({w["device"] for w in chosen})
    expected = sorted(d for d, r in roles.items() if r == role)
    if devices != expected:
        raise ValueError(f"role {role} is missing eligible equipment")
    return chosen


def normalized_thermal(window, reader, normalizer, ct_channel="CT1"):
    """Reader-verified raw window -> normalized thermal only; the sensor array is dropped."""
    sensor, thermal, labels = reader(window)
    if labels.shape != (detectors.WINDOW_TICKS,) or not (labels == 0).all():
        raise ValueError("pilot window is not 30-tick pure Normal")
    _, t = controls.normalized_pair(sensor, thermal, normalizer, ct_channel)
    array = np.ascontiguousarray(t, dtype=np.float32)
    if array.shape != (detectors.WINDOW_TICKS, *detectors.THERMAL_SHAPE):
        raise ValueError("normalized thermal window has the wrong shape")
    if not np.isfinite(array).all():
        raise ValueError("normalized thermal window is not finite")
    return array


class ThermalCache:
    """float32 memmap of normalized thermal windows, built once with raw hash checks.

    Re-reading raw every epoch would re-hash and re-decode the same 30 files per window for
    30 epochs. Raw content is verified once at build time by the shared reader, then the
    bytes that actually landed on disk are streamed back and digested, so a write that
    silently truncated or corrupted the file is caught. The digest covers the npy header and
    every array byte, which makes a shape-preserving payload edit detectable.

    At ~6 GB the read-back is not free, so it happens exactly once: :meth:`build` verifies
    after flushing and records the cost, and :meth:`attach` re-verifies when a later process
    reuses the file. Neither path trusts an unverified cache.
    """

    def __init__(self, path: Path, count: int, file_sha256=None):
        self.path, self.count = Path(path), count
        self.shape = (count, detectors.WINDOW_TICKS, *detectors.THERMAL_SHAPE)
        self.file_sha256 = file_sha256
        self.source_payload_sha256 = self.payload_sha256 = None
        self.build_io_seconds = self.verify_seconds = self.build_wall_seconds = 0.0
        self.bytes = 0
        self.order_sha256 = None
        self.verified_after_flush = False
        self.verified_at_attach = False
        self._data = None

    @property
    def payload_bytes(self):
        return int(np.prod(self.shape)) * 4

    @staticmethod
    def _file_digest(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _digests(cls, path):
        """File and payload digests from a single pass; the payload skips the npy header."""
        offset = np.load(path, mmap_mode="r").offset
        file_digest, payload_digest = hashlib.sha256(), hashlib.sha256()
        with open(path, "rb") as handle:
            remaining = offset
            while remaining:
                block = handle.read(min(1 << 22, remaining))
                if not block:
                    raise ValueError("cache file ended inside its header")
                file_digest.update(block)
                remaining -= len(block)
            for block in iter(lambda: handle.read(1 << 22), b""):
                file_digest.update(block)
                payload_digest.update(block)
        return file_digest.hexdigest(), payload_digest.hexdigest()

    @classmethod
    def required_bytes(cls, count):
        return int(np.prod((count, detectors.WINDOW_TICKS, *detectors.THERMAL_SHAPE))) * 4

    @classmethod
    def build(cls, path, windows, reader, normalizer, *, ct_channel="CT1", progress=None,
              free_space_margin=1.05):
        cache = cls(path, len(windows))
        if cache.path.exists():
            raise ValueError("cache path already exists; the pilot never overwrites")
        cache.path.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(cache.path.parent).free
        needed = int(cache.payload_bytes * free_space_margin)
        if free < needed:
            raise ValueError(f"cache needs about {needed} bytes but {free} are free at "
                             f"{cache.path.parent}")
        cache.order_sha256 = sa.digest_json([w["base_window_id"] for w in windows])
        started = time.perf_counter()
        out = np.lib.format.open_memmap(cache.path, mode="w+", dtype=np.float32, shape=cache.shape)
        io_seconds = 0.0
        source = hashlib.sha256()
        try:
            for index, window in enumerate(windows):
                read_started = time.perf_counter()
                array = normalized_thermal(window, reader, normalizer, ct_channel)
                io_seconds += time.perf_counter() - read_started
                # Digest what we intended to store, so the read-back can be compared to it.
                source.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
                out[index] = array
                if progress and (index + 1) % 100 == 0:
                    progress(index + 1, len(windows))
            out.flush()
        finally:
            del out
        verify_started = time.perf_counter()
        cache.file_sha256, stored = cls._digests(cache.path)
        cache.source_payload_sha256 = source.hexdigest()
        cache.payload_sha256 = stored
        cache.verify_seconds = time.perf_counter() - verify_started
        if stored != cache.source_payload_sha256:
            raise ValueError("cache payload read back from disk differs from the normalized "
                             "windows that were written")
        cache.verified_after_flush = True
        cache.build_io_seconds = io_seconds
        cache.build_wall_seconds = time.perf_counter() - started
        cache.bytes = cache.path.stat().st_size
        if cache.bytes < cache.payload_bytes:
            raise ValueError("cache file is smaller than its declared payload")
        return cache

    @classmethod
    def attach(cls, path, count, file_sha256):
        """Reuse a cache written earlier; the recorded digest must still hold."""
        cache = cls(path, count, file_sha256=file_sha256)
        started = time.perf_counter()
        actual, cache.payload_sha256 = cls._digests(cache.path)
        cache.verify_seconds = time.perf_counter() - started
        if actual != file_sha256:
            raise ValueError("cache file content changed since it was built")
        cache.verified_at_attach = True
        cache.bytes = cache.path.stat().st_size
        return cache

    def open(self):
        if not (self.verified_after_flush or self.verified_at_attach):
            raise ValueError("cache was never verified; rebuild or attach with its digest")
        self._data = np.load(self.path, mmap_mode="r")
        if self._data.shape != self.shape or self._data.dtype != np.float32:
            raise ValueError("cache shape or dtype disagrees with the declared windows")
        return self

    def __call__(self, index: int) -> np.ndarray:
        if self._data is None:
            raise ValueError("cache must be opened, and so verified, before use")
        return np.asarray(self._data[index], dtype=np.float32)

    def record(self):
        return {"path": str(self.path), "windows": self.count, "file_bytes": self.bytes,
                "payload_bytes": self.payload_bytes, "file_sha256": self.file_sha256,
                "payload_sha256": self.payload_sha256,
                "source_payload_sha256": self.source_payload_sha256,
                "stored_payload_matches_source": (
                    self.source_payload_sha256 is not None
                    and self.source_payload_sha256 == self.payload_sha256),
                "window_order_sha256": self.order_sha256,
                "raw_content_verified_at_build": True,
                "verified_after_flush": self.verified_after_flush,
                "verified_at_attach": self.verified_at_attach,
                "digest_scope": "one_pass_file_and_payload_digests_payload_compared_to_source_arrays",
                "build_wall_seconds": self.build_wall_seconds,
                "build_raw_io_seconds": self.build_io_seconds,
                "digest_verify_seconds": self.verify_seconds}


def train_pilot(*, train_count, dev_count, load_thermal, device, seed=SEED, epochs=EPOCHS,
                nominal_batch=NOMINAL_EFFECTIVE_BATCH, micro_batch=MICRO_BATCH,
                arm=ARM, lr=1e-3, weight_decay=0.01, on_epoch=None, on_improvement=None):
    """Train one AE arm for the full epoch budget. Returns (record, selected state_dict).

    ``load_thermal(index)`` yields a normalized float32 window; indices ``0..train_count-1``
    are fit and ``train_count..train_count+dev_count-1`` are dev. The split is positional so
    this core never sees a role string, a device ID or a label.

    ``on_epoch`` and ``on_improvement`` let the caller persist each epoch and each new best
    checkpoint as they happen, so a run that dies at epoch 29 does not lose everything.
    """
    if train_count < 1 or dev_count < 1:
        raise ValueError("both a fit and a dev partition are required")
    if not isinstance(micro_batch, int) or isinstance(micro_batch, bool) or micro_batch < 1:
        raise ValueError("micro batch size must be a positive int")
    if not isinstance(nominal_batch, int) or isinstance(nominal_batch, bool) or nominal_batch < 1:
        raise ValueError("nominal batch size must be a positive int")
    determinism = enable_determinism()
    torch.manual_seed(seed)
    models, arm_manifest = detectors.build_arm_set(OBJECTIVE, seed=seed)
    model = models[arm].to(device)
    initial_sha = detectors.state_tensor_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=weight_decay, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    timing = Counter()

    def load_batch(indices):
        started = time.perf_counter()
        stacked = np.stack([load_thermal(int(i)) for i in indices])
        timing["io"] += time.perf_counter() - started
        started = time.perf_counter()
        tensor = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
        _sync(device)
        timing["transfer"] += time.perf_counter() - started
        return tensor

    def forward_loss(tensor):
        outputs = model(thermal=tensor)
        total, _ = detectors.autoencoder_loss(outputs, thermal=tensor, arm=arm)
        return total

    def check_finite(value, message, context):
        if not np.isfinite(value):
            raise PilotFailure(message, dict(context, value=repr(value), finite=False))

    epochs_record, best, run_peak = [], None, 0
    batch_size_counts = Counter()
    run_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        epoch_started = time.perf_counter()
        before = {key: timing[key] for key in ("io", "transfer", "compute")}
        model.train()
        order = epoch_permutation(train_count, seed=seed, epoch=epoch)
        train_sum, steps, batches = 0.0, 0, 0
        for position in range(0, train_count, nominal_batch):
            batch = order[position:position + nominal_batch]
            actual = len(batch)
            batch_size_counts[actual] += 1
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for start, stop, weight in partition_microbatches(actual, micro_batch):
                tensor = load_batch(batch[start:stop])
                compute_started = time.perf_counter()
                loss = forward_loss(tensor) * weight
                value = float(loss.detach())
                check_finite(value, "non-finite training loss",
                             {"epoch": epoch, "batch": batches, "micro": [start, stop],
                              "actual_batch": actual, "phase": "train_forward"})
                loss.backward()
                _sync(device)
                timing["compute"] += time.perf_counter() - compute_started
                batch_loss += value
            compute_started = time.perf_counter()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise PilotFailure("parameter detached from the loss",
                                       {"epoch": epoch, "batch": batches, "parameter": name,
                                        "phase": "gradient_check"})
                if not torch.isfinite(parameter.grad).all():
                    raise PilotFailure("non-finite gradient",
                                       {"epoch": epoch, "batch": batches, "parameter": name,
                                        "phase": "gradient_check", "finite": False})
            optimizer.step()
            _sync(device)
            timing["compute"] += time.perf_counter() - compute_started
            steps += 1
            batches += 1
            train_sum += batch_loss * actual
        lr_used = scheduler.get_last_lr()[0]
        scheduler.step()
        train_loss = train_sum / train_count
        check_finite(train_loss, "non-finite epoch training loss",
                     {"epoch": epoch, "phase": "train_epoch_mean"})

        model.eval()
        dev_sum = 0.0
        with torch.no_grad():
            for position in range(0, dev_count, nominal_batch):
                indices = list(range(train_count + position,
                                     train_count + min(position + nominal_batch, dev_count)))
                for start, stop, weight in partition_microbatches(len(indices), micro_batch):
                    tensor = load_batch(indices[start:stop])
                    compute_started = time.perf_counter()
                    value = float(forward_loss(tensor))
                    _sync(device)
                    timing["compute"] += time.perf_counter() - compute_started
                    check_finite(value, "non-finite dev loss",
                                 {"epoch": epoch, "dev_position": position,
                                  "micro": [start, stop], "phase": "dev_forward"})
                    dev_sum += value * (stop - start)
        dev_loss = dev_sum / dev_count
        check_finite(dev_loss, "non-finite dev epoch loss",
                     {"epoch": epoch, "phase": "dev_epoch_mean"})

        _sync(device)
        epoch_wall = time.perf_counter() - epoch_started
        parts = {key: timing[key] - before[key] for key in ("io", "transfer", "compute")}
        entry = {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
                 "lr_used_this_epoch": lr_used,
                 "next_epoch_lr": scheduler.get_last_lr()[0], "optimizer_steps": steps,
                 "batches": batches, "wall_seconds": epoch_wall,
                 "io_seconds": parts["io"], "host_to_device_seconds": parts["transfer"],
                 "compute_seconds": parts["compute"],
                 "unaccounted_seconds": epoch_wall - sum(parts.values())}
        if device.type == "cuda":
            entry["peak_gpu_bytes_this_epoch"] = int(torch.cuda.max_memory_allocated(device))
            run_peak = max(run_peak, entry["peak_gpu_bytes_this_epoch"])
        epochs_record.append(entry)
        # Strict improvement only, so an equal dev loss keeps the earlier epoch.
        if best is None or dev_loss < best["dev_loss"]:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = {"epoch": epoch, "dev_loss": dev_loss, "state": state}
            if on_improvement:
                on_improvement(state, epoch, dev_loss)
        if on_epoch:
            on_epoch(entry)

    ties = [e["epoch"] for e in epochs_record
            if e["dev_loss"] == best["dev_loss"] and e["epoch"] != best["epoch"]]
    selected = best["state"]
    record = {
        "pilot_id": PILOT_ID, "arm": arm, "objective": OBJECTIVE, "fold": FOLD, "seed": seed,
        "epochs_requested": epochs, "epochs_completed": len(epochs_record),
        "full_epoch_budget_completed": len(epochs_record) == EPOCHS,
        "optimizer": {"name": "AdamW", "lr": lr, "weight_decay": weight_decay,
                      "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False},
        "lr_schedule": {"name": "CosineAnnealingLR", "T_max": epochs, "step": "per_epoch"},
        "precision": "FP32", "determinism": determinism, "device": str(device),
        "nominal_effective_batch": nominal_batch, "micro_batch": micro_batch,
        "partial_batch_normalisation": "actual_batch_size_not_nominal",
        "actual_batch_size_counts": {str(k): v for k, v in sorted(batch_size_counts.items())},
        "num_workers": 0, "worker_processes_used": False,
        "window_order": "PCG64(SeedSequence([seed, epoch])).permutation",
        "train_windows": train_count, "dev_windows": dev_count,
        "thermal_output_elements_per_window": THERMAL_OUTPUT_ELEMENTS,
        "encoder_parameters": sum(p.numel() for p in model.thermal_encoder.parameters()),
        "decoder_parameters": sum(p.numel() for p in model.thermal_decoder.parameters()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "arm_set_manifest": {k: arm_manifest[k] for k in
                             ("initial_state_sha256", "shared_branch_sha256", "sha256")},
        "initial_weights_sha256": initial_sha,
        "checkpoint_state_sha256": detectors.state_tensor_sha256(selected),
        "state_digest_schema": "sorted_key_then_shape_dtype_then_raw_bytes",
        "dev_checkpoint_epoch": best["epoch"], "dev_checkpoint_loss": best["dev_loss"],
        "dev_checkpoint_tie_rule": "earlier_epoch",
        "dev_checkpoint_tied_later_epochs": ties,
        "epochs": epochs_record,
        "optimizer_steps_total": sum(e["optimizer_steps"] for e in epochs_record),
        "batches_total": sum(e["batches"] for e in epochs_record),
        "train_loop_wall_seconds": time.perf_counter() - run_started,
        "train_loop_wall_excludes": ["cache_build", "input_validation"],
        "train_loop_wall_includes": ["epoch_history_and_checkpoint_callbacks"],
        "total_io_seconds": timing["io"], "total_host_to_device_seconds": timing["transfer"],
        "total_compute_seconds": timing["compute"],
        "sensor_modality_used": False,
        "calibration_window_read": False, "heldout_window_read": False,
        "read_flag_scope": ("refers to reading raw calibration/held-out WINDOWS and to model "
                           "selection or evaluation on them; reading the controls/manifest "
                           "JSON for pre-run validation is not a window read"),
        "model_selected_on": "dev_Normal_reconstruction_loss_only",
        "heldout_model_performance_computed": False,
        "cost_basis": f"measured_over_{len(epochs_record)}_epochs",
        "cost_basis_is_full_budget": len(epochs_record) == EPOCHS,
        "status": "P1a_uncommitted_diagnostic",
        "full_P0_gate": "NOT_EVALUATED",
    }
    if device.type == "cuda":
        record["peak_gpu_bytes_run_max"] = run_peak
        record["gpu_name"] = torch.cuda.get_device_name(device)
        record["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        record["cuda_device_count"] = torch.cuda.device_count()
        record["exclusive_gpu_access_enforced"] = False
    record["sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return record, selected


def write_failure(out: Path, exception, stage, context=None):
    """Record a failure next to whatever the run already produced. Never retried."""
    payload = {"exception": type(exception).__name__, "message": str(exception), "stage": stage,
               "context": context if context is not None else getattr(exception, "context", None),
               "retried": False, "learning_rate_adjusted": False, "out_of_memory_retry": False,
               "run_abandoned": True,
               "preserved": sorted(p.name for p in Path(out).iterdir() if p.is_file())}
    (Path(out) / "failure.json").write_text(
        json.dumps(payload, indent=2, default=repr, allow_nan=False) + "\n")
    return payload


def save_checkpoint_atomically(path: Path, state, epoch, dev_loss):
    """Write to a temporary file and replace, so a failure cannot leave a torn checkpoint."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": state, "epoch": epoch, "dev_loss": dev_loss}, temporary)
    os.replace(temporary, path)
    return path


def run_training(out: Path, *, train_count, dev_count, load_thermal, device,
                 verbose=True, **kwargs):
    """Train while persisting each epoch and each new best checkpoint as they happen.

    A run that dies at epoch 29 keeps its history and its best checkpoint so far, and the
    failure is recorded rather than retried with a different learning rate or batch size.
    """
    out = Path(out)
    history_path, checkpoint = out / "epochs.jsonl", out / "checkpoint.pt"
    history = history_path.open("x")

    def persist_epoch(entry):
        history.write(json.dumps(entry, allow_nan=False) + "\n")
        history.flush()
        os.fsync(history.fileno())
        if verbose:
            print(json.dumps(entry), flush=True)

    def persist_best(state, epoch, dev_loss):
        save_checkpoint_atomically(checkpoint, state, epoch, dev_loss)

    try:
        record, state = train_pilot(train_count=train_count, dev_count=dev_count,
                                    load_thermal=load_thermal, device=device,
                                    on_epoch=persist_epoch, on_improvement=persist_best, **kwargs)
    except PilotFailure as failure:
        history.close()
        write_failure(out, failure, "training", failure.context)
        raise
    except Exception as exception:
        history.close()
        write_failure(out, exception, "training")
        raise
    else:
        history.close()
    record["epoch_history_file"] = str(history_path)
    record["checkpoint_file"] = str(checkpoint)
    record["checkpoint_file_sha256"] = ThermalCache._file_digest(checkpoint)
    record["checkpoint_save"] = "temporary_file_then_os_replace"
    return record, state


def _verify_digest(payload, label):
    body = dict(payload)
    claimed = body.pop("sha256")
    if sa.digest_json(body) != claimed:
        raise ValueError(f"{label} digest mismatch")
    return claimed


def resolve_evidence(report, evidence_root):
    """Re-point recorded evidence paths at this machine without touching their hashes.

    A packet moved between the Jetson collaboration directory and the server changes paths,
    never content. So the report's ``file_sha256`` stays authoritative: the file at the
    substituted path must match it, and a mismatch is an error rather than an occasion to
    rewrite the hash.
    """
    resolved = {}
    for name, entry in sorted(report["evidence"].items()):
        recorded = Path(entry["path"])
        path = (recorded if evidence_root is None
                else Path(evidence_root).resolve() / recorded.parent.name / recorded.name)
        if not path.is_file():
            raise ValueError(f"evidence {name!r} is missing at {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["file_sha256"]:
            raise ValueError(f"evidence {name!r} content differs from the prerequisite report; "
                             "the recorded hash is authoritative and is never rewritten")
        resolved[name] = {"recorded_path": str(recorded), "resolved_path": str(path),
                          "file_sha256": actual, "path_substituted": str(path) != str(recorded)}
    return resolved


def _check_fixture_pair(resolved, summary_key, provenance_key, code_files, expected_tests=None,
                        *, require_zero_fixture_steps=False):
    """A fixture PASS plus the exact code digests it was produced from.

    Missing counts must not read as zero, an all-skipped run must not read as a pass, and
    code edited after the fixtures ran must not inherit the fixture's verdict.
    """
    for key in (summary_key, provenance_key):
        if key not in resolved:
            raise ValueError(f"prerequisite report is missing {key!r} evidence")
    summary = json.loads(Path(resolved[summary_key]["resolved_path"]).read_text())
    for field in ("gate", "tests_run", "failures", "errors", "skipped"):
        if field not in summary:
            raise ValueError(f"{summary_key} evidence omits {field!r}")
    if summary["gate"] != "PASS":
        raise ValueError(f"{summary_key} evidence is not a PASS")
    for field in ("failures", "errors", "skipped"):
        if not isinstance(summary[field], int) or isinstance(summary[field], bool) or summary[field] != 0:
            raise ValueError(f"{summary_key} evidence reports {field}={summary[field]!r}")
    if not isinstance(summary["tests_run"], int) or summary["tests_run"] < 1:
        raise ValueError(f"{summary_key} evidence reports no tests")
    if expected_tests is not None and summary["tests_run"] != expected_tests:
        raise ValueError(f"{summary_key} ran {summary['tests_run']} tests, expected {expected_tests}")
    # "optimizer_steps"/"real_data_optimizer_steps" mean steps taken on REAL data and must be
    # zero. A runner fixture legitimately steps on synthetic arrays, so that count lives in
    # "fixture_optimizer_steps" and is recorded rather than forced to zero.
    real_steps = summary.get("real_data_optimizer_steps", summary.get("optimizer_steps"))
    if real_steps is None:
        raise ValueError(f"{summary_key} evidence omits 'real_data_optimizer_steps'")
    if not isinstance(real_steps, int) or isinstance(real_steps, bool) or real_steps != 0:
        raise ValueError(f"{summary_key} evidence records {real_steps!r} real-data optimizer steps")
    fixture_steps = summary.get("fixture_optimizer_steps", 0)
    if not isinstance(fixture_steps, int) or isinstance(fixture_steps, bool) or fixture_steps < 0:
        raise ValueError(f"{summary_key} evidence has an invalid fixture_optimizer_steps")
    if require_zero_fixture_steps and fixture_steps != 0:
        raise ValueError(f"{summary_key} should take no optimizer step at all, "
                         f"but records {fixture_steps}")
    provenance = json.loads(Path(resolved[provenance_key]["resolved_path"]).read_text())
    live = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in code_files}
    for rel, digest in live.items():
        if provenance.get("code_sha256", {}).get(rel) != digest:
            raise ValueError(f"{rel} differs from the fixture-verified copy; re-run the fixtures")
    return {"summary": summary, "code_sha256": live,
            "real_data_optimizer_steps": real_steps, "fixture_optimizer_steps": fixture_steps}


def validate_inputs(prepared: Path, prerequisites: Path, bank_result: Path, *, evidence_root=None):
    """Every gate the pilot depends on, checked before a single window is read."""
    report = json.loads(Path(prerequisites).read_text())
    _verify_digest(report, "prerequisite report")
    if report.get("stage") != "P1a_data_and_model_prerequisites" or report.get("gate") != "PASS":
        raise ValueError("prerequisite report is not a P1a PASS")
    for flag in ("neural_training_executed", "heldout_scored", "protocol_frozen"):
        if report.get(flag):
            raise ValueError(f"prerequisite report asserts {flag}; the pilot refuses to proceed")
    if report.get("optimizer_steps") != 0:
        raise ValueError("prerequisite report already records optimizer steps")
    if not report.get("training_runner_verified"):
        raise ValueError("training_runner_verified is false; this runner must be reviewed and "
                         "fixture-verified before it touches real data")
    resolved = resolve_evidence(report, evidence_root)
    for required in ("data", "fold0_controls", "fold0_bank"):
        if required not in resolved:
            raise ValueError(f"prerequisite report is missing {required!r} evidence")

    # The model fixtures build and differentiate models but never step an optimiser; the
    # runner fixtures do step, on synthetic arrays only.
    model_fixtures = _check_fixture_pair(resolved, "neural_fixtures", "neural_provenance",
                                        MODEL_CODE_FILES, EXPECTED_MODEL_FIXTURE_TESTS,
                                        require_zero_fixture_steps=True)
    runner_fixtures = _check_fixture_pair(resolved, "runner_fixtures", "runner_provenance",
                                         RUNNER_CODE_FILES)

    manifest_path = prepared / "split_manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != resolved["data"]["file_sha256"]:
        raise ValueError("--prepared manifest is not the file the prerequisite report validated")
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = _verify_digest(manifest, "split manifest")
    if manifest["data_gate"] != "PASS" or manifest["protocol_id"] != sa.PROTOCOL:
        raise ValueError("prepared manifest is not a PASS for this protocol")
    raw = json.loads((prepared / "raw_manifest.json").read_text())
    if sa.digest_json(raw) != manifest["source_manifest_sha256"]:
        raise ValueError("raw manifest hash mismatch")

    fold = manifest["folds"][str(FOLD)]
    normalizer_sha = _verify_digest(dict(fold["normalizer"]), "fold normalizer")

    found = json.loads(Path(resolved["fold0_controls"]["resolved_path"]).read_text())
    _verify_digest(found, "controls result")
    if (found["calibration_gate"] != "PASS" or found["fold"] != FOLD or found["seed"] != SEED
            or found["protocol_id"] != sa.PROTOCOL or found["ct_channel"] != "CT1"
            or found["normalizer_sha256"] != normalizer_sha
            or found["gradient_training_runs"] != 0 or found["heldout_scored"]):
        raise ValueError("fold 0 controls evidence does not satisfy the pilot precondition")

    summary = json.loads(Path(resolved["fold0_bank"]["resolved_path"]).read_text())
    if summary.get("full_P0_gate") != "NOT_EVALUATED":
        raise ValueError("fold 0 bank summary claims a P0 gate result")
    for role in ALLOWED_ROLES:
        coverage = summary["coverage"][role]
        if coverage["bank_gate"] != "PASS" or coverage["missing_accepted_cells"]:
            raise ValueError(f"fold 0 bank coverage is not a PASS for {role}")
    payload = json.loads((Path(bank_result) / "bank.json").read_text())
    claimed = _verify_digest(payload, "bank result")
    if claimed != summary["bank_payload_sha256"]:
        raise ValueError("bank.json payload digest disagrees with the checked summary")
    if (payload["fold"] != FOLD or payload["seed"] != SEED
            or payload["protocol_id"] != sa.PROTOCOL or payload["ct_channel"] != "CT1"
            or payload["normalizer_sha256"] != normalizer_sha
            or payload["heldout_read"] or payload["calibration_read"]
            or payload["bank_gate"] != "PASS"):
        raise ValueError("bank.json does not satisfy the pilot precondition")

    return {"manifest": manifest, "raw": raw, "fold": fold,
            "evidence": {"prerequisite_report_sha256": report["sha256"],
                         "prepared_payload_sha256": manifest_sha,
                         "normalizer_sha256": normalizer_sha,
                         "controls_payload_sha256": found["sha256"],
                         "bank_payload_sha256": claimed,
                         "bank_normalizer_binding_verified": True,
                         "model_fixtures": model_fixtures, "runner_fixtures": runner_fixtures,
                         "resolved_evidence": resolved,
                         "evidence_hashes_recomputed_after_path_substitution": False}}


def execute_pilot(out: Path, inputs, *, device, micro_batch=MICRO_BATCH, ct_channel="CT1",
                  epochs=EPOCHS, cli_started=None, reader_factory=None, verbose=True):
    """Everything after the gates: select, cache, train, and serialise the record.

    Split out of :func:`main` so the final artifact path is reachable from a fixture. A
    failure anywhere here, including while writing ``pilot.json``, is recorded rather than
    leaving an output directory that cannot be told apart from a crash before any work.
    """
    out = Path(out)
    cli_started = time.perf_counter() if cli_started is None else cli_started
    manifest, fold = inputs["manifest"], inputs["fold"]
    try:
        train_windows = select_windows(manifest, fold["roles"], "fit")
        dev_windows = select_windows(manifest, fold["roles"], "dev")
        provenance = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
            "environment_profile": "SERVER-TRAINING", "python_version": platform.python_version(),
            "numpy_version": np.__version__, "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda, "command": sys.argv,
            "purpose": "P1a_T_AE_pilot_uncommitted_diagnostic",
            "git_commit_sha": manifest["provenance"]["git_commit_sha"],
            "git_dirty": True,
            "git_dirty_basis": ("conservative_declaration_isolated_uncommitted_pilot_packet_"
                                "not_a_git_status_measurement"),
            "code_sha256": {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in
                            ("src/train_anomaly_pilot.py", "src/models/anomaly_detectors.py",
                             "src/models/anomaly_baselines.py", "src/models/v2_plus.py",
                             "src/fit_anomaly_controls.py", "src/data/synthetic_anomaly.py")},
            "input_evidence": inputs["evidence"], "raw_modified": False, "packages_installed": [],
            "device": str(device),
            "fit_devices": sorted({w["device"] for w in train_windows}),
            "dev_devices": sorted({w["device"] for w in dev_windows}),
            "fit_windows": len(train_windows), "dev_windows": len(dev_windows),
            "fit_window_ids_sha256": sa.digest_json([w["base_window_id"] for w in train_windows]),
            "dev_window_ids_sha256": sa.digest_json([w["base_window_id"] for w in dev_windows]),
            "calibration_or_heldout_windows_read": False,
        }
        if device.type == "cuda":
            provenance.update(
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cuda_device_count=torch.cuda.device_count(),
                gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                exclusive_gpu_access_enforced=False)
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

        data_root = Path(manifest["provenance"]["data_root"]).resolve()
        make = reader_factory or controls.make_reader
        reader = make(data_root, inputs["raw"])
        cache_started = time.perf_counter()
        cache = ThermalCache.build(
            out / "thermal_cache.npy", train_windows + dev_windows, reader, fold["normalizer"],
            ct_channel=ct_channel,
            progress=(lambda n, total: print(f"cache {n}/{total}", flush=True)) if verbose else None)
        cache.open()
        cache_seconds = time.perf_counter() - cache_started

        record, _ = run_training(out, train_count=len(train_windows), dev_count=len(dev_windows),
                                 load_thermal=cache, device=device, micro_batch=micro_batch,
                                 epochs=epochs, verbose=verbose)
        # run_training already recorded checkpoint_file, checkpoint_file_sha256 and
        # epoch_history_file. Re-deriving them from a local that no longer existed here is
        # exactly what killed the first real 30-epoch run after it had finished training.
        record["ct_channel"] = ct_channel
        record["input_evidence"] = inputs["evidence"]
        record["window_cache"] = cache.record()
        record["cache_stage_wall_seconds"] = cache_seconds
        record["cli_wall_seconds"] = time.perf_counter() - cli_started
        record["preparation_wall_seconds"] = (record["cli_wall_seconds"]
                                              - record["train_loop_wall_seconds"])
        record["sha256"] = hashlib.sha256(
            json.dumps({k: v for k, v in record.items() if k != "sha256"},
                       sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (out / "pilot.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    except PilotFailure as failure:
        if not (out / "failure.json").exists():
            write_failure(out, failure, "training", failure.context)
        raise
    except Exception as exception:
        if not (out / "failure.json").exists():
            write_failure(out, exception, "preparation_or_serialisation")
        raise
    if verbose:
        print(json.dumps({k: record[k] for k in
                          ("dev_checkpoint_epoch", "dev_checkpoint_loss", "train_loop_wall_seconds",
                           "cli_wall_seconds", "optimizer_steps_total",
                           "full_epoch_budget_completed")}), flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prerequisites", type=Path, required=True,
                        help="exp004-p1a-prerequisite-validation.json")
    parser.add_argument("--bank-result", type=Path, required=True,
                        help="directory holding the fold 0 fit/dev bank.json")
    parser.add_argument("--evidence-root", type=Path, default=None,
                        help="directory holding the evidence run folders on this machine")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, default=MICRO_BATCH)
    parser.add_argument("--ct-channel", choices=["CT1"], default="CT1",
                        help="P1a is fixed to CT1")
    args = parser.parse_args(argv)
    cli_started = time.perf_counter()

    if (platform.machine() != "x86_64" or Path("/etc/nv_tegra_release").exists()
            or Path(sys.prefix).resolve() != (Path.home() / "venvs/factory_training").resolve()
            or os.environ.get("PYTHONNOUSERSITE") != "1"):
        parser.error("the pilot requires the audited SERVER-TRAINING venv")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; the pilot never falls back to CPU for real training")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        parser.error("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the process")
    if args.micro_batch < 1:
        parser.error("--micro-batch must be a positive int")
    if args.ct_channel != "CT1":
        parser.error("P1a is fixed to CT1")

    inputs = validate_inputs(args.prepared, args.prerequisites, args.bank_result,
                             evidence_root=args.evidence_root)
    data_root = Path(inputs["manifest"]["provenance"]["data_root"]).resolve()
    out = args.out.resolve()
    if out.exists() or out == data_root or data_root in out.parents or out in data_root.parents:
        parser.error("output must be new and separate from raw data")
    out.mkdir(parents=True, exist_ok=False)

    execute_pilot(out, inputs, device=torch.device("cuda"), micro_batch=args.micro_batch,
                  ct_channel=args.ct_channel, cli_started=cli_started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
