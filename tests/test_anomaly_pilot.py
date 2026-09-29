"""P1a pilot contract: gates, accumulation arithmetic and a tiny no-data training fixture.

No test here reads the dataset, the calibration split or the held-out split. Training
fixtures run on synthetic arrays on CPU for 1-2 epochs; the real 30-epoch run is a
separate operation that requires CUDA and a verified prerequisite report.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

LOCAL_DEPENDENCIES = ("src/train_anomaly_pilot.py", "src/models/anomaly_detectors.py",
                      "src/models/anomaly_baselines.py", "src/models/v2_plus.py",
                      "src/fit_anomaly_controls.py", "src/data/synthetic_anomaly.py",
                      "tests/test_anomaly_detectors.py")

try:
    import numpy as np
    import torch
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised on Jetson
    np = torch = None
    IMPORT_ERROR = repr(exc)


def leaf(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if torch is not None:
    pilot = leaf("_pilot_under_test", "src/train_anomaly_pilot.py")
    ad = pilot.detectors
else:  # pragma: no cover
    pilot = ad = None

requires_torch = unittest.skipIf(torch is None, f"torch unavailable (expected on Jetson): {IMPORT_ERROR}")


def tiny_loader(count, *, seed=0):
    """Deterministic synthetic thermal windows; no dataset is touched."""
    rng = np.random.default_rng(seed)
    windows = rng.standard_normal((count, ad.WINDOW_TICKS, *ad.THERMAL_SHAPE)).astype(np.float32)
    return lambda index: windows[index], windows


@requires_torch
class AccumulationTests(unittest.TestCase):
    def test_weights_sum_to_one_and_use_the_actual_batch_size(self):
        for actual in (1, 2, 3, 15, 16, 17, 32):
            for micro in (1, 2, 4, 16):
                with self.subTest(actual=actual, micro=micro):
                    chunks = pilot.partition_microbatches(actual, micro)
                    self.assertEqual(chunks[0][0], 0)
                    self.assertEqual(chunks[-1][1], actual)
                    self.assertAlmostEqual(sum(w for _, _, w in chunks), 1.0, places=12)
                    for (_, stop, _), (start, _, _) in zip(chunks, chunks[1:]):
                        self.assertEqual(stop, start)
                    for start, stop, weight in chunks:
                        self.assertAlmostEqual(weight, (stop - start) / actual, places=12)

    def test_trailing_partial_batch_is_not_normalised_by_the_nominal_16(self):
        chunks = pilot.partition_microbatches(5, 2)
        self.assertEqual([(s, e) for s, e, _ in chunks], [(0, 2), (2, 4), (4, 5)])
        self.assertAlmostEqual(chunks[-1][2], 1 / 5, places=12)
        self.assertNotAlmostEqual(chunks[-1][2], 1 / 16, places=6)

    def test_invalid_sizes_are_refused(self):
        for actual, micro in ((0, 2), (-1, 2), (4, 0), (True, 2), (4, True), (2.0, 2)):
            with self.subTest(actual=actual, micro=micro):
                with self.assertRaises(ValueError):
                    pilot.partition_microbatches(actual, micro)

    def test_accumulated_gradient_equals_the_whole_batch_gradient(self):
        """Element-mean MSE decomposes exactly, so accumulation must not change the gradient."""
        torch.manual_seed(5)
        reference = ad.AnomalyDetector("S", "AE").eval()
        batch = torch.randn(5, ad.WINDOW_TICKS, ad.SENSOR_DIM)

        def grads(partition_micro):
            model = copy.deepcopy(reference)
            model.zero_grad(set_to_none=True)
            if partition_micro is None:
                total, _ = ad.autoencoder_loss(model(sensor=batch), sensor=batch, arm="S")
                total.backward()
            else:
                for start, stop, weight in pilot.partition_microbatches(len(batch), partition_micro):
                    chunk = batch[start:stop]
                    total, _ = ad.autoencoder_loss(model(sensor=chunk), sensor=chunk, arm="S")
                    (total * weight).backward()
            return {n: p.grad.clone() for n, p in model.named_parameters()}

        whole = grads(None)
        for micro in (1, 2, 3, 4, 5):
            with self.subTest(micro=micro):
                for name, grad in grads(micro).items():
                    self.assertTrue(torch.allclose(whole[name], grad, atol=1e-6, rtol=1e-5),
                                    f"micro={micro} {name}")

    def test_accumulated_gradient_equivalence_holds_for_the_p1a_thermal_arm(self):
        torch.manual_seed(7)
        reference = ad.AnomalyDetector("T", "AE").eval()
        batch = torch.randn(2, ad.WINDOW_TICKS, *ad.THERMAL_SHAPE)
        whole = copy.deepcopy(reference)
        total, _ = ad.autoencoder_loss(whole(thermal=batch), thermal=batch, arm="T")
        total.backward()
        split = copy.deepcopy(reference)
        for start, stop, weight in pilot.partition_microbatches(2, 1):
            chunk = batch[start:stop]
            part, _ = ad.autoencoder_loss(split(thermal=chunk), thermal=chunk, arm="T")
            (part * weight).backward()
        for (name, a), (_, b) in zip(whole.named_parameters(), split.named_parameters()):
            self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-6, rtol=1e-4), name)


@requires_torch
class PermutationTests(unittest.TestCase):
    def test_order_depends_only_on_seed_and_epoch(self):
        a = pilot.epoch_permutation(50, seed=42, epoch=3)
        self.assertTrue(np.array_equal(a, pilot.epoch_permutation(50, seed=42, epoch=3)))
        self.assertFalse(np.array_equal(a, pilot.epoch_permutation(50, seed=42, epoch=4)))
        self.assertFalse(np.array_equal(a, pilot.epoch_permutation(50, seed=123, epoch=3)))
        self.assertEqual(sorted(a.tolist()), list(range(50)))


@requires_torch
class TinyTrainingFixtureTests(unittest.TestCase):
    """Runs the real loop on synthetic arrays. No dataset, no CUDA, few epochs."""

    def train(self, *, epochs=2, train_count=5, dev_count=2, micro_batch=1, nominal=2, loader=None):
        load, windows = tiny_loader(train_count + dev_count) if loader is None else loader
        return pilot.train_pilot(train_count=train_count, dev_count=dev_count, load_thermal=load,
                                 device=torch.device("cpu"), epochs=epochs, nominal_batch=nominal,
                                 micro_batch=micro_batch), windows

    def test_record_reports_the_budget_and_costs_it_actually_ran(self):
        (record, state), _ = self.train(epochs=2)
        self.assertEqual(record["epochs_completed"], 2)
        self.assertEqual(len(record["epochs"]), 2)
        self.assertFalse(record["full_epoch_budget_completed"])  # 2 != the 30 budget
        self.assertEqual(sum(e["optimizer_steps"] for e in record["epochs"]),
                         sum(e["batches"] for e in record["epochs"]))
        self.assertGreater(record["train_loop_wall_seconds"], 0)
        self.assertGreaterEqual(record["total_io_seconds"], 0)
        self.assertGreaterEqual(record["total_host_to_device_seconds"], 0)
        self.assertEqual(record["cost_basis"], "measured_over_2_epochs")
        self.assertFalse(record["cost_basis_is_full_budget"])
        self.assertEqual(record["optimizer_steps_total"],
                         sum(e["optimizer_steps"] for e in record["epochs"]))
        self.assertEqual(record["train_loop_wall_excludes"], ["cache_build", "input_validation"])
        self.assertIn("epoch_history_and_checkpoint_callbacks", record["train_loop_wall_includes"])
        for entry in record["epochs"]:
            self.assertAlmostEqual(
                entry["wall_seconds"],
                entry["io_seconds"] + entry["host_to_device_seconds"]
                + entry["compute_seconds"] + entry["unaccounted_seconds"], places=6)
        self.assertEqual(record["thermal_output_elements_per_window"], 576000)
        self.assertEqual(record["encoder_parameters"] + record["decoder_parameters"],
                         record["total_parameters"])
        self.assertEqual(set(state), set(ad.AnomalyDetector("T", "AE").state_dict()))

    def test_declared_contract_fields_are_recorded_as_measured_facts(self):
        (record, _), _ = self.train()
        self.assertEqual(record["arm"], "T")
        self.assertEqual(record["objective"], "AE")
        self.assertEqual(record["fold"], 0)
        self.assertEqual(record["seed"], 42)
        self.assertEqual(record["precision"], "FP32")
        self.assertEqual(record["num_workers"], 0)
        self.assertFalse(record["worker_processes_used"])
        self.assertFalse(record["sensor_modality_used"])
        self.assertFalse(record["calibration_window_read"])
        self.assertFalse(record["heldout_window_read"])
        self.assertFalse(record["heldout_model_performance_computed"])
        self.assertIn("window read", record["read_flag_scope"])
        self.assertEqual(record["model_selected_on"], "dev_Normal_reconstruction_loss_only")
        self.assertEqual(record["full_P0_gate"], "NOT_EVALUATED")
        self.assertEqual(record["status"], "P1a_uncommitted_diagnostic")
        self.assertEqual(record["partial_batch_normalisation"], "actual_batch_size_not_nominal")
        self.assertEqual(record["optimizer"], {"name": "AdamW", "lr": 1e-3, "weight_decay": 0.01,
                                               "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False})
        self.assertEqual(record["lr_schedule"]["name"], "CosineAnnealingLR")
        self.assertTrue(record["determinism"]["requested"])
        self.assertTrue(record["determinism"]["strict"])
        self.assertFalse(record["determinism"]["warn_only"])
        self.assertEqual(record["determinism"]["nondeterministic_op_policy"], "raise_immediately")
        self.assertTrue(record["determinism"]["torch_deterministic_algorithms"])
        self.assertTrue(record["determinism"]["cudnn_deterministic"])
        self.assertFalse(record["determinism"]["cudnn_benchmark"])
        self.assertFalse(record["determinism"]["tf32_matmul"])

    def test_reported_learning_rate_separates_the_epoch_used_from_the_next(self):
        """scheduler.step() runs at the end of an epoch, so one value is not both."""
        (record, _), _ = self.train(epochs=3)
        self.assertAlmostEqual(record["epochs"][0]["lr_used_this_epoch"], 1e-3, places=12)
        for earlier, later in zip(record["epochs"], record["epochs"][1:]):
            self.assertAlmostEqual(earlier["next_epoch_lr"], later["lr_used_this_epoch"], places=12)
        self.assertLess(record["epochs"][-1]["next_epoch_lr"],
                        record["epochs"][0]["lr_used_this_epoch"])

    def test_actual_batch_sizes_include_the_trailing_partial_batch(self):
        (record, _), _ = self.train(train_count=5, nominal=2)
        self.assertEqual(record["actual_batch_size_counts"], {"1": 2, "2": 4})

    def test_two_runs_with_the_same_seed_produce_the_same_checkpoint(self):
        load, _ = tiny_loader(7, seed=3)
        kwargs = dict(train_count=5, dev_count=2, load_thermal=load,
                      device=torch.device("cpu"), epochs=2, nominal_batch=2, micro_batch=1)
        first, _ = pilot.train_pilot(**kwargs)
        second, _ = pilot.train_pilot(**kwargs)
        self.assertEqual(first["checkpoint_state_sha256"], second["checkpoint_state_sha256"])
        self.assertEqual(first["initial_weights_sha256"], second["initial_weights_sha256"])
        self.assertEqual([e["train_loss"] for e in first["epochs"]],
                         [e["train_loss"] for e in second["epochs"]])

    def test_checkpoint_keeps_the_earlier_epoch_on_an_exact_dev_loss_tie(self):
        """lr=0 freezes the weights, so every epoch scores identically and ties for real."""
        load, _ = tiny_loader(7, seed=11)
        record, _ = pilot.train_pilot(train_count=5, dev_count=2, load_thermal=load,
                                      device=torch.device("cpu"), epochs=3, nominal_batch=2,
                                      micro_batch=1, lr=0.0, weight_decay=0.0)
        losses = [e["dev_loss"] for e in record["epochs"]]
        self.assertEqual(len(set(losses)), 1, f"lr=0 should freeze the dev loss, got {losses}")
        self.assertEqual(record["dev_checkpoint_epoch"], 1)
        self.assertEqual(record["dev_checkpoint_tied_later_epochs"], [2, 3])
        self.assertEqual(record["dev_checkpoint_tie_rule"], "earlier_epoch")

    def test_checkpoint_tracks_the_minimum_when_losses_differ(self):
        (record, _), _ = self.train(epochs=3)
        best = min(e["dev_loss"] for e in record["epochs"])
        earliest = min(e["epoch"] for e in record["epochs"] if e["dev_loss"] == best)
        self.assertEqual(record["dev_checkpoint_epoch"], earliest)
        self.assertEqual(record["dev_checkpoint_loss"], best)
        self.assertNotIn(record["dev_checkpoint_epoch"], record["dev_checkpoint_tied_later_epochs"])

    def test_improvement_and_epoch_callbacks_fire_so_progress_can_be_persisted(self):
        load, _ = tiny_loader(7, seed=13)
        seen, saved = [], []
        record, _ = pilot.train_pilot(train_count=5, dev_count=2, load_thermal=load,
                                      device=torch.device("cpu"), epochs=2, nominal_batch=2,
                                      micro_batch=1, on_epoch=seen.append,
                                      on_improvement=lambda state, epoch, loss: saved.append(epoch))
        self.assertEqual([e["epoch"] for e in seen], [1, 2])
        self.assertIn(record["dev_checkpoint_epoch"], saved)
        self.assertEqual(saved[0], 1)

    def test_non_finite_window_is_preserved_and_stops_the_run_without_retry(self):
        load, windows = tiny_loader(7, seed=1)
        windows[0, 0, 0, 0] = np.float32("nan")
        with self.assertRaises(pilot.PilotFailure) as caught:
            pilot.train_pilot(train_count=5, dev_count=2, load_thermal=load,
                              device=torch.device("cpu"), epochs=1, nominal_batch=2, micro_batch=1)
        context = caught.exception.context
        self.assertEqual(context["epoch"], 1)
        self.assertFalse(context["finite"])
        self.assertIsInstance(context["value"], str)
        json.dumps(context)  # the failure record must be writable as standard JSON

    def test_non_finite_dev_window_is_caught_in_the_dev_pass(self):
        load, windows = tiny_loader(7, seed=2)
        windows[6, 0, 0, 0] = np.float32("inf")  # index 6 is dev when train_count=5
        with self.assertRaises(pilot.PilotFailure) as caught:
            pilot.train_pilot(train_count=5, dev_count=2, load_thermal=load,
                              device=torch.device("cpu"), epochs=1, nominal_batch=2, micro_batch=1)
        self.assertIn("dev", caught.exception.context["phase"])

    def test_invalid_batch_sizes_are_refused_before_any_epoch(self):
        load, _ = tiny_loader(7)
        for kwargs in ({"micro_batch": 0}, {"nominal_batch": 0}, {"micro_batch": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    pilot.train_pilot(train_count=5, dev_count=2, load_thermal=load,
                                      device=torch.device("cpu"), epochs=1, **kwargs)

    def test_both_partitions_are_required(self):
        load, _ = tiny_loader(4)
        for train_count, dev_count in ((0, 2), (3, 0)):
            with self.assertRaises(ValueError):
                pilot.train_pilot(train_count=train_count, dev_count=dev_count,
                                  load_thermal=load, device=torch.device("cpu"), epochs=1)


@requires_torch
class RoleBoundaryTests(unittest.TestCase):
    def manifest(self):
        windows, roles = [], {}
        for index, (device, role) in enumerate([("agv01", "fit"), ("oht01", "fit"),
                                               ("agv02", "dev"), ("oht02", "dev"),
                                               ("agv03", "calibration"), ("oht03", "heldout")]):
            roles[device] = role
            windows.append({"device": device, "base_window_id": f"w{index}", "hard_valid": True,
                            "pure_Normal": True, "start_index": 0, "raw_base_ids": []})
        return {"windows": windows}, roles

    def test_only_fit_and_dev_may_be_selected(self):
        manifest, roles = self.manifest()
        self.assertEqual(len(pilot.select_windows(manifest, roles, "fit")), 2)
        self.assertEqual(len(pilot.select_windows(manifest, roles, "dev")), 2)
        for forbidden in ("calibration", "heldout", "heldout-eval", "test"):
            with self.subTest(role=forbidden):
                with self.assertRaisesRegex(ValueError, "refused role"):
                    pilot.select_windows(manifest, roles, forbidden)

    def test_ineligible_windows_are_excluded_and_missing_equipment_is_an_error(self):
        manifest, roles = self.manifest()
        manifest["windows"][0]["pure_Normal"] = False
        with self.assertRaisesRegex(ValueError, "missing eligible equipment"):
            pilot.select_windows(manifest, roles, "fit")
        manifest["windows"][0]["pure_Normal"] = True
        manifest["windows"][1]["hard_valid"] = False
        with self.assertRaisesRegex(ValueError, "missing eligible equipment"):
            pilot.select_windows(manifest, roles, "fit")

    def test_allowed_roles_are_exactly_fit_and_dev(self):
        self.assertEqual(pilot.ALLOWED_ROLES, ("fit", "dev"))


@requires_torch
class FailurePreservationTests(unittest.TestCase):
    """Behaviour, not source strings: a dying run keeps what it already produced."""

    def loader_failing_at(self, calls):
        state = {"n": 0}
        windows, _ = tiny_loader(7, seed=21)

        def load(index):
            state["n"] += 1
            if state["n"] > calls:
                raise RuntimeError("injected CUDA out of memory")
            return windows(index)
        return load

    def test_history_and_best_checkpoint_survive_a_mid_run_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            with self.assertRaisesRegex(RuntimeError, "injected CUDA out of memory"):
                pilot.run_training(out, train_count=5, dev_count=2,
                                   load_thermal=self.loader_failing_at(9),
                                   device=torch.device("cpu"), epochs=3,
                                   nominal_batch=2, micro_batch=1, verbose=False)
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["exception"], "RuntimeError")
            self.assertEqual(failure["stage"], "training")
            self.assertFalse(failure["retried"])
            self.assertFalse(failure["learning_rate_adjusted"])
            self.assertFalse(failure["out_of_memory_retry"])
            self.assertTrue(failure["run_abandoned"])
            history = [json.loads(line) for line in
                       (out / "epochs.jsonl").read_text().splitlines() if line]
            self.assertEqual([e["epoch"] for e in history], [1])
            self.assertIn("epochs.jsonl", failure["preserved"])
            self.assertIn("checkpoint.pt", failure["preserved"])
            saved = torch.load(out / "checkpoint.pt", weights_only=False)
            self.assertEqual(saved["epoch"], 1)
            self.assertAlmostEqual(saved["dev_loss"], history[0]["dev_loss"], places=12)

    def test_a_pilot_failure_is_recorded_with_its_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            load, windows = tiny_loader(7, seed=22)
            windows[0, 0, 0, 0] = np.float32("nan")
            with self.assertRaises(pilot.PilotFailure):
                pilot.run_training(out, train_count=5, dev_count=2, load_thermal=load,
                                   device=torch.device("cpu"), epochs=1,
                                   nominal_batch=2, micro_batch=1, verbose=False)
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["exception"], "PilotFailure")
            self.assertFalse(failure["context"]["finite"])
            self.assertFalse(failure["retried"])

    def test_successful_run_records_the_atomic_save_and_leaves_no_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            load, _ = tiny_loader(7, seed=23)
            record, _ = pilot.run_training(out, train_count=5, dev_count=2, load_thermal=load,
                                           device=torch.device("cpu"), epochs=2,
                                           nominal_batch=2, micro_batch=1, verbose=False)
            self.assertFalse((out / "failure.json").exists())
            self.assertEqual(record["checkpoint_save"], "temporary_file_then_os_replace")
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["checkpoint.pt", "epochs.jsonl"])
            self.assertEqual(record["checkpoint_file_sha256"],
                             pilot.ThermalCache._file_digest(out / "checkpoint.pt"))
            history = (out / "epochs.jsonl").read_text().splitlines()
            self.assertEqual(len(history), 2)

    def test_history_file_is_never_reopened_over_an_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            (out / "epochs.jsonl").write_text("")
            load, _ = tiny_loader(7, seed=24)
            with self.assertRaises(FileExistsError):
                pilot.run_training(out, train_count=5, dev_count=2, load_thermal=load,
                                   device=torch.device("cpu"), epochs=1,
                                   nominal_batch=2, micro_batch=1, verbose=False)


@requires_torch
class EndToEndArtifactTests(unittest.TestCase):
    """Run the whole post-gate path on synthetic arrays, through to pilot.json.

    The first real 30-epoch run finished training and then died serialising its record,
    because a refactor left main referring to a local that had moved into run_training.
    Unit tests of train_pilot and run_training could not see that: only a test that reaches
    the final artifact can. Nothing here touches the dataset, CUDA or the CLI gates.
    """

    def inputs(self, tmp, fit_devices=("agv01", "oht01"), dev_devices=("agv02",), per_device=2):
        windows, roles = [], {}
        for device, role in [(d, "fit") for d in fit_devices] + [(d, "dev") for d in dev_devices]:
            roles[device] = role
            for k in range(per_device):
                windows.append({"device": device, "base_window_id": f"{device}:{k}",
                                "session_id": f"s/{device}", "start_index": k * 30,
                                "raw_base_ids": [f"{device}/{k}/{i}" for i in range(30)],
                                "hard_valid": True, "pure_Normal": True, "base_QC_flags": []})
        manifest = {"windows": windows,
                    "provenance": {"data_root": str(tmp / "raw"), "git_commit_sha": "0" * 40}}
        normalizer = {"sensor_mean": [0.0] * 8, "sensor_scale": [1.0] * 8,
                      "thermal_mean": 40.0, "thermal_scale": 3.0}
        fold = {"roles": roles, "normalizer": normalizer}
        (tmp / "raw").mkdir(parents=True, exist_ok=True)
        return {"manifest": manifest, "raw": [], "fold": fold,
                "evidence": {"synthetic_fixture": True}}

    def reader_factory(self, seed=0):
        def make(data_root, raw_manifest):
            def reader(window):
                # Python's hash() of a str is salted per process, so it cannot seed a
                # reproducible fixture. Derive the namespace from SHA-256 instead.
                digest = hashlib.sha256(f"{window['base_window_id']}|{seed}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
                sensor = rng.standard_normal((ad.WINDOW_TICKS, 8))
                thermal = 40.0 + 3.0 * rng.standard_normal((ad.WINDOW_TICKS, *ad.THERMAL_SHAPE))
                return sensor, thermal, np.zeros(ad.WINDOW_TICKS, dtype=int)
            return reader
        return make

    def test_full_post_gate_path_writes_a_complete_pilot_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "run"
            out.mkdir()
            record = pilot.execute_pilot(out, self.inputs(tmp), device=torch.device("cpu"),
                                         epochs=2, micro_batch=1, reader_factory=self.reader_factory(),
                                         verbose=False)
            self.assertFalse((out / "failure.json").exists())
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["checkpoint.pt", "epochs.jsonl", "pilot.json", "provenance.json",
                              "thermal_cache.npy"])
            saved = json.loads((out / "pilot.json").read_text())
            self.assertEqual(saved["sha256"], record["sha256"])
            # The fields that the broken run never reached.
            for key in ("checkpoint_file", "checkpoint_file_sha256", "epoch_history_file",
                        "window_cache", "cache_stage_wall_seconds", "cli_wall_seconds",
                        "preparation_wall_seconds", "ct_channel", "input_evidence"):
                self.assertIn(key, saved, key)
            self.assertEqual(saved["checkpoint_file"], str(out / "checkpoint.pt"))
            self.assertEqual(saved["checkpoint_file_sha256"],
                             pilot.ThermalCache._file_digest(out / "checkpoint.pt"))
            self.assertEqual(saved["train_windows"], 4)
            self.assertEqual(saved["dev_windows"], 2)
            self.assertTrue(saved["window_cache"]["stored_payload_matches_source"])
            self.assertGreater(saved["cli_wall_seconds"], saved["train_loop_wall_seconds"])
            self.assertGreater(saved["preparation_wall_seconds"], 0)

    def test_record_is_valid_json_without_nan_and_digest_covers_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "run"
            out.mkdir()
            pilot.execute_pilot(out, self.inputs(tmp), device=torch.device("cpu"), epochs=1,
                                micro_batch=1, reader_factory=self.reader_factory(), verbose=False)
            text = (out / "pilot.json").read_text()
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            saved = json.loads(text)
            recomputed = hashlib.sha256(json.dumps(
                {k: v for k, v in saved.items() if k != "sha256"},
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.assertEqual(recomputed, saved["sha256"])

    def test_provenance_is_written_before_training_so_a_crash_still_leaves_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "run"
            out.mkdir()
            inputs = self.inputs(tmp)
            # Keep the role: removing it would also drop the device from `expected`, so the
            # equality check would still pass. Make its windows ineligible instead.
            for window in inputs["manifest"]["windows"]:
                if window["device"] == "agv01":
                    window["pure_Normal"] = False
            with self.assertRaisesRegex(ValueError, "missing eligible equipment"):
                pilot.execute_pilot(out, inputs, device=torch.device("cpu"), epochs=1,
                                    micro_batch=1, reader_factory=self.reader_factory(),
                                    verbose=False)
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["stage"], "preparation_or_serialisation")
            self.assertFalse(failure["retried"])

    def test_a_serialisation_failure_is_recorded_not_silent(self):
        """The exact class of bug that killed the first run: dying after training."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "run"
            out.mkdir()
            original = pilot.ThermalCache.record

            def exploding(self):
                raise RuntimeError("injected serialisation failure")
            pilot.ThermalCache.record = exploding
            try:
                with self.assertRaisesRegex(RuntimeError, "injected serialisation failure"):
                    pilot.execute_pilot(out, self.inputs(tmp), device=torch.device("cpu"),
                                        epochs=1, micro_batch=1,
                                        reader_factory=self.reader_factory(), verbose=False)
            finally:
                pilot.ThermalCache.record = original
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["stage"], "preparation_or_serialisation")
            # Training output is preserved even though the record was never written.
            self.assertIn("epochs.jsonl", failure["preserved"])
            self.assertIn("checkpoint.pt", failure["preserved"])
            self.assertFalse((out / "pilot.json").exists())

    def test_reader_fixture_is_reproducible_across_processes(self):
        """Guards the fixture itself; a salted hash would make failures irreproducible."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            window = {"base_window_id": "agv01:0"}
            a = self.reader_factory()(tmp, [])(window)[1]
            b = self.reader_factory()(tmp, [])(window)[1]
            self.assertTrue(np.array_equal(a, b))
            c = self.reader_factory(seed=1)(tmp, [])(window)[1]
            self.assertFalse(np.array_equal(a, c))

    def test_verbose_is_consumed_by_run_training_and_never_reaches_train_pilot(self):
        import inspect
        seen = {}
        original = pilot.train_pilot

        def spy(**kwargs):
            seen.update(kwargs)
            return original(**kwargs)
        pilot.train_pilot = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                out = tmp / "run"
                out.mkdir()
                pilot.execute_pilot(out, self.inputs(tmp), device=torch.device("cpu"), epochs=1,
                                    micro_batch=1, reader_factory=self.reader_factory(),
                                    verbose=False)
        finally:
            pilot.train_pilot = original
        self.assertNotIn("verbose", seen)
        self.assertEqual(seen["epochs"], 1)
        self.assertEqual(seen["micro_batch"], 1)
        self.assertNotIn("verbose", inspect.signature(original).parameters)


@requires_torch
class CacheTests(unittest.TestCase):
    def write_cache(self, path, count=2):
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                          shape=(count, ad.WINDOW_TICKS, *ad.THERMAL_SHAPE))
        array[:] = (np.arange(array.size, dtype=np.float32) % 7).reshape(array.shape)
        array.flush()
        del array
        return pilot.ThermalCache._file_digest(path)

    def test_required_bytes_matches_the_declared_geometry(self):
        self.assertEqual(pilot.ThermalCache.required_bytes(1),
                         ad.WINDOW_TICKS * 120 * 160 * 4)
        # fold 0 fit+dev is 2088+525 windows
        self.assertEqual(pilot.ThermalCache.required_bytes(2613), 6_020_352_000)

    def test_attached_cache_is_usable_and_an_unverified_one_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npy"
            digest = self.write_cache(path)
            cache = pilot.ThermalCache.attach(path, 2, digest).open()
            self.assertTrue(cache.verified_at_attach)
            self.assertEqual(cache(0).shape, (ad.WINDOW_TICKS, *ad.THERMAL_SHAPE))
            self.assertEqual(cache(0).dtype, np.float32)
            with self.assertRaisesRegex(ValueError, "never verified"):
                pilot.ThermalCache(path, 2).open()
            with self.assertRaisesRegex(ValueError, "must be opened"):
                pilot.ThermalCache.attach(path, 2, digest)(0)

    def test_shape_preserving_payload_edit_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npy"
            digest = self.write_cache(path)
            data = np.load(path, mmap_mode="r+")
            data[1, 5, 10, 10] = np.float32(99.0)
            data.flush()
            del data
            reopened = np.load(path, mmap_mode="r")
            self.assertEqual(reopened.shape, (2, ad.WINDOW_TICKS, *ad.THERMAL_SHAPE))
            self.assertEqual(reopened.dtype, np.float32)
            del reopened
            with self.assertRaisesRegex(ValueError, "changed since it was built"):
                pilot.ThermalCache.attach(path, 2, digest)

    def test_window_count_disagreement_is_caught_at_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npy"
            digest = self.write_cache(path, count=2)
            with self.assertRaisesRegex(ValueError, "disagrees with the declared windows"):
                pilot.ThermalCache.attach(path, 3, digest).open()

    def build_with_constant_windows(self, path, count=2, value=0.0):
        """Build a cache without touching any dataset by replacing the window reader."""
        original = pilot.normalized_thermal
        shaped = np.full((ad.WINDOW_TICKS, *ad.THERMAL_SHAPE), value, dtype=np.float32)
        pilot.normalized_thermal = lambda *a, **k: shaped
        try:
            return pilot.ThermalCache.build(
                path, [{"base_window_id": f"w{i}"} for i in range(count)], None, None)
        finally:
            pilot.normalized_thermal = original

    def test_build_compares_the_stored_payload_against_the_source_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.build_with_constant_windows(Path(tmp) / "cache.npy", value=1.5)
            self.assertTrue(cache.verified_after_flush)
            self.assertEqual(cache.source_payload_sha256, cache.payload_sha256)
            self.assertTrue(cache.record()["stored_payload_matches_source"])
            self.assertNotEqual(cache.file_sha256, cache.payload_sha256)  # header is included
            self.assertTrue(np.all(cache.open()(0) == 1.5))

    def test_build_refuses_to_overwrite_an_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npy"
            self.build_with_constant_windows(path)
            with self.assertRaisesRegex(ValueError, "never overwrites"):
                self.build_with_constant_windows(path)

    def test_source_payload_mismatch_is_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npy"
            windows = [{"base_window_id": "w0"}]
            shaped = np.zeros((ad.WINDOW_TICKS, *ad.THERMAL_SHAPE), dtype=np.float32)
            original = pilot.normalized_thermal
            real_digest = pilot.ThermalCache._digests

            def lying_digest(target):
                file_sha, _ = real_digest(target)
                return file_sha, "0" * 64  # pretend the disk holds something else
            pilot.normalized_thermal = lambda *a, **k: shaped
            pilot.ThermalCache._digests = staticmethod(lying_digest)
            try:
                with self.assertRaisesRegex(ValueError, "differs from the normalized"):
                    pilot.ThermalCache.build(path, windows, None, None)
            finally:
                pilot.normalized_thermal = original
                pilot.ThermalCache._digests = real_digest

    def test_record_separates_read_cost_from_verification_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npy"
            digest = self.write_cache(path)
            record = pilot.ThermalCache.attach(path, 2, digest).record()
            self.assertEqual(record["file_sha256"], digest)
            self.assertEqual(record["payload_bytes"], pilot.ThermalCache.required_bytes(2))
            self.assertGreaterEqual(record["digest_verify_seconds"], 0)
            self.assertIn("payload_compared_to_source_arrays", record["digest_scope"])


@requires_torch
class EvidenceTests(unittest.TestCase):
    def live_code(self, files):
        return {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in files}

    def report(self, directory, *, summaries=None, provenances=None, **overrides):
        payloads = {
            "data": {"d": 1}, "fold0_controls": {"c": 1}, "fold0_bank": {"b": 1},
            "neural_fixtures": {"gate": "PASS", "tests_run": pilot.EXPECTED_MODEL_FIXTURE_TESTS,
                                "failures": 0, "errors": 0, "skipped": 0,
                                "real_data_optimizer_steps": 0, "fixture_optimizer_steps": 0},
            "neural_provenance": {"code_sha256": self.live_code(pilot.MODEL_CODE_FILES)},
            "runner_fixtures": {"gate": "PASS", "tests_run": 7, "failures": 0, "errors": 0,
                                "skipped": 0, "real_data_optimizer_steps": 0,
                                "fixture_optimizer_steps": 84},
            "runner_provenance": {"code_sha256": self.live_code(pilot.RUNNER_CODE_FILES)},
        }
        payloads.update(summaries or {})
        payloads.update(provenances or {})
        files = {}
        for name, payload in payloads.items():
            path = directory / f"run_{name}" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            files[name] = {"path": str(path),
                           "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        report = {"stage": "P1a_data_and_model_prerequisites", "gate": "PASS",
                  "training_runner_verified": True, "neural_training_executed": False,
                  "heldout_scored": False, "protocol_frozen": False, "optimizer_steps": 0,
                  "evidence": files}
        report.update(overrides)
        report["sha256"] = pilot.sa.digest_json({k: v for k, v in report.items() if k != "sha256"})
        return report, files

    def write_report(self, directory, **kwargs):
        report, files = self.report(directory / "evidence", **kwargs)
        path = directory / "report.json"
        path.write_text(json.dumps(report))
        return path, report, files

    def test_substituted_paths_must_still_match_the_recorded_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, files = self.report(Path(tmp) / "jetson")
            moved = Path(tmp) / "server"
            for name, entry in files.items():
                target = moved / f"run_{name}"
                target.mkdir(parents=True, exist_ok=True)
                (target / f"{name}.json").write_text(Path(entry["path"]).read_text())
            resolved = pilot.resolve_evidence(original, moved)
            for name, entry in resolved.items():
                self.assertTrue(entry["path_substituted"])
                self.assertEqual(entry["file_sha256"], files[name]["file_sha256"])
            (moved / "run_data" / "data.json").write_text('{"d": 2}')
            with self.assertRaisesRegex(ValueError, "authoritative"):
                pilot.resolve_evidence(original, moved)

    def test_missing_evidence_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, _ = self.report(Path(tmp) / "jetson")
            with self.assertRaisesRegex(ValueError, "missing"):
                pilot.resolve_evidence(original, Path(tmp) / "absent")

    def test_unverified_runner_blocks_real_data_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = self.write_report(Path(tmp), training_runner_verified=False)
            with self.assertRaisesRegex(ValueError, "training_runner_verified"):
                pilot.validate_inputs(Path(tmp), path, Path(tmp))

    def test_report_digest_and_gates_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self.report(Path(tmp) / "evidence")
            tampered = Path(tmp) / "tampered.json"
            tampered.write_text(json.dumps(dict(report, optimizer_steps=1)))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                pilot.validate_inputs(Path(tmp), tampered, Path(tmp))
            for overrides, pattern in (({"gate": "FAIL"}, "not a P1a PASS"),
                                       ({"stage": "other"}, "not a P1a PASS"),
                                       ({"neural_training_executed": True}, "neural_training_executed"),
                                       ({"heldout_scored": True}, "heldout_scored"),
                                       ({"protocol_frozen": True}, "protocol_frozen"),
                                       ({"optimizer_steps": 3}, "optimizer steps")):
                with self.subTest(overrides=overrides):
                    path, _, _ = self.write_report(Path(tmp) / json.dumps(overrides), **overrides)
                    with self.assertRaisesRegex(ValueError, pattern):
                        pilot.validate_inputs(Path(tmp), path, Path(tmp))

    def test_fixture_evidence_must_carry_real_counts_and_the_matching_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, _ = self.report(Path(tmp) / "good")
            resolved = pilot.resolve_evidence(good, None)
            pair = pilot._check_fixture_pair(resolved, "neural_fixtures", "neural_provenance",
                                             pilot.MODEL_CODE_FILES,
                                             pilot.EXPECTED_MODEL_FIXTURE_TESTS,
                                             require_zero_fixture_steps=True)
            self.assertEqual(pair["summary"]["failures"], 0)
            self.assertEqual(pair["fixture_optimizer_steps"], 0)
            # The runner fixtures really do step an optimiser, on synthetic arrays only.
            runner = pilot._check_fixture_pair(resolved, "runner_fixtures", "runner_provenance",
                                              pilot.RUNNER_CODE_FILES)
            self.assertEqual(runner["real_data_optimizer_steps"], 0)
            self.assertGreater(runner["fixture_optimizer_steps"], 0)

            cases = [
                ({"gate": "FAIL"}, "not a PASS"),
                ({"failures": 1}, "failures=1"),
                ({"errors": 2}, "errors=2"),
                ({"skipped": pilot.EXPECTED_MODEL_FIXTURE_TESTS}, "skipped="),
                ({"tests_run": 0}, "no tests"),
                ({"tests_run": 39}, "expected 40"),
                ({"real_data_optimizer_steps": 1}, "real-data optimizer steps"),
                ({"fixture_optimizer_steps": 3}, "no optimizer step at all"),
            ]
            for index, (change, pattern) in enumerate(cases):
                with self.subTest(change=change):
                    summary = {"gate": "PASS", "tests_run": pilot.EXPECTED_MODEL_FIXTURE_TESTS,
                               "failures": 0, "errors": 0, "skipped": 0,
                               "real_data_optimizer_steps": 0, "fixture_optimizer_steps": 0}
                    summary.update(change)
                    report, _ = self.report(Path(tmp) / f"case{index}",
                                            summaries={"neural_fixtures": summary})
                    with self.assertRaisesRegex(ValueError, pattern):
                        pilot._check_fixture_pair(pilot.resolve_evidence(report, None),
                                                  "neural_fixtures", "neural_provenance",
                                                  pilot.MODEL_CODE_FILES,
                                                  pilot.EXPECTED_MODEL_FIXTURE_TESTS,
                                                  require_zero_fixture_steps=True)

            for index, field in enumerate(("gate", "tests_run", "failures", "errors", "skipped")):
                with self.subTest(omitted=field):
                    summary = {"gate": "PASS", "tests_run": 40, "failures": 0, "errors": 0,
                               "skipped": 0, "real_data_optimizer_steps": 0}
                    summary.pop(field)
                    report, _ = self.report(Path(tmp) / f"omit{index}",
                                            summaries={"neural_fixtures": summary})
                    with self.assertRaisesRegex(ValueError, f"omits '{field}'"):
                        pilot._check_fixture_pair(pilot.resolve_evidence(report, None),
                                                  "neural_fixtures", "neural_provenance",
                                                  pilot.MODEL_CODE_FILES, 40)

    def test_code_changed_after_the_fixtures_ran_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale = dict(self.live_code(pilot.MODEL_CODE_FILES))
            stale["src/models/anomaly_detectors.py"] = "0" * 64
            report, _ = self.report(Path(tmp) / "stale",
                                    provenances={"neural_provenance": {"code_sha256": stale}})
            with self.assertRaisesRegex(ValueError, "re-run the fixtures"):
                pilot._check_fixture_pair(pilot.resolve_evidence(report, None),
                                          "neural_fixtures", "neural_provenance",
                                          pilot.MODEL_CODE_FILES,
                                          pilot.EXPECTED_MODEL_FIXTURE_TESTS)

    def test_runner_binding_covers_the_transitive_dependencies(self):
        for rel in ("src/fit_anomaly_controls.py", "src/data/synthetic_anomaly.py",
                    "src/models/anomaly_baselines.py"):
            self.assertIn(rel, pilot.RUNNER_CODE_FILES)

    def test_runner_evidence_is_required_not_just_a_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, files = self.report(Path(tmp) / "evidence")
            del report["evidence"]["runner_fixtures"]
            report["sha256"] = pilot.sa.digest_json(
                {k: v for k, v in report.items() if k != "sha256"})
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "runner_fixtures"):
                pilot.validate_inputs(Path(tmp), path, Path(tmp))

    def test_code_files_cover_the_runner_and_the_model(self):
        self.assertIn("src/train_anomaly_pilot.py", pilot.RUNNER_CODE_FILES)
        self.assertIn("tests/test_anomaly_pilot.py", pilot.RUNNER_CODE_FILES)
        for rel in pilot.MODEL_CODE_FILES + pilot.RUNNER_CODE_FILES:
            self.assertTrue((ROOT / rel).is_file(), rel)


class PolicyTests(unittest.TestCase):
    """Runs everywhere, including where torch is absent."""

    def source(self):
        return (ROOT / "src/train_anomaly_pilot.py").read_text()

    def test_declared_local_dependencies_all_exist(self):
        for rel in LOCAL_DEPENDENCIES:
            self.assertTrue((ROOT / rel).is_file(), f"server packet would be missing {rel}")

    def test_runner_requires_cuda_and_never_falls_back_to_cpu(self):
        text = self.source()
        self.assertIn("CUDA is required", text)
        self.assertIn("never falls back to CPU", text)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", text)
        self.assertIn("factory_training", text)

    def test_runner_installs_nothing_and_uses_no_mixed_precision(self):
        text = self.source()
        for forbidden in ("pip install", "subprocess", "empty_cache", "autocast",
                          "GradScaler", "ReduceLROnPlateau", "half()", "bfloat16"):
            self.assertNotIn(forbidden, text, forbidden)

    def test_failure_record_declares_that_nothing_was_retried(self):
        text = self.source()
        for declaration in ('"retried": False', '"learning_rate_adjusted": False',
                            '"out_of_memory_retry": False', '"run_abandoned": True'):
            self.assertIn(declaration, text, declaration)

    def test_determinism_is_strict_rather_than_warn_only(self):
        text = self.source()
        self.assertIn("torch.use_deterministic_algorithms(True)", text)
        self.assertNotIn("use_deterministic_algorithms(True, warn_only", text)

    def test_runner_marks_itself_an_uncommitted_diagnostic(self):
        text = self.source()
        self.assertIn("P1a_uncommitted_diagnostic", text)
        self.assertIn('"full_P0_gate": "NOT_EVALUATED"', text)


if __name__ == "__main__":
    unittest.main()
