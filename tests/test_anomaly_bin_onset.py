"""Onset design gates and update/observer oracles on explicitly artificial data.

No project raw windows, dev forward, GPU or real-data optimizer step is needed
by these CPU fixtures. Real CLI never exposes tiny-budget/CPU fixture options.
"""
import argparse
from collections import Counter
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_test_bin_onset", ROOT / "src/diagnose_anomaly_bin_onset.py")
onset = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = onset
spec.loader.exec_module(onset)
try:
    import torch
except ImportError:
    torch = None

FIXTURE_OPTIMIZER_STEPS = 0
ARTIFICIAL_RUNTIME = dict(hostname="fixture-host", python_version="3.12", numpy_version="2",
                          torch_version="2.6.0+cu124", cuda_runtime_version="12.4", device="cuda:0",
                          cuda_device_index=0, gpu_name="fixture-gpu", gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
                          nvidia_smi_index=0, cuda_visible_devices=None, cublas_workspace_config=":4096:8",
                          cudnn_version=90100, driver_version="fixture-driver")


def artificial_inputs(count=20):
    """Direct sampler construction is confined to this named artificial fixture."""
    windows, bases = [], []
    for i in range(count):
        device, base_id = f"fixture-device{i % 2}", f"artificial-window{i:03}"
        windows.append(dict(base_window_id=base_id, device=device, role="fit"))
        events = tuple(onset.bs.EventRef(onset.sa.digest_json([i, f]), onset.sa.digest_json([i, f, "CT1"]),
                                        f, 4, f != "thermal-only", f != "sensor-only")
                       for f in ("sensor-only", "thermal-only", "coupled"))
        bases.append(onset.bs.BaseEvents(base_id, device, i, events))
    index = onset.bs.EventIndex("fit", 0, 42, "CT1", "a" * 64, "b" * 64,
                                tuple(bases), tuple(w["base_window_id"] for w in windows), "c" * 64)
    return {"indexes": {"fit": index}, "windows": {"fit": windows},
            "cache_manifest": {"windows": windows}}


def measurement(fractions=(0.0, 0.0, 0.0, 0.0), std=0.001, gap=0.001):
    return {"layers": {"0": {g: {"abs_gt6_fraction": v} for g, v in zip(("i", "f", "g", "o"), fractions)}},
            "groups": {"thermal-only": {"thermal_z_mean_feature_std": std, "max_abs_logit_gap": gap}}}


class DesignTests(unittest.TestCase):
    def test_probe_selection_is_fit_only_and_reproducible(self):
        inputs = artificial_inputs()
        rows = onset.selected_probes(inputs)
        self.assertEqual(rows, onset.selected_probes(inputs))
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["role"] for r in rows}, {"fit"})
        pairs = onset.resolve_probe_pairs(inputs, rows)
        self.assertEqual({p.family for p in pairs}, {"sensor-only", "thermal-only"})

    def test_foreign_role_probe_is_rejected(self):
        inputs = artificial_inputs()
        rows = onset.selected_probes(inputs)
        rows[0]["role"] = "dev"
        with self.assertRaisesRegex(ValueError, "fit-only"):
            onset.resolve_probe_pairs(inputs, rows)

    def test_probe_event_instance_change_is_rejected(self):
        inputs = artificial_inputs()
        rows = onset.selected_probes(inputs)
        rows[0]["event_instance_id"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "fit-only"):
            onset.resolve_probe_pairs(inputs, rows)

    def test_probe_window_digest_change_is_rejected(self):
        inputs = artificial_inputs()
        rows = onset.selected_probes(inputs)
        rows[0]["window_record_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "fit-only"):
            onset.resolve_probe_pairs(inputs, rows)

    def test_missing_accepted_probe_cell_is_rejected(self):
        inputs = artificial_inputs(1)
        base = inputs["indexes"]["fit"].bases[0]
        from dataclasses import replace
        inputs["indexes"]["fit"] = replace(inputs["indexes"]["fit"], bases=(replace(base, events=(base.events[0],)),))
        with self.assertRaisesRegex(ValueError, "probe cell"):
            onset.selected_probes(inputs)

    def test_healthy_collapse_and_intermediate_are_distinct(self):
        self.assertEqual(onset.classify_snapshot(measurement()), "responsive_at_probe")
        self.assertEqual(onset.classify_snapshot(measurement((0.95,) * 4, 1e-9, 0)), "collapse_like")
        self.assertEqual(onset.classify_snapshot(measurement((0.95,) * 4, 0.001, 0)), "intermediate")
        self.assertEqual(onset.classify_snapshot(measurement((0.1,) * 4, 1e-9, 0)), "collapse_like")

    def test_rule_boundaries_do_not_hide_intermediate_cases(self):
        self.assertEqual(onset.classify_snapshot(measurement((0.9,) * 4, 1e-6, 1e-6)), "collapse_like")
        self.assertEqual(onset.classify_snapshot(measurement((0.99,) * 4)), "responsive_at_probe")
        self.assertEqual(onset.classify_snapshot(measurement(std=1e-4)), "intermediate")

    def test_nonfinite_or_invalid_criterion_fails(self):
        for value in (float("nan"), float("inf"), -0.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                onset.classify_snapshot(measurement(std=value))

    def test_onset_is_interval_not_exact_step(self):
        rows = [{"step": 0, "classification": "responsive_at_probe"},
                {"step": 10, "classification": "intermediate"},
                {"step": 20, "classification": "collapse_like"}]
        self.assertEqual(onset.onset_interval(rows), dict(lower_exclusive=10, upper_inclusive=20,
                                                         baseline_already_collapse_like=False))

    def test_censoring_does_not_assert_prevention(self):
        value = onset.onset_interval([{"step": 261, "classification": "responsive_at_probe"}])
        self.assertEqual(value, dict(not_observed_through_step=261, prevention_established=False))

    def test_baseline_collapse_is_not_assigned_to_step1(self):
        value = onset.onset_interval([{"step": 0, "classification": "collapse_like"}])
        self.assertIsNone(value["lower_exclusive"])
        self.assertTrue(value["baseline_already_collapse_like"])

    def test_draft_fails_before_torch_or_cache_attachment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"schema": onset.PLAN_SCHEMA, "recipe_id": onset.RECIPE, "status": "DRAFT"}
            plan["sha256"] = onset.sa.digest_json(plan)
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            args = SimpleNamespace(plan=path, plan_sha256=onset.p1b.file_sha(path))
            with mock.patch.object(onset.p1b, "validate_inputs", return_value={}), \
                    mock.patch.object(onset.p1b, "_training") as training, \
                    mock.patch.object(onset.p1b.rc.RawNormalCache, "attach") as attach:
                with self.assertRaisesRegex(ValueError, "separately frozen"):
                    onset.validate_inputs(args)
                training.assert_not_called()
                attach.assert_not_called()

    def test_pinned_plan_payload_tamper_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(dict(status="FROZEN_DIAGNOSTIC", sha256="a" * 64)))
            args = SimpleNamespace(plan=path, plan_sha256=onset.p1b.file_sha(path))
            with mock.patch.object(onset.p1b, "validate_inputs", return_value={}):
                with self.assertRaisesRegex(ValueError, "payload digest"):
                    onset.validate_inputs(args)

    def test_failure_record_preserves_first_exception_and_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "steps.jsonl").write_text('{"step":1}\n')
            onset._record_failure(out, KeyboardInterrupt("fixture interrupted"),
                                   dict(optimizer_steps_completed=1, stage="first_epoch"))
            result = json.loads((out / "failure.json").read_text())
            self.assertEqual(result["schema"], onset.SCHEMA)
            self.assertEqual(result["exception"], "KeyboardInterrupt")
            self.assertEqual(result["partial_files_preserved"], ["steps.jsonl"])
            self.assertFalse(result["retried"])

    def test_failure_record_write_error_does_not_replace_original(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(onset.p1b, "_write_json", side_effect=OSError("disk")), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            onset._record_failure(Path(directory), KeyboardInterrupt("original"), {})
            self.assertIn("original", stderr.getvalue())

    def test_original_replay_uses_exact_unrounded_aggregates(self):
        record = dict(initial_lr=1e-3, initial_state_tensor_sha256="a" * 64, train_loss=0.6936151724437187,
                      optimizer_steps_total=261, fit_order_manifest_sha256="b" * 64)
        ref = dict(initial_state_sha256="a" * 64, train_loss=record["train_loss"],
                   optimizer_steps=261, fit_order_manifest_sha256="b" * 64)
        self.assertEqual(onset.check_baseline_replay(record, ref)["gate"], "PASS_EXACT_AGGREGATES")
        for key, value in (("train_loss", 0.6936152), ("optimizer_steps", 260),
                           ("fit_order_manifest_sha256", "f" * 64), ("initial_state_sha256", "f" * 64)):
            with self.subTest(key=key), self.assertRaisesRegex(onset.p1b.P1bFailure, "aggregate consistency"):
                onset.check_baseline_replay(record, {**ref, key: value})

    def test_changed_lr_is_not_required_to_reproduce_original_loss(self):
        self.assertEqual(onset.check_baseline_replay({"initial_lr": 1e-4}, {})["gate"], "NOT_APPLICABLE_DIFFERENT_LR")

    def test_runtime_gate_checks_device_software_driver_and_identity(self):
        expected = dict(ARTIFICIAL_RUNTIME)
        self.assertEqual(onset.check_runtime_identity(expected, expected)["gate"], "PASS_EXACT_RUNTIME_IDENTITY")
        for key, value in (("device", "cuda:1"), ("cuda_device_index", 1), ("torch_version", "2.5.0"),
                           ("cudnn_version", 90101), ("driver_version", "changed"), ("gpu_uuid", "different-GPU"),
                           ("numpy_version", "1"), ("cuda_visible_devices", "1"), ("cublas_workspace_config", ":16:8")):
            with self.subTest(key=key), self.assertRaisesRegex(onset.p1b.P1bFailure, "runtime identity differs"):
                onset.check_runtime_identity({**expected, key: value}, expected)
        with self.assertRaisesRegex(ValueError, "fields differ"):
            onset.check_runtime_identity({"device": "cuda:0"}, expected)

    def test_driver_lookup_uses_physical_uuid_not_csv_row_order(self):
        rows = "1, GPU-second, Same GPU, 580.1\n0, GPU-first, Same GPU, 580.2\n"
        self.assertEqual(onset._driver_identity(rows, "GPU-first"),
                         dict(nvidia_smi_index=0, gpu_name="Same GPU", driver_version="580.2"))
        for output, gpu in ((rows, "GPU-absent"), (rows + rows, "GPU-first"), ("0, GPU-first, Same GPU, \n", "GPU-first")):
            with self.subTest(output=output, gpu=gpu), self.assertRaises(ValueError):
                onset._driver_identity(output, gpu)


class ArtificialLoader:
    def __init__(self):
        self.counts, self.seconds = Counter(), Counter()
        self.context = {}

    def begin_phase(self):
        pass

    def load(self, role, ref):
        if role != "fit":
            raise AssertionError("diagnostic fixture attempted a non-fit window")
        base = (ref.window_index % 7 - 3) / 7
        sensor = np.full((30, 5), base + 0.4 * ref.label * ref.sensor_changed, np.float32)
        thermal = np.broadcast_to(np.float32(base + 0.4 * ref.label * ref.thermal_changed), (30, 120, 160))
        self.counts["fit_examples"] += 1
        return sensor, thermal


@unittest.skipUnless(torch is not None, "SERVER-TRAINING torch required")
class UpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        onset.p1b._training()
        self.inputs = artificial_inputs()

    def factory(self, seed, dropout=0.2):
        class TinyBin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.thermal_encoder = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Tanh(),
                                                           torch.nn.Dropout(dropout))
                self.head = torch.nn.Linear(4, 1)

            def forward(self, thermal):
                z = self.thermal_encoder(thermal.mean((1, 2, 3)).unsqueeze(1))
                return {"logit": self.head(z).squeeze(-1)}
        torch.manual_seed(seed)
        model = TinyBin()
        self.last_model = model
        return model, {"fixture": "tiny artificial BIN; not a project model"}

    def run_fixture(self, **options):
        global FIXTURE_OPTIMIZER_STEPS
        args = dict(initial_lr=1e-3, device=torch.device("cpu"), scope="synthetic_fixture",
                    nominal_batch=2, micro_batch=1, probe_steps=(), model_factory=self.factory)
        args.update(options)
        record, state = onset.train_first_epoch(self.inputs, ArtificialLoader(), **args)
        FIXTURE_OPTIMIZER_STEPS += record["fixture_optimizer_steps"]
        return record, state

    def test_probe_enabled_disabled_produce_identical_updates_and_rng(self):
        states_a, states_b = [], []
        record_a, state_a = self.run_fixture(on_step=lambda e: states_a.append(onset.p1b.detectors.state_tensor_sha256(self.last_model)))
        rng_a = torch.get_rng_state().clone()
        def observer(model, step):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            return {"step": step, "value": model(thermal=torch.ones(2, 30, 120, 160))["logit"].tolist()}
        record_b, state_b = self.run_fixture(probe_steps=(0, 1, 2, 5, 10, 20), observer=observer,
                                             on_step=lambda e: states_b.append(onset.p1b.detectors.state_tensor_sha256(self.last_model)))
        self.assertEqual(states_a, states_b)
        self.assertEqual(record_a["final_state_tensor_sha256"], record_b["final_state_tensor_sha256"])
        self.assertTrue(torch.equal(rng_a, torch.get_rng_state()))
        self.assertEqual(record_b["probe_steps_completed"], [0, 1, 2, 5, 10, 20])
        self.assertEqual(record_b["real_data_optimizer_steps"], 0)

    def test_observer_restores_mixed_modes_and_gradients_on_exception(self):
        model, _ = self.factory(42)
        model.train()
        model.thermal_encoder[1].eval()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        modes = [m.training for m in model.modules()]
        grads = [p.grad.clone() for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "original"), onset.observation_mode(model, torch):
            model(thermal=torch.ones(1, 30, 120, 160))
            raise RuntimeError("original")
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a, p.grad) for a, p in zip(grads, model.parameters())))

    def test_no_rng_setter_is_called_for_unchanged_noop_observation(self):
        model, _ = self.factory(42)
        with mock.patch.object(torch, "set_rng_state", wraps=torch.set_rng_state) as cpu_setter, \
                mock.patch.object(torch.cuda, "set_rng_state", wraps=torch.cuda.set_rng_state) as cuda_setter, \
                mock.patch.object(torch.cuda, "set_rng_state_all", wraps=torch.cuda.set_rng_state_all) as all_setter:
            with onset.observation_mode(model, torch):
                pass
            cpu_setter.assert_not_called()
            cuda_setter.assert_not_called()
            all_setter.assert_not_called()

    def test_rng_consuming_observer_is_rejected_before_first_update(self):
        progress = {}
        with self.assertRaisesRegex(ValueError, "consumed model RNG"):
            self.run_fixture(probe_steps=(0,), progress=progress,
                             observer=lambda model, step: {"draw": float(torch.rand(()))})
        self.assertEqual(progress["optimizer_steps_completed"], 0)

    def test_invalid_duplicate_or_outside_probe_steps_fail(self):
        for steps in ((1, 1), (2, 1), (-1,), (21,)):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "probe step"):
                self.run_fixture(probe_steps=steps)

    def test_actual_batch_denominator_and_cosine30_are_preserved(self):
        record, _ = self.run_fixture(nominal_batch=6, micro_batch=2)
        self.assertEqual(record["actual_batch_sizes"], {"6": 6, "4": 1})
        self.assertEqual(record["exposures"], {"fit_Normal": 20, "fit_positive": 20})
        self.assertEqual(record["cosine_T_max"], 30)
        self.assertAlmostEqual(record["next_epoch_lr"], 1e-3 * (1 + math.cos(math.pi / 30)) / 2)
        self.assertFalse(record["dev_window_read"])

    def test_same_sampler_and_initial_state_across_lrs(self):
        a, _ = self.run_fixture(initial_lr=1e-3)
        b, _ = self.run_fixture(initial_lr=1e-4)
        for key in ("initial_state_tensor_sha256", "fit_order_manifest_sha256", "fit_example_stream_sha256", "exposures"):
            self.assertEqual(a[key], b[key])
        self.assertNotEqual(a["final_state_tensor_sha256"], b["final_state_tensor_sha256"])

    def test_gradient_norm_reads_actual_backward_without_mutation(self):
        model, _ = self.factory(42, dropout=0)
        x = torch.linspace(-1, 1, 4).reshape(4, 1, 1, 1).expand(4, 30, 120, 160)
        logits = model(thermal=x)["logit"]
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0., 1.])).backward()
        expected = {k: math.sqrt(sum(float(p.grad.double().square().sum()) for n, p in model.named_parameters()
                                     if n.startswith(k + "."))) for k in ("thermal_encoder", "head")}
        before = [p.grad.clone() for p in model.parameters()]
        result = onset.gradient_norms(model, torch)
        for key in expected:
            self.assertEqual(result[key]["l2"], expected[key])
        self.assertTrue(all(torch.equal(g, p.grad) for g, p in zip(before, model.parameters())))

    def test_partial_batch_update_matches_independent_logistic_gradient(self):
        self.inputs = artificial_inputs(2)
        class LinearBin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(1, 1)
                with torch.no_grad():
                    self.head.weight.fill_(0.7)
                    self.head.bias.fill_(0.2)
            def forward(self, thermal):
                return {"logit": self.head(thermal.mean((1, 2, 3)).unsqueeze(1)).squeeze(-1)}
        captured = []
        def factory(seed):
            model = LinearBin()
            self.last_model = model
            return model, {"fixture": "analytic logistic-gradient oracle"}
        batches = onset.bs.BasePairedSampler(self.inputs["indexes"]["fit"], 42, 8, 30).epoch_batches(1)
        refs = onset.p1b.paired_examples(batches[0])
        loader = ArtificialLoader()
        x = torch.from_numpy(np.stack([loader.load("fit", r)[1] for r in refs])).mean((1, 2, 3))
        target = torch.tensor([r.label for r in refs], dtype=torch.float32)
        residual = torch.sigmoid(0.7 * x + 0.2) - target
        expected_w, expected_b = (residual * x).mean(), residual.mean()
        expected = LinearBin()
        expected.head.weight.grad = expected_w.reshape(1, 1).clone()
        expected.head.bias.grad = expected_b.reshape(1).clone()
        reference_optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-3, weight_decay=0.01)
        reference_optimizer.step()
        record, state = self.run_fixture(nominal_batch=16, micro_batch=2,
                                         model_factory=factory, on_step=captured.append)
        self.assertEqual(record["actual_batch_sizes"], {"4": 1})
        self.assertAlmostEqual(captured[0]["gradient_norms"]["head"]["l2"],
                               math.hypot(float(expected_w), float(expected_b)), places=6)
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.allclose(state[name], value, rtol=1e-6, atol=5e-8), name)

    def test_missing_and_nonfinite_gradients_are_rejected(self):
        model, _ = self.factory(42)
        with self.assertRaisesRegex(ValueError, "missing/nonfinite"):
            onset.gradient_norms(model, torch)
        for p in model.parameters():
            p.grad = torch.full_like(p, float("nan"))
        with self.assertRaisesRegex(ValueError, "missing/nonfinite"):
            onset.gradient_norms(model, torch)

    def test_callback_failure_preserves_completed_optimizer_count(self):
        progress = {}
        def interrupt(entry):
            raise KeyboardInterrupt("fixture after first update")
        with self.assertRaisesRegex(KeyboardInterrupt, "after first update"):
            self.run_fixture(on_step=interrupt, progress=progress)
        self.assertEqual(progress["optimizer_steps_completed"], 1)
        self.assertEqual(progress["phase"], "step_record")

    def test_projection_hook_is_removed_when_actual_forward_fails(self):
        class MalformedBin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.thermal_encoder = torch.nn.Module()
                self.thermal_encoder.fc_thermal = torch.nn.Linear(1, 4)
                self.head = torch.nn.Linear(4, 2)
            def forward(self, thermal):
                z = self.thermal_encoder.fc_thermal(thermal.mean((1, 2, 3)).unsqueeze(1))
                return {"logit": self.head(z)}
        def factory(seed):
            model = MalformedBin()
            self.last_model = model
            return model, {"fixture": "malformed forward to check hook cleanup"}
        progress = {}
        with self.assertRaisesRegex(ValueError, "logit shape"):
            self.run_fixture(model_factory=factory, progress=progress)
        self.assertEqual(progress["optimizer_steps_completed"], 0)
        self.assertFalse(self.last_model.thermal_encoder.fc_thermal._forward_hooks)

    def main_fixture(self, root, *, incomplete=False, change_final_binding=False, interrupt_step=False, runtime_mismatch=False):
        out = root / "output"
        pairs = onset.resolve_probe_pairs(self.inputs, onset.selected_probes(self.inputs))
        inputs = {**self.inputs, "cache_path": root / "artificial-cache", "candidate_source_files": {"fixture": "fabricated main input"},
                  "candidate_git_commit_sha": "a" * 40, "diagnostic_fixture_payload_sha256": "b" * 64,
                  "evidence": {"purpose": "fabricated main gate fixture, not research evidence"},
                  "plan": {"sha256": "c" * 64, "runtime_reference": dict(ARTIFICIAL_RUNTIME)},
                  "probe_pairs": pairs, "baseline_reference": {}, "baseline_provenance_file_sha256": "f" * 64,
                  "cache_manifest": {**self.inputs["cache_manifest"], "sha256": "d" * 64}}
        final = copy.deepcopy(inputs)
        if change_final_binding:
            final["candidate_source_files"]["fixture"] = "changed during fixture"
        class FakeCache:
            manifest = inputs["cache_manifest"]
            _data = None
            def open(self):
                return self
            def reader(self, **kwargs):
                return lambda window: (_ for _ in ()).throw(AssertionError("no real cache reads"))
        argv = []
        for flag in ("prepared", "bank-result", "cache-run", "sampler-fixtures", "runner-fixtures",
                     "diagnostic-fixtures", "plan", "baseline-result", "out"):
            argv.extend(("--" + flag, str(out if flag == "out" else root / flag)))
        for flag in ("cache-result-sha256", "sampler-fixtures-sha256", "runner-fixtures-sha256",
                     "diagnostic-fixtures-sha256", "plan-sha256", "baseline-result-sha256"):
            argv.extend(("--" + flag, "e" * 64))
        argv.extend(("--initial-lr", "0.001"))
        original_train = onset.train_first_epoch
        def train(*args, **kwargs):
            kwargs.update(scope="synthetic_fixture", model_factory=self.factory, probe_steps=(0, 1, 2, 3))
            if interrupt_step:
                persist = kwargs["on_step"]
                def stop(entry):
                    persist(entry)
                    raise KeyboardInterrupt("main fixture after actual step")
                kwargs["on_step"] = stop
            return original_train(*args, **kwargs)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(onset.p1b, "_check_output", return_value=out))
            stack.enter_context(mock.patch.object(onset, "validate_inputs", side_effect=[inputs, final]))
            stack.enter_context(mock.patch.object(onset.p1b, "runtime_environment",
                                                   return_value=(torch.device("cpu"), {"environment_profile": "SYNTHETIC_FIXTURE"}, "synthetic_fixture")))
            identity = {**ARTIFICIAL_RUNTIME, "driver_version": "changed"} if runtime_mismatch else dict(ARTIFICIAL_RUNTIME)
            stack.enter_context(mock.patch.object(onset, "runtime_identity", return_value=identity))
            stack.enter_context(mock.patch.object(onset.p1b.rc.RawNormalCache, "attach", return_value=FakeCache(),
                                                   side_effect=AssertionError("runtime gate did not stop before cache") if runtime_mismatch else None))
            stack.enter_context(mock.patch.object(onset.p1b, "PairLoader", side_effect=lambda *a: ArtificialLoader()))
            stack.enter_context(mock.patch.object(onset, "train_first_epoch", side_effect=train))
            stack.enter_context(mock.patch.object(onset, "observe_fit", side_effect=lambda model, *a, **k: {"fixture_observation": True}))
            stack.enter_context(mock.patch.object(onset, "check_baseline_replay", return_value={"gate": "ARTIFICIAL_MAIN_FIXTURE_ONLY"}))
            stack.enter_context(mock.patch.object(onset, "PROBE_STEPS", (0, 1, 2, 3)))
            if not incomplete:
                stack.enter_context(mock.patch.object(onset, "REAL_STEPS", 3))
            with contextlib.redirect_stdout(io.StringIO()):
                return onset.main(argv)

    def test_main_persists_reloadable_final_state_and_binds_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.main_fixture(root), 0)
            out = root / "output"
            result = json.loads((out / "result.json").read_text())
            onset.p1b._signed(result, "fixture result")
            self.assertEqual(result["fixture_optimizer_steps"], 3)
            self.assertEqual(result["real_data_optimizer_steps"], 0)
            self.assertEqual(result["checkpoint_selection"], "NONE_FINAL_STATE_ONLY")
            self.assertEqual(result["runtime_identity"], ARTIFICIAL_RUNTIME)
            self.assertEqual(result["runtime_gate"]["gate"], "PASS_EXACT_RUNTIME_IDENTITY")
            for name, key in (("final-state.pt", "final_state_file_sha256"), ("steps.jsonl", "steps_file_sha256"),
                              ("probes.jsonl", "probes_file_sha256"), ("provenance.json", "provenance_file_sha256")):
                self.assertEqual(onset.p1b.file_sha(out / name), result[key])
            self.assertFalse((out / "failure.json").exists())

    def test_main_runtime_mismatch_stops_before_cache_or_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(onset.p1b.P1bFailure, "runtime identity differs"):
                self.main_fixture(root, runtime_mismatch=True)
            failed = json.loads((root / "output/failure.json").read_text())
            self.assertEqual(failed["context"]["stage"], "environment")
            self.assertEqual(failed["context"]["optimizer_steps_completed"], 0)
            self.assertIn("driver_version", failed["context"]["runtime_mismatches"])
            self.assertEqual(failed["partial_files_preserved"], [])

    def test_main_incomplete_budget_is_preserved_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "incomplete diagnostic budget"):
                self.main_fixture(root, incomplete=True)
            record = json.loads((root / "output/failure.json").read_text())
            self.assertEqual(record["context"]["optimizer_steps_completed"], 3)
            self.assertFalse((root / "output/result.json").exists())

    def test_main_revalidates_source_binding_after_final_state_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "sources, plan or evidence changed"):
                self.main_fixture(root, change_final_binding=True)
            record = json.loads((root / "output/failure.json").read_text())
            self.assertEqual(record["context"]["stage"], "final_inputs")
            self.assertTrue((root / "output/final-state.pt").exists())
            self.assertFalse((root / "output/result.json").exists())

    def test_main_interrupt_preserves_actual_completed_step_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(KeyboardInterrupt, "actual step"):
                self.main_fixture(root, interrupt_step=True)
            record = json.loads((root / "output/failure.json").read_text())
            self.assertEqual(record["context"]["optimizer_steps_completed"], 1)
            self.assertEqual(len((root / "output/steps.jsonl").read_text().splitlines()), 1)
            self.assertFalse((root / "output/result.json").exists())


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "audited SERVER CUDA required")
class GpuObserverTests(unittest.TestCase):
    def test_runtime_identity_records_actual_cuda_uuid_driver_and_cudnn(self):
        identity = onset.runtime_identity(torch.device("cuda:0"))
        self.assertEqual(set(identity), set(onset.RUNTIME_FIELDS))
        properties = torch.cuda.get_device_properties(0)
        self.assertEqual(identity["gpu_uuid"].removeprefix("GPU-"), str(properties.uuid).removeprefix("GPU-"))
        self.assertEqual(identity["gpu_name"], properties.name)
        self.assertEqual(identity["cudnn_version"], torch.backends.cudnn.version())
        self.assertEqual(identity["cuda_device_index"], 0)
        self.assertTrue(identity["driver_version"])

    def test_native_thermal_lstm_probe_preserves_real_model_gpu_updates(self):
        global FIXTURE_OPTIMIZER_STEPS
        torch.set_num_threads(1)
        onset.p1b._training()
        inputs = artificial_inputs(2)
        pairs = onset.resolve_probe_pairs(inputs, onset.selected_probes(inputs))
        device = torch.device("cuda:0")
        def execute(observed):
            probes, states = [], []
            holder = {}
            def factory(seed):
                models, manifest = onset.p1b.detectors.build_arm_set("BIN", seed=seed)
                holder["model"] = models["T"]
                return models["T"], manifest
            def step_record(entry):
                self.assertEqual(entry["train_fc_thermal"]["count"], 2 * 30 * 128)
                self.assertGreaterEqual(entry["train_fc_thermal"]["abs_max"], entry["train_fc_thermal"]["abs_mean"])
                states.append(onset.p1b.detectors.state_tensor_sha256(holder["model"]))
            kwargs = dict(initial_lr=1e-3, device=device, scope="synthetic_fixture", nominal_batch=2,
                          micro_batch=2, probe_steps=(0, 1, 2) if observed else (), model_factory=factory,
                          on_step=step_record)
            if observed:
                probe_loader = ArtificialLoader()
                kwargs.update(observer=lambda model, step: onset.observe_fit(model, probe_loader, pairs,
                                                                              arm="T", device=device),
                              on_probe=probes.append)
            record, state = onset.train_first_epoch(inputs, ArtificialLoader(), **kwargs)
            return record, states, probes, torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
        control, control_states, _, cpu_rng, gpu_rng = execute(False)
        observed, observed_states, probes, cpu_rng2, gpu_rng2 = execute(True)
        FIXTURE_OPTIMIZER_STEPS += 4
        self.assertEqual(control_states, observed_states)
        self.assertEqual(control["final_state_tensor_sha256"], observed["final_state_tensor_sha256"])
        self.assertTrue(torch.equal(cpu_rng, cpu_rng2))
        self.assertTrue(torch.equal(gpu_rng, gpu_rng2))
        self.assertEqual([p["step"] for p in probes], [0, 1, 2])
        for p in probes:
            for value in p["measurement"]["parity"].values():
                self.assertTrue(value["passed"])
            self.assertEqual(p["measurement"]["groups"]["sensor-only"]["exact_equal_logit_pairs"], 2)
            values = p["measurement"]["logits_normal_then_positive"]
            indices = [i for i, pair in enumerate(pairs) if pair.family == "thermal-only"]
            expected_bce = np.mean([0.5 * (np.logaddexp(0, values[2 * i]) + np.logaddexp(0, -values[2 * i + 1])) for i in indices])
            self.assertEqual(p["measurement"]["groups"]["thermal-only"]["balanced_probe_BCE"], expected_bce)
            self.assertIn("BCE_change_from_step0", p["measurement"]["groups"]["thermal-only"])
        # Independent implementation reference: preserved P1b's first epoch on
        # artificial fit/dev arrays. Its post-epoch dev evaluation selects the
        # only state; neither its scheduler nor dev forward affects epoch1 updates.
        from dataclasses import replace
        original_inputs = {**inputs, "windows": {**inputs["windows"], "dev": inputs["windows"]["fit"]},
                           "indexes": {**inputs["indexes"], "dev": replace(inputs["indexes"]["fit"], role="dev")}}
        class OriginalReferenceLoader(ArtificialLoader):
            def load(self, role, ref):
                if role not in ("fit", "dev"):
                    raise AssertionError("foreign reference fixture role")
                return super().load("fit", ref)
        original, original_state = onset.p1b.train_loop(
            original_inputs, OriginalReferenceLoader(), arm="T", objective="BIN", seed=42,
            device=device, scope="synthetic_fixture", epochs=1, nominal_batch=2, micro_batch=2)
        FIXTURE_OPTIMIZER_STEPS += original["fixture_optimizer_steps"]
        self.assertEqual(onset.p1b.detectors.state_tensor_sha256(original_state), observed["final_state_tensor_sha256"])
        self.assertEqual(original["actual_fit_event_stream_sha256"], observed["fit_example_stream_sha256"])
        self.assertEqual(original["epochs"][0]["train_loss"], observed["train_loss"])


if __name__ == "__main__":
    unittest.main()
