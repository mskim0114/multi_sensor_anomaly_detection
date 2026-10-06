"""Stability gates, negative controls, epoch semantics and persistence on artificial data."""
import argparse
from collections import Counter
import contextlib
import copy
from dataclasses import replace
import hashlib
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
spec = importlib.util.spec_from_file_location("_test_bin_stability", ROOT / "src/diagnose_anomaly_bin_stability.py")
st = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = st
spec.loader.exec_module(st)
spec2 = importlib.util.spec_from_file_location("_stability_artificial_onset", ROOT / "tests/test_anomaly_bin_onset.py")
aux = importlib.util.module_from_spec(spec2)
sys.modules[spec2.name] = aux
spec2.loader.exec_module(aux)
try:
    import torch
except ImportError:
    torch = None


def artificial_inputs(count=2):
    inputs = aux.artificial_inputs(count)
    inputs["indexes"]["dev"] = replace(inputs["indexes"]["fit"], role="dev")
    inputs["windows"]["dev"] = [{**w, "role": "dev"} for w in inputs["windows"]["fit"]]
    return inputs


class ArtificialLoader(aux.ArtificialLoader):
    def load(self, role, ref):
        if role not in ("fit", "dev"):
            raise AssertionError("foreign fixture role")
        pair = super().load("fit", ref)
        self.counts[role + "_calls"] += 1
        return pair


def measurement(bce=0.69, gain=0.002, category="responsive_at_probe"):
    std, gap = (1e-8, 0) if category == "collapse_like" else (0.01, 0.02)
    return {"groups": {"thermal-only": {"thermal_z_mean_feature_std": std, "max_abs_logit_gap": gap,
            "balanced_probe_BCE": bce, "mean_abs_logit_gap": gap, "contrast_BCE_gain": gain}}}


def write_signed(path, value):
    value = copy.deepcopy(value)
    value.pop("sha256", None)
    value["sha256"] = st.sa.digest_json(value)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    return st.p1b.file_sha(path), value


class MetadataTests(unittest.TestCase):
    def make_gate(self, root):
        inputs = artificial_inputs()
        inputs.update(source_files=st.p1b.source_hashes(), evidence={"artificial": True},
                      candidate_source_files=st.onset.source_hashes(), probe_pairs=())
        original = {"sha256": "b"*64, "input_file_sha256": {"fixture": "c"*64},
                    "health_probes": [{"role": "fit"}], "runtime_reference": dict(aux.ARTIFICIAL_RUNTIME)}
        inputs["plan"] = original
        state_path = root/"final-state.pt"
        state_path.write_bytes(b"artificial metadata fixture, not a checkpoint")
        (root/"provenance.json").write_text('{"artificial":true}\n')
        replay = {"schema": st.onset.SCHEMA, "gate": "PASS", "scope": "P1c_fit_diagnostic", "initial_lr": 1e-4,
                  "arm": "T", "objective": "BIN", "real_data_optimizer_steps": 261,
                  "source_files": st.onset.source_hashes(), "input_evidence": inputs["evidence"],
                  "plan_payload_sha256": original["sha256"], "runtime_identity": original["runtime_reference"],
                  "initial_state_tensor_sha256": "1"*64, "final_state_tensor_sha256": "2"*64,
                  "train_loss": 0.69, "optimizer_steps_total": 261, "fit_order_manifest_sha256": "3"*64,
                  "fit_example_stream_sha256": "4"*64, "final_state_file_sha256": st.p1b.file_sha(state_path),
                  "provenance_file_sha256": st.p1b.file_sha(root/"provenance.json")}
        replay_sha, replay = write_signed(root/"replay.json", replay)
        expected = sum(isinstance(n, st.ast.FunctionDef) and n.name.startswith("test_")
                       for p in st.TEST_FILES for n in st.ast.walk(st.ast.parse((ROOT/p).read_text())))
        fixture = {"schema": st.FIXTURE_SCHEMA, "gate": "PASS", "source_files": st.source_hashes(),
                   "unit_tests": dict(tests_run=expected, failures=0, errors=0, skipped=0),
                   "test_files": list(st.TEST_FILES), "fixture_optimizer_steps": 1, "real_data_optimizer_steps": 0,
                   "runtime_identity": original["runtime_reference"], "git_commit_sha": "a"*40}
        fixture_sha, fixture = write_signed(root/"fixture.json", fixture)
        plan = {**copy.deepcopy(st.FROZEN_VALUES), "diagnostic_rules": copy.deepcopy(st.onset.RULES),
                "proposed_fit_signal_screen": dict(st.SCREEN), "candidate_source_files": st.source_hashes(),
                "baseline_source_files": inputs["source_files"], "input_file_sha256": original["input_file_sha256"],
                "health_probes": original["health_probes"], "runtime_reference": original["runtime_reference"],
                "onset_plan_file_sha256": "9"*64, "onset_plan_payload_sha256": original["sha256"],
                "first_epoch_replay_reference": st.replay_reference(replay, replay_sha),
                "candidate_fixture_evidence": dict(file_sha256=fixture_sha, payload_sha256=fixture["sha256"])}
        plan_sha, plan = write_signed(root/"plan.json", plan)
        args = argparse.Namespace(plan=root/"plan.json", plan_sha256=plan_sha,
                 onset_plan=root/"original-plan.json", onset_plan_sha256="9"*64,
                 replay_result=root/"replay.json", replay_result_sha256=replay_sha,
                 stability_fixtures=root/"fixture.json", stability_fixtures_sha256=fixture_sha)
        return inputs, args, plan, fixture, replay

    def validate(self, inputs, args):
        with mock.patch.object(st.onset, "validate_inputs", side_effect=lambda a: copy.deepcopy(inputs)):
            return st.validate_inputs(args)

    def test_metadata_gate_accepts_bound_fixture_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs,args,_,_,_ = self.make_gate(Path(tmp))
            result = self.validate(inputs,args)
            self.assertEqual(result["first_epoch_reference"]["optimizer_steps"],261)
            self.assertEqual(len(result["candidate_source_files"]),22)

    def test_recipe_budget_roles_resume_and_stop_policy_are_exact(self):
        plan = {**copy.deepcopy(st.FROZEN_VALUES), "diagnostic_rules": st.onset.RULES, "proposed_fit_signal_screen": st.SCREEN}
        changes = {"epochs": 1, "micro_batch": 1, "initial_lr": 1e-3, "arm": "F", "training_seed": 43,
                   "actual_roles_read": ["fit","dev","cal"], "resume": True, "retry": True,
                   "collapse_early_stop": True, "negative_control_required": False, "first_epoch_gate_before_dev": False,
                   "paper_result_eligible": True, "threshold_generation": True, "epochs_bool": True}
        for key,value in changes.items():
            actual = "epochs" if key == "epochs_bool" else key
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,"recipe differs"):
                st.check_recipe({**plan,actual:value})

    def test_probe_schedule_and_exploratory_screen_are_pinned(self):
        plan = {**copy.deepcopy(st.FROZEN_VALUES), "diagnostic_rules": st.onset.RULES, "proposed_fit_signal_screen": st.SCREEN}
        self.assertEqual(len(st.PROBE_STEPS),31)
        for epoch in (1,2,3,5,10,15,20,25,30):
            self.assertIn(261*epoch,st.PROBE_STEPS)
        for changed in ({**plan,"probe_steps": [0,261,7830]},
                        {**plan,"proposed_fit_signal_screen": {**st.SCREEN,"contrast_BCE_gain_min": 0.001}},
                        {**plan,"diagnostic_rules": {}}):
            with self.assertRaises(ValueError):
                st.check_recipe(changed)

    def test_changed_candidate_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs,args,plan,_,_ = self.make_gate(Path(tmp))
            plan["candidate_source_files"][st.NEW_FILES[0]] = "f"*64
            args.plan_sha256,_ = write_signed(args.plan,plan)
            with self.assertRaisesRegex(ValueError,"source/input/onset"):
                self.validate(inputs,args)

    def test_runtime_probe_or_input_pin_change_is_rejected(self):
        for key in ("runtime_reference","health_probes","input_file_sha256","onset_plan_payload_sha256"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                inputs,args,plan,_,_ = self.make_gate(Path(tmp))
                plan[key] = {} if isinstance(plan[key],dict) else []
                args.plan_sha256,_ = write_signed(args.plan,plan)
                with self.assertRaisesRegex(ValueError,"source/input/onset"):
                    self.validate(inputs,args)

    def test_skipped_failed_or_foreign_source_fixture_is_rejected(self):
        for change in ("skip","source","scope","test_files"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                inputs,args,plan,fixture,_ = self.make_gate(Path(tmp))
                if change == "skip": fixture["unit_tests"]["skipped"] = 1
                if change == "source": fixture["source_files"][st.NEW_FILES[0]] = "f"*64
                if change == "scope": fixture["real_data_optimizer_steps"] = 1
                if change == "test_files": fixture["test_files"] = []
                args.stability_fixtures_sha256,fixture = write_signed(args.stability_fixtures,fixture)
                plan["candidate_fixture_evidence"] = dict(file_sha256=args.stability_fixtures_sha256,payload_sha256=fixture["sha256"])
                args.plan_sha256,_ = write_signed(args.plan,plan)
                with self.assertRaisesRegex(ValueError,"fixture evidence"):
                    self.validate(inputs,args)

    def test_reference_file_corruption_is_detected_without_tensor_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs,args,_,_,_ = self.make_gate(Path(tmp))
            (Path(tmp)/"final-state.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"replay artifact changed"):
                self.validate(inputs,args)

    def test_wrong_lr_or_replay_source_is_rejected(self):
        for key in ("initial_lr","source_files"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                inputs,args,_,_,replay = self.make_gate(Path(tmp))
                replay[key] = 1e-3 if key == "initial_lr" else {}
                args.replay_result_sha256,_ = write_signed(args.replay_result,replay)
                with self.assertRaisesRegex(ValueError,"replay does not match"):
                    self.validate(inputs,args)

    def test_cli_has_no_budget_lr_resume_or_fixture_override(self):
        for flag in ("--epochs","--initial-lr","--resume","--fixture"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                st.main([flag,"1"])


class MeasurementTests(unittest.TestCase):
    def refs(self):
        return [SimpleNamespace(label=0,base_window_id="fixture",family=None,event_id=None),
                SimpleNamespace(label=1,base_window_id="fixture",family="sensor-only",event_id="event")]

    def test_equal_logits_have_exact_zero_contrast_even_at_extreme_bias(self):
        for n in ([-1000.,0.,1000.],[-0.,0.,1e-8],[0.01,0.3,-0.6]):
            metrics = st.contrast_metrics(n,n)
            self.assertEqual(metrics["contrast_BCE_gain"],0)
            self.assertTrue(math.isfinite(metrics["balanced_probe_BCE"]))

    def test_contrast_direction_and_midpoint_decomposition(self):
        value = st.contrast_metrics([-1.,-2.],[1.,2.])
        reversed_value = st.contrast_metrics([1.,2.],[-1.,-2.])
        self.assertGreater(value["contrast_BCE_gain"],0)
        self.assertLess(reversed_value["contrast_BCE_gain"],0)
        self.assertEqual(value["paired_midpoint_BCE"],math.log(2))
        self.assertAlmostEqual(value["balanced_probe_BCE"],np.mean([np.log1p(np.exp(-1)),np.log1p(np.exp(-2))]))

    def test_nonfinite_mismatched_or_empty_logits_are_rejected(self):
        for n,p in (([],[]),([1],[1,2]),([float("nan")],[0]),([[0]],[[0]])):
            with self.subTest(n=n), self.assertRaises(ValueError):
                st.contrast_metrics(n,p)

    def test_byte_control_reads_each_loaded_input_once(self):
        array = np.zeros((2,3),np.float32)
        loader = SimpleNamespace(load=mock.Mock(return_value=(None,array)),begin_phase=mock.Mock())
        checked = st.NegativeControlLoader(loader)
        checked.begin_phase()
        for ref in self.refs(): checked.load("fit",ref)
        self.assertEqual(loader.load.call_count,2)
        self.assertEqual(len(checked.rows),1)
        self.assertEqual(checked.rows[0]["normal_thermal_sha256"],checked.rows[0]["positive_thermal_sha256"])

    def test_numeric_equal_signed_zero_byte_difference_is_rejected(self):
        a,b = np.zeros((2,3),np.float32),np.zeros((2,3),np.float32)
        b[0,0] = -0.0
        self.assertTrue(np.array_equal(a,b))
        checked = st.NegativeControlLoader(SimpleNamespace(load=mock.Mock(side_effect=[(None,a),(None,b)])))
        normal,positive = self.refs()
        checked.load("fit",normal)
        with self.assertRaisesRegex(st.p1b.P1bFailure,"thermal bytes"):
            checked.load("fit",positive)

    def test_negative_control_rejects_wrong_dtype_shape_or_role(self):
        normal,positive = self.refs()
        for role,a,b in (("dev",np.zeros(2,np.float32),np.zeros(2,np.float32)),
                         ("fit",np.zeros(2,np.float64),np.zeros(2,np.float64)),
                         ("fit",np.zeros(2,np.float32),np.zeros((1,2),np.float32))):
            checked = st.NegativeControlLoader(SimpleNamespace(load=mock.Mock(side_effect=[(None,a),(None,b)])))
            with self.subTest(role=role,dtype=str(a.dtype)), self.assertRaises((ValueError,st.p1b.P1bFailure)):
                checked.load(role,normal)
                checked.load(role,positive)

    def test_observer_rejects_sensor_only_logit_leakage_after_equal_inputs(self):
        inputs = artificial_inputs()
        pairs = st.onset.resolve_probe_pairs(inputs,st.onset.selected_probes(inputs))
        def corrupted(model,loader,refs,**kwargs):
            for ref in st.p1b.paired_examples(refs): loader.load("fit",ref)
            values = [v for pair in refs for v in (0.0,0.1)]
            groups = {f:{"pair_count":2,"exact_equal_logit_pairs":0,"max_abs_logit_gap":0.1,
                         "balanced_probe_BCE":st.contrast_metrics([0,0],[0.1,0.1])["balanced_probe_BCE"]}
                      for f in ("sensor-only","thermal-only")}
            return {"logits_normal_then_positive":values,"groups":groups}
        with mock.patch.object(st.onset,"observe_fit",side_effect=corrupted), self.assertRaisesRegex(st.p1b.P1bFailure,"logits/contrast"):
            st.observe_fit(None,ArtificialLoader(),pairs,device=None)

    def test_observer_rejects_inconsistent_native_bce_before_screening(self):
        inputs = artificial_inputs()
        pairs = st.onset.resolve_probe_pairs(inputs,st.onset.selected_probes(inputs))
        bad = {"logits_normal_then_positive":[0.0]*8,
               "groups":{"sensor-only":{"balanced_probe_BCE":0.1}}}
        with mock.patch.object(st.onset,"observe_fit",return_value=bad), self.assertRaisesRegex(ValueError,"arithmetic differ"):
            st.observe_fit(None,ArtificialLoader(),pairs,device=None)

    def test_first_epoch_gate_checks_every_endpoint_and_stream_field(self):
        reference = {"initial_state_tensor_sha256":"a","final_state_tensor_sha256":"b","train_loss":0.693,
                     "optimizer_steps":261,"fit_order_manifest_sha256":"c","fit_example_stream_sha256":"d"}
        self.assertEqual(st.check_first_epoch(reference,reference)["gate"],"PASS_EXACT_FIRST_EPOCH")
        for key in reference:
            with self.subTest(key=key), self.assertRaises(st.p1b.P1bFailure):
                st.check_first_epoch({**reference,key:None},reference)

    def test_gpu_inventory_identifies_own_and_foreign_processes(self):
        value = st.parse_gpu_activity("GPU-a, 25, 100, 24000\nGPU-b, 0, 1, 24000",
                                     "GPU-a, 123, python, 50\nGPU-a, 124, training, 40\nGPU-b, 125, other, 80", "GPU-a",123)
        self.assertEqual(value["other_compute_process_count"],1)
        self.assertTrue(value["own_process_present"])
        self.assertEqual(len(value["processes"]),2)
        self.assertFalse(value["exclusive_access_enforced"])

    def test_gpu_inventory_accepts_empty_processes_but_rejects_ambiguous_device(self):
        value = st.parse_gpu_activity("GPU-a, N/A, 0, 24000","","GPU-a",123)
        self.assertIsNone(value["utilization_percent"])
        self.assertFalse(value["own_process_present"])
        for text in ("GPU-b,0,0,1","GPU-a,0,0,1\nGPU-a,0,0,1"):
            with self.assertRaises(ValueError): st.parse_gpu_activity(text,"","GPU-a",123)

    def test_trajectory_reports_recovery_without_early_stop(self):
        rows = [{"step":i,"classification":c} for i,c in enumerate(
                ("responsive_at_probe","collapse_like","intermediate","responsive_at_probe"))]
        result = st.trajectory_summary(rows,screen=False)
        self.assertTrue(result["responsive_recovery_sampled"])
        self.assertFalse(result["collapse_persistent_since_first_sample"])
        self.assertEqual(result["onset"]["upper_inclusive"],1)

    def test_screen_requires_both_late_epochs_and_does_not_mean_generalization(self):
        rows = [{"step":261*e,"classification":"responsive_at_probe","measurement":measurement(0.65,0.03)} for e in (25,30)]
        self.assertEqual(st.trajectory_summary(rows)["fit_signal_screen"]["gate"],"PASS")
        rows[1]["measurement"]["groups"]["thermal-only"]["contrast_BCE_gain"] = 0.01
        value = st.trajectory_summary(rows)["fit_signal_screen"]
        self.assertEqual(value["gate"],"UNMET")
        self.assertIn("memorization",value["scope"])


class FakeDiagnostics:
    """Stands in for the cached diagnostics module; records the tolerance it receives."""

    def __init__(self, passed=True):
        self.calls, self.passed = [], passed

    def thermal_lstm_diagnostics(self, model, thermal, *, rtol=1e-5, atol=1e-6):
        self.calls.append(dict(rtol=rtol, atol=atol))
        record = {"max_abs_error": 2e-6, "max_rel_error": 3e-3, "rtol": rtol, "atol": atol,
                  "count": 10, "violations": 0 if self.passed else 1, "passed": self.passed}
        return {"parity_sequence": dict(record), "parity_encoder_output": dict(record)}


@contextlib.contextmanager
def fake_diagnostics_module(fake):
    name = st.DIAGNOSTICS_MODULE[0]
    previous = sys.modules.get(name)
    sys.modules[name] = fake
    try:
        yield fake
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class ProbeParityToleranceTests(unittest.TestCase):
    """r3: the probe's gate recomputation uses the frozen PROBE_PARITY, nothing else changes."""

    def test_tolerance_is_frozen_in_the_recipe_and_checked(self):
        self.assertEqual(st.PROBE_PARITY, {"rtol": 1e-4, "atol": 1e-4})
        self.assertEqual(st.FROZEN_VALUES["probe_parity_tolerance"], st.PROBE_PARITY)
        plan = copy.deepcopy(st.FROZEN_VALUES)
        plan["diagnostic_rules"] = dict(st.onset.RULES)
        plan["proposed_fit_signal_screen"] = dict(st.SCREEN)
        st.check_recipe(plan)
        for bad in (None, {"rtol": 1e-5, "atol": 1e-6}, {"rtol": 1e-4}, {"rtol": 1e-4, "atol": 1e-3}):
            with self.subTest(tolerance=bad):
                changed = copy.deepcopy(plan)
                if bad is None:
                    changed.pop("probe_parity_tolerance")
                else:
                    changed["probe_parity_tolerance"] = bad
                with self.assertRaisesRegex(ValueError, "probe_parity_tolerance"):
                    st.check_recipe(changed)

    def test_swap_passes_the_frozen_tolerance_and_always_restores(self):
        fake = FakeDiagnostics()
        original = fake.thermal_lstm_diagnostics
        with fake_diagnostics_module(fake):
            with st._ProbeParityTolerance() as parity:
                fake.thermal_lstm_diagnostics("model", "thermal")
            self.assertEqual(fake.calls, [st.PROBE_PARITY])
            self.assertEqual(parity.last["parity_sequence"]["atol"], 1e-4)
            self.assertEqual(fake.thermal_lstm_diagnostics, original)
            with self.assertRaises(KeyboardInterrupt):
                with st._ProbeParityTolerance():
                    raise KeyboardInterrupt
            self.assertEqual(fake.thermal_lstm_diagnostics, original)

    def test_parity_failure_stops_with_the_measured_numbers(self):
        fake = FakeDiagnostics(passed=False)
        original = fake.thermal_lstm_diagnostics

        def failing_observer(model, loader, pairs, **kwargs):
            measured = sys.modules[st.DIAGNOSTICS_MODULE[0]].thermal_lstm_diagnostics(model, "thermal")
            assert measured["parity_sequence"]["passed"] is False
            raise ValueError("manual LSTM diagnostic failed native elementwise parity")
        with fake_diagnostics_module(fake), mock.patch.object(st.onset, "observe_fit", failing_observer):
            with self.assertRaises(st.p1b.P1bFailure) as caught:
                st.observe_fit("model", ArtificialLoader(), [], device="cpu")
            self.assertEqual(fake.thermal_lstm_diagnostics, original)
        context = caught.exception.context
        self.assertEqual(context["probe_parity_tolerance"], st.PROBE_PARITY)
        last = context["last_microbatch_parity"]["parity_sequence"]
        self.assertEqual((last["max_abs_error"], last["violations"], last["passed"]), (2e-6, 1, False))
        json.dumps(context, allow_nan=False)

    def test_other_observer_errors_pass_through_unchanged(self):
        fake = FakeDiagnostics()
        with fake_diagnostics_module(fake), mock.patch.object(
                st.onset, "observe_fit", side_effect=ValueError("probe logit shape/nonfinite mismatch")):
            with self.assertRaisesRegex(ValueError, "probe logit shape"):
                st.observe_fit("model", ArtificialLoader(), [], device="cpu")

    @unittest.skipIf(torch is None, "torch unavailable (expected on Jetson)")
    def test_a_failed_probe_preserves_the_exact_model_state_and_the_original_error(self):
        model = torch.nn.Linear(3, 2)
        failure = st.p1b.P1bFailure("probe manual-LSTM parity exceeded the frozen tolerance", {"x": 1})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(st, "observe_fit", side_effect=failure):
            with self.assertRaises(st.p1b.P1bFailure) as caught:
                st.observe_and_preserve(model, 2610, Path(tmp), None, [], device="cpu")
            self.assertIs(caught.exception, failure)
            saved_name = caught.exception.context["probe_failure_state"]
            self.assertEqual(saved_name, "failure-probe-state-step-2610.pt")
            saved = torch.load(Path(tmp) / saved_name, map_location="cpu", weights_only=True)
            self.assertEqual(saved["selection"], "PROBE_FAILURE_STATE_NOT_DEV_SELECTED")
            self.assertEqual(saved["optimizer_step"], 2610)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(saved["model_state_dict"][k], v))

    @unittest.skipIf(torch is None, "torch unavailable (expected on Jetson)")
    def test_a_state_save_failure_never_replaces_the_probe_error(self):
        model = torch.nn.Linear(3, 2)
        failure = st.p1b.P1bFailure("probe failed", {})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(st, "observe_fit", side_effect=failure), \
                mock.patch.object(st, "save_state", side_effect=OSError("disk full")):
            with self.assertRaises(st.p1b.P1bFailure) as caught:
                st.observe_and_preserve(model, 7, Path(tmp), None, [], device="cpu")
            self.assertIs(caught.exception, failure)
            self.assertIn("state save failed: OSError", caught.exception.context["probe_failure_state"])

    def test_a_successful_probe_writes_no_failure_state(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(st, "observe_fit", return_value={"ok": 1}):
            self.assertEqual(st.observe_and_preserve(object(), 5, Path(tmp), None, [], device="cpu"), {"ok": 1})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    @unittest.skipIf(torch is None, "torch unavailable (expected on Jetson)")
    def test_the_looser_tolerance_still_rejects_a_real_mismatch(self):
        """A 1e-3 output shift is far below a real defect and still fails PROBE_PARITY."""
        diag = st.p1b._leaf(*st.DIAGNOSTICS_MODULE)
        det = st.p1b._leaf("_stability_parity_detectors", "src/models/anomaly_detectors.py")
        torch.manual_seed(42)
        model = det.AnomalyDetector("T", "BIN")
        lstm = model.thermal_encoder.lstm

        class Shifted(torch.nn.LSTM):
            def forward(self, x, hx=None):
                out, state = super().forward(x, hx)
                return out + 1e-3, state
        shifted = Shifted(lstm.input_size, lstm.hidden_size, lstm.num_layers, batch_first=True, dropout=lstm.dropout)
        shifted.load_state_dict(lstm.state_dict())
        thermal = torch.randn(2, 30, 120, 160, generator=torch.Generator().manual_seed(1))
        healthy = diag.thermal_lstm_diagnostics(model, thermal, **st.PROBE_PARITY)
        self.assertTrue(healthy["parity_sequence"]["passed"] and healthy["parity_encoder_output"]["passed"])
        model.thermal_encoder.lstm = shifted
        broken = diag.thermal_lstm_diagnostics(model, thermal, **st.PROBE_PARITY)
        self.assertFalse(broken["parity_sequence"]["passed"])
        self.assertFalse(broken["parity_encoder_output"]["passed"])
        # Realistic re-implementation defects still fail PROBE_PARITY by a wide margin.
        # (A single weight nudged by 0.05 on the untrained model gives ~5e-6 and is no
        # longer detected at 1e-4; review (54) records that trade-off.)
        model.thermal_encoder.lstm = lstm.eval()
        x = 0.8 * torch.randn(4, 30, lstm.input_size, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            reference, _ = lstm(x)
            no_bias_hh, swapped = copy.deepcopy(lstm), copy.deepcopy(lstm)
            for layer in range(lstm.num_layers):
                getattr(no_bias_hh, f"bias_hh_l{layer}").zero_()
                for name in (f"weight_ih_l{layer}", f"weight_hh_l{layer}", f"bias_ih_l{layer}", f"bias_hh_l{layer}"):
                    p = getattr(swapped, name)
                    i, f, g, o = p.data.chunk(4, 0)
                    p.data.copy_(torch.cat([f, i, g, o], 0))
            for label, wrong in (("bias_hh dropped", no_bias_hh), ("input/forget swapped", swapped)):
                with self.subTest(defect=label):
                    manual, _ = diag.lstm_recurrence(wrong, x)
                    record = diag._parity(manual, reference, **st.PROBE_PARITY)
                    self.assertFalse(record["passed"])
                    self.assertGreater(record["max_abs_error"], 10 * st.PROBE_PARITY["atol"])


@unittest.skipUnless(torch is not None,"SERVER-TRAINING torch required")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        st.p1b._training()
        self.inputs = artificial_inputs(2)

    def factory(self,seed):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc_thermal = torch.nn.Linear(1,4)
                self.dropout = torch.nn.Dropout(0.2)
            def forward(self,x):
                return self.dropout(self.fc_thermal(x[:,0,0,0].unsqueeze(1)).tanh())
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.thermal_encoder,self.head = Encoder(),torch.nn.Linear(4,1)
            def forward(self,thermal):
                return {"logit":self.head(self.thermal_encoder(thermal)).squeeze(-1)}
        torch.manual_seed(seed)
        self.model = Tiny()
        return self.model,{"artificial":"tiny BIN"}

    def run_fixture(self,**options):
        kwargs = dict(device=torch.device("cpu"),scope="synthetic_fixture",epochs=3,nominal_batch=2,
                      micro_batch=2,probe_steps=(),model_factory=self.factory)
        kwargs.update(options)
        return st.train_loop(self.inputs,ArtificialLoader(),**kwargs)

    def test_multiple_epochs_keep_cosine30_sampler30_and_distinct_terminal_states(self):
        record,_ = self.run_fixture()
        self.assertEqual(record["optimizer_steps_total"],6)
        self.assertEqual(record["real_data_optimizer_steps"],0)
        self.assertEqual(record["cosine_T_max"],30)
        self.assertEqual(len({e["terminal_state_tensor_sha256"] for e in record["epochs"]}),3)
        for epoch,entry in enumerate(record["epochs"],1):
            self.assertAlmostEqual(entry["next_epoch_lr"],1e-4*(1+math.cos(math.pi*epoch/30))/2)
        self.assertNotEqual(record["epochs"][0]["cumulative_fit_example_stream_sha256"],record["epochs"][1]["cumulative_fit_example_stream_sha256"])

    def test_first_epoch_matches_preserved_onset_core_exactly(self):
        original,state = st.onset.train_first_epoch(self.inputs,ArtificialLoader(),initial_lr=1e-4,
            device=torch.device("cpu"),scope="synthetic_fixture",nominal_batch=2,micro_batch=2,
            probe_steps=(),model_factory=self.factory)
        record,_ = self.run_fixture(epochs=1)
        reference = st.replay_reference({**original,"sha256":"fixture"},"fixture")
        self.assertEqual(record["epochs"][0]["terminal_state_tensor_sha256"],original["final_state_tensor_sha256"])
        self.assertEqual(record["epochs"][0]["train_loss"],original["train_loss"])
        self.assertEqual(record["actual_fit_event_stream_sha256"],original["fit_example_stream_sha256"])
        checked,_ = self.run_fixture(epochs=1,replay=reference)
        self.assertEqual(checked["first_epoch_replay"]["gate"],"PASS_EXACT_FIRST_EPOCH")

    def test_failed_first_epoch_gate_stops_before_any_dev_or_epoch2(self):
        original,_ = self.run_fixture(epochs=1)
        first = original["epochs"][0]
        reference = dict(initial_state_tensor_sha256=original["initial_state_tensor_sha256"],
            final_state_tensor_sha256="changed",train_loss=first["train_loss"],optimizer_steps=2,
            fit_order_manifest_sha256=first["fit_order_manifest_sha256"],fit_example_stream_sha256=original["actual_fit_event_stream_sha256"])
        progress,terminals = {},[]
        with mock.patch.object(st,"evaluate_dev",side_effect=AssertionError("dev reached")), self.assertRaisesRegex(st.p1b.P1bFailure,"before dev"):
            self.run_fixture(replay=reference,progress=progress,on_terminal=lambda s,e:terminals.append(e))
        self.assertEqual(progress["optimizer_steps_completed"],2)
        self.assertEqual(len(terminals),1)

    def test_exact_dev_ties_keep_earliest_checkpoint(self):
        calls = []
        with mock.patch.object(st,"evaluate_dev",return_value=0.5):
            record,_ = self.run_fixture(on_improvement=lambda s,e,l:calls.append(e))
        self.assertEqual(calls,[1])
        self.assertEqual(record["dev_checkpoint_epoch"],1)
        self.assertEqual(record["dev_checkpoint_ties"],[2,3])

    def test_dev_weighted_bce_matches_independent_numpy_sum(self):
        model,_ = self.factory(42)
        refs = st.bs.dev_examples(self.inputs["indexes"]["dev"])
        loader = ArtificialLoader()
        _,helpers,_ = st.p1b._training()
        # Disable dropout for both the independent computation and production dev.
        model.eval()
        with torch.no_grad():
            logits = [float(model(thermal=torch.from_numpy(loader.load("dev",r)[1].copy()).unsqueeze(0))["logit"]) for r in refs]
        expected = math.fsum(float(np.logaddexp(0,l)-r.label*l)*r.weight for l,r in zip(logits,refs))
        actual = st.evaluate_dev(model,ArtificialLoader(),refs,torch.device("cpu"),torch,helpers,Counter(),Counter(),{},2)
        self.assertAlmostEqual(actual,expected,places=7)

    def test_nonfinite_dev_loss_is_rejected_before_checkpoint_selection(self):
        model,_ = self.factory(42)
        model.forward = lambda thermal:{"logit":torch.full((len(thermal),),float("nan"))}
        refs = st.bs.dev_examples(self.inputs["indexes"]["dev"])
        _,helpers,_ = st.p1b._training()
        with self.assertRaisesRegex(st.p1b.P1bFailure,"nonfinite dev BCE"):
            st.evaluate_dev(model,ArtificialLoader(),refs,torch.device("cpu"),torch,helpers,Counter(),Counter(),{},2)

    def test_collapsed_probe_never_reduces_the_epoch_budget(self):
        rows = []
        record,_ = self.run_fixture(probe_steps=(0,2,4,6),observer=lambda m,s:measurement(category="collapse_like"),on_probe=rows.append)
        self.assertEqual(record["optimizer_steps_total"],6)
        self.assertEqual(len(rows),4)
        self.assertTrue(record["trajectory"]["collapse_persistent_since_first_sample"])

    def test_probe_rng_consumption_fails_before_first_update(self):
        progress = {}
        def observer(m,s):
            torch.rand(())
            return measurement()
        with self.assertRaisesRegex(ValueError,"consumed model RNG"):
            self.run_fixture(probe_steps=(0,),observer=observer,progress=progress)
        self.assertEqual(progress["optimizer_steps_completed"],0)

    def test_hook_removed_when_forward_raises_and_no_update_occurs(self):
        progress = {}
        def factory(seed):
            model,manifest = self.factory(seed)
            def broken(**kwargs): raise RuntimeError("fixture forward failure")
            model.forward = broken
            return model,manifest
        with self.assertRaisesRegex(RuntimeError,"fixture forward failure"):
            self.run_fixture(model_factory=factory,progress=progress)
        self.assertEqual(progress["optimizer_steps_completed"],0)
        self.assertFalse(self.model.thermal_encoder.fc_thermal._forward_hooks)

    def test_serialization_distinguishes_selected_and_terminal_states(self):
        record,state = self.run_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"terminal.pt"
            st.save_state(path,state,record["dev_checkpoint_epoch"],selection="ARTIFICIAL_SELECTED")
            saved = torch.load(path,weights_only=True)
            self.assertEqual(saved["selection"],"ARTIFICIAL_SELECTED")
            self.assertEqual(st.p1b.detectors.state_tensor_sha256(saved["model_state_dict"]),record["selected_state_tensor_sha256"])

    def test_write_failure_preserves_original_exception_and_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/"steps.jsonl").write_text('{"step":1}\n')
            st.record_failure(out,RuntimeError("first technical failure"),dict(optimizer_steps_completed=1))
            failed = json.loads((out/"failure.json").read_text())
            self.assertFalse(failed["retried"])
            self.assertEqual(failed["exception"],"RuntimeError")
            self.assertIn("steps.jsonl",failed["partial_files_preserved"])

    def main_fixture(self,root,*,runtime_mismatch=False,final_binding_change=False,interrupt=False,corrupt_checkpoint=False,
                     probe_failure=False):
        original,_ = st.onset.train_first_epoch(self.inputs,ArtificialLoader(),initial_lr=1e-4,
            device=torch.device("cpu"),scope="synthetic_fixture",nominal_batch=2,micro_batch=2,
            probe_steps=(),model_factory=self.factory)
        reference = st.replay_reference({**original,"sha256":"fixture"},"fixture")
        inputs = {**self.inputs,"candidate_source_files":st.source_hashes(),"evidence":{"artificial":True},
                  "candidate_git_commit_sha":"a"*40,"stability_fixture_payload_sha256":"b"*64,
                  "plan":{"sha256":"c"*64,"runtime_reference":dict(aux.ARTIFICIAL_RUNTIME)},
                  "first_epoch_reference":reference,"probe_pairs":(),"cache_path":root/"artificial-cache",
                  "cache_manifest":{"sha256":"d"*64}}
        out = root/"runs"/"case"
        argv = []
        for flag in ("prepared","bank-result","cache-run","sampler-fixtures","runner-fixtures","diagnostic-fixtures",
                     "onset-plan","baseline-result","replay-result","stability-fixtures","plan","out"):
            argv.extend(["--"+flag,str(out if flag == "out" else root/"inputs"/flag)])
        for flag in ("cache-result","sampler-fixtures","runner-fixtures","diagnostic-fixtures","onset-plan",
                     "baseline-result","replay-result","stability-fixtures","plan"):
            argv.extend(["--"+flag+"-sha256","e"*64])
        class Cache:
            manifest = {"sha256":"d"*64}
            _data = True
            def open(self): return self
            def reader(self,**kwargs): return None
        cache = Cache()
        core,append = st.train_loop,st.onset._append_json
        def run(*args,**kwargs):
            kwargs.update(scope="synthetic_fixture",nominal_batch=2,micro_batch=2,model_factory=self.factory)
            record,state = core(*args,**kwargs)
            if corrupt_checkpoint:
                p = out/"checkpoint.pt"
                saved = torch.load(p,weights_only=True)
                saved["epoch"] = 999
                torch.save(saved,p)
            return record,state
        def writes(path,value):
            append(path,value)
            if interrupt and path.name == "steps.jsonl" and value["step"] == 2:
                raise KeyboardInterrupt("artificial interrupt after committed second update")
        identity = dict(aux.ARTIFICIAL_RUNTIME)
        if runtime_mismatch: identity["driver_version"] = "changed"
        final = {**inputs,"candidate_source_files":{}} if final_binding_change else inputs
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(st.p1b,"OUTPUT_PARENT",root/"runs"))
            stack.enter_context(mock.patch.object(st,"EPOCHS",3))
            stack.enter_context(mock.patch.object(st,"REAL_STEPS",6))
            stack.enter_context(mock.patch.object(st,"PROBE_STEPS",tuple(range(7))))
            stack.enter_context(mock.patch.object(st,"validate_inputs",side_effect=[inputs,final]))
            stack.enter_context(mock.patch.object(st.p1b,"runtime_environment",return_value=(torch.device("cpu"),{},"artificial")))
            stack.enter_context(mock.patch.object(st.onset,"runtime_identity",return_value=identity))
            stack.enter_context(mock.patch.object(st,"gpu_activity",return_value={"gpu_uuid":identity["gpu_uuid"],"other_compute_process_count":0,"own_process_present":True}))
            attached = stack.enter_context(mock.patch.object(st.p1b.rc.RawNormalCache,"attach",return_value=cache,
                       side_effect=AssertionError("runtime failure did not precede cache") if runtime_mismatch else None))
            stack.enter_context(mock.patch.object(st.p1b,"PairLoader",side_effect=lambda *a:ArtificialLoader()))
            calls = Counter()
            def observed(*a,**k):
                calls["probe"] += 1
                if probe_failure and calls["probe"] == 4:   # the fourth probe is step 3
                    raise st.p1b.P1bFailure("probe manual-LSTM parity exceeded the frozen tolerance",
                                            {"last_microbatch_parity": {"artificial": True}})
                return measurement()
            stack.enter_context(mock.patch.object(st,"observe_fit",side_effect=observed))
            stack.enter_context(mock.patch.object(st,"train_loop",side_effect=run))
            stack.enter_context(mock.patch.object(st.onset,"_append_json",side_effect=writes))
            with contextlib.redirect_stdout(io.StringIO()):
                st.main(argv)
        self.assertIsNone(cache._data)
        return out

    def test_main_persists_history_selected_checkpoint_and_all_terminal_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.main_fixture(Path(tmp))
            result = json.loads((out/"result.json").read_text())
            self.assertEqual(result["gate"],"PASS")
            self.assertEqual(result["real_data_optimizer_steps"],0)
            self.assertEqual(len(result["terminal_states"]),3)
            self.assertEqual(result["first_epoch_replay"]["gate"],"PASS_EXACT_FIRST_EPOCH")
            self.assertEqual(len((out/"gpu-activity.jsonl").read_text().splitlines()),4)
            for name,digest in result["artifact_file_sha256"].items():
                self.assertEqual(st.p1b.file_sha(out/name),digest)

    def test_main_probe_failure_preserves_that_step_state_without_retry(self):
        """r3: the real main path wires observe_and_preserve, so a failed probe keeps its state."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(st.p1b.P1bFailure):
                self.main_fixture(Path(tmp),probe_failure=True)
            out = Path(tmp)/"runs"/"case"
            failure = json.loads((out/"failure.json").read_text())
            self.assertEqual(failure["context"]["probe_failure_state"],"failure-probe-state-step-0003.pt")
            self.assertEqual(failure["context"]["optimizer_steps_completed"],3)
            saved = torch.load(out/"failure-probe-state-step-0003.pt",map_location="cpu",weights_only=True)
            self.assertEqual(saved["optimizer_step"],3)
            self.assertFalse((out/"result.json").exists())

    def test_main_runtime_mismatch_stops_before_cache_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(st.p1b.P1bFailure,"runtime identity differs"):
                self.main_fixture(root,runtime_mismatch=True)
            failed = json.loads((root/"runs/case/failure.json").read_text())
            self.assertEqual(failed["context"]["optimizer_steps_completed"],0)
            self.assertFalse((root/"runs/case/result.json").exists())

    def test_main_rejects_changed_final_source_binding_after_preserving_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError,"evidence changed"):
                self.main_fixture(root,final_binding_change=True)
            failed = json.loads((root/"runs/case/failure.json").read_text())
            self.assertEqual(failed["schema"],st.SCHEMA)
            self.assertEqual(failed["context"]["optimizer_steps_completed"],6)
            self.assertIn("terminal-states/epoch-03.pt",failed["partial_files_preserved"])
            self.assertFalse((root/"runs/case/result.json").exists())

    def test_main_checkpoint_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError,"persisted checkpoint/history"):
                self.main_fixture(root,corrupt_checkpoint=True)
            failed = json.loads((root/"runs/case/failure.json").read_text())
            self.assertEqual(failed["stage"],"artifact_verification")
            self.assertFalse(failed["retried"])

    def test_main_keyboard_interrupt_preserves_committed_updates_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(KeyboardInterrupt):
                self.main_fixture(root,interrupt=True)
            failed = json.loads((root/"runs/case/failure.json").read_text())
            self.assertEqual(failed["exception"],"KeyboardInterrupt")
            self.assertEqual(failed["context"]["optimizer_steps_completed"],2)
            self.assertEqual(len((root/"runs/case/steps.jsonl").read_text().splitlines()),2)
            self.assertFalse(failed["retried"])


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),"SERVER CUDA required")
class GpuStabilityTests(unittest.TestCase):
    def test_native_multiepoch_observation_control_rng_and_first_epoch_oracle(self):
        torch.set_num_threads(1)
        st.p1b._training()
        inputs,device = artificial_inputs(),torch.device("cuda:0")
        pairs = st.onset.resolve_probe_pairs(inputs,st.onset.selected_probes(inputs))
        def execute(observed):
            holder,states,probes = {},[],[]
            def factory(seed):
                models,manifest = st.p1b.detectors.build_arm_set("BIN",seed=seed)
                holder["model"] = models["T"]
                return models["T"],manifest
            kwargs = dict(device=device,scope="synthetic_fixture",epochs=3,nominal_batch=2,micro_batch=2,
                          model_factory=factory,probe_steps=tuple(range(7)) if observed else (),
                          on_step=lambda e:states.append(st.p1b.detectors.state_tensor_sha256(holder["model"])),on_probe=probes.append)
            if observed:
                kwargs["observer"] = lambda m,s:st.observe_fit(m,ArtificialLoader(),pairs,device=device)
            record,_ = st.train_loop(inputs,ArtificialLoader(),**kwargs)
            return record,states,probes,torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone()
        control,states,_,cpu,gpu = execute(False)
        repeated,states2,_,cpu2,gpu2 = execute(False)
        observed,states3,probes,cpu3,gpu3 = execute(True)
        self.assertEqual(states,states2)
        self.assertEqual(states,states3)
        self.assertTrue(torch.equal(cpu,cpu2) and torch.equal(cpu,cpu3))
        self.assertTrue(torch.equal(gpu,gpu2) and torch.equal(gpu,gpu3))
        for p in probes:
            self.assertEqual(p["measurement"]["negative_control"]["gate"],"PASS_EXACT_INPUT_LOGIT_CONTRAST")
            self.assertEqual(p["measurement"]["groups"]["sensor-only"]["contrast_BCE_gain"],0)
        original,_ = st.onset.train_first_epoch(inputs,ArtificialLoader(),initial_lr=1e-4,device=device,
                        scope="synthetic_fixture",nominal_batch=2,micro_batch=2,probe_steps=())
        first = observed["epochs"][0]
        self.assertEqual(first["terminal_state_tensor_sha256"],original["final_state_tensor_sha256"])
        self.assertEqual(first["train_loss"],original["train_loss"])
        self.assertEqual(first["cumulative_fit_example_stream_sha256"],original["fit_example_stream_sha256"])


if __name__ == "__main__":
    unittest.main()
