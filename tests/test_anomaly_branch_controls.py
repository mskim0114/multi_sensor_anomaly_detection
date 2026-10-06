"""Branch controls: exact head replay, symmetric S/T/F controls, artificial F cases, invariance."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    TORCH_ERROR = None
except Exception as exc:  # pragma: no cover - exercised on Jetson
    torch = None
    TORCH_ERROR = repr(exc)
requires_torch = unittest.skipIf(torch is None, f"torch unavailable (expected on Jetson): {TORCH_ERROR}")
requires_cuda = unittest.skipIf(torch is None or not torch.cuda.is_available(), "needs a CUDA device")


def leaf(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if torch is not None:
    det = leaf("_branch_controls_detectors", "src/models/anomaly_detectors.py")
    bc = leaf("_branch_controls_under_test", "src/models/anomaly_branch_controls.py")


def bin_model(arm, seed=42):
    torch.manual_seed(seed)
    return det.AnomalyDetector(arm, "BIN")


def inputs(n, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 30, 5, generator=g).to(device),
            torch.randn(n, 30, 120, 160, generator=g).to(device))


def native_forward(model, sensor, thermal, micro_batch=2):
    """What the observer does: one eval forward per micro-batch, branch z captured by hooks."""
    captured, hooks = {"sensor": [], "thermal": []}, []
    for branch in ("sensor", "thermal"):
        encoder = getattr(model, branch + "_encoder", None)
        if encoder is not None:
            hooks.append(encoder.register_forward_hook(
                lambda m, i, o, b=branch: captured[b].append(o.detach().clone())))
    model.eval()
    logits = []
    try:
        with torch.no_grad():
            for start in range(0, sensor.shape[0], micro_batch):
                kwargs = {}
                if "sensor" in model.modalities:
                    kwargs["sensor"] = sensor[start:start + micro_batch]
                if "thermal" in model.modalities:
                    kwargs["thermal"] = thermal[start:start + micro_batch]
                logits.append(model(**kwargs)["logit"])
    finally:
        for hook in hooks:
            hook.remove()
    z = {f"{k}_z": torch.cat(v) for k, v in captured.items() if v}
    return torch.cat(logits), z


def state_hash(model):
    digest = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@requires_torch
class ReplayTests(unittest.TestCase):
    def test_head_replay_is_exact_on_cpu_for_every_arm(self):
        sensor, thermal = inputs(4)
        for arm in ("S", "T", "F"):
            with self.subTest(arm=arm):
                model = bin_model(arm)
                logits, z = native_forward(model, sensor, thermal)
                record = bc.analyze_branch_controls(model, logits, **z)
                self.assertTrue(record["head_replay"]["passed"])
                self.assertTrue(record["head_replay"]["exact_zero"])
                self.assertEqual(record["head_replay"]["max_abs"], 0.0)

    @requires_cuda
    def test_head_replay_is_exact_on_cuda_for_every_arm(self):
        sensor, thermal = inputs(4, device="cuda")
        for arm in ("S", "T", "F"):
            with self.subTest(arm=arm):
                model = bin_model(arm).cuda()
                logits, z = native_forward(model, sensor, thermal)
                record = bc.analyze_branch_controls(model, logits, **z)
                self.assertTrue(record["head_replay"]["exact_zero"])

    def test_a_replay_mismatch_fails_instead_of_measuring(self):
        sensor, thermal = inputs(4)
        model = bin_model("F")
        logits, z = native_forward(model, sensor, thermal)
        bad = logits.clone()
        bad[1] = torch.nextafter(bad[1], torch.tensor(float("inf")))
        with self.assertRaisesRegex(ValueError, "head replay differs"):
            bc.analyze_branch_controls(model, bad, **z)
        # an explicit tolerance is honoured, and recorded
        record = bc.analyze_branch_controls(model, bad, replay_atol=1e-5, **z)
        self.assertEqual(record["head_replay"]["atol"], 1e-5)
        self.assertFalse(record["head_replay"]["exact_zero"])

    def test_single_branch_arms_mark_the_missing_branch_not_applicable(self):
        sensor, thermal = inputs(4)
        for arm, missing in (("S", "thermal"), ("T", "sensor")):
            with self.subTest(arm=arm):
                model = bin_model(arm)
                logits, z = native_forward(model, sensor, thermal)
                record = bc.analyze_branch_controls(model, logits, **z)
                self.assertEqual(record["branches"][missing], {"status": "not_applicable"})
                present = "sensor" if missing == "thermal" else "thermal"
                self.assertEqual(record["branches"][present]["status"], "measured")


@requires_torch
class ArtificialFusionTests(unittest.TestCase):
    """F cases where the right answer is known in advance."""

    def test_identical_inputs_give_zero_spread_and_zero_permutation_effect(self):
        sensor, thermal = inputs(1)
        sensor, thermal = sensor.repeat(4, 1, 1), thermal.repeat(4, 1, 1, 1)
        model = bin_model("F")
        logits, z = native_forward(model, sensor, thermal)
        record = bc.analyze_branch_controls(model, logits, **z)
        for branch in ("sensor", "thermal"):
            b = record["branches"][branch]
            self.assertEqual(b["feature_population_std_mean"], 0.0)
            self.assertTrue(b["permutation_logit_delta"]["exact_zero"])
            self.assertEqual(b["pairs_exactly_equal"], 2)
            self.assertTrue(b["categories"]["collapse_like_representation"])
            self.assertTrue(b["categories"]["negligible_head_contribution"])

    def test_an_ignored_branch_has_exactly_no_permutation_effect(self):
        sensor, thermal = inputs(4)
        for ignored, columns in (("thermal", slice(128, 256)), ("sensor", slice(0, 128))):
            with self.subTest(ignored=ignored):
                model = bin_model("F")
                with torch.no_grad():
                    model.fusion.weight[:, columns] = 0.0
                logits, z = native_forward(model, sensor, thermal)
                record = bc.analyze_branch_controls(model, logits, **z)
                kept = "sensor" if ignored == "thermal" else "thermal"
                self.assertTrue(record["branches"][ignored]["permutation_logit_delta"]["exact_zero"])
                self.assertTrue(record["branches"][ignored]["zero_logit_delta"]["exact_zero"])
                self.assertTrue(record["branches"][ignored]["categories"]["negligible_head_contribution"])
                # the ignored branch still varies across inputs: not a collapsed representation
                self.assertFalse(record["branches"][ignored]["categories"]["collapse_like_representation"])
                self.assertGreater(record["branches"][kept]["permutation_logit_delta"]["max_abs"], 0.0)

    def test_a_constant_nonzero_branch_moves_the_logit_only_through_zeroing(self):
        """The bias trap: zeroing a constant branch changes the logit without any input signal."""
        sensor, thermal = inputs(4)
        model = bin_model("F")
        constant = torch.full((1, 128), 0.37)
        original = model.thermal_encoder.forward
        model.thermal_encoder.forward = lambda x: constant.expand(x.shape[0], 128).clone()
        try:
            logits, z = native_forward(model, sensor, thermal)
            record = bc.analyze_branch_controls(model, logits, **z)
        finally:
            model.thermal_encoder.forward = original
        thermal_record = record["branches"]["thermal"]
        self.assertEqual(thermal_record["feature_population_std_mean"], 0.0)
        self.assertTrue(thermal_record["permutation_logit_delta"]["exact_zero"])
        self.assertGreater(thermal_record["zero_logit_delta"]["max_abs"], 0.0)
        self.assertTrue(thermal_record["categories"]["collapse_like_representation"])
        self.assertTrue(thermal_record["categories"]["negligible_head_contribution"])
        self.assertIn("not evidence", thermal_record["zero_control_note"])


@requires_torch
class ContractTests(unittest.TestCase):
    def setUp(self):
        self.sensor, self.thermal = inputs(4)

    def test_wrong_branch_inputs_and_shapes_are_refused(self):
        model_s, model_f = bin_model("S"), bin_model("F")
        logits_s, z_s = native_forward(model_s, self.sensor, self.thermal)
        logits_f, z_f = native_forward(model_f, self.sensor, self.thermal)
        cases = {
            "S given thermal_z": (model_s, logits_s, dict(z_s, thermal_z=z_f["thermal_z"])),
            "F missing sensor_z": (model_f, logits_f, {"thermal_z": z_f["thermal_z"]}),
            "float64 z": (model_f, logits_f, {k: v.double() for k, v in z_f.items()}),
            "wrong width": (model_f, logits_f, dict(z_f, sensor_z=z_f["sensor_z"][:, :64])),
            "length mismatch": (model_f, logits_f[:2], z_f),
            "nonfinite z": (model_f, logits_f, dict(z_f, sensor_z=z_f["sensor_z"].clone().fill_(float("nan")))),
            "odd paired": (model_f, logits_f[:3], {k: v[:3] for k, v in z_f.items()}),
        }
        for label, (model, logits, z) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    bc.analyze_branch_controls(model, logits, **z)
        with self.assertRaises(ValueError):
            bc.analyze_branch_controls(det.AnomalyDetector("T", "AE"), logits_s, thermal_z=z_f["thermal_z"])
        for bad in (0, True, 2.5):
            with self.assertRaises(ValueError):
                bc.analyze_branch_controls(model_f, logits_f, micro_batch=bad, **z_f)
        for bad in (float("inf"), -1.0, True):
            with self.assertRaises(ValueError):
                bc.analyze_branch_controls(model_f, logits_f, replay_atol=bad, **z_f)

    def test_spread_and_pair_statistics_match_hand_computed_values(self):
        model = bin_model("T")
        z = torch.zeros(4, 128)
        z[0, 0], z[1, 0], z[2, 0], z[3, 0] = 0.0, 3.0, 1.0, 1.5
        logits = model.head(z).squeeze(-1).detach()
        record = bc.analyze_branch_controls(model, logits, thermal_z=z)["branches"]["thermal"]
        # population std of feature 0 over (0, 3, 1, 1.5) is sqrt(1.171875); other features are 0
        self.assertAlmostEqual(record["feature_population_std_max"], 1.171875 ** 0.5, places=12)
        self.assertAlmostEqual(record["feature_population_std_mean"], 1.171875 ** 0.5 / 128, places=12)
        self.assertEqual(record["pair_count"], 2)
        self.assertEqual(record["pair_max_abs_feature_change_max"], 3.0)
        self.assertEqual(record["pair_max_abs_feature_change_mean"], 1.75)
        self.assertEqual(record["pairs_exactly_equal"], 0)

    def test_replay_runs_with_every_module_in_eval_mode(self):
        model = bin_model("F")
        logits, z = native_forward(model, self.sensor, self.thermal)
        model.train()
        seen = []
        hooks = [model.head.register_forward_pre_hook(lambda m, i: seen.append(m.training)),
                 model.fusion.register_forward_pre_hook(lambda m, i: seen.append(m.training))]
        try:
            bc.analyze_branch_controls(model, logits, **z)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertTrue(seen)
        self.assertFalse(any(seen))
        self.assertTrue(model.training)

    def test_unpaired_mode_skips_pair_statistics(self):
        model = bin_model("T")
        logits, z = native_forward(model, self.sensor[:3], self.thermal[:3], micro_batch=2)
        record = bc.analyze_branch_controls(model, logits, paired=False, **z)
        self.assertNotIn("pair_count", record["branches"]["thermal"])

    def test_observation_changes_no_flag_rng_parameter_or_gradient(self):
        model = bin_model("F").train()
        model.head.eval()
        loss = model(sensor=self.sensor[:2], thermal=self.thermal[:2])["logit"].sum()
        loss.backward()
        logits, z = native_forward(model, self.sensor, self.thermal)
        model.train()
        model.head.eval()
        flags = {n: m.training for n, m in model.named_modules()}
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        before, cpu_rng = state_hash(model), torch.get_rng_state()
        setter = mock.Mock(side_effect=AssertionError("RNG must not be rewritten"))
        with mock.patch.object(torch, "set_rng_state", setter), \
                mock.patch.object(torch.cuda, "set_rng_state_all", setter):
            bc.analyze_branch_controls(model, logits, **z)
        setter.assert_not_called()
        self.assertEqual({n: m.training for n, m in model.named_modules()}, flags)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        self.assertEqual(state_hash(model), before)
        for n, p in model.named_parameters():
            if n in grads:
                self.assertTrue(torch.equal(p.grad, grads[n]), n)

    def test_rng_consumption_during_replay_is_a_failure_and_flags_come_back(self):
        model = bin_model("T").train()
        logits, z = native_forward(model, self.sensor, self.thermal)
        model.train()
        original = model.head.forward

        def consuming(x):
            torch.rand(1)
            return original(x)
        model.head.forward = consuming
        with self.assertRaisesRegex(RuntimeError, "consumed CPU RNG"):
            bc.analyze_branch_controls(model, logits, **z)
        self.assertTrue(model.training)

    def test_record_is_strict_json_with_frozen_thresholds(self):
        model = bin_model("F")
        logits, z = native_forward(model, self.sensor, self.thermal)
        record = bc.analyze_branch_controls(model, logits, **z)
        json.loads(json.dumps(record, allow_nan=False))
        self.assertEqual(record["category_thresholds"],
                         {"collapse_like_representation_std_mean_max": 1e-6,
                          "negligible_head_contribution_permutation_max_abs": 1e-6})
        self.assertEqual(record["shuffle_policy"], bc.SHUFFLE_POLICY)
        self.assertTrue(record["categories_are_labels_not_conclusions"])


class LeafTests(unittest.TestCase):
    def test_module_imports_only_torch_and_the_standard_library(self):
        tree = ast.parse((ROOT / "src/models/anomaly_branch_controls.py").read_text())
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                roots.add(node.module.split(".")[0])
        self.assertLessEqual(roots - set(sys.stdlib_module_names), {"torch"})


if __name__ == "__main__":
    unittest.main()
