"""Thermal-branch BIN diagnostics: parity is real, observation changes nothing, microbatches combine."""
import copy
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
    from torch import nn
    TORCH_ERROR = None
except Exception as exc:  # pragma: no cover - exercised on Jetson
    torch = None
    TORCH_ERROR = repr(exc)
requires_torch = unittest.skipIf(torch is None, f"torch unavailable (expected on Jetson): {TORCH_ERROR}")


def leaf(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if torch is not None:
    det = leaf("_bin_diag_detectors", "src/models/anomaly_detectors.py")
    diag = leaf("_bin_diag_under_test", "src/models/anomaly_bin_diagnostics.py")


def thermal_batch(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, 30, 120, 160, generator=generator, dtype=torch.float32)


def bin_model(arm, seed=42):
    torch.manual_seed(seed)
    return det.AnomalyDetector(arm, "BIN")


if torch is not None:
    class PerturbedLSTM(nn.LSTM):
        """An nn.LSTM whose forward output is shifted, so the manual recurrence must disagree."""

        shift = 1e-3

        def forward(self, x, hx=None):
            out, state = super().forward(x, hx)
            return out + self.shift, state


@requires_torch
class RecurrenceTests(unittest.TestCase):
    def test_manual_recurrence_matches_nn_lstm_on_a_short_multilayer_sequence(self):
        torch.manual_seed(3)
        for layers, steps in ((1, 1), (2, 3), (3, 5)):
            with self.subTest(layers=layers, steps=steps):
                lstm = nn.LSTM(4, 6, layers, batch_first=True, dropout=0.0).eval()
                x = torch.randn(7, steps, 4)
                with torch.no_grad():
                    reference, _ = lstm(x)
                    manual, stats = diag.lstm_recurrence(lstm, x)
                record = diag._parity(manual, reference, diag.DEFAULT_RTOL, diag.DEFAULT_ATOL)
                self.assertTrue(record["passed"], record)
                self.assertEqual(record["violations"], 0)
                self.assertEqual(len(stats), layers)
                for layer in stats:
                    for gate in diag.GATES:
                        self.assertEqual(layer["gates"][gate]["total_count"], 7 * steps * 6)

    def test_a_perturbed_weight_breaks_parity(self):
        torch.manual_seed(4)
        lstm = nn.LSTM(4, 6, 2, batch_first=True).eval()
        x = torch.randn(5, 3, 4)
        wrong = copy.deepcopy(lstm)
        with torch.no_grad():
            wrong.weight_hh_l1[0, 0] += 0.05
            reference, _ = lstm(x)
            manual, _ = diag.lstm_recurrence(wrong, x)
        record = diag._parity(manual, reference, diag.DEFAULT_RTOL, diag.DEFAULT_ATOL)
        self.assertFalse(record["passed"])
        self.assertGreater(record["violations"], 0)
        self.assertGreater(record["max_abs_error"], diag.DEFAULT_ATOL)

    def test_unsupported_lstm_configurations_are_refused(self):
        cases = (nn.LSTM(4, 6, 1, batch_first=False), nn.LSTM(4, 6, 1, batch_first=True, bidirectional=True),
                 nn.LSTM(4, 6, 1, batch_first=True, bias=False), nn.LSTM(4, 6, 1, batch_first=True, proj_size=3),
                 nn.GRU(4, 6, 1, batch_first=True))
        for lstm in cases:
            with self.subTest(lstm=repr(lstm)):
                with self.assertRaises(ValueError):
                    diag.lstm_recurrence(lstm, torch.randn(2, 3, 4))

    def test_saturation_count_follows_the_pre_activation_threshold(self):
        torch.manual_seed(5)
        lstm = nn.LSTM(3, 4, 1, batch_first=True).eval()
        with torch.no_grad():
            lstm.weight_ih_l0.fill_(0.0)
            lstm.weight_hh_l0.fill_(0.0)
            lstm.bias_hh_l0.fill_(0.0)
            lstm.bias_ih_l0.copy_(torch.tensor([7.0] * 4 + [5.0] * 4 + [-7.0] * 4 + [6.0] * 4))
            _, stats = diag.lstm_recurrence(lstm, torch.zeros(2, 3, 3))
        gates = stats[0]["gates"]
        self.assertEqual(gates["input"]["abs_gt6_count"], gates["input"]["total_count"])
        self.assertEqual(gates["forget"]["abs_gt6_count"], 0)
        self.assertEqual(gates["cell"]["abs_gt6_count"], gates["cell"]["total_count"])
        self.assertEqual(gates["output"]["abs_gt6_count"], 0)  # strictly greater than 6


@requires_torch
class ThermalDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thermal = thermal_batch(4)

    def test_parity_passes_on_both_t_and_f_and_reports_the_native_z(self):
        for arm in ("T", "F"):
            with self.subTest(arm=arm):
                model = bin_model(arm)
                record = diag.thermal_lstm_diagnostics(model, self.thermal)
                self.assertTrue(record["parity_sequence"]["passed"], record["parity_sequence"])
                self.assertTrue(record["parity_encoder_output"]["passed"], record["parity_encoder_output"])
                with torch.no_grad():
                    z = model.eval().thermal_encoder(self.thermal)
                self.assertEqual(torch.tensor(record["thermal_z_values"], dtype=torch.float64).tolist(),
                                 z.double().tolist())
                self.assertEqual(record["examples"], 4)
                self.assertEqual(len(record["layers"]), model.thermal_encoder.lstm.num_layers)
                self.assertEqual(record["fc_thermal"]["count"], 4 * 30 * 128)

    def test_a_mismatched_reference_lstm_fails_parity_instead_of_passing_silently(self):
        model = bin_model("T")
        original = model.thermal_encoder.lstm
        perturbed = PerturbedLSTM(original.input_size, original.hidden_size, original.num_layers,
                                  batch_first=True, dropout=original.dropout)
        perturbed.load_state_dict(original.state_dict())
        model.thermal_encoder.lstm = perturbed
        record = diag.thermal_lstm_diagnostics(model, self.thermal[:2])
        self.assertFalse(record["parity_sequence"]["passed"])
        self.assertFalse(record["parity_encoder_output"]["passed"])
        self.assertFalse(diag.summarize(record)["parity_passed"])

    def test_observation_restores_training_flags_rng_and_leaves_gradients_alone(self):
        model = bin_model("F").train()
        model.head.eval()  # a mixed state must come back exactly
        loss = model(sensor=torch.randn(2, 30, 5), thermal=self.thermal[:2])["logit"].sum()
        loss.backward()
        flags = {name: m.training for name, m in model.named_modules()}
        grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        cpu_rng = torch.get_rng_state()
        diag.thermal_lstm_diagnostics(model, self.thermal[:2])
        self.assertEqual({name: m.training for name, m in model.named_modules()}, flags)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        for name, p in model.named_parameters():
            if name in grads:
                self.assertTrue(torch.equal(p.grad, grads[name]), name)

    def test_tolerances_must_be_finite_and_non_negative(self):
        model = bin_model("T")
        for rtol, atol in ((float("inf"), 1e-6), (1e-5, float("inf")), (float("nan"), 1e-6),
                           (-1e-5, 1e-6), (1e-5, -1.0), (True, 1e-6), ("1e-5", 1e-6)):
            with self.subTest(rtol=rtol, atol=atol):
                with self.assertRaisesRegex(ValueError, "finite number"):
                    diag.thermal_lstm_diagnostics(model, self.thermal[:1], rtol=rtol, atol=atol)
        diag.thermal_lstm_diagnostics(model, self.thermal[:1], rtol=0, atol=0.0)

    def test_a_cpu_call_does_not_initialize_cuda(self):
        """Run in a fresh interpreter so no earlier test has touched CUDA."""
        import subprocess
        script = (
            "import importlib.util, sys, torch\n"
            "def leaf(n, r):\n"
            f"    s = importlib.util.spec_from_file_location(n, {str(ROOT)!r} + '/' + r)\n"
            "    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m\n"
            "det = leaf('d', 'src/models/anomaly_detectors.py'); diag = leaf('g', 'src/models/anomaly_bin_diagnostics.py')\n"
            "torch.manual_seed(42); model = det.AnomalyDetector('T', 'BIN')\n"
            "before = torch.cuda.is_initialized()\n"
            "diag.thermal_lstm_diagnostics(model, torch.zeros(1, 30, 120, 160))\n"
            "print(before, torch.cuda.is_initialized(), torch.cuda.is_available())\n")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        before, after, _available = result.stdout.split()
        self.assertEqual((before, after), ("False", "False"))

    def test_rng_is_compared_never_rewritten(self):
        """Rewriting even an identical CUDA RNG state re-initializes cuDNN LSTM dropout (review (44))."""
        model = bin_model("T")
        refuse = mock.Mock(side_effect=AssertionError("RNG state must not be rewritten"))
        with mock.patch.object(torch, "set_rng_state", refuse), \
                mock.patch.object(torch.cuda, "set_rng_state_all", refuse), \
                mock.patch.object(torch.cuda, "set_rng_state", refuse):
            diag.thermal_lstm_diagnostics(model, self.thermal[:1])
        refuse.assert_not_called()

    def test_an_observation_that_consumes_rng_is_a_failure(self):
        model = bin_model("T")
        original = model.thermal_encoder.lstm.forward

        def consuming(x, hx=None):
            torch.rand(1)
            return original(x, hx)
        model.thermal_encoder.lstm.forward = consuming
        with self.assertRaisesRegex(RuntimeError, "consumed CPU RNG"):
            diag.thermal_lstm_diagnostics(model, self.thermal[:1])
        self.assertTrue(model.training)  # a fresh module is in train mode; the flag came back

    def test_a_fresh_model_gets_no_gradients(self):
        model = bin_model("T")
        diag.thermal_lstm_diagnostics(model, self.thermal[:1])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_state_is_restored_even_when_the_call_fails(self):
        model = bin_model("T").train()
        cpu_rng = torch.get_rng_state()
        bad = self.thermal[:1].clone()
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite thermal input"):
            diag.thermal_lstm_diagnostics(model, bad)
        self.assertTrue(all(m.training for m in model.modules()))
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))

    def test_non_finite_weights_are_a_failure(self):
        model = bin_model("T")
        with torch.no_grad():
            model.thermal_encoder.fc_thermal.weight[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite parameter"):
            diag.thermal_lstm_diagnostics(model, self.thermal[:1])

    def test_wrong_models_and_inputs_are_refused(self):
        cases = {
            "AE objective": (lambda: (torch.manual_seed(0), det.AnomalyDetector("T", "AE"))[1], self.thermal[:1]),
            "sensor arm": (lambda: bin_model("S"), self.thermal[:1]),
            "float64": (lambda: bin_model("T"), self.thermal[:1].double()),
            "wrong shape": (lambda: bin_model("T"), torch.zeros(1, 30, 60, 80)),
            "frames axis": (lambda: bin_model("T"), torch.zeros(1, 30, 1, 120, 160)),
        }
        for label, (make, x) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    diag.thermal_lstm_diagnostics(make(), x)

    def test_microbatches_combine_to_the_whole_batch(self):
        for blown in (False, True):
            with self.subTest(saturated=blown):
                self._check_combination(blown)

    def _check_combination(self, blown):
        model = bin_model("T")
        if blown:  # non-zero saturation counts, so a part that is dropped would show
            with torch.no_grad():
                model.thermal_encoder.fc_thermal.weight.mul_(2000.0)
        whole = diag.thermal_lstm_diagnostics(model, self.thermal)
        parts = diag.combine_diagnostics([diag.thermal_lstm_diagnostics(model, self.thermal[:1]),
                                          diag.thermal_lstm_diagnostics(model, self.thermal[1:3]),
                                          diag.thermal_lstm_diagnostics(model, self.thermal[3:])])
        self.assertEqual(parts["examples"], whole["examples"])
        self.assertEqual(parts["fc_thermal"]["count"], whole["fc_thermal"]["count"])
        self.assertAlmostEqual(parts["fc_thermal"]["abs_sum"], whole["fc_thermal"]["abs_sum"],
                               delta=1e-9 * whole["fc_thermal"]["abs_sum"])
        self.assertEqual(parts["fc_thermal"]["abs_max"], whole["fc_thermal"]["abs_max"])
        for la, lb in zip(parts["layers"], whole["layers"]):
            for gate in diag.GATES:
                self.assertEqual(la["gates"][gate]["total_count"], lb["gates"][gate]["total_count"])
                self.assertEqual(la["gates"][gate]["abs_gt6_count"], lb["gates"][gate]["abs_gt6_count"])
        if blown:
            self.assertGreater(parts["layers"][0]["gates"]["input"]["abs_gt6_count"], 0)
        a, b = diag.summarize(parts), diag.summarize(whole)
        self.assertAlmostEqual(a["thermal_z_across_example_std"]["mean"],
                               b["thermal_z_across_example_std"]["mean"], places=9)
        self.assertEqual(len(parts["thermal_z_values"]), 4)
        self.assertEqual(parts["parity_sequence"]["count"], whole["parity_sequence"]["count"])

    def test_combining_refuses_mixed_runs(self):
        t = diag.thermal_lstm_diagnostics(bin_model("T"), self.thermal[:1])
        f = diag.thermal_lstm_diagnostics(bin_model("F"), self.thermal[:1])
        with self.assertRaisesRegex(ValueError, "different arm"):
            diag.combine_diagnostics([t, f])
        with self.assertRaises(ValueError):
            diag.combine_diagnostics([])

    def test_record_is_strict_json(self):
        record = diag.thermal_lstm_diagnostics(bin_model("F"), self.thermal[:2])
        text = json.dumps(record, allow_nan=False)
        self.assertEqual(json.loads(text)["schema"], diag.SCHEMA)
        json.dumps(diag.summarize(record), allow_nan=False)

    def test_a_blown_up_projection_shows_first_layer_saturation(self):
        """The P1b failure pattern: a huge fc_thermal output saturates layer 0 and flattens z."""
        model = bin_model("T")
        healthy = diag.summarize(diag.thermal_lstm_diagnostics(model, self.thermal))
        with torch.no_grad():
            model.thermal_encoder.fc_thermal.weight.mul_(2000.0)
        blown = diag.summarize(diag.thermal_lstm_diagnostics(model, self.thermal))
        self.assertEqual(max(healthy["layers"][0]["gate_abs_gt6_fraction"].values()), 0.0)
        self.assertGreater(min(blown["layers"][0]["gate_abs_gt6_fraction"].values()), 0.5)
        self.assertGreater(blown["fc_thermal_abs_mean"], 100 * healthy["fc_thermal_abs_mean"])


@requires_torch
@unittest.skipIf(torch is None or not torch.cuda.is_available(), "needs a CUDA device")
class CudaUpdateInvarianceTests(unittest.TestCase):
    """The P1c fixture failure: observing between steps must not change later GPU updates."""

    @staticmethod
    def _train(observe, steps=3):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = det.AnomalyDetector("T", "BIN").cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        generator = torch.Generator().manual_seed(7)
        thermal = torch.randn(4, 30, 120, 160, generator=generator).cuda()
        label = torch.tensor([0.0, 1.0, 0.0, 1.0], device="cuda")
        loss_fn = nn.BCEWithLogitsLoss()
        hashes = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(thermal=thermal)["logit"], label).backward()
            optimizer.step()
            digest = hashlib.sha256()
            for p in model.parameters():
                digest.update(p.detach().cpu().numpy().tobytes())
            hashes.append(digest.hexdigest())
            if observe:
                diag.thermal_lstm_diagnostics(model, thermal[:2])
        return hashes

    def test_gpu_updates_are_identical_with_and_without_observation(self):
        deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True
        try:
            control = self._train(False)
            self.assertEqual(self._train(False), control, "control is not reproducible on this device")
            self.assertEqual(self._train(True), control)
        finally:
            torch.backends.cudnn.deterministic = deterministic


class LeafTests(unittest.TestCase):
    def test_module_imports_only_torch_and_the_standard_library(self):
        import ast
        tree = ast.parse((ROOT / "src/models/anomaly_bin_diagnostics.py").read_text())
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
