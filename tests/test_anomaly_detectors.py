"""Structure contract for the S/T/F anomaly detector arms. No training is executed.

Skipped in full on JETSON-RUNTIME, where torch is deliberately absent: these arms are
trained on the server and only exported models run on the Jetson. Nothing here installs
torch or launches an optimiser step beyond a single backward pass for finiteness.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Every repository file these tests load. A server verification packet that omits one of
# these will fail before reading any data, so the list is asserted, not just documented.
LOCAL_DEPENDENCIES = ("src/models/anomaly_detectors.py", "src/models/v2_plus.py")

try:
    import torch
    TORCH_ERROR = None
except Exception as exc:  # pragma: no cover - exercised on Jetson
    torch, TORCH_ERROR = None, repr(exc)


def leaf(name, path):
    """Load a module file directly, bypassing src/models/__init__.py."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if torch is not None:
    ad = leaf("_anomaly_detectors", "src/models/anomaly_detectors.py")
    v2 = leaf("_v2_plus_ref", "src/models/v2_plus.py")
else:  # pragma: no cover
    ad = v2 = None

requires_torch = unittest.skipIf(torch is None, f"torch unavailable (expected on Jetson): {TORCH_ERROR}")


def sensor_batch(b=2, dim=None):
    return torch.randn(b, ad.WINDOW_TICKS, dim or ad.SENSOR_DIM)


def thermal_batch(b=1):
    return torch.randn(b, ad.WINDOW_TICKS, *ad.THERMAL_SHAPE)


@requires_torch
class ShapeTests(unittest.TestCase):
    def test_ae_reconstruction_shapes_match_the_declared_inputs(self):
        s, t = sensor_batch(1), thermal_batch(1)
        for arm, want_s, want_t in [("S", True, False), ("T", False, True), ("F", True, True)]:
            with self.subTest(arm=arm):
                model = ad.AnomalyDetector(arm, "AE").eval()
                out = model(sensor=s, thermal=t)
                self.assertEqual(tuple(out["bottleneck"].shape), (1, ad.HIDDEN_DIM))
                if want_s:
                    self.assertEqual(tuple(out["sensor_reconstruction"].shape),
                                     (1, ad.WINDOW_TICKS, ad.SENSOR_DIM))
                else:
                    self.assertIsNone(out["sensor_reconstruction"])
                if want_t:
                    self.assertEqual(tuple(out["thermal_reconstruction"].shape),
                                     (1, ad.WINDOW_TICKS, *ad.THERMAL_SHAPE))
                else:
                    self.assertIsNone(out["thermal_reconstruction"])
                self.assertIsNone(out["logit"])

    def test_bin_emits_one_logit_per_window_and_no_reconstruction(self):
        s, t = sensor_batch(3), thermal_batch(3)
        for arm in ad.ARMS:
            with self.subTest(arm=arm):
                out = ad.AnomalyDetector(arm, "BIN").eval()(sensor=s, thermal=t)
                self.assertEqual(tuple(out["logit"].shape), (3,))
                self.assertIsNone(out["sensor_reconstruction"])
                self.assertIsNone(out["thermal_reconstruction"])

    def test_wrong_window_shape_is_refused_rather_than_broadcast(self):
        model = ad.AnomalyDetector("F", "AE").eval()
        with self.assertRaises(ValueError):
            model(sensor=torch.randn(1, 29, ad.SENSOR_DIM), thermal=thermal_batch(1))
        with self.assertRaises(ValueError):
            model(sensor=sensor_batch(1), thermal=torch.randn(1, ad.WINDOW_TICKS, 120, 161))
        with self.assertRaises(ValueError):
            model(sensor=torch.randn(1, ad.WINDOW_TICKS, 8), thermal=thermal_batch(1))


@requires_torch
class ModalityIsolationTests(unittest.TestCase):
    def test_single_modality_arms_ignore_the_removed_modality(self):
        torch.manual_seed(7)
        s, t1, t2 = sensor_batch(1), thermal_batch(1), thermal_batch(1)
        self.assertFalse(torch.equal(t1, t2))
        model = ad.AnomalyDetector("S", "AE").eval()
        a = model(sensor=s, thermal=t1)
        b = model(sensor=s, thermal=t2)
        self.assertTrue(torch.equal(a["bottleneck"], b["bottleneck"]))
        self.assertTrue(torch.equal(a["sensor_reconstruction"], b["sensor_reconstruction"]))

        s2 = sensor_batch(1)
        self.assertFalse(torch.equal(s, s2))
        tm = ad.AnomalyDetector("T", "AE").eval()
        self.assertTrue(torch.equal(tm(sensor=s, thermal=t1)["thermal_reconstruction"],
                                    tm(sensor=s2, thermal=t1)["thermal_reconstruction"]))

    def test_fused_arm_depends_on_both_modalities(self):
        torch.manual_seed(11)
        model = ad.AnomalyDetector("F", "AE").eval()
        s, s2, t, t2 = sensor_batch(1), sensor_batch(1), thermal_batch(1), thermal_batch(1)
        base = model(sensor=s, thermal=t)["bottleneck"]
        self.assertFalse(torch.equal(base, model(sensor=s2, thermal=t)["bottleneck"]))
        self.assertFalse(torch.equal(base, model(sensor=s, thermal=t2)["bottleneck"]))

    def test_missing_required_modality_is_an_error_not_a_zero_tensor(self):
        with self.assertRaises(ValueError):
            ad.AnomalyDetector("S", "AE").eval()(thermal=thermal_batch(1))
        with self.assertRaises(ValueError):
            ad.AnomalyDetector("T", "AE").eval()(sensor=sensor_batch(1))
        with self.assertRaises(ValueError):
            ad.AnomalyDetector("F", "BIN").eval()(sensor=sensor_batch(1))


@requires_torch
class BranchInitialisationTests(unittest.TestCase):
    def _assert_identical(self, a, b):
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(sorted(sa), sorted(sb))
        for k in sa:
            self.assertTrue(torch.equal(sa[k], sb[k]), f"{k} differs")

    def test_explicit_copy_makes_every_shared_branch_identical(self):
        for objective in ad.OBJECTIVES:
            with self.subTest(objective=objective):
                models, manifest = ad.build_arm_set(objective, seed=42)
                self._assert_identical(models["S"].sensor_encoder, models["F"].sensor_encoder)
                self._assert_identical(models["T"].thermal_encoder, models["F"].thermal_encoder)
                if objective == "AE":
                    self._assert_identical(models["S"].sensor_decoder, models["F"].sensor_decoder)
                    self._assert_identical(models["T"].thermal_decoder, models["F"].thermal_decoder)
                self.assertTrue(manifest["copy_is_explicit_not_rng_order"])
                self.assertEqual(manifest["gradient_training_runs"], 0)

    def test_seeding_alone_would_not_have_matched_the_thermal_branch(self):
        """The reason §3.3(3) forbids relying on RNG call order; copying is load-bearing."""
        torch.manual_seed(42)
        t_alone = ad.AnomalyDetector("T", "AE")
        torch.manual_seed(42)
        fused = ad.AnomalyDetector("F", "AE")
        torch.manual_seed(42)
        s_alone = ad.AnomalyDetector("S", "AE")
        # Sensor happens to match because F draws it first; thermal does not.
        self._assert_identical(s_alone.sensor_encoder, fused.sensor_encoder)
        differs = any(not torch.equal(a, b) for a, b in
                      zip(t_alone.thermal_encoder.state_dict().values(),
                          fused.thermal_encoder.state_dict().values()))
        self.assertTrue(differs, "RNG order coincidentally matched; the copy is still required")

    def test_fusion_and_bin_heads_are_reported_as_uncopied(self):
        _, ae = ad.build_arm_set("AE", seed=42)
        self.assertEqual(ae["uncopied_F_parameters"], ["fusion.bias", "fusion.weight"])
        _, binm = ad.build_arm_set("BIN", seed=42)
        self.assertEqual(binm["uncopied_F_parameters"],
                         ["fusion.bias", "fusion.weight", "head.bias", "head.weight"])

    def test_shared_branches_are_identical_by_tensor_bytes_not_just_names(self):
        for objective in ad.OBJECTIVES:
            with self.subTest(objective=objective):
                models, manifest = ad.build_arm_set(objective, seed=42)
                for branch, per_arm in manifest["shared_branch_sha256"].items():
                    digests = set(per_arm.values())
                    self.assertEqual(len(digests), 1, f"{branch} initial bytes differ: {per_arm}")
                    for arm, digest in per_arm.items():
                        self.assertEqual(digest, ad.state_tensor_sha256(getattr(models[arm], branch)))
                self.assertEqual(set(manifest["initial_state_sha256"]), set(ad.ARMS))

    def test_tensor_digest_detects_a_weight_change_that_a_name_hash_would_miss(self):
        a = ad.AnomalyDetector("S", "AE")
        before = ad.state_tensor_sha256(a)
        with torch.no_grad():
            a.sensor_encoder.fc_sensor.bias[0] += 1e-6
        self.assertNotEqual(before, ad.state_tensor_sha256(a))

    def test_arm_set_is_reproducible_by_tensor_bytes_and_seed_sensitive(self):
        _, m42a = ad.build_arm_set("AE", seed=42)
        _, m42b = ad.build_arm_set("AE", seed=42)
        _, m123 = ad.build_arm_set("AE", seed=123)
        self.assertEqual(m42a["initial_state_sha256"], m42b["initial_state_sha256"])
        for arm in ad.ARMS:
            self.assertNotEqual(m42a["initial_state_sha256"][arm], m123["initial_state_sha256"][arm])

    def test_arm_set_is_reproducible_and_seed_sensitive(self):
        a, ma = ad.build_arm_set("AE", seed=42)
        b, mb = ad.build_arm_set("AE", seed=42)
        self.assertEqual(ma["sha256"], mb["sha256"])
        self._assert_identical(a["F"], b["F"])
        _, mc = ad.build_arm_set("AE", seed=123)
        c, _ = ad.build_arm_set("AE", seed=123)
        self.assertEqual(ma["parameter_counts"], mc["parameter_counts"])
        self.assertFalse(torch.equal(a["S"].sensor_encoder.fc_sensor.weight,
                                     c["S"].sensor_encoder.fc_sensor.weight))

    def test_fused_arm_is_larger_so_gains_are_not_attributed_to_fusion_alone(self):
        _, manifest = ad.build_arm_set("AE", seed=42)
        counts = manifest["parameter_counts"]
        self.assertGreater(counts["F"], counts["S"])
        self.assertGreater(counts["F"], counts["T"])


@requires_torch
class LossTests(unittest.TestCase):
    def test_element_mean_mse_does_not_scale_with_element_count(self):
        """A summed MSE would let 576,000 thermal elements dominate 150 sensor elements."""
        s = torch.zeros(2, ad.WINDOW_TICKS, ad.SENSOR_DIM)
        t = torch.zeros(2, ad.WINDOW_TICKS, *ad.THERMAL_SHAPE)
        self.assertAlmostEqual(ad.reconstruction_error(s + 3.0, s).item(), 9.0, places=5)
        self.assertAlmostEqual(ad.reconstruction_error(t + 3.0, t).item(), 9.0, places=5)

    def test_fused_loss_weights_each_modality_one_half(self):
        s = torch.zeros(1, ad.WINDOW_TICKS, ad.SENSOR_DIM)
        t = torch.zeros(1, ad.WINDOW_TICKS, *ad.THERMAL_SHAPE)
        outputs = {"sensor_reconstruction": s + 2.0, "thermal_reconstruction": t + 4.0}
        total, parts = ad.autoencoder_loss(outputs, sensor=s, thermal=t, arm="F")
        self.assertAlmostEqual(parts["sensor"].item(), 4.0, places=5)
        self.assertAlmostEqual(parts["thermal"].item(), 16.0, places=5)
        self.assertAlmostEqual(total.item(), 0.5 * 4.0 + 0.5 * 16.0, places=4)

    def test_single_modality_loss_is_that_modality_alone(self):
        s = torch.zeros(1, ad.WINDOW_TICKS, ad.SENSOR_DIM)
        t = torch.zeros(1, ad.WINDOW_TICKS, *ad.THERMAL_SHAPE)
        total, parts = ad.autoencoder_loss({"sensor_reconstruction": s + 2.0}, sensor=s, arm="S")
        self.assertAlmostEqual(total.item(), 4.0, places=5)
        self.assertEqual(set(parts), {"sensor"})
        total, parts = ad.autoencoder_loss({"thermal_reconstruction": t + 5.0}, thermal=t, arm="T")
        self.assertAlmostEqual(total.item(), 25.0, places=4)
        self.assertEqual(set(parts), {"thermal"})

    def test_per_sample_score_is_the_same_formula_as_the_training_loss(self):
        torch.manual_seed(3)
        s = sensor_batch(4)
        model = ad.AnomalyDetector("S", "AE").eval()
        score = ad.residual_score(model, sensor=s)
        self.assertEqual(tuple(score.shape), (4,))
        out = model(sensor=s)
        batch_loss, _ = ad.autoencoder_loss(out, sensor=s, arm="S")
        self.assertTrue(torch.allclose(score.mean(), batch_loss, atol=1e-6))

    def test_mismatched_reconstruction_shape_is_refused(self):
        s = torch.zeros(1, ad.WINDOW_TICKS, ad.SENSOR_DIM)
        with self.assertRaises(ValueError):
            ad.reconstruction_error(torch.zeros(1, ad.WINDOW_TICKS, 4), s)

    def test_residual_score_refuses_train_mode_because_dropout_breaks_reproducibility(self):
        model = ad.AnomalyDetector("S", "AE").train()
        with self.assertRaisesRegex(ValueError, "eval"):
            ad.residual_score(model, sensor=sensor_batch(1))
        model.eval()
        s = sensor_batch(1)
        first = ad.residual_score(model, sensor=s)
        self.assertTrue(torch.equal(first, ad.residual_score(model, sensor=s)))
        self.assertFalse(first.requires_grad)

    def test_residual_score_is_undefined_for_the_binary_objective(self):
        with self.assertRaises(ValueError):
            ad.residual_score(ad.AnomalyDetector("S", "BIN").eval(), sensor=sensor_batch(1))


@requires_torch
class DecoderBypassTests(unittest.TestCase):
    def test_decoders_accept_the_bottleneck_only(self):
        import inspect
        for decoder in (ad.SensorDecoder(), ad.ThermalDecoder()):
            with self.subTest(decoder=type(decoder).__name__):
                params = list(inspect.signature(decoder.forward).parameters)
                self.assertEqual(params, ["z"])

    def test_reconstruction_is_a_pure_function_of_the_bottleneck(self):
        torch.manual_seed(5)
        model = ad.AnomalyDetector("F", "AE").eval()
        s, t = sensor_batch(1), thermal_batch(1)
        out = model(sensor=s, thermal=t)
        z = model.bottleneck(sensor=s, thermal=t)
        self.assertTrue(torch.equal(out["sensor_reconstruction"], model.sensor_decoder(z)))
        self.assertTrue(torch.equal(out["thermal_reconstruction"], model.thermal_decoder(z)))
        # A decoder run on an unrelated bottleneck still works: it needs no input window.
        free = model.sensor_decoder(torch.randn(1, ad.HIDDEN_DIM))
        self.assertEqual(tuple(free.shape), (1, ad.WINDOW_TICKS, ad.SENSOR_DIM))

    def test_decoder_refuses_a_sequence_in_place_of_a_bottleneck(self):
        with self.assertRaises(ValueError):
            ad.SensorDecoder()(torch.randn(1, ad.WINDOW_TICKS, ad.HIDDEN_DIM))

    def test_positions_are_the_only_time_varying_decoder_input(self):
        """Identical z at every step, so any step-to-step variation comes from position."""
        torch.manual_seed(9)
        decoder = ad.SensorDecoder().eval()
        out = decoder(torch.randn(1, ad.HIDDEN_DIM))[0]
        self.assertFalse(torch.allclose(out[0], out[-1]))


@requires_torch
class BackwardTests(unittest.TestCase):
    def test_ae_backward_produces_finite_gradients_for_every_arm(self):
        torch.manual_seed(13)
        for arm in ad.ARMS:
            with self.subTest(arm=arm):
                model = ad.AnomalyDetector(arm, "AE")
                s, t = sensor_batch(1), thermal_batch(1)
                total, _ = ad.autoencoder_loss(model(sensor=s, thermal=t),
                                               sensor=s, thermal=t, arm=arm)
                total.backward()
                trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                self.assertGreater(len(trainable), 0)
                for name, p in trainable:
                    self.assertIsNotNone(p.grad, f"{arm}/{name} is detached from the loss")
                    self.assertTrue(torch.isfinite(p.grad).all(), f"{arm}/{name}")

    def test_bin_backward_with_bce_produces_finite_gradients(self):
        torch.manual_seed(17)
        model = ad.AnomalyDetector("F", "BIN")
        s, t = sensor_batch(2), thermal_batch(2)
        labels = torch.tensor([0.0, 1.0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(sensor=s, thermal=t)["logit"], labels)
        loss.backward()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.assertIsNotNone(p.grad, f"{name} is detached from the BCE loss")
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_unused_branch_receives_no_gradient_in_a_single_modality_arm(self):
        torch.manual_seed(19)
        model = ad.AnomalyDetector("S", "AE")
        s, t = sensor_batch(1), thermal_batch(1)
        total, _ = ad.autoencoder_loss(model(sensor=s, thermal=t), sensor=s, arm="S")
        total.backward()
        self.assertFalse(hasattr(model, "thermal_encoder"))
        self.assertFalse(hasattr(model, "thermal_decoder"))


@requires_torch
class ContractTests(unittest.TestCase):
    def test_se_block_matches_the_v2_plus_structure(self):
        mine, theirs = ad.SEBlock(ad.HIDDEN_DIM), v2.SEBlock(ad.HIDDEN_DIM)
        self.assertEqual([type(m).__name__ for m in mine.fc],
                         [type(m).__name__ for m in theirs.fc])
        self.assertEqual([tuple(p.shape) for p in mine.parameters()],
                         [tuple(p.shape) for p in theirs.parameters()])

    def test_frame_cnn_matches_the_v2_plus_thermal_encoder(self):
        mine, theirs = ad.ThermalEncoder(), v2.V2Plus()
        self.assertEqual([type(m).__name__ for m in mine.thermal_conv],
                         [type(m).__name__ for m in theirs.thermal_conv])
        self.assertEqual(mine.fc_thermal.weight.shape, theirs.fc_thermal.weight.shape)

    def test_field_channel_view_selects_one_ct_channel(self):
        self.assertEqual(ad.field_channel_indices("CT1"), (0, 1, 2, 3, 4))
        self.assertEqual(ad.field_channel_indices("CT2"), (0, 1, 2, 3, 5))
        with self.assertRaises(ValueError):
            ad.field_channel_indices("CT3")
        self.assertEqual(len(ad.FIELD_SENSOR_CHANNELS), ad.SENSOR_DIM)

    def test_no_supcon_or_four_class_head_is_present(self):
        record = ad.structure_manifest()
        self.assertFalse(record["supcon_or_4class_head"])
        self.assertEqual(record["status"],
                         "implementation_candidate_pending_review_and_server_verification")
        self.assertFalse(record["protocol_frozen"])
        self.assertEqual(record["full_P0_gate"], "NOT_EVALUATED")
        self.assertFalse(record["training_contract"]["pretrained_or_supcon_weights_used"])
        self.assertFalse(record["training_contract"]["heldout_used_for_any_selection"])
        for arm in ad.ARMS:
            model = ad.AnomalyDetector(arm, "AE")
            names = [n for n, _ in model.named_modules()]
            self.assertNotIn("classifier", names)

    def test_structure_manifest_is_stable(self):
        self.assertEqual(ad.structure_manifest()["sha256"], ad.structure_manifest()["sha256"])

    def test_declared_budget_matches_the_plan(self):
        c = ad.TRAINING_CONTRACT
        self.assertEqual((c["epochs"], c["optimizer"], c["learning_rate"], c["weight_decay"]),
                         (30, "AdamW", 1e-3, 0.01))
        self.assertEqual(c["effective_batch"], 16)
        self.assertEqual(c["seeds"], (42, 123, 456))
        self.assertTrue(c["deterministic"])
        self.assertEqual(c["precision"], "FP32")
        self.assertEqual(c["lr_schedule"], "fixed_cosine")

    def test_decoder_structure_is_recorded_as_an_explicit_choice(self):
        d = ad.structure_manifest()["decoder"]
        self.assertFalse(d["skip_connections"])
        self.assertFalse(d["teacher_forcing"])
        self.assertEqual(d["input"], "bottleneck_and_tick_position_only")

    def test_unknown_arm_or_objective_is_refused(self):
        with self.assertRaises(ValueError):
            ad.AnomalyDetector("X", "AE")
        with self.assertRaises(ValueError):
            ad.AnomalyDetector("S", "CLASSIFY")
        with self.assertRaises(ValueError):
            ad.build_arm_set("AE", seed=True)


class JetsonPolicyTests(unittest.TestCase):
    """Runs everywhere, including where torch is absent."""

    def test_module_does_not_reach_for_a_gpu_or_install_anything(self):
        text = (ROOT / "src/models/anomaly_detectors.py").read_text()
        for forbidden in ("pip install", "subprocess", ".cuda(", "device=\"cuda",
                          "load_state_dict(torch.load", "requires_grad_(False)"):
            self.assertNotIn(forbidden, text, forbidden)

    def test_declared_local_dependencies_all_exist(self):
        for rel in LOCAL_DEPENDENCIES:
            self.assertTrue((ROOT / rel).is_file(), f"server packet would be missing {rel}")

    def test_structure_is_not_claimed_to_be_protocol_frozen(self):
        text = (ROOT / "src/models/anomaly_detectors.py").read_text()
        self.assertIn("implementation_candidate_pending_review_and_server_verification", text)
        self.assertNotIn("structure_frozen_no_training_executed", text)

    def test_skip_is_explicit_when_torch_is_missing(self):
        if torch is None:
            self.assertIsNotNone(TORCH_ERROR)
        else:
            self.assertIsNone(TORCH_ERROR)


if __name__ == "__main__":
    unittest.main()
