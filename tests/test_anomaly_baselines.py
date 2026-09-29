"""Fixture tests for the EXP-20260928-004 statistical baselines. No hardware, no dataset.

`src/models/__init__.py` imports torch, which this Jetson does not have, so the module is
loaded as a leaf by path exactly as `test_data_portability.py` does.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_leaf(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ab = load_leaf("baselines_under_test", "src/models/anomaly_baselines.py")


def sensor_windows(n=6, t=30, c=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, t, c))


def thermal_windows(n=4, t=5, h=6, w=7, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, t, h, w))


class LabelBarrierTests(unittest.TestCase):
    def test_no_public_callable_accepts_a_label_argument(self):
        """The structural version of design §7-1: severity cannot reach a fitting path."""
        checked = ab.assert_no_label_arguments(ab)
        self.assertGreater(len(checked), 15, "guard should be inspecting the real surface")
        for expected in ("MahalanobisBaseline.fit", "EcdfScaler.fit", "LateFusionBaseline.fit"):
            self.assertIn(expected, checked)

    def test_guard_actually_fires(self):
        """A guard that never fails proves nothing, so check it rejects a planted case."""
        import types

        fake = types.ModuleType("fake_baselines")

        def fit_threshold(scores, severity_labels):      # the thing we must never allow
            return 0.0

        fit_threshold.__module__ = fake.__name__
        fake.fit_threshold = fit_threshold
        with self.assertRaisesRegex(AssertionError, "severity_labels"):
            ab.assert_no_label_arguments(fake)


class EqualEquipmentWeightTests(unittest.TestCase):
    def test_every_machine_gets_the_same_total_weight(self):
        ids = ["agv01"] * 2 + ["oht05"] * 7 + ["agv09"] * 1
        w = ab.equal_equipment_weights(ids)
        self.assertAlmostEqual(w.sum(), 1.0, places=12)
        for dev in set(ids):
            mask = np.array([d == dev for d in ids])
            self.assertAlmostEqual(w[mask].sum(), 1.0 / 3, places=12,
                                   msg=f"{dev} should hold 1/3 of the weight")
        # the single-window machine must outweigh each window of the seven-window machine
        self.assertGreater(w[-1], w[2])

    def test_empty_input_is_an_error(self):
        with self.assertRaises(ValueError):
            ab.equal_equipment_weights([])


class NtcAndThermalScoreTests(unittest.TestCase):
    def test_b_ntc_is_max_abs_of_the_ntc_channel_only(self):
        x = np.zeros((2, 30, 5))
        x[0, 7, 0] = -4.5          # NTC, negative: the score is on |.|
        x[0, 9, 3] = 99.0          # a different channel must not be read
        x[1, 0, 0] = 2.0
        s = ab.score_b_ntc(x)
        np.testing.assert_allclose(s, [4.5, 2.0])

    def test_b_ntc_honours_the_channel_index(self):
        x = np.zeros((1, 30, 5))
        x[0, 3, 2] = 7.0
        self.assertAlmostEqual(ab.score_b_ntc(x, ntc_index=2)[0], 7.0)
        self.assertAlmostEqual(ab.score_b_ntc(x, ntc_index=0)[0], 0.0)
        with self.assertRaises(ValueError):
            ab.score_b_ntc(x, ntc_index=5)

    def test_b_therm_is_max_abs_over_time_and_pixels(self):
        x = np.zeros((2, 4, 3, 3))
        x[0, 2, 1, 1] = -6.25
        x[1, 0, 0, 0] = 1.5
        np.testing.assert_allclose(ab.score_b_therm(x), [6.25, 1.5])

    def test_wrong_rank_is_rejected(self):
        with self.assertRaises(ValueError):
            ab.score_b_ntc(np.zeros((30, 5)))
        with self.assertRaises(ValueError):
            ab.score_b_therm(np.zeros((4, 3, 3)))

    def test_non_finite_input_is_rejected_not_repaired(self):
        x = np.zeros((1, 30, 5))
        x[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ab.score_b_ntc(x)
        t = np.zeros((1, 2, 2, 2))
        t[0, 0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ab.score_b_therm(t)


class MdFeatureTests(unittest.TestCase):
    def test_feature_values_and_order(self):
        x = np.zeros((1, 4, 5))
        x[0, :, 0] = [1.0, 3.0, -1.0, 1.0]        # mean 1.0, std(ddof=0) sqrt(2), maxabs 3
        f = ab.md_features(x)[0]
        self.assertEqual(f.shape, (15,))
        self.assertAlmostEqual(f[0], 1.0)
        self.assertAlmostEqual(f[1], np.sqrt(2.0))
        self.assertAlmostEqual(f[2], 3.0)
        np.testing.assert_allclose(f[3:], 0.0)

    def test_names_line_up_with_columns(self):
        names = ab.md_feature_names(["NTC", "PM1.0", "PM2.5", "PM10", "CT1"])
        self.assertEqual(len(names), 15)
        self.assertEqual(names[0], "NTC.temporal_mean")
        self.assertEqual(names[1], "NTC.temporal_std_ddof0")
        self.assertEqual(names[2], "NTC.temporal_max_abs")
        self.assertEqual(names[3], "PM1.0.temporal_mean")
        self.assertEqual(names[14], "CT1.temporal_max_abs")

    def test_std_uses_ddof_zero(self):
        x = np.zeros((1, 2, 5))
        x[0, :, 0] = [0.0, 2.0]                   # ddof=0 -> 1.0, ddof=1 would be sqrt(2)
        self.assertAlmostEqual(ab.md_features(x)[0, 1], 1.0)


class MahalanobisTests(unittest.TestCase):
    def setUp(self):
        self.x = sensor_windows(n=40, seed=3)
        self.ids = ["agv01"] * 12 + ["oht05"] * 20 + ["agv09"] * 8
        self.md = ab.MahalanobisBaseline.fit(self.x, self.ids)

    def test_shrinkage_formula_is_applied_exactly(self):
        c, p = self.md.covariance, ab.MD_FEATURE_COUNT
        expected = (1 - ab.MD_LAMBDA) * c + ab.MD_LAMBDA * (np.trace(c) / p) * np.eye(p) \
            + ab.MD_EPSILON * np.eye(p)
        np.testing.assert_allclose(self.md.covariance_regularised, expected, atol=1e-12)
        self.assertEqual((self.md.lam, self.md.epsilon), (0.1, 1e-6))

    def test_standardisation_puts_unit_variance_on_the_diagonal(self):
        np.testing.assert_allclose(np.diag(self.md.covariance), 1.0, atol=1e-10)

    def test_score_matches_an_independent_inverse(self):
        """The solve path must agree with the textbook form `(z-mu)' C^-1 (z-mu)`."""
        z = (ab.md_features(self.x) - self.md.feature_mean) / self.md.feature_scale
        d = z - self.md.feature_centroid
        inv = np.linalg.inv(self.md.covariance_regularised)
        expected = np.einsum("ij,jk,ik->i", d, inv, d)
        np.testing.assert_allclose(self.md.score(self.x), expected, rtol=1e-9, atol=1e-9)

    def test_the_distribution_centroid_scores_about_zero(self):
        """Covariance and distance must be centred on the same point."""
        w = ab.equal_equipment_weights(self.ids)
        z = (ab.md_features(self.x) - self.md.feature_mean) / self.md.feature_scale
        centre = w @ z
        L = np.linalg.cholesky(self.md.covariance_regularised)
        y = np.linalg.solve(L, (centre - self.md.feature_centroid)[:, None])
        self.assertLess(float((y ** 2).sum()), 1e-16)

    def test_a_one_ulp_feature_spread_makes_the_centroid_material(self):
        """Why the centroid is stored: with a near-constant feature it is not ~1e-16.

        The channel is constant inside each window and differs by a single ULP across
        windows, so its feature scale is ~1e-16 and the float residue in the stored mean,
        divided by that scale, becomes order one. Without subtracting the centroid the
        distribution's own centre scored 9.26 instead of 0.
        """
        vals = [1.0, np.nextafter(1.0, 2.0), 1.0, np.nextafter(1.0, 2.0), 1.0]
        x = np.zeros((5, 30, 5))
        for i, v in enumerate(vals):
            x[i, :, 0] = v
        x[:, :, 1:] = np.random.default_rng(0).normal(size=(5, 30, 4))
        ids = ["a", "a", "a", "b", "b"]                 # unbalanced on purpose
        md = ab.MahalanobisBaseline.fit(x, ids)
        self.assertLess(md.feature_scale[0], 1e-15)
        self.assertGreater(np.abs(md.feature_centroid).max(), 0.1,
                           "this fixture exists because the centroid is material here")
        w = ab.equal_equipment_weights(ids)
        z = (ab.md_features(x) - md.feature_mean) / md.feature_scale
        L = np.linalg.cholesky(md.covariance_regularised)
        uncentred = float((np.linalg.solve(L, (w @ z)[:, None]) ** 2).sum())
        centred = float((np.linalg.solve(L, ((w @ z) - md.feature_centroid)[:, None]) ** 2).sum())
        self.assertGreater(uncentred, 1.0, "without the centroid the centre is far from 0")
        self.assertLess(centred, 1e-16)
        self.assertTrue(np.all(np.isfinite(md.score(x))))

    def test_ordinary_data_leaves_the_centroid_negligible(self):
        """The fix must not disturb the well-scaled case."""
        self.assertLess(np.abs(self.md.feature_centroid).max(), 1e-12)

    def test_score_is_non_negative_and_larger_further_out(self):
        s_in = self.md.score(self.x)
        self.assertTrue(np.all(s_in >= 0))
        far = self.x.copy()
        far[:, :, 0] += 50.0
        self.assertTrue(np.all(self.md.score(far) > s_in))

    def test_constant_features_get_scale_one_and_are_recorded(self):
        """A channel that never moves must not divide by zero or dominate the distance."""
        x = sensor_windows(n=20, seed=4)
        x[:, :, 4] = 2.5                       # CT1 pinned: mean const, std 0, maxabs const
        md = ab.MahalanobisBaseline.fit(x, ["agv01"] * 10 + ["oht01"] * 10)
        self.assertEqual(md.constant_features, [12, 13, 14])
        np.testing.assert_allclose(md.feature_scale[[12, 13, 14]], 1.0)
        self.assertTrue(np.all(np.isfinite(md.score(x))))
        # the constant columns must contribute exactly nothing, not float dust
        z = (ab.md_features(x) - md.feature_mean) / md.feature_scale
        np.testing.assert_array_equal(z[:, [12, 13, 14]], np.zeros((len(x), 3)))

    def test_all_constant_input_degenerates_predictably(self):
        x = np.full((8, 30, 5), 1.25)
        md = ab.MahalanobisBaseline.fit(x, ["agv01"] * 4 + ["oht01"] * 4)
        self.assertEqual(len(md.constant_features), ab.MD_FEATURE_COUNT)
        np.testing.assert_allclose(md.covariance, 0.0, atol=1e-12)
        # trace 0, so the whole regularised covariance is epsilon * I
        np.testing.assert_allclose(md.covariance_regularised, ab.MD_EPSILON * np.eye(15), atol=1e-15)
        np.testing.assert_allclose(md.score(x), 0.0, atol=1e-8)

    def test_fitting_on_a_non_fit_role_is_refused(self):
        for role in ("dev", "cal", "heldout"):
            with self.assertRaisesRegex(ValueError, "role 'fit'"):
                ab.MahalanobisBaseline.fit(self.x, self.ids, role=role)

    def test_device_id_length_must_match(self):
        with self.assertRaises(ValueError):
            ab.MahalanobisBaseline.fit(self.x, self.ids[:-1])

    def test_round_trip_preserves_scores_and_hash(self):
        again = ab.MahalanobisBaseline.from_dict(json.loads(json.dumps(self.md.to_dict())))
        np.testing.assert_array_equal(again.feature_centroid, self.md.feature_centroid)
        np.testing.assert_allclose(again.score(self.x), self.md.score(self.x), rtol=1e-12)
        self.assertEqual(again.params_sha256(), self.md.params_sha256())

    def test_hash_changes_when_a_parameter_changes(self):
        d = self.md.to_dict()
        d["feature_mean"] = [v + 1e-9 for v in d["feature_mean"]]
        self.assertNotEqual(ab.params_sha256(d), self.md.params_sha256())


class EcdfTests(unittest.TestCase):
    def test_ties_are_counted_and_range_is_bounded(self):
        dev = np.array([1.0, 2.0, 2.0, 5.0])
        e = ab.EcdfScaler.fit(dev, ["d"] * 4, component="B-MD")
        got = e.transform(np.array([0.5, 1.0, 1.5, 2.0, 4.9, 5.0, 9.0]))
        np.testing.assert_allclose(got, [0.0, 0.25, 0.25, 0.75, 0.75, 1.0, 1.0])

    def test_equal_equipment_weighting_beats_window_counts(self):
        """One machine with many windows must not outvote a machine with few."""
        dev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        ids = ["busy"] * 9 + ["quiet"] * 1
        e = ab.EcdfScaler.fit(dev, ids, component="B-MD")
        # 'busy' holds 1/2 of the weight across its nine zeros, 'quiet' 1/2 on its single 10
        np.testing.assert_allclose(e.transform(np.array([0.0])), [0.5])
        np.testing.assert_allclose(e.transform(np.array([10.0])), [1.0])

    def test_no_extrapolation_below_or_above(self):
        e = ab.EcdfScaler.fit(np.array([3.0, 4.0]), ["d", "d"], component="B-THERM")
        np.testing.assert_allclose(e.transform(np.array([-1e9, 1e9])), [0.0, 1.0])

    def test_saturation_report(self):
        e = ab.EcdfScaler.fit(np.array([1.0, 2.0, 3.0, 4.0]), ["d"] * 4, component="B-MD")
        rep = e.saturation(np.array([0.0, 2.5, 4.0, 5.0]))
        self.assertAlmostEqual(rep["below_dev_min_fraction"], 0.25)
        self.assertAlmostEqual(rep["at_or_above_dev_max_fraction"], 0.5)
        self.assertAlmostEqual(rep["u_eq_0_fraction"], 0.25)
        self.assertAlmostEqual(rep["u_eq_1_fraction"], 0.5)
        self.assertFalse(rep["degenerate_dev_scores"])

    def test_degenerate_dev_scores_are_flagged(self):
        e = ab.EcdfScaler.fit(np.full(5, 7.0), ["d"] * 5, component="B-MD")
        self.assertTrue(e.degenerate)
        np.testing.assert_allclose(e.transform(np.array([6.9, 7.0, 7.1])), [0.0, 1.0, 1.0])

    def test_fitting_on_a_non_dev_role_is_refused(self):
        for role in ("fit", "cal", "heldout"):
            with self.assertRaisesRegex(ValueError, "role 'dev'"):
                ab.EcdfScaler.fit(np.array([1.0, 2.0]), ["d", "d"], component="B-MD", role=role)

    def test_round_trip(self):
        e = ab.EcdfScaler.fit(np.array([1.0, 2.0, 2.0, 5.0]), ["a", "a", "b", "b"],
                             component="B-MD")
        again = ab.EcdfScaler.from_dict(json.loads(json.dumps(e.to_dict())))
        q = np.array([0.0, 2.0, 6.0])
        np.testing.assert_allclose(again.transform(q), e.transform(q))
        self.assertEqual(again.params_sha256(), e.params_sha256())


class LateFusionTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["agv01"] * 2 + ["oht01"] * 2
        self.md_dev = np.array([1.0, 2.0, 3.0, 4.0])
        self.th_dev = np.array([10.0, 20.0, 30.0, 40.0])
        self.late = ab.LateFusionBaseline.fit(self.md_dev, self.th_dev, self.ids)

    def test_combination_is_the_declared_equal_weighting(self):
        md_q = np.array([2.0, 4.0, 0.0])
        th_q = np.array([40.0, 10.0, 0.0])
        expected = 0.5 * self.late.ecdf_a.transform(md_q) + 0.5 * self.late.ecdf_b.transform(th_q)
        np.testing.assert_allclose(self.late.score(md_q, th_q), expected)
        self.assertEqual(self.late.weights, (0.5, 0.5))

    def test_score_is_bounded_to_the_unit_interval(self):
        s = self.late.score(np.array([-1e9, 1e9]), np.array([1e9, -1e9]))
        self.assertTrue(np.all((s >= 0.0) & (s <= 1.0)))
        np.testing.assert_allclose(s, [0.5, 0.5])

    def test_mismatched_component_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            self.late.score(np.array([1.0, 2.0]), np.array([1.0]))

    def test_weights_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            ab.LateFusionBaseline(ecdf_a=self.late.ecdf_a, ecdf_b=self.late.ecdf_b,
                                  weights=(0.6, 0.6))

    def test_round_trip(self):
        again = ab.LateFusionBaseline.from_dict(json.loads(json.dumps(self.late.to_dict())))
        q_a, q_b = np.array([2.5]), np.array([25.0])
        np.testing.assert_allclose(again.score(q_a, q_b), self.late.score(q_a, q_b))
        self.assertEqual(again.params_sha256(), self.late.params_sha256())


class NormalizerTests(unittest.TestCase):
    def test_sensor_normalizer_is_equal_equipment_weighted(self):
        ticks = np.zeros((4, 5))
        ticks[:, 0] = [0.0, 0.0, 0.0, 8.0]          # three on one machine, one on another
        p = ab.fit_run_normalizer(ticks, ["a", "a", "a", "b"])
        self.assertAlmostEqual(p["mean"][0], 4.0, places=12)   # (0 + 8) / 2, not 2.0
        self.assertEqual(p["role"], "fit")
        self.assertEqual(p["n_devices"], 2)

    def test_exactly_constant_channel_gives_exactly_zero_std_and_output(self):
        """Offset centring means an exact constant needs no tolerance to be detected."""
        ticks = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (6, 1))
        p = ab.fit_run_normalizer(ticks, ["a"] * 3 + ["b"] * 3)
        self.assertEqual(p["constant_channels"], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(p["std"], [0.0] * 5)
        np.testing.assert_allclose(p["scale"], 1.0)
        self.assertEqual(p["constant_rule"], "exact_ptp_zero_no_tolerance")
        out = ab.apply_sensor_normalizer(ticks, p)
        np.testing.assert_array_equal(out, np.zeros_like(out))   # exactly zero, not 1e-16

    def test_a_genuinely_tiny_spread_is_preserved_not_called_constant(self):
        """A real but small spread must keep its scale; a tolerance would erase it."""
        ticks = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (6, 1))
        ticks[3, 0] += 1e-13                      # far below any plausible rtol
        p = ab.fit_run_normalizer(ticks, ["a"] * 3 + ["b"] * 3)
        self.assertNotIn(0, p["constant_channels"], "a real spread was called constant")
        self.assertGreater(p["std"][0], 0.0)
        self.assertEqual(p["scale"][0], p["std"][0])
        self.assertEqual(p["constant_channels"], [1, 2, 3, 4])
        out = ab.apply_sensor_normalizer(ticks, p)
        self.assertGreater(np.abs(out[:, 0]).max(), 0.1, "the spread should be visible in z")

    def test_variance_underflow_is_an_error_not_a_silent_scale_one(self):
        """ptp > 0 with zero variance is underflow; substituting 1.0 would erase a spread."""
        ticks = np.tile(np.array([1e-170, 2.0, 3.0, 4.0, 5.0]), (4, 1))
        ticks[1, 0] = 2e-170          # difference 1e-170 is representable; its square is not
        self.assertGreater(np.ptp(ticks[:, 0]), 0.0)
        with self.assertRaisesRegex(ValueError, "underflow"):
            ab.fit_run_normalizer(ticks, ["a", "a", "b", "b"])

    def test_thermal_normalizer_is_a_single_global_pair(self):
        frames = np.zeros((2, 2, 2))
        frames[1] = 4.0
        p = ab.fit_thermal_normalizer(frames, ["a", "b"])
        self.assertAlmostEqual(p["mean"], 2.0, places=12)
        self.assertAlmostEqual(p["scale"], 2.0, places=12)
        self.assertFalse(p["constant"])
        np.testing.assert_allclose(ab.apply_thermal_normalizer(frames, p),
                                   np.where(frames == 0.0, -1.0, 1.0))

    def test_constant_thermal_is_flagged_with_exactly_zero_std(self):
        p = ab.fit_thermal_normalizer(np.full((3, 2, 2), 9.0), ["a", "b", "c"])
        self.assertTrue(p["constant"])
        self.assertEqual(p["std"], 0.0)
        self.assertEqual(p["scale"], 1.0)
        frames = np.full((3, 2, 2), 9.0)
        np.testing.assert_array_equal(ab.apply_thermal_normalizer(frames, p),
                                      np.zeros_like(frames))

    def test_thermal_variance_underflow_is_an_error(self):
        frames = np.full((2, 2, 2), 1e-170)
        frames[1, 1, 1] = 2e-170
        self.assertGreater(np.ptp(frames), 0.0)
        with self.assertRaisesRegex(ValueError, "underflow"):
            ab.fit_thermal_normalizer(frames, ["a", "b"])

    def test_helper_is_marked_as_not_the_protocol_normalizer(self):
        """v3 fits the raw normaliser on unique ticks; this helper weights by machine."""
        self.assertIn("Not the protocol normaliser", ab.fit_run_normalizer.__doc__)

    def test_select_primary_channels_from_an_eight_channel_normalizer(self):
        eight = {"kind": "sensor_zscore", "role": "fit",
                 "mean": list(range(8)), "scale": [1.0] * 8, "constant_channels": [5, 7]}
        five = ab.select_primary_channels(eight, [0, 1, 2, 3, 4])
        self.assertEqual(five["mean"], [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(five["constant_channels"], [])
        self.assertEqual(five["source_channel_count"], 8)
        # a primary slot that maps onto a constant source channel must stay flagged
        with_ct2 = ab.select_primary_channels(eight, [0, 1, 2, 3, 5])
        self.assertEqual(with_ct2["constant_channels"], [4])

    def test_select_primary_channels_rejects_bad_index_sets(self):
        eight = {"mean": list(range(8)), "scale": [1.0] * 8}
        with self.assertRaises(ValueError):
            ab.select_primary_channels(eight, [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            ab.select_primary_channels(eight, [0, 1, 2, 3, 99])
        with self.assertRaises(ValueError):
            ab.select_primary_channels(eight, [0, 0, 1, 2, 3])

    def test_non_finite_raw_input_is_rejected(self):
        ticks = np.zeros((3, 5))
        ticks[1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ab.fit_run_normalizer(ticks, ["a", "a", "b"])


class ManifestAndHashTests(unittest.TestCase):
    def test_manifest_carries_every_parameter_hash(self):
        x = sensor_windows(n=20, seed=9)
        ids = ["agv01"] * 10 + ["oht01"] * 10
        md = ab.MahalanobisBaseline.fit(x, ids)
        late = ab.LateFusionBaseline.fit(md.score(x), ab.score_b_ntc(x), ids)
        sn = ab.fit_run_normalizer(x.reshape(-1, 5), [d for d in ids for _ in range(30)])
        tn = ab.fit_thermal_normalizer(thermal_windows(n=2).reshape(2, -1, 7)[:, :6, :], ["a", "b"])
        m = ab.baseline_manifest(fold=0, sensor_normalizer=sn, thermal_normalizer=tn,
                                 md=md, late=late)
        self.assertEqual(m["gradient_training_runs"], 0)
        self.assertEqual(m["component_set"], ["B-NTC", "B-MD", "B-THERM", "B-LATE"])
        for key in ("sensor_normalizer_sha256", "thermal_normalizer_sha256",
                    "b_md_sha256", "b_late_sha256"):
            self.assertRegex(m[key], r"^[0-9a-f]{64}$")
        json.dumps(m)          # must be plain data

    def test_hash_is_stable_across_processes(self):
        """Python salts str hashing per process; the stored hash must not depend on that."""
        payload = json.dumps({"b": [1.0, 2.0], "a": "x"})
        code = (
            "import importlib.util,sys,json;"
            f"spec=importlib.util.spec_from_file_location('ab', {str(ROOT / 'src/models/anomaly_baselines.py')!r});"
            "m=importlib.util.module_from_spec(spec);sys.modules['ab']=m;spec.loader.exec_module(m);"
            f"print(m.params_sha256(json.loads({payload!r})))"
        )
        runs = set()
        for seed in ("0", "1"):
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin",
                                      "PYTHONNOUSERSITE": "1"})
            self.assertEqual(out.returncode, 0, out.stderr)
            runs.add(out.stdout.strip())
        self.assertEqual(len(runs), 1, f"hash differed across PYTHONHASHSEED: {runs}")

    def test_key_order_does_not_change_the_hash(self):
        self.assertEqual(ab.params_sha256({"a": 1, "b": 2}), ab.params_sha256({"b": 2, "a": 1}))

    def test_nan_in_parameters_is_refused(self):
        with self.assertRaises(ValueError):
            ab.params_sha256({"a": float("nan")})


if __name__ == "__main__":
    unittest.main()
