"""Fixture tests for the EXP-20260928-004 NQ and SYN threshold policies.

Loaded as a leaf so no torch is needed, matching `test_data_portability.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
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


cal = load_leaf("calibration_under_test", "src/evaluation/anomaly_calibration.py")
base = load_leaf("baselines_for_weight_check", "src/models/anomaly_baselines.py")


def synthetic_block(devices=("agv01", "oht01"), families=("sensor-only", "thermal-only", "coupled"),
                    strengths=(1, 2, 4), per_cell=2, start=100.0):
    """One complete (device, family, strength) grid, `per_cell` windows each."""
    dev, fam, stg, scores = [], [], [], []
    v = start
    for d in devices:
        for f in families:
            for s in strengths:
                for _ in range(per_cell):
                    dev.append(d); fam.append(f); stg.append(s); scores.append(v)
                    v += 1.0
    return np.array(scores), dev, fam, stg


class LabelBarrierTests(unittest.TestCase):
    def test_no_public_callable_accepts_a_label_argument(self):
        checked = cal.assert_no_label_arguments(cal)
        for expected in ("calibrate_nq", "calibrate_syn", "calibrate_with_sensitivity",
                         "stratified_cell_weights"):
            self.assertIn(expected, checked)

    def test_guard_fires_on_a_planted_violation(self):
        import types

        fake = types.ModuleType("fake_cal")

        def calibrate(scores, real_severity):
            return 0.0

        calibrate.__module__ = fake.__name__
        fake.calibrate = calibrate
        with self.assertRaisesRegex(AssertionError, "real_severity"):
            cal.assert_no_label_arguments(fake)


class RoleTests(unittest.TestCase):
    def test_only_calibration_may_fit_a_threshold(self):
        s, ids = np.array([1.0, 2.0, 3.0, 4.0]), ["a", "a", "b", "b"]
        sy, dv, fm, st = synthetic_block(per_cell=1)
        # the role check must fire before any grid check, so these ids need not align
        for role in ("fit", "dev", "heldout", "heldout-eval"):
            with self.assertRaisesRegex(ValueError, "role 'calibration'"):
                cal.calibrate_nq(s, ids, role=role)
            with self.assertRaisesRegex(ValueError, "role 'calibration'"):
                cal.calibrate_syn(s, ids, sy, dv, fm, st, role=role)


class WeightingTests(unittest.TestCase):
    def test_matches_the_baseline_module_implementation(self):
        """The duplicate exists only to avoid importing torch; it must not drift."""
        for ids in (["a"], ["a", "b"], ["a"] * 5 + ["b"], ["x", "y", "z"] * 3 + ["y"]):
            np.testing.assert_allclose(cal.equal_equipment_weights(ids),
                                       base.equal_equipment_weights(ids), rtol=0, atol=0)

    def test_cells_carry_equal_weight(self):
        sy, dv, fm, st = synthetic_block(per_cell=3)
        w, acc = cal.stratified_cell_weights(dv, fm, st, expected_device_ids=sorted(set(dv)))
        self.assertAlmostEqual(w.sum(), 1.0, places=12)
        self.assertEqual(acc["cells_expected"], 2 * 3 * 3)
        self.assertEqual(acc["cells_observed"], 2 * 3 * 3)
        self.assertEqual(acc["cells_missing"], [])
        self.assertFalse(acc["weighting_degraded_by_missing_cells"])
        keys = list(zip(dv, fm, st))
        for cell in set(keys):
            mask = np.array([k == cell for k in keys])
            self.assertAlmostEqual(w[mask].sum(), 1.0 / 18, places=12)

    def test_uneven_cell_sizes_still_balance_by_cell(self):
        dv = ["agv01"] * 4 + ["oht01"] * 1
        fm = ["sensor-only"] * 5
        st = [1] * 4 + [1]
        w, acc = cal.stratified_cell_weights(dv, fm, st, expected_device_ids=sorted(set(dv)),
                                             expected_families=["sensor-only"],
                                             expected_strengths=[1])
        self.assertEqual(acc["cells_observed"], 2)
        self.assertAlmostEqual(w[:4].sum(), 0.5, places=12)
        self.assertAlmostEqual(w[4], 0.5, places=12)
        self.assertEqual(acc["windows_per_cell"], {"min": 1, "max": 4})

    def test_a_missing_cell_is_an_error_by_default(self):
        sy, dv, fm, st = synthetic_block(per_cell=1)
        drop = [i for i, (f, s) in enumerate(zip(fm, st)) if (f, s) != ("coupled", 4)]
        with self.assertRaisesRegex(ValueError, "produced no"):
            cal.stratified_cell_weights([dv[i] for i in drop], [fm[i] for i in drop],
                                        [st[i] for i in drop], expected_device_ids=sorted(set(dv)))

    def test_a_missing_cell_can_be_accepted_but_is_recorded(self):
        sy, dv, fm, st = synthetic_block(per_cell=1)
        drop = [i for i, (f, s) in enumerate(zip(fm, st)) if (f, s) != ("coupled", 4)]
        w, acc = cal.stratified_cell_weights([dv[i] for i in drop], [fm[i] for i in drop],
                                             [st[i] for i in drop],
                                             expected_device_ids=sorted(set(dv)),
                                             on_missing_cells="report")
        self.assertTrue(acc["weighting_degraded_by_missing_cells"])
        self.assertEqual(len(acc["cells_missing"]), 2)          # one per machine
        self.assertAlmostEqual(w.sum(), 1.0, places=12)

    def test_a_whole_family_going_missing_is_detected(self):
        """An inferred grid would not contain thermal-only either, and would say nothing."""
        sy, dv, fm, st = synthetic_block(families=("sensor-only", "coupled"), per_cell=2)
        with self.assertRaisesRegex(ValueError, r"families \['thermal-only'\]"):
            cal.stratified_cell_weights(dv, fm, st, expected_device_ids=sorted(set(dv)))
        w, acc = cal.stratified_cell_weights(dv, fm, st, expected_device_ids=sorted(set(dv)),
                                             on_missing_cells="report")
        self.assertEqual(acc["entirely_absent_families"], ["thermal-only"])
        self.assertEqual(acc["cells_expected"], 2 * 3 * 3)
        self.assertEqual(acc["cells_observed"], 2 * 2 * 3)

    def test_a_whole_strength_going_missing_is_detected(self):
        sy, dv, fm, st = synthetic_block(strengths=(1, 2), per_cell=2)
        with self.assertRaisesRegex(ValueError, r"strengths \[4\]"):
            cal.stratified_cell_weights(dv, fm, st, expected_device_ids=sorted(set(dv)))
        w, acc = cal.stratified_cell_weights(dv, fm, st, expected_device_ids=sorted(set(dv)),
                                             on_missing_cells="report")
        self.assertEqual(acc["entirely_absent_strengths"], [4])

    def test_a_whole_machine_going_missing_is_detected(self):
        """Only possible because the machine axis is declared from calibration-Normal."""
        sy, dv, fm, st = synthetic_block(devices=("agv01",), per_cell=2)
        with self.assertRaisesRegex(ValueError, r"machines \['oht01'\]"):
            cal.stratified_cell_weights(dv, fm, st, expected_device_ids=["agv01", "oht01"])
        w, acc = cal.stratified_cell_weights(dv, fm, st,
                                             expected_device_ids=["agv01", "oht01"],
                                             on_missing_cells="report")
        self.assertEqual(acc["entirely_absent_devices"], ["oht01"])

    def test_calibrate_syn_takes_its_machine_axis_from_calibration_normal(self):
        """A machine present in Normal but absent from the bank must not slip through."""
        sn = np.arange(20, dtype=np.float64)
        nids = ["agv01"] * 10 + ["oht01"] * 10
        sy, dv, fm, st = synthetic_block(devices=("agv01",), per_cell=1, start=100.0)
        with self.assertRaisesRegex(ValueError, r"machines \['oht01'\]"):
            cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=0.05)

    def test_an_unknown_family_or_strength_is_refused_not_absorbed(self):
        sy, dv, fm, st = synthetic_block(per_cell=1)
        bad_fam = list(fm); bad_fam[0] = "made-up-family"
        with self.assertRaisesRegex(ValueError, "outside the declared grid"):
            cal.stratified_cell_weights(dv, bad_fam, st, expected_device_ids=sorted(set(dv)))
        bad_stg = list(st); bad_stg[0] = 3
        with self.assertRaisesRegex(ValueError, "outside the declared grid"):
            cal.stratified_cell_weights(dv, fm, bad_stg, expected_device_ids=sorted(set(dv)))

    def test_a_non_integer_strength_is_refused_not_truncated(self):
        sy, dv, fm, st = synthetic_block(per_cell=1)
        bad = list(st); bad[0] = 1.9
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            cal.stratified_cell_weights(dv, fm, bad, expected_device_ids=sorted(set(dv)))
        worse = list(st); worse[0] = True
        with self.assertRaisesRegex(ValueError, "boolean"):
            cal.stratified_cell_weights(dv, fm, worse, expected_device_ids=sorted(set(dv)))

    def test_the_expected_axes_are_recorded_as_declared(self):
        sy, dv, fm, st = synthetic_block(per_cell=1)
        _, acc = cal.stratified_cell_weights(dv, fm, st, expected_device_ids=sorted(set(dv)))
        self.assertEqual(acc["expected_axes_source"], "declared_not_inferred")
        self.assertEqual(acc["expected_families"], list(cal.V3_FAMILIES))
        self.assertEqual(acc["expected_strengths"], list(cal.V3_STRENGTHS))

    def test_mismatched_descriptor_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagree in length"):
            cal.stratified_cell_weights(["a", "b"], ["sensor-only"], [1, 1],
                                        expected_device_ids=["a", "b"])


class NqTests(unittest.TestCase):
    def test_exceedance_never_passes_the_budget(self):
        # one machine, 100 distinct scores: the weighted exceedance steps by 0.01
        s = np.arange(100, dtype=np.float64)
        ids = ["a"] * 100
        r = cal.calibrate_nq(s, ids, alpha=0.01)
        self.assertLessEqual(r["achieved_normal_fpr"], 0.01 + 1e-12)
        self.assertAlmostEqual(r["tau"], 98.0)                  # only score 99 exceeds -> 0.01
        self.assertAlmostEqual(r["achieved_normal_fpr"], 0.01, places=12)

    def test_the_chosen_tau_is_the_smallest_feasible_one(self):
        s = np.arange(100, dtype=np.float64)
        ids = ["a"] * 100
        r = cal.calibrate_nq(s, ids, alpha=0.01)
        below = r["tau"] - 1.0
        self.assertGreater(cal.weighted_rate_above(s, cal.equal_equipment_weights(ids), below),
                           0.01, "a smaller tau must break the constraint")

    def test_ties_on_the_boundary_do_not_overshoot(self):
        """Five windows all at the top: an interpolated quantile would allow 5 exceedances."""
        s = np.array([0.0, 1.0, 2.0] + [9.0] * 5)
        ids = ["a"] * 8
        r = cal.calibrate_nq(s, ids, alpha=0.1)
        self.assertEqual(r["tau"], 9.0)                         # not 8.x
        self.assertEqual(r["achieved_normal_fpr"], 0.0)
        self.assertTrue(r["tau_is_at_normal_max"])

    def test_equal_equipment_weighting_is_used(self):
        """A machine with many Normal windows must not set the budget on its own."""
        s = np.concatenate([np.zeros(99), np.array([5.0])])
        ids = ["busy"] * 99 + ["quiet"]
        r = cal.calibrate_nq(s, ids, alpha=0.4)
        # 'quiet' holds half the weight on its single score of 5.0, so excluding it costs 0.5
        self.assertEqual(r["tau"], 5.0)
        self.assertEqual(r["achieved_normal_fpr"], 0.0)

    def test_degenerate_identical_scores(self):
        r = cal.calibrate_nq(np.full(6, 3.0), ["a"] * 3 + ["b"] * 3, alpha=0.01)
        self.assertTrue(r["degenerate_all_normal_scores_equal"])
        self.assertEqual(r["tau"], 3.0)
        self.assertEqual(r["achieved_normal_fpr"], 0.0)

    def test_synthetic_plays_no_part(self):
        r = cal.calibrate_nq(np.arange(10.0), ["a"] * 10, alpha=0.2)
        self.assertFalse(r["synthetic_used"])

    def test_alpha_and_input_validation(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                cal.calibrate_nq(np.arange(5.0), ["a"] * 5, alpha=bad)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            cal.calibrate_nq(np.array([1.0, np.nan]), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "empty"):
            cal.calibrate_nq(np.array([]), [])
        with self.assertRaises(ValueError):
            cal.calibrate_nq(np.arange(5.0), ["a"] * 4)


class SynTests(unittest.TestCase):
    def setUp(self):
        self.sn = np.arange(100, dtype=np.float64)
        self.nids = ["agv01"] * 50 + ["oht01"] * 50      # must match the synthetic machines

    def test_separable_case_detects_everything_within_the_budget(self):
        sy, dv, fm, st = synthetic_block(per_cell=2, start=200.0)
        r = cal.calibrate_syn(self.sn, self.nids, sy, dv, fm, st, alpha=0.01)
        self.assertAlmostEqual(r["achieved_synthetic_tpr"], 1.0, places=12)
        self.assertLessEqual(r["achieved_normal_fpr"], 0.01 + 1e-12)
        self.assertAlmostEqual(r["objective_value"], 0.5 * r["achieved_normal_fpr"], places=12)
        self.assertFalse(r["degenerate_always_normal"])

    def test_the_fpr_constraint_is_respected_even_when_it_costs_detection(self):
        """Synthetic hidden inside the Normal range: detection must not buy FPR."""
        sy, dv, fm, st = synthetic_block(per_cell=1, start=10.0)   # inside 0..99
        r = cal.calibrate_syn(self.sn, self.nids, sy, dv, fm, st, alpha=0.01)
        self.assertLessEqual(r["achieved_normal_fpr"], 0.01 + 1e-12)
        self.assertLess(r["achieved_synthetic_tpr"], 1.0)

    def test_always_normal_degeneracy_is_named(self):
        """Every synthetic score below the whole Normal range: the optimum flags nothing."""
        sy, dv, fm, st = synthetic_block(per_cell=1, start=-500.0)
        r = cal.calibrate_syn(self.sn, self.nids, sy, dv, fm, st, alpha=0.01)
        self.assertEqual(r["achieved_synthetic_tpr"], 0.0)
        self.assertTrue(r["degenerate_always_normal"])
        self.assertTrue(r["tau_is_at_normal_max"])

    def test_equal_objective_resolves_to_the_lower_fpr(self):
        """The first tiebreak, exercised on a case where two taus score identically."""
        sn = np.array([0.0, 1.0])
        nids = ["a", "a"]                                  # weights 0.5, 0.5
        sy = np.array([0.5, 2.0])
        dv, fm, st = ["a", "a"], ["sensor-only"] * 2, [1, 1]
        # tau=0 -> FPR .5, TPR 1.0, objective .25
        # tau=1 -> FPR  0, TPR  .5, objective .25   <- same objective, lower FPR
        # one family and one strength only, so the incomplete grid is accepted explicitly
        r = cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=0.6, on_missing_cells="report")
        self.assertAlmostEqual(r["objective_value"], 0.25, places=12)
        self.assertEqual(r["tau"], 1.0)
        self.assertEqual(r["achieved_normal_fpr"], 0.0)
        self.assertGreater(r["tied_at_optimum"], 1)

    def test_only_observed_scores_are_candidates_so_tau_lands_on_one(self):
        """No interpolation: the reported tau is always an observed score."""
        sn = np.array([0.0, 1.0, 2.0, 3.0])
        nids = ["a", "a", "b", "b"]
        sy = np.array([10.0, 10.0])
        dv, fm, st = ["a", "b"], ["sensor-only"] * 2, [1, 1]
        r = cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=0.01,
                              on_missing_cells="report")
        self.assertIn(r["tau"], set(np.concatenate([sn, sy]).tolist()))
        self.assertEqual(r["tau"], 3.0)                    # FPR 0 and TPR 1 together
        self.assertEqual(r["achieved_normal_fpr"], 0.0)
        self.assertEqual(r["achieved_synthetic_tpr"], 1.0)
        self.assertEqual(r["objective_value"], 0.0)

    def test_objective_matches_its_definition_at_the_chosen_tau(self):
        sy, dv, fm, st = synthetic_block(per_cell=2, start=50.0)
        r = cal.calibrate_syn(self.sn, self.nids, sy, dv, fm, st, alpha=0.05)
        expected = 0.5 * r["achieved_normal_fpr"] + 0.5 * (1.0 - r["achieved_synthetic_tpr"])
        self.assertAlmostEqual(r["objective_value"], expected, places=12)
        self.assertEqual(r["objective"], "0.5*FPR_normal+0.5*(1-TPR_synthetic)")

    def test_no_candidate_below_the_optimum_does_better(self):
        """Exhaustiveness: nothing feasible beats the reported objective."""
        sy, dv, fm, st = synthetic_block(per_cell=2, start=40.0)
        r = cal.calibrate_syn(self.sn, self.nids, sy, dv, fm, st, alpha=0.02)
        wn = cal.equal_equipment_weights(self.nids)
        wy, _ = cal.stratified_cell_weights(dv, fm, st, expected_device_ids=sorted(set(dv)))
        for tau in np.unique(np.concatenate([self.sn, sy])):
            fpr = cal.weighted_rate_above(self.sn, wn, tau)
            if fpr > 0.02:
                continue
            tpr = cal.weighted_rate_above(sy, wy, tau)
            self.assertLessEqual(r["objective_value"] - 1e-12, 0.5 * fpr + 0.5 * (1 - tpr))

    def test_per_stratum_detection_is_reported(self):
        sy, dv, fm, st = synthetic_block(per_cell=2, start=200.0)
        r = cal.calibrate_syn(self.sn, self.nids, sy, dv, fm, st, alpha=0.01)
        strata = r["synthetic_tpr_by_stratum"]
        self.assertEqual(len(strata), 3 * 3)
        for key, entry in strata.items():
            self.assertAlmostEqual(entry["tpr"], 1.0, places=12)
            self.assertEqual(entry["windows"], 4)
            self.assertEqual(entry["devices"], 2)

    def test_per_stratum_reveals_a_bank_detected_through_one_family_only(self):
        sy, dv, fm, st = synthetic_block(per_cell=2, start=0.0)
        sy = np.array(sy)
        sy[:] = 10.0                                  # inside the Normal range
        for i, f in enumerate(fm):
            if f == "coupled":
                sy[i] = 500.0                         # only this family is separable
        r = cal.calibrate_syn(self.sn, self.nids, sy, dv, fm, st, alpha=0.01)
        strata = r["synthetic_tpr_by_stratum"]
        self.assertTrue(all(v["tpr"] == 1.0 for k, v in strata.items() if k.startswith("coupled")))
        self.assertTrue(all(v["tpr"] == 0.0 for k, v in strata.items()
                            if not k.startswith("coupled")))
        self.assertAlmostEqual(r["achieved_synthetic_tpr"], 1.0 / 3, places=12)

    def test_missing_cells_reach_the_caller_from_calibrate_syn(self):
        sy, dv, fm, st = synthetic_block(per_cell=1)
        keep = [i for i, (f, s) in enumerate(zip(fm, st)) if (f, s) != ("coupled", 4)]
        args = (self.sn, self.nids, np.array(sy)[keep], [dv[i] for i in keep],
                [fm[i] for i in keep], [st[i] for i in keep])
        with self.assertRaisesRegex(ValueError, "produced no"):
            cal.calibrate_syn(*args, alpha=0.01)
        r = cal.calibrate_syn(*args, alpha=0.01, on_missing_cells="report")
        self.assertTrue(r["synthetic"]["weighting_degraded_by_missing_cells"])


class NqVersusSynTests(unittest.TestCase):
    def test_a_separable_bank_lets_syn_raise_tau_above_nq_for_free(self):
        """This is the whole point of SYN: the synthetic shows the budget can be unspent.

        NQ must spend its full alpha because it only sees Normal scores. SYN sees that
        every synthetic window sits far above the Normal range, so it can push tau to the
        Normal maximum, reaching FPR 0 while keeping detection at 1.
        """
        sn = np.arange(100, dtype=np.float64)
        nids = ["agv01"] * 50 + ["oht01"] * 50
        sy, dv, fm, st = synthetic_block(per_cell=2, start=1000.0)
        out = cal.calibrate_with_sensitivity(sn, nids, sy, dv, fm, st, alpha=0.01)
        nq, syn = out["primary"]["NQ"], out["primary"]["SYN"]
        self.assertEqual(nq["tau"], 98.0)
        self.assertAlmostEqual(nq["achieved_normal_fpr"], 0.01, places=12)
        self.assertEqual(syn["tau"], 99.0)
        self.assertEqual(syn["achieved_normal_fpr"], 0.0)
        self.assertEqual(syn["achieved_synthetic_tpr"], 1.0)
        self.assertGreater(syn["tau"], nq["tau"])
        self.assertFalse(out["nq_equals_syn"])

    def test_an_overlapping_bank_leaves_syn_at_the_nq_point(self):
        """When raising tau costs detection, SYN spends the budget and the two coincide."""
        sn = np.arange(100, dtype=np.float64)
        nids = ["agv01"] * 50 + ["oht01"] * 50
        sy, dv, fm, st = synthetic_block(per_cell=1, start=95.0)   # overlaps the Normal tail
        out = cal.calibrate_with_sensitivity(sn, nids, sy, dv, fm, st, alpha=0.01)
        nq, syn = out["primary"]["NQ"], out["primary"]["SYN"]
        self.assertEqual(syn["tau"], nq["tau"])
        self.assertTrue(out["nq_equals_syn"])
        self.assertLess(syn["achieved_synthetic_tpr"], 1.0)
        self.assertAlmostEqual(syn["achieved_normal_fpr"], 0.01, places=12)

    def test_sensitivity_covers_the_planned_alphas_and_is_monotone(self):
        sn = np.arange(100, dtype=np.float64)
        nids = ["agv01"] * 50 + ["oht01"] * 50
        sy, dv, fm, st = synthetic_block(per_cell=1, start=90.0)
        out = cal.calibrate_with_sensitivity(sn, nids, sy, dv, fm, st)
        self.assertEqual([e["alpha"] for e in out["sensitivity"]], [0.005, 0.02])
        taus = {e["alpha"]: e["NQ"]["tau"] for e in out["sensitivity"]}
        taus[0.01] = out["primary"]["NQ"]["tau"]
        # a looser budget can only lower or hold the threshold
        self.assertGreaterEqual(taus[0.005], taus[0.01])
        self.assertGreaterEqual(taus[0.01], taus[0.02])

    def test_result_records_that_no_real_label_was_used(self):
        sn = np.arange(20.0)
        sy, dv, fm, st = synthetic_block(per_cell=1, start=100.0)
        out = cal.calibrate_with_sensitivity(sn, ["agv01"] * 10 + ["oht01"] * 10,
                                             sy, dv, fm, st)
        self.assertFalse(out["real_labels_used"])
        self.assertFalse(out["selection_used_heldout"])
        self.assertRegex(out["sha256"], r"^[0-9a-f]{64}$")


class SynIsNeverMoreSensitiveThanNqTests(unittest.TestCase):
    """A property of the two rules, provable without knowing any distribution.

    NQ is the smallest tau whose weighted Normal exceedance is at most alpha. For any real
    u lying between two consecutive Normal scores, the exceedance equals the exceedance at
    the lower one, so the smallest feasible tau over all reals is exactly tau_NQ. SYN
    optimises over candidates drawn from the same feasibility rule, so

        tau_SYN >= tau_NQ                       always, at the same alpha and weights

    and therefore, on ANY evaluation scores, `{s > tau_SYN}` is a subset of `{s > tau_NQ}`:
    SYN can only lower both recall and the false alarm rate, never raise either. Q-C is
    therefore a trade-off measurement, not a question of whether calibration raises recall
    on a fixed checkpoint.
    """

    def _bank(self, seed):
        rng = np.random.default_rng(seed)
        n_dev = int(rng.integers(2, 5))
        devices = [f"m{i:02d}" for i in range(n_dev)]
        nids, sn = [], []
        for d in devices:
            k = int(rng.integers(5, 40))
            nids += [d] * k
            sn.append(rng.normal(size=k) * float(rng.uniform(0.5, 3.0)))
        sn = np.concatenate(sn)
        shift = float(rng.uniform(-1.0, 4.0))          # sometimes overlapping, sometimes not
        dv, fm, st, sy = [], [], [], []
        for d in devices:
            for f in cal.V3_FAMILIES:
                for g in cal.V3_STRENGTHS:
                    k = int(rng.integers(1, 4))
                    dv += [d] * k
                    fm += [f] * k
                    st += [g] * k
                    sy.append(rng.normal(size=k) + shift + 0.3 * g)
        return sn, nids, np.concatenate(sy), dv, fm, st

    def test_tau_ordering_and_subset_hold_over_many_random_banks(self):
        rng = np.random.default_rng(12345)
        for seed in range(30):
            sn, nids, sy, dv, fm, st = self._bank(seed)
            for alpha in (0.005, 0.01, 0.02, 0.1):
                nq = cal.calibrate_nq(sn, nids, alpha=alpha)
                syn = cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=alpha)
                with self.subTest(seed=seed, alpha=alpha):
                    self.assertGreaterEqual(syn["tau"], nq["tau"],
                                            "tau_SYN < tau_NQ contradicts the feasibility argument")
                    # the positive set must nest on arbitrary third-party scores
                    probe = rng.normal(size=400) * 3.0
                    pos_nq = cal.decide(probe, nq["tau"])
                    pos_syn = cal.decide(probe, syn["tau"])
                    self.assertTrue(np.all(pos_syn <= pos_nq),
                                    "SYN flagged something NQ did not")
                    # and both rates can only fall
                    self.assertLessEqual(syn["achieved_normal_fpr"], nq["achieved_normal_fpr"] + 1e-15)
                    wy, _ = cal.stratified_cell_weights(dv, fm, st,
                                                        expected_device_ids=sorted(set(nids)))
                    tpr_nq = cal.weighted_rate_above(sy, wy, nq["tau"])
                    self.assertLessEqual(syn["achieved_synthetic_tpr"], tpr_nq + 1e-15)

    def test_the_ordering_is_not_vacuous(self):
        """At least some banks must make SYN strictly more conservative than NQ."""
        strict = 0
        for seed in range(30):
            sn, nids, sy, dv, fm, st = self._bank(seed)
            nq = cal.calibrate_nq(sn, nids, alpha=0.01)
            syn = cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=0.01)
            strict += syn["tau"] > nq["tau"]
        self.assertGreater(strict, 0, "if SYN never differs, the property test proves nothing")

    def test_a_synthetic_candidate_below_tau_nq_is_never_feasible(self):
        """The step that makes the enlarged candidate set safe."""
        sn = np.array([0.0, 10.0])
        nids = ["a", "b"]                                   # weights 0.5, 0.5
        wn = cal.equal_equipment_weights(nids)
        nq = cal.calibrate_nq(sn, nids, alpha=0.4)
        self.assertEqual(nq["tau"], 10.0)
        for u in (-1.0, 0.0, 5.0, 9.999):                   # every candidate below tau_NQ
            self.assertGreater(cal.weighted_rate_above(sn, wn, u), 0.4)


class CandidatePruningTests(unittest.TestCase):
    """Pruning below tau_NQ must be exact, not merely close.

    It rests on the same fact as the tau ordering: the weighted Normal exceedance is
    constant between consecutive Normal scores, so nothing below tau_NQ is feasible. The
    surviving candidates go through identical arithmetic in identical order, so every
    reported number must match the exhaustive search bit for bit.
    """

    KEYS = ("tau", "achieved_normal_fpr", "achieved_synthetic_tpr", "objective_value",
            "candidates_feasible", "tied_at_optimum", "degenerate_always_normal")

    def _assert_identical(self, a, b, msg=""):
        for key in self.KEYS:
            self.assertEqual(a[key], b[key], f"{key} differs{msg}")
        self.assertEqual(a["synthetic_tpr_by_stratum"], b["synthetic_tpr_by_stratum"], msg)

    def _bank(self, seed, jitter=None):
        rng = np.random.default_rng(seed)
        devices = ["agv01", "agv05", "oht01", "oht05"]
        nids, sn = [], []
        for d in devices:
            k = int(rng.integers(20, 60))
            nids += [d] * k
            sn.append(rng.normal(size=k))
        sn = np.concatenate(sn)
        dv, fm, st, sy = [], [], [], []
        shift = float(rng.uniform(-0.5, 3.0))
        for d in devices:
            for f in cal.V3_FAMILIES:
                for g in cal.V3_STRENGTHS:
                    k = int(rng.integers(2, 8))
                    dv += [d] * k
                    fm += [f] * k
                    st += [g] * k
                    sy.append(rng.normal(size=k) + shift)
        sy = np.concatenate(sy)
        if jitter == "ties":
            sn = np.round(sn, 1)          # force many exact ties on both sides
            sy = np.round(sy, 1)
        elif jitter == "one_ulp":
            sn = np.where(np.arange(len(sn)) % 2 == 0, 1.0, np.nextafter(1.0, 2.0))
            sy = np.where(np.arange(len(sy)) % 3 == 0, 1.0,
                          np.nextafter(np.nextafter(1.0, 2.0), 2.0))
        return sn, nids, sy, dv, fm, st

    def test_identical_to_the_exhaustive_search_over_random_banks(self):
        for seed in range(20):
            sn, nids, sy, dv, fm, st = self._bank(seed)
            for alpha in (0.005, 0.01, 0.02, 0.1, 0.5):
                with self.subTest(seed=seed, alpha=alpha):
                    fast = cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=alpha,
                                             candidate_pruning=True)
                    slow = cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=alpha,
                                             candidate_pruning=False)
                    self._assert_identical(fast, slow, f" (seed {seed}, alpha {alpha})")

    def test_identical_when_scores_are_heavily_tied(self):
        for seed in range(10):
            sn, nids, sy, dv, fm, st = self._bank(seed, jitter="ties")
            for alpha in (0.01, 0.1):
                with self.subTest(seed=seed, alpha=alpha):
                    self._assert_identical(
                        cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=alpha, candidate_pruning=True),
                        cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=alpha, candidate_pruning=False))

    def test_identical_when_scores_differ_by_one_ulp(self):
        """The case where a float tie in the objective could actually arise."""
        for alpha in (0.01, 0.2, 0.6):
            sn, nids, sy, dv, fm, st = self._bank(3, jitter="one_ulp")
            with self.subTest(alpha=alpha):
                self._assert_identical(
                    cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=alpha, candidate_pruning=True),
                    cal.calibrate_syn(sn, nids, sy, dv, fm, st, alpha=alpha, candidate_pruning=False))

    def test_pruning_accounting_is_honest(self):
        sn, nids, sy, dv, fm, st = self._bank(1)
        fast = cal.calibrate_syn(sn, nids, sy, dv, fm, st, candidate_pruning=True)
        slow = cal.calibrate_syn(sn, nids, sy, dv, fm, st, candidate_pruning=False)
        self.assertTrue(fast["candidate_pruning"])
        self.assertFalse(slow["candidate_pruning"])
        self.assertEqual(fast["candidates_total"], slow["candidates_total"])
        self.assertEqual(slow["candidates_evaluated"], slow["candidates_total"])
        self.assertEqual(fast["candidates_evaluated"] + fast["candidates_pruned_below_tau_nq"],
                         fast["candidates_total"])
        self.assertLess(fast["candidates_evaluated"], fast["candidates_total"],
                        "this fixture should give pruning something to do")
        self.assertIsNone(slow["pruning_floor_tau_nq"])
        nq = cal.calibrate_nq(sn, nids)
        self.assertEqual(fast["pruning_floor_tau_nq"], nq["tau"])
        self.assertGreaterEqual(fast["tau"], fast["pruning_floor_tau_nq"])

    def test_sensitivity_wrapper_threads_the_flag(self):
        sn, nids, sy, dv, fm, st = self._bank(2)
        fast = cal.calibrate_with_sensitivity(sn, nids, sy, dv, fm, st, candidate_pruning=True)
        slow = cal.calibrate_with_sensitivity(sn, nids, sy, dv, fm, st, candidate_pruning=False)
        for a, b in [(fast["primary"], slow["primary"])] + list(zip(fast["sensitivity"],
                                                                   slow["sensitivity"])):
            self.assertEqual(a["SYN"]["tau"], b["SYN"]["tau"])
            self.assertEqual(a["NQ"]["tau"], b["NQ"]["tau"])


class ComparatorTests(unittest.TestCase):
    def test_decision_is_strictly_greater_than_tau(self):
        np.testing.assert_array_equal(cal.decide([0.9, 1.0, 1.1], 1.0),
                                      [False, False, True])

    def test_weighted_rate_above_is_strict_too(self):
        s = np.array([1.0, 2.0, 2.0, 3.0])
        w = cal.equal_equipment_weights(["a"] * 4)
        self.assertAlmostEqual(cal.weighted_rate_above(s, w, 2.0), 0.25)
        self.assertAlmostEqual(cal.weighted_rate_above(s, w, 1.999), 0.75)


class HashTests(unittest.TestCase):
    def test_hash_is_present_and_changes_with_the_threshold(self):
        a = cal.calibrate_nq(np.arange(50.0), ["a"] * 50, alpha=0.02)
        b = cal.calibrate_nq(np.arange(50.0), ["a"] * 50, alpha=0.1)
        self.assertRegex(a["sha256"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.assertNotEqual(a["tau"], b["tau"])


if __name__ == "__main__":
    unittest.main()
