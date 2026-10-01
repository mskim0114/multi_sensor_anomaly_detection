"""BIN exposure, split admission and dev weighting on artificial metadata only."""
from collections import Counter
import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_bin_sampler_under_test",
                                            ROOT / "src/data/anomaly_bin_sampler.py")
bs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bs
spec.loader.exec_module(bs)

NORMALIZER = "a" * 64
FIT_IDS = "b" * 64
ROLES = {"agv01": "fit", "oht01": "fit", "agv02": "dev", "oht02": "dev",
         "agv03": "calibration", "oht03": "heldout"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fixture(role="fit", counts=(6, 7, 8, 9), ct="CT1", fold=0, generator_seed=42,
            devices=None):
    if devices is None:
        devices = ("agv01", "oht01") if role == "fit" else ("agv02", "oht02")
    windows, events = [], []
    # With four bases each device has two; with three, the first has two.
    for i, count in enumerate(counts):
        device = devices[i % len(devices)]
        identifier = f"Training/{device}_0901_0900{i:02d}:30"
        windows.append({"base_window_id": identifier, "device": device,
                        "hard_valid": True, "pure_Normal": True, "base_QC_flags": []})
        for j, (family, strength) in enumerate((f, s) for f in
                                             ("sensor-only", "thermal-only", "coupled")
                                             for s in (1, 2, 4)):
            namespace = ["synthetic-modality-v3-proposed", fold, role,
                         identifier, family, strength, generator_seed]
            sensor, thermal = {"sensor-only": (True, False),
                               "thermal-only": (False, True),
                               "coupled": (True, True)}[family]
            events.append({"base_window_id": identifier, "device": device, "role": role,
                           "family": family, "strength": strength, "namespace": namespace,
                           "event_id": digest(namespace), "ct_channel": ct,
                           "normalizer_sha256": NORMALIZER, "fit_ids_sha256": FIT_IDS,
                           "accepted": j < count, "sensor_changed": sensor,
                           "thermal_changed": thermal,
                           "rejection_reasons": [] if j < count else ["fixture_constraint"]})
    return windows, events


def build(windows=None, events=None, *, role="fit", roles=None, **kwargs):
    if windows is None:
        windows, events = fixture(role=role)
    options = dict(fold=0, generator_seed=42, ct_channel="CT1",
                   normalizer_sha256=NORMALIZER, fit_ids_sha256=FIT_IDS)
    options.update(kwargs)
    return bs.build_event_index(windows, ROLES if roles is None else roles, events,
                                role=role, **options)


class FitScheduleTests(unittest.TestCase):
    def test_each_epoch_has_one_normal_and_one_accepted_per_base(self):
        index = build()
        sampler = bs.BasePairedSampler(index, training_seed=42)
        expected = {b.base_window_id: {e.event_id for e in b.events} for b in index.bases}
        for epoch in range(1, 31):
            pairs = sampler.epoch_pairs(epoch)
            self.assertEqual(Counter(p.base_window_id for p in pairs),
                             Counter({identifier: 1 for identifier in expected}))
            for pair in pairs:
                self.assertIn(pair.event_id, expected[pair.base_window_id])
                self.assertEqual(index.selected_window_ids[pair.window_index], pair.base_window_id)

    def test_without_replacement_per_cycle_and_30epoch_exposure(self):
        index = build()
        schedule = [bs.BasePairedSampler(index, 42).epoch_pairs(e) for e in range(1, 31)]
        counts = Counter(p.event_id for pairs in schedule for p in pairs)
        for base in index.bases:
            k = len(base.events)
            sequence = [next(p.event_id for p in pairs if p.base_window_id == base.base_window_id)
                        for pairs in schedule]
            for start in range(0, 30, k):
                self.assertEqual(len(set(sequence[start:start + k])), len(sequence[start:start + k]))
            q, r = divmod(30, k)
            self.assertEqual(Counter(counts[e.event_id] for e in base.events),
                             Counter({q: k - r, q + 1: r}) if r else Counter({q: k}))

    def test_direct_restart_out_of_order_and_global_rng_independence(self):
        index = build()
        sampler = bs.BasePairedSampler(index, 42)
        expected = sampler.epoch_manifest(30)
        for e in (30, 1, 4, 14, 3, 30):
            sampler.epoch_pairs(e)
        np.random.seed(999)
        np.random.random(100)
        self.assertEqual(sampler.epoch_manifest(30), expected)
        self.assertEqual(bs.BasePairedSampler(index, 42).epoch_manifest(30), expected)

    def test_input_order_and_later_input_mutation_do_not_change_schedule(self):
        windows, events = fixture()
        index = build(windows, events)
        other = build(windows, list(reversed(events)))
        self.assertEqual(index.sha256, other.sha256)
        original = bs.BasePairedSampler(index, 42).epoch_manifest(17)
        windows[0]["base_QC_flags"].append("changed")
        events[0]["accepted"] = False
        self.assertEqual(bs.BasePairedSampler(index, 42).epoch_manifest(17), original)
        self.assertEqual(bs.BasePairedSampler(other, 42).epoch_manifest(17), original)

    def test_window_reordering_changes_binding_positions_not_semantic_schedule(self):
        windows, events = fixture()
        a, b = build(windows, events), build(windows[::-1], events)
        self.assertNotEqual(a.sha256, b.sha256)
        for epoch in (1, 9, 30):
            pa, pb = [bs.BasePairedSampler(index, 42).epoch_pairs(epoch) for index in (a, b)]
            self.assertEqual([(p.base_window_id, p.event_id) for p in pa],
                             [(p.base_window_id, p.event_id) for p in pb])
            for pair in pb:
                self.assertEqual(windows[::-1][pair.window_index]["base_window_id"], pair.base_window_id)

    def test_all_three_arms_share_the_full_positive_schedule(self):
        index = build()
        arms = {arm: bs.BasePairedSampler(index, training_seed=123).epoch_manifest(1)
                for arm in ("S", "T", "F")}
        self.assertEqual(len({json.dumps(record, sort_keys=True) for record in arms.values()}), 1)
        refs = {e.event_id: e for b in index.bases for e in b.events}
        positive_ids = {p.event_id for epoch in range(1, 10)
                        for p in bs.BasePairedSampler(index, 42).epoch_pairs(epoch)}
        self.assertTrue(any(not refs[i].sensor_changed for i in positive_ids))
        self.assertTrue(any(not refs[i].thermal_changed for i in positive_ids))

    def test_paired_partial_batch_has_no_dropped_or_split_pair(self):
        sampler = bs.BasePairedSampler(build(), 42, paired_bases_per_batch=3)
        batches = sampler.epoch_batches(4)
        self.assertEqual([len(b) for b in batches], [3, 1])
        self.assertEqual(tuple(p for batch in batches for p in batch), sampler.epoch_pairs(4))
        record = sampler.epoch_manifest(4)
        self.assertEqual(record["optimizer_steps"], 2)
        self.assertEqual(record["normal_examples"], 4)
        self.assertEqual(record["synthetic_examples"], 4)
        self.assertEqual(record["example_order_within_pair"], ["Normal", "synthetic"])
        payload = dict(record)
        claimed = payload.pop("sha256")
        self.assertEqual(digest(payload), claimed)

    def test_training_seed_and_fold_are_distinct_training_namespaces(self):
        index = build()
        original = [bs.BasePairedSampler(index, 42).epoch_pairs(e) for e in range(1, 6)]
        self.assertNotEqual(original, [bs.BasePairedSampler(index, 123).epoch_pairs(e)
                                      for e in range(1, 6)])
        windows, events = fixture(fold=1)
        other = build(windows, events, fold=1)
        self.assertNotEqual([tuple(p.base_window_id for p in pairs) for pairs in original],
                            [tuple(p.base_window_id for p in bs.BasePairedSampler(other, 42).epoch_pairs(e))
                             for e in range(1, 6)])

    def test_documented_rng_namespace_matches_independent_30epoch_reference(self):
        # Independent oracle from the agreed namespace, not sampler internals.
        # Unequal k plus 30 epochs exercise per-base keys and new cycle keys.
        windows, events = fixture()
        ids = sorted(w["base_window_id"] for w in windows)
        accepted = {identifier: sorted(e["event_id"] for e in events
                                        if e["accepted"] and e["base_window_id"] == identifier)
                    for identifier in ids}
        prefix = ["exp004-base-paired-bin-v1", "synthetic-modality-v3-proposed", 0]
        for seed in (42, 123, 456):
            observed, expected = [], []
            sampler = bs.BasePairedSampler(build(windows, events), seed)
            for epoch in range(1, 31):
                e = epoch - 1
                key = prefix + [seed, "fit", "base-order", e]
                order = np.random.Generator(np.random.PCG64(int(digest(key)[:32], 16))).permutation(len(ids))
                row = []
                for position in order:
                    identifier = ids[int(position)]
                    choices = accepted[identifier]
                    key = prefix + [seed, "fit", "event-cycle", identifier, e // len(choices)]
                    permutation = np.random.Generator(np.random.PCG64(int(digest(key)[:32], 16))).permutation(len(choices))
                    row.append((identifier, choices[int(permutation[e % len(choices)])]))
                expected.append(row)
                observed.append([(p.base_window_id, p.event_id) for p in sampler.epoch_pairs(epoch)])
            with self.subTest(seed=seed):
                self.assertEqual(observed, expected)

    def test_ct_sensitivity_keeps_event_id_order_and_binds_instances(self):
        ct1 = build()
        windows, events = fixture(ct="CT2")
        ct2 = build(windows, events, ct_channel="CT2")
        for epoch in (1, 9, 30):
            a, b = [bs.BasePairedSampler(index, 42).epoch_pairs(epoch) for index in (ct1, ct2)]
            self.assertEqual([(p.base_window_id, p.event_id) for p in a],
                             [(p.base_window_id, p.event_id) for p in b])
            self.assertTrue(all(x.event_instance_id != y.event_instance_id for x, y in zip(a, b)))
            self.assertTrue(all(p.ct_channel == "CT1" for p in a))
            self.assertTrue(all(p.ct_channel == "CT2" for p in b))
        self.assertNotEqual(ct1.sha256, ct2.sha256)

    def test_immutable_index_pair_and_sampler(self):
        index = build()
        sampler = bs.BasePairedSampler(index, 42)
        pair = sampler.epoch_pairs(1)[0]
        for obj, key, value in ((index, "role", "dev"), (sampler, "training_seed", 123),
                                (pair, "event_id", "changed"),
                                (index.bases[0].events[0], "sensor_changed", False)):
            with self.subTest(key=key), self.assertRaises(FrozenInstanceError):
                setattr(obj, key, value)

    def test_pairs_carry_the_complete_regeneration_identity(self):
        windows, events = fixture()
        by_id = {e["event_id"]: e for e in events}
        sampler = bs.BasePairedSampler(build(windows, events), 42)
        for pair in sampler.epoch_pairs(1):
            event = by_id[pair.event_id]
            key = ["synthetic-modality-v3-proposed", 0, "fit", pair.base_window_id,
                   pair.family, pair.strength, 42]
            self.assertEqual(pair.event_id, digest(key))
            self.assertEqual(pair.event_instance_id, digest([key, pair.ct_channel]))
            self.assertTrue(event["accepted"])
            self.assertEqual((pair.sensor_changed, pair.thermal_changed),
                             (event["sensor_changed"], event["thermal_changed"]))
            self.assertEqual(windows[pair.window_index]["base_window_id"], pair.base_window_id)
            self.assertEqual(windows[pair.window_index]["device"], pair.device)

    def test_explicit_seed_and_no_modality_or_model_arguments(self):
        parameters = inspect.signature(bs.BasePairedSampler).parameters
        self.assertIs(parameters["training_seed"].default, inspect.Parameter.empty)
        self.assertEqual(parameters["paired_bases_per_batch"].default, 8)
        self.assertNotIn("modality", parameters)
        self.assertNotIn("model", parameters)
        for key in ("modality", "model"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                bs.BasePairedSampler(build(), 42, **{key: "S"})

    def test_epoch_conversion_and_planned_limit(self):
        sampler = bs.BasePairedSampler(build(), 42, planned_epochs=3)
        self.assertEqual(sampler.epoch_manifest(1)["epoch_zero_based"], 0)
        self.assertEqual(sampler.epoch_manifest(3)["epoch_zero_based"], 2)
        for epoch in (0, 4):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                sampler.epoch_pairs(epoch)


class DevWeightsTests(unittest.TestCase):
    def test_device_count_is_derived_for_three_and_four_devices(self):
        for devices in (("agv02", "oht02", "agv04"),
                        ("agv02", "oht02", "agv04", "oht04")):
            counts = (6, 7, 8, 9, 6)
            windows, events = fixture(role="dev", counts=counts, devices=devices)
            roles = {**ROLES, **{device: "dev" for device in devices}}
            examples = bs.dev_examples(build(windows, events, role="dev", roles=roles))
            bases_per_device = Counter(w["device"] for w in windows)
            for label in (0, 1):
                self.assertAlmostEqual(math.fsum(e.weight for e in examples if e.label == label),
                                       0.5, delta=1e-12)
                for device in devices:
                    self.assertAlmostEqual(math.fsum(e.weight for e in examples
                                                    if e.device == device and e.label == label),
                                           0.5 / len(devices), delta=1e-12)
            for i, window in enumerate(windows):
                target = 0.5 / (len(devices) * bases_per_device[window["device"]])
                for example in examples:
                    if example.base_window_id == window["base_window_id"]:
                        self.assertAlmostEqual(example.weight, target / counts[i] if example.label else target,
                                               delta=1e-12)

    def test_hierarchy_on_unequal_device_base_and_event_counts(self):
        windows, events = fixture(role="dev", counts=(6, 7, 9))
        index = build(windows, events, role="dev")
        examples = bs.dev_examples(index)
        weights = {(e.base_window_id, e.label): e.weight for e in examples}
        # agv02 has 2 bases: each has Normal mass 1/8 and positive mass 1/8.
        # oht02 has 1 base: its Normal and positive masses are each 1/4.
        for i, k in enumerate((6, 7, 9)):
            base_id = windows[i]["base_window_id"]
            normal = 0.25 if i == 1 else 0.125
            self.assertAlmostEqual(weights[(base_id, 0)], normal)
            self.assertAlmostEqual(weights[(base_id, 1)], normal / k)
            for label in (0, 1):
                self.assertAlmostEqual(math.fsum(e.weight for e in examples
                                                if e.base_window_id == base_id and e.label == label), normal)
        for label in (0, 1):
            self.assertAlmostEqual(math.fsum(e.weight for e in examples if e.label == label), 0.5)
            for device in ("agv02", "oht02"):
                self.assertAlmostEqual(math.fsum(e.weight for e in examples
                                                if e.label == label and e.device == device), 0.25)

    def test_all_accepted_events_kept_with_positive_label_and_normal_sentinel(self):
        for ct in ("CT1", "CT2"):
            windows, events = fixture(role="dev", ct=ct)
            by_id = {e["event_id"]: e for e in events}
            index = build(windows, events, role="dev", ct_channel=ct)
            examples = bs.dev_examples(index)
            self.assertEqual(Counter(e.event_id for e in examples if e.label == 1),
                             Counter(e.event_id for b in index.bases for e in b.events))
            self.assertEqual(sum(e.label == 0 for e in examples), 4)
            for example in examples:
                self.assertEqual(example.event_id is None, example.label == 0)
                self.assertEqual(example.event_instance_id is None, example.label == 0)
                self.assertEqual(example.family is None, example.label == 0)
                self.assertEqual(example.strength is None, example.label == 0)
                self.assertEqual(example.ct_channel, ct)
                self.assertEqual(index.selected_window_ids[example.window_index], example.base_window_id)
                self.assertEqual(windows[example.window_index]["device"], example.device)
                self.assertGreater(example.weight, 0)
                if example.label == 1:
                    expected = by_id[example.event_id]
                    self.assertTrue(expected["accepted"])
                    for field in ("family", "strength", "sensor_changed", "thermal_changed"):
                        self.assertEqual(getattr(example, field), expected[field])
                    self.assertEqual(example.event_instance_id, digest([expected["namespace"], ct]))
                else:
                    self.assertEqual((example.sensor_changed, example.thermal_changed), (False, False))

    def test_weighted_bce_matches_hand_calculated_target(self):
        windows, events = fixture(role="dev", counts=(6, 7, 9))
        examples = bs.dev_examples(build(windows, events, role="dev"))
        # Per-base loss values are known; positive events all use the same loss
        # within their base. This must give equal base mass, independent of k.
        losses = {windows[i]["base_window_id"]: (i + 1, 4 * (i + 1)) for i in range(3)}
        value = math.fsum(e.weight * losses[e.base_window_id][e.label] for e in examples)
        self.assertAlmostEqual(value, 0.125 * (1 + 4) + 0.25 * (2 + 8) + 0.125 * (3 + 12))
        unweighted = math.fsum(losses[e.base_window_id][e.label] for e in examples) / len(examples)
        self.assertNotAlmostEqual(value, unweighted)

    def test_dev_manifest_digest_and_order_are_input_order_independent(self):
        windows, events = fixture(role="dev")
        a = bs.dev_weight_manifest(build(windows, events, role="dev"))
        b = bs.dev_weight_manifest(build(windows, events[::-1], role="dev"))
        self.assertEqual(a, b)
        self.assertEqual(a["policy"], "class_then_equipment_then_base_then_event")
        self.assertEqual(a["class_weight_sum"], {"0": 0.5, "1": 0.5})
        claimed = a.pop("sha256")
        self.assertEqual(digest(a), claimed)


class AdmissionTests(unittest.TestCase):
    def test_foreign_roles_and_non_normal_windows_fail(self):
        for key, value in (("device", "agv03"), ("device", "oht03"),
                            ("hard_valid", False), ("hard_valid", 1),
                            ("pure_Normal", False), ("pure_Normal", 1)):
            windows, events = fixture()
            windows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                build(windows, events)
        for role in ("calibration", "heldout", "heldout-eval", None):
            with self.subTest(role=role), self.assertRaises(ValueError):
                build(role=role)

    def test_empty_missing_equipment_duplicate_and_extra_bases_fail(self):
        windows, events = fixture()
        cases = [(windows[:1], events[:9]), ([], []),
                 (windows + [windows[0]], events), (windows, events + [events[0]])]
        other = copy.deepcopy(events)
        other[0]["base_window_id"] = "foreign"
        cases.append((windows, other))
        # An additional foreign event must fail even when the selected grid is complete.
        foreign = copy.deepcopy(events[0])
        foreign["base_window_id"] = "foreign"
        cases.append((windows, events + [foreign]))
        for w, e in cases:
            with self.subTest(windows=len(w), events=len(e)), self.assertRaises(ValueError):
                build(w, e)

    def test_rejected_record_content_is_part_of_source_binding(self):
        windows, events = fixture()
        original = build(windows, events)
        events[8]["rejection_reasons"] = ["changed_fixture_constraint"]
        other = build(windows, events)
        self.assertEqual(original.bases, other.bases)
        self.assertNotEqual(original.sha256, other.sha256)

    def test_complete_grid_and_at_least_one_accepted_per_base_required(self):
        windows, events = fixture()
        for shortened in (events[:-1], events[9:]):
            with self.subTest(count=len(shortened)), self.assertRaises(ValueError):
                build(windows, shortened)
        for event in events[:9]:
            event["accepted"] = False
            event["rejection_reasons"] = ["fixture_constraint"]
        with self.assertRaisesRegex(ValueError, "at least one accepted"):
            build(windows, events)

    def test_identity_and_provenance_corruption_fail_even_for_rejected_event(self):
        corruptions = (("role", "dev"), ("device", "oht01"), ("ct_channel", "CT2"),
                       ("family", "foreign"), ("strength", True),
                       ("event_id", "c" * 64), ("event_instance_id", "c" * 64),
                       ("normalizer_sha256", "c" * 64), ("fit_ids_sha256", "c" * 64),
                       ("namespace", ["foreign"]), ("accepted", 1),
                       ("sensor_changed", 1), ("thermal_changed", 0))
        for position in (0, 8):
            for key, value in corruptions:
                windows, events = fixture()
                events[position][key] = value
                with self.subTest(position=position, key=key), self.assertRaises(ValueError):
                    build(windows, events)

    def test_namespace_fold_seed_and_boolean_aliases_fail(self):
        for field, value in ((1, False), (6, True), (5, True), (1, 1), (6, 123)):
            windows, events = fixture()
            events[0]["namespace"][field] = value
            events[0]["event_id"] = digest(events[0]["namespace"])
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                build(windows, events)

    def test_noop_wrong_family_flags_and_inconsistent_reasons_fail(self):
        for key, value in (("sensor_changed", False), ("thermal_changed", True),
                            ("rejection_reasons", ["no_op"]), ("rejection_reasons", None)):
            windows, events = fixture()
            events[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                build(windows, events)
        windows, events = fixture()
        events[8]["rejection_reasons"] = []
        with self.assertRaises(ValueError):
            build(windows, events)

    def test_supplied_instance_id_matches_explicit_legacy_derivation(self):
        windows, events = fixture()
        legacy = build(windows, events)
        for event in events:
            event["event_instance_id"] = digest([event["namespace"], event["ct_channel"]])
        current = build(windows, events)
        self.assertEqual(legacy.bases, current.bases)
        self.assertNotEqual(legacy.sha256, current.sha256)

    def test_invalid_options_epoch_and_batch_fail(self):
        for options in ({"fold": True}, {"fold": 4}, {"generator_seed": -1},
                        {"generator_seed": True}, {"ct_channel": "CT4"},
                        {"normalizer_sha256": "x"}, {"fit_ids_sha256": None}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build(**options)
        index = build()
        for value in (-1, True, 1.0, "1"):
            with self.subTest(seed=value), self.assertRaises(ValueError):
                bs.BasePairedSampler(index, training_seed=value)
            with self.subTest(epoch=value), self.assertRaises(ValueError):
                bs.BasePairedSampler(index, 42).epoch_pairs(value)
        for value in (0, -1, True, 1.5):
            with self.subTest(batch=value), self.assertRaises(ValueError):
                bs.BasePairedSampler(index, 42, paired_bases_per_batch=value)
        for value in (0, -1, True, 1.5):
            with self.subTest(planned_epochs=value), self.assertRaises(ValueError):
                bs.BasePairedSampler(index, 42, planned_epochs=value)

    def test_fit_and_dev_interfaces_refuse_wrong_role(self):
        with self.assertRaises(ValueError):
            bs.BasePairedSampler(build(role="dev"), 42)
        with self.assertRaises(ValueError):
            bs.dev_examples(build())


if __name__ == "__main__":
    unittest.main()
