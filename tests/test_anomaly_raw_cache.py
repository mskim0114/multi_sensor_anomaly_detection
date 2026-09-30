"""Raw Normal-window cache contract: fit/dev only, dtype-exact, verified, byte-identical.

The reference is the canonical reader itself (``fit_anomaly_controls.make_reader``), run
on a tiny synthetic dataset written to a temporary directory in the real raw layout. No
test reads the project dataset. Thermal ``.bin`` files are written once as float64 (what
the EXP-004 data holds) and once as float32, so both dtype-preservation cases are checked
against the canonical path rather than against a stand-in.
"""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def leaf(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rc = leaf("_raw_cache_under_test", "src/data/anomaly_raw_cache.py")
controls = leaf("_raw_cache_controls", "src/fit_anomaly_controls.py")
sa = controls.sa

try:
    import torch  # noqa: F401
    TORCH_ERROR = None
except Exception as exc:  # pragma: no cover - exercised on Jetson
    TORCH_ERROR = repr(exc)
requires_torch = unittest.skipIf(TORCH_ERROR is not None,
                                 f"torch unavailable (expected on Jetson): {TORCH_ERROR}")

ROLES = {"agv01": "fit", "oht01": "fit", "agv02": "dev", "oht02": "calibration",
         "agv03": "heldout"}
TICKS = {"agv01": 40, "oht01": 40, "agv02": 40, "oht02": 30}  # agv03 has no files at all
NORMALIZER = {"sensor_mean": [40.1, 5.0, 7.0, 9.0, 11.0, 12.0, 13.0, 14.0],
              "sensor_std": [1.3, 0.7, 0.9, 1.1, 2.0, 2.1, 2.2, 2.3],
              "sensor_scale": [1.3, 0.7, 0.9, 1.1, 2.0, 2.1, 2.2, 2.3],
              "thermal_mean": 40.123456789, "thermal_std": 3.003847327976131,
              "thermal_scale": 3.003847327976131, "fit_ids_sha256": "f" * 64}
NORMALIZER["sha256"] = sa.digest_json(NORMALIZER)
F64 = {"sensor": "float64", "thermal": "float64", "labels": "int64"}
F32 = {"sensor": "float64", "thermal": "float32", "labels": "int64"}


def tick_name(device, k):
    return f"{device}_0101_00{k // 60:02d}{k % 60:02d}"


def write_dataset(root, thermal_dtype, seed=7):
    """Raw files in the canonical layout, plus the ledger and window manifest entries."""
    root = Path(root).resolve()
    source, labels = root / "Training" / "원천데이터", root / "Training" / "라벨링데이터"
    source.mkdir(parents=True)
    labels.mkdir(parents=True)
    ledger = []
    for device, count in TICKS.items():
        digest = hashlib.sha256(f"{device}|{seed}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        for k in range(count):
            name = tick_name(device, k)
            pm = np.cumsum(rng.uniform(0.5, 3.0, 3))
            sensor = [rng.normal(40, 1.3), *pm, *np.abs(rng.normal(12, 2, 4))]
            thermal = rng.normal(40.0, 3.0, sa.THERMAL_SHAPE).astype(thermal_dtype)
            npy = io.BytesIO()
            np.save(npy, thermal, allow_pickle=False)
            payloads = {
                source / f"{name}.csv": (",".join(sa.CHANNELS) + "\n"
                                         + ",".join(repr(float(v)) for v in sensor)
                                         + "\n").encode("utf-8"),
                source / f"{name}.bin": npy.getvalue(),
                labels / f"{name}.json": json.dumps(
                    {"annotations": [{"tagging": [{"state": "0"}]}]}).encode(),
            }
            for path, payload in payloads.items():
                path.write_bytes(payload)
                ledger.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
    return ledger


def manifest_windows():
    windows = []
    for device in ROLES:
        count = TICKS.get(device, 30)
        for start in range(0, count - 29, 10):
            ids = [f"Training/{tick_name(device, k)}" for k in range(start, start + 30)]
            windows.append({"base_window_id": ids[0] + ":30", "session_id": ids[0],
                            "device": device, "start_index": start, "raw_base_ids": ids,
                            "pure_class": 0, "pure_Normal": True, "majority_label": 0,
                            "mixed": False, "hard_valid": True, "hard_errors": [],
                            "base_QC_flags": []})
    return windows


class CountingReader:
    """Wraps a reader so a test can assert exactly which windows were read."""

    def __init__(self, reader):
        self.inner, self.calls = reader, []

    def __call__(self, window):
        self.calls.append(window["base_window_id"])
        return self.inner(window)


class Fixture:
    """One synthetic dataset and the canonical reader over it."""

    def __init__(self, tmp, thermal_dtype, dtypes):
        self.tmp = Path(tmp)
        self.data_root = (self.tmp / f"raw_{np.dtype(thermal_dtype).name}").resolve()
        self.ledger = write_dataset(self.data_root, thermal_dtype)
        self.source_sha = sa.digest_json(self.ledger)
        self.windows = manifest_windows()
        self.selected = rc.select_windows(self.windows, ROLES)
        self.canonical = controls.make_reader(self.data_root, self.ledger)
        self.dtypes = dtypes
        self.serial = 0

    def build(self, windows=None, reader=None, out=None, **overrides):
        self.serial += 1
        out = out or self.tmp / f"cache_{self.data_root.name}_{self.serial}"
        kwargs = dict(data_root=self.data_root, raw_manifest=self.ledger,
                      source_manifest_sha256=self.source_sha, dtypes=self.dtypes)
        kwargs.update(overrides)
        return rc.RawNormalCache.build(out, self.selected if windows is None else windows,
                                       ROLES, reader or self.canonical, **kwargs)


class SharedFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.f64 = Fixture(cls._tmp.name, np.float64, F64)
        cls.f32 = Fixture(cls._tmp.name, np.float32, F32)
        cls.cache64 = cls.f64.build().open()
        cls.cache32 = cls.f32.build().open()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def pairs(self):
        """(label, fixture, cache) for both thermal dtypes."""
        return (("float64", self.f64, self.cache64), ("float32", self.f32, self.cache32))

    def assert_bytes_equal(self, left, right, label):
        self.assertIs(type(left), np.ndarray, label)
        self.assertIs(type(right), np.ndarray, label)
        self.assertEqual(left.dtype.str, right.dtype.str, label)
        self.assertEqual(left.shape, right.shape, label)
        self.assertEqual(left.tobytes(), right.tobytes(), label)


class RoleBoundaryTests(SharedFixture):
    def test_selection_is_fit_then_dev_in_manifest_order(self):
        selected = self.f64.selected
        roles = [ROLES[w["device"]] for w in selected]
        self.assertEqual(roles, sorted(roles, key=rc.ALLOWED_ROLES.index))
        self.assertEqual({ROLES[w["device"]] for w in selected}, {"fit", "dev"})
        order = [w["base_window_id"] for w in self.f64.windows]
        for role in rc.ALLOWED_ROLES:
            ids = [w["base_window_id"] for w in selected if ROLES[w["device"]] == role]
            self.assertEqual(ids, [i for i in order if i in ids])

    @staticmethod
    def interleaved_manifest():
        """Metadata only: devices interleaved and ineligible windows mixed in."""
        # ids run against manifest order, so "manifest order" and "id order" differ.
        rows = (("w10", "agv02", True, True), ("w09", "agv01", True, True),
                ("w08", "oht02", True, True), ("w07", "oht01", True, True),
                ("w06", "agv01", False, True), ("w05", "agv02", True, True),
                ("w04", "oht01", True, False), ("w03", "agv01", True, True),
                ("w02", "agv03", True, True), ("w01", "oht01", True, True),
                ("w00", "agv02", True, False))
        return [{"base_window_id": i, "device": d, "hard_valid": h, "pure_Normal": p}
                for i, d, h, p in rows]

    def test_selection_keeps_manifest_order_within_each_role_on_interleaved_devices(self):
        chosen = rc.select_windows(self.interleaved_manifest(), ROLES)
        self.assertEqual([w["base_window_id"] for w in chosen],
                         ["w09", "w07", "w03", "w01", "w10", "w05"])

    @requires_torch
    def test_selection_is_exactly_the_pilot_selection(self):
        pilot = leaf("_raw_cache_pilot", "src/train_anomaly_pilot.py")
        for windows in (self.interleaved_manifest(), self.f64.windows):
            expected = [w["base_window_id"] for role in rc.ALLOWED_ROLES
                        for w in pilot.select_windows({"windows": windows}, ROLES, role)]
            self.assertEqual([w["base_window_id"] for w in rc.select_windows(windows, ROLES)],
                             expected)

    def test_calibration_and_heldout_windows_are_refused_before_any_read(self):
        for device in ("oht02", "agv03"):
            with self.subTest(device=device):
                intruder = next(w for w in self.f64.windows if w["device"] == device)
                reader = CountingReader(self.f64.canonical)
                out = Path(self._tmp.name) / f"refused_{device}"
                with self.assertRaisesRegex(ValueError, "refused role"):
                    self.f64.build(self.f64.selected + [intruder], reader=reader, out=out)
                self.assertEqual(reader.calls, [])
                self.assertFalse(out.exists())

    def test_calibration_window_is_refused_even_when_it_comes_first(self):
        intruder = next(w for w in self.f64.windows if w["device"] == "oht02")
        reader = CountingReader(self.f64.canonical)
        with self.assertRaisesRegex(ValueError, "refused role 'calibration'"):
            self.f64.build([intruder] + self.f64.selected, reader=reader)
        self.assertEqual(reader.calls, [])

    def test_unassigned_device_and_ineligible_windows_are_refused(self):
        base = self.f64.selected[0]
        cases = {"refused role None": dict(base, device="agv09"),
                 "not hard-valid pure Normal": dict(base, pure_Normal=False),
                 "not hard-valid pure Normal ": dict(base, hard_valid=False),
                 "not hard-valid pure Normal  ": dict(base, pure_Normal=1),
                 "raw ticks": dict(base, raw_base_ids=base["raw_base_ids"][:29])}
        for message, window in cases.items():
            with self.subTest(message=message):
                reader = CountingReader(self.f64.canonical)
                with self.assertRaisesRegex(ValueError, message.strip()):
                    self.f64.build([window], reader=reader)
                self.assertEqual(reader.calls, [])

    def test_duplicate_window_is_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate window"):
            self.f64.build([self.f64.selected[0], self.f64.selected[0]])

    def test_selection_requires_every_fit_and_dev_device(self):
        windows = [w for w in self.f64.windows if w["device"] != "agv02"]
        with self.assertRaisesRegex(ValueError, "no eligible pure-Normal window in role dev"):
            rc.select_windows(windows, ROLES)
        windows = [dict(w, pure_Normal=False) if w["device"] == "oht01" else w
                   for w in self.f64.windows]
        with self.assertRaisesRegex(ValueError, "role fit is missing eligible equipment"):
            rc.select_windows(windows, ROLES)

    def test_cache_reader_refuses_windows_it_does_not_hold(self):
        read = self.cache64.reader()
        for device in ("oht02", "agv03"):
            window = next(w for w in self.f64.windows if w["device"] == device)
            with self.assertRaisesRegex(ValueError, "not in this fit/dev cache"):
                read(window)
        changed = dict(self.f64.selected[0])
        changed["raw_base_ids"] = list(reversed(changed["raw_base_ids"]))
        with self.assertRaisesRegex(ValueError, "differs from the one cached"):
            read(changed)
        with self.assertRaisesRegex(ValueError, "differs from the one cached"):
            read(dict(self.f64.selected[0], pure_Normal=False))
        with self.assertRaisesRegex(ValueError, "differs from the one cached"):
            read(dict(self.f64.selected[0], hard_valid=False))
        original = self.f64.selected[0]
        without_session = {k: v for k, v in original.items() if k != "session_id"}
        for label, window in (("session_id", dict(original,
                                                  session_id=original["session_id"] + ":x")),
                              ("start_index", dict(original,
                                                   start_index=original["start_index"] + 1)),
                              ("missing session_id", without_session),
                              ("numpy start_index", dict(
                                  original, start_index=np.int64(original["start_index"]))),
                              ("device", dict(original, device="oht01"))):
            with self.subTest(changed=label):
                with self.assertRaisesRegex(ValueError, "differs from the one cached"):
                    read(window)
        self.assertEqual(len(read(copy.deepcopy(original))), 3)

    def test_manifest_records_only_fit_and_dev(self):
        manifest = self.cache64.manifest
        self.assertEqual({w["role"] for w in manifest["windows"]}, {"fit", "dev"})
        self.assertEqual(manifest["role_counts"],
                         {"fit": sum(ROLES[w["device"]] == "fit" for w in self.f64.selected),
                          "dev": sum(ROLES[w["device"]] == "dev" for w in self.f64.selected)})


class DtypePreservationTests(SharedFixture):
    def test_every_array_matches_the_canonical_reader_byte_for_byte(self):
        for label, fixture, cache in self.pairs():
            read = cache.reader(verify_each_read=True)
            for window in fixture.selected:
                for name, expected, got in zip(rc.ARRAYS, fixture.canonical(window), read(window)):
                    self.assert_bytes_equal(expected, got, f"{label} {name} {window['base_window_id']}")
                    self.assertTrue(got.flags.c_contiguous)

    def test_float32_thermal_is_stored_as_float32_not_promoted(self):
        self.assertEqual(self.cache32.manifest["arrays"]["thermal"]["dtype"], "<f4")
        self.assertEqual(self.cache64.manifest["arrays"]["thermal"]["dtype"], "<f8")
        _, thermal, _ = self.cache32.window(0)
        self.assertEqual(thermal.dtype, np.float32)
        self.assertEqual(self.cache32.manifest["arrays"]["thermal"]["payload_bytes"],
                         len(self.f32.selected) * 30 * 120 * 160 * 4)
        self.assertEqual(np.load(self.cache32.out / "thermal.npy", mmap_mode="r").dtype, np.float32)

    def test_declared_dtype_mismatch_fails_in_both_directions(self):
        cases = ((self.f32, F64, "<f4 differs from declared <f8"),   # would be a promotion
                 (self.f64, F32, "<f8 differs from declared <f4"))   # would be a demotion
        for fixture, dtypes, message in cases:
            with self.subTest(message=message):
                out = Path(self._tmp.name) / f"mismatch_{fixture.serial}_{dtypes['thermal']}"
                with self.assertRaisesRegex(ValueError, message):
                    fixture.build(dtypes=dtypes, out=out)
                failure = json.loads((out / "failure.json").read_text())
                self.assertEqual(failure["stage"], "read")
                self.assertEqual(failure["window_index"], 0)
                self.assertFalse((out / "manifest.json").exists())

    def reader_returning(self, **replacements):
        def read(window):
            arrays = dict(zip(rc.ARRAYS, self.f64.canonical(window)))
            for name, change in replacements.items():
                arrays[name] = change(arrays[name])
            return tuple(arrays[name] for name in rc.ARRAYS)
        return read

    def test_reader_output_that_is_not_exactly_as_declared_is_refused(self):
        cases = {
            "big-endian": (dict(thermal=lambda a: a.astype(">f8")), "differs from declared"),
            "fortran order": (dict(thermal=np.asfortranarray), "not C-contiguous"),
            "non-contiguous": (dict(thermal=lambda a: np.concatenate([a, a], 2)[:, :, ::2]),
                               "not C-contiguous"),
            "wrong shape": (dict(sensor=lambda a: a[:, :5].copy()), "shape"),
            "int32 labels": (dict(labels=lambda a: a.astype(np.int32)), "labels dtype"),
            "list": (dict(sensor=lambda a: a.tolist()), "not an ndarray"),
            "non-finite": (dict(thermal=lambda a: np.where(a > 0, np.inf, a)), "not finite"),
            "non-Normal": (dict(labels=lambda a: a + 1), "pure Normal"),
        }
        for label, (change, message) in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    self.f64.build(self.f64.selected[:1], reader=self.reader_returning(**change))

    def test_reader_must_return_the_three_arrays(self):
        with self.assertRaisesRegex(ValueError, r"\(sensor, thermal, labels\)"):
            self.f64.build(self.f64.selected[:1], reader=lambda w: self.f64.canonical(w)[:2])

    def test_dtype_declaration_is_validated_before_anything_else(self):
        reader = CountingReader(self.f64.canonical)
        bad = ({"sensor": "float64", "thermal": "float64"},
               dict(F64, thermal=">f8"), dict(F64, labels="float64"), dict(F64, thermal="O"))
        for dtypes in bad:
            with self.subTest(dtypes=dtypes):
                with self.assertRaises((ValueError, TypeError)):
                    self.f64.build(dtypes=dtypes, reader=reader)
        self.assertEqual(reader.calls, [])

    def test_required_bytes_matches_the_fold0_geometry(self):
        sizes = rc.required_bytes(2613, F64)
        self.assertEqual(sizes["thermal"], 12_040_704_000)   # review update (23) B
        self.assertEqual(sizes["sensor"], 2613 * 30 * 8 * 8)
        self.assertEqual(sizes["labels"], 2613 * 30 * 8)
        self.assertEqual(rc.required_bytes(2613, F32)["thermal"], 6_020_352_000)


class DownstreamBitIdentityTests(SharedFixture):
    """What P1b will actually consume must not change by a single bit."""

    def test_normal_normalization_and_final_float32_are_byte_identical(self):
        for label, fixture, cache in self.pairs():
            read = cache.reader()
            for window in fixture.selected:
                for ct in ("CT1", "CT2"):
                    raw_c, raw_k = fixture.canonical(window), read(window)
                    s_c, t_c = controls.normalized_pair(raw_c[0], raw_c[1], NORMALIZER, ct)
                    s_k, t_k = controls.normalized_pair(raw_k[0], raw_k[1], NORMALIZER, ct)
                    tag = f"{label} {ct} {window['base_window_id']}"
                    self.assert_bytes_equal(s_c, s_k, "sensor " + tag)
                    self.assert_bytes_equal(t_c, t_k, "thermal " + tag)
                    # The pilot's final tensor conversion, spelled as it spells it.
                    self.assert_bytes_equal(np.ascontiguousarray(t_c, dtype=np.float32),
                                            np.ascontiguousarray(t_k, dtype=np.float32),
                                            "final float32 " + tag)

    def test_float32_promotion_would_change_the_final_tensor(self):
        """The fixture is sensitive: a promoting cache would not pass the test above."""
        _, thermal, _ = self.f32.canonical(self.f32.selected[0])
        _, native = controls.normalized_pair(np.zeros((30, 8)), thermal, NORMALIZER, "CT1")
        _, promoted = controls.normalized_pair(np.zeros((30, 8)), thermal.astype(np.float64),
                                               NORMALIZER, "CT1")
        native = np.ascontiguousarray(native, dtype=np.float32)
        promoted = np.ascontiguousarray(promoted, dtype=np.float32)
        self.assertGreater(int((native != promoted).sum()), 0)
        _, cached, _ = self.cache32.reader()(self.f32.selected[0])
        _, from_cache = controls.normalized_pair(np.zeros((30, 8)), cached, NORMALIZER, "CT1")
        self.assertEqual(np.ascontiguousarray(from_cache, dtype=np.float32).tobytes(),
                         native.tobytes())

    def test_inject_blind_results_are_byte_identical(self):
        for label, fixture, cache in self.pairs():
            read = cache.reader()
            for window in fixture.selected:
                role = ROLES[window["device"]]
                raw_c, raw_k = fixture.canonical(window), read(window)
                for family in ("sensor-only", "thermal-only", "coupled"):
                    for strength in (1, 2, 4):
                        kwargs = dict(normalizer=NORMALIZER, protocol=sa.PROTOCOL, fold=0,
                                      role=role, base_window_id=window["base_window_id"],
                                      family=family, strength=strength, generator_seed=42,
                                      ct_channel="CT1")
                        s_c, t_c, m_c = sa.inject_blind(raw_c[0], raw_c[1],
                                                        tick_labels=raw_c[2], **kwargs)
                        s_k, t_k, m_k = sa.inject_blind(raw_k[0], raw_k[1],
                                                        tick_labels=raw_k[2], **kwargs)
                        tag = f"{label} {family} s{strength} {window['base_window_id']}"
                        self.assert_bytes_equal(s_c, s_k, "sensor " + tag)
                        self.assert_bytes_equal(t_c, t_k, "thermal " + tag)
                        self.assertEqual(json.dumps(m_c, sort_keys=True),
                                         json.dumps(m_k, sort_keys=True), tag)
                        # Normalizing the injected pair must agree as well.
                        n_c = controls.normalized_pair(s_c, t_c, NORMALIZER, "CT1")
                        n_k = controls.normalized_pair(s_k, t_k, NORMALIZER, "CT1")
                        for a, b in zip(n_c, n_k):
                            self.assert_bytes_equal(np.ascontiguousarray(a, dtype=np.float32),
                                                    np.ascontiguousarray(b, dtype=np.float32),
                                                    "normalized " + tag)

    def test_fixture_events_are_mostly_accepted_so_the_comparison_is_not_vacuous(self):
        window = self.f64.selected[0]
        s, t, labels = self.f64.canonical(window)
        accepted = [sa.inject_blind(s, t, tick_labels=labels, normalizer=NORMALIZER,
                                    protocol=sa.PROTOCOL, fold=0, role="fit",
                                    base_window_id=window["base_window_id"], family=family,
                                    strength=strength, generator_seed=42)[2]["accepted"]
                    for family in ("thermal-only", "coupled") for strength in (1, 2, 4)]
        self.assertTrue(all(accepted))

    @requires_torch
    def test_pilot_normalized_thermal_is_byte_identical(self):
        pilot = leaf("_raw_cache_pilot", "src/train_anomaly_pilot.py")
        for label, fixture, cache in self.pairs():
            read = cache.reader()
            for window in fixture.selected:
                expected = pilot.normalized_thermal(window, fixture.canonical, NORMALIZER)
                got = pilot.normalized_thermal(window, read, NORMALIZER)
                self.assert_bytes_equal(expected, got, f"{label} {window['base_window_id']}")


class ManifestTests(SharedFixture):
    def test_source_manifest_digest_must_match_before_anything_is_created(self):
        reader = CountingReader(self.f64.canonical)
        out = Path(self._tmp.name) / "wrong_source"
        with self.assertRaisesRegex(ValueError, "raw manifest content differs"):
            self.f64.build(reader=reader, out=out, source_manifest_sha256="0" * 64)
        self.assertEqual(reader.calls, [])
        self.assertFalse(out.exists())

    def test_window_absent_from_the_ledger_is_refused_before_any_read(self):
        missing = self.f64.selected[0]["raw_base_ids"][3].split("/")[1]
        ledger = [e for e in self.f64.ledger if not e["path"].endswith(f"{missing}.bin")]
        reader = CountingReader(self.f64.canonical)
        with self.assertRaisesRegex(ValueError, "absent from the source manifest"):
            self.f64.build(reader=reader, raw_manifest=ledger,
                           source_manifest_sha256=sa.digest_json(ledger))
        self.assertEqual(reader.calls, [])

    def test_raw_content_change_is_caught_by_the_canonical_reader_at_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp, np.float64, F64)
            name = fixture.selected[1]["raw_base_ids"][0].split("/")[1]
            path = fixture.data_root / "Training" / "원천데이터" / f"{name}.bin"
            path.write_bytes(path.read_bytes()[:-8] + b"\0" * 8)
            out = Path(tmp) / "changed"
            with self.assertRaisesRegex(ValueError, "raw content changed"):
                fixture.build(out=out)
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["window_index"], 0)  # window 0 overlaps window 1
            self.assertEqual(failure["stage"], "read")

    def test_per_window_records_and_order_digests(self):
        manifest = self.cache64.manifest
        ledger = {e["path"]: e["sha256"] for e in self.f64.ledger}
        for record, window, arrays in zip(manifest["windows"], self.f64.selected,
                                          map(self.f64.canonical, self.f64.selected)):
            self.assertEqual(record["base_window_id"], window["base_window_id"])
            self.assertEqual(record["raw_base_ids"], window["raw_base_ids"])
            self.assertEqual(record["raw_files_sha256"],
                             sa.digest_json(rc.raw_sources(window, self.f64.data_root, ledger)))
            for name, array in zip(rc.ARRAYS, arrays):
                self.assertEqual(record["array_sha256"][name],
                                 hashlib.sha256(array.tobytes()).hexdigest())
        ids = [w["base_window_id"] for w in self.f64.selected]
        self.assertEqual(manifest["base_order_sha256"], sa.digest_json(ids))
        self.assertEqual(manifest["role_order_sha256"],
                         sa.digest_json([ROLES[w["device"]] for w in self.f64.selected]))
        identities = [{"base_window_id": w["base_window_id"], "device": w["device"],
                       "role": ROLES[w["device"]], "session_id": w["session_id"],
                       "start_index": w["start_index"], "raw_base_ids": w["raw_base_ids"]}
                      for w in self.f64.selected]
        self.assertEqual(manifest["window_order_sha256"], sa.digest_json(identities))
        self.assertEqual(manifest["window_sources_sha256"], sa.digest_json(
            [sa.digest_json(rc.raw_sources(w, self.f64.data_root, ledger))
             for w in self.f64.selected]))
        self.assertEqual(manifest["source_manifest_sha256"], self.f64.source_sha)
        self.assertEqual(len(rc.raw_sources(self.f64.selected[0], self.f64.data_root, ledger)),
                         90)

    def test_window_order_changes_the_order_digests_but_not_the_windows(self):
        reordered = self.f64.build(list(reversed(self.f64.selected))).open()
        a, b = self.cache64.manifest, reordered.manifest
        for key in ("base_order_sha256", "role_order_sha256", "window_order_sha256",
                    "window_sources_sha256"):
            self.assertNotEqual(a[key], b[key], key)
        self.assertEqual(a["source_manifest_sha256"], b["source_manifest_sha256"])
        by_id = {r["base_window_id"]: r for r in a["windows"]}
        for record in b["windows"]:
            self.assertEqual(record["array_sha256"], by_id[record["base_window_id"]]["array_sha256"])
        window = self.f64.selected[0]
        for x, y in zip(self.cache64.reader()(window), reordered.reader()(window)):
            self.assertEqual(x.tobytes(), y.tobytes())

    def test_manifest_digest_covers_its_content(self):
        manifest = copy.deepcopy(self.cache64.manifest)
        claimed = manifest.pop("sha256")
        self.assertEqual(sa.digest_json(manifest), claimed)
        for name in rc.ARRAYS:
            entry = manifest["arrays"][name]
            self.assertEqual(entry["payload_sha256"], entry["source_payload_sha256"])
            self.assertNotEqual(entry["file_sha256"], entry["payload_sha256"])
            self.assertTrue(entry["c_contiguous"])
            self.assertFalse(entry["fortran_order"])
        self.assertTrue(manifest["verified_after_flush"])

    def test_build_reads_each_window_exactly_once_in_order(self):
        reader = CountingReader(self.f64.canonical)
        self.f64.build(reader=reader)
        self.assertEqual(reader.calls, [w["base_window_id"] for w in self.f64.selected])


class OutputAndFailureTests(SharedFixture):
    def test_existing_output_is_never_overwritten(self):
        out = self.cache64.out
        before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
        reader = CountingReader(self.f64.canonical)
        with self.assertRaisesRegex(ValueError, "never overwrites"):
            self.f64.build(reader=reader, out=out)
        self.assertEqual(reader.calls, [])
        self.assertEqual({p.name: p.stat().st_mtime_ns for p in out.iterdir()}, before)

    def test_output_must_be_separate_from_the_raw_data_root(self):
        for out in (self.f64.data_root / "cache", self.f64.data_root.parent):
            with self.subTest(out=out):
                with self.assertRaisesRegex(ValueError, "never overwrites|separate from the raw"):
                    self.f64.build(out=out)
        with self.assertRaisesRegex(ValueError, "separate from the raw"):
            self.f64.build(out=self.f64.data_root / "nested" / "cache")

    def test_reader_failure_is_preserved_with_the_partial_files(self):
        def failing(window):
            if len(failing.seen) == 2:
                raise OSError("simulated read failure")
            failing.seen.append(window["base_window_id"])
            return self.f64.canonical(window)
        failing.seen = []
        out = Path(self._tmp.name) / "failed_read"
        with self.assertRaisesRegex(OSError, "simulated read failure"):
            self.f64.build(reader=failing, out=out)
        failure = json.loads((out / "failure.json").read_text())
        self.assertEqual(failure["stage"], "read")
        self.assertEqual(failure["exception"], "OSError")
        self.assertEqual(failure["window_index"], 2)
        self.assertEqual(failure["base_window_id"], self.f64.selected[2]["base_window_id"])
        self.assertEqual(failure["windows_written"], 2)
        self.assertEqual(sorted(failure["partial_files_preserved"]),
                         ["labels.npy", "sensor.npy", "thermal.npy"])
        self.assertFalse((out / "manifest.json").exists())
        partial = np.load(out / "thermal.npy", mmap_mode="r")
        self.assertEqual(partial[1].tobytes(), self.f64.canonical(self.f64.selected[1])[1].tobytes())
        del partial
        with self.assertRaisesRegex(ValueError, "recorded a failure"):
            rc.RawNormalCache.attach(out)

    def assert_failure_recorded(self, out, stage, exception):
        failure = json.loads((out / "failure.json").read_text())
        self.assertEqual(failure["stage"], stage)
        self.assertEqual(failure["exception"], exception)
        self.assertFalse((out / "manifest.json").exists())
        with self.assertRaisesRegex(ValueError, "recorded a failure"):
            rc.RawNormalCache.attach(out)
        return failure

    def test_an_interrupt_is_recorded_like_any_other_failure(self):
        def interrupted(window):
            if len(interrupted.seen) == 2:
                raise KeyboardInterrupt
            interrupted.seen.append(window["base_window_id"])
            return self.f64.canonical(window)
        interrupted.seen = []
        out = Path(self._tmp.name) / "interrupted"
        with self.assertRaises(KeyboardInterrupt):
            self.f64.build(reader=interrupted, out=out)
        failure = self.assert_failure_recorded(out, "read", "KeyboardInterrupt")
        self.assertEqual(failure["windows_written"], 2)
        self.assertEqual(sorted(failure["partial_files_preserved"]),
                         ["labels.npy", "sensor.npy", "thermal.npy"])

    def test_failures_in_every_later_stage_are_recorded(self):
        cases = {
            "open": mock.patch.object(rc.np.lib.format, "open_memmap",
                                      side_effect=OSError(27, "File too large")),
            "flush": mock.patch.object(np.memmap, "flush",
                                       side_effect=OSError(28, "No space left on device")),
            "manifest": mock.patch.object(rc, "_write_json_atomically",
                                          side_effect=OSError(28, "No space left on device")),
        }
        for stage, patch in cases.items():
            with self.subTest(stage=stage):
                out = Path(self._tmp.name) / f"failed_{stage}"
                with patch, self.assertRaises(OSError):
                    self.f64.build(self.f64.selected[:2], out=out)
                failure = self.assert_failure_recorded(out, stage, "OSError")
                if stage != "open":
                    self.assertEqual(sorted(failure["partial_files_preserved"]),
                                     ["labels.npy", "sensor.npy", "thermal.npy"])

    def test_provenance_is_stored_as_it_reads_back_so_the_cache_attaches(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.f64.build(self.f64.selected[:1], out=Path(tmp) / "c",
                                   provenance={"seeds": {2: "a", 10: "b"}, "run": ("x", 1)})
            self.assertEqual(cache.manifest["provenance"],
                             {"seeds": {"2": "a", "10": "b"}, "run": ["x", 1]})
            attached = rc.RawNormalCache.attach(cache.out)
            self.assertEqual(attached.manifest["sha256"], cache.manifest["sha256"])

    def test_provenance_that_is_not_plain_json_is_refused_before_anything_is_created(self):
        for provenance in ({"git_dirty": np.bool_(True)}, {"loss": float("nan")},
                           {1: "a", "1": "b"}):
            with self.subTest(provenance=provenance):
                reader = CountingReader(self.f64.canonical)
                out = Path(self._tmp.name) / "bad_provenance"
                with self.assertRaisesRegex(ValueError, "provenance is not plain JSON"):
                    self.f64.build(reader=reader, out=out, provenance=provenance)
                self.assertEqual(reader.calls, [])
                self.assertFalse(out.exists())

    def test_disk_space_is_checked_before_anything_is_created(self):
        out = Path(self._tmp.name) / "no_space"
        reader = CountingReader(self.f64.canonical)
        usage = mock.Mock(free=10)
        with mock.patch.object(rc.shutil, "disk_usage", return_value=usage):
            with self.assertRaisesRegex(ValueError, "are free"):
                self.f64.build(reader=reader, out=out)
        self.assertEqual(reader.calls, [])
        self.assertFalse(out.exists())


class VerificationTests(SharedFixture):
    def test_payload_that_differs_from_the_source_after_flush_is_caught(self):
        real = rc.file_digests

        def lying(path):
            file_sha, _, offset = real(path)
            return file_sha, "0" * 64, offset
        out = Path(self._tmp.name) / "lying_payload"
        with mock.patch.object(rc, "file_digests", lying):
            with self.assertRaisesRegex(ValueError, "payload read back from disk differs"):
                self.f64.build(out=out)
        failure = json.loads((out / "failure.json").read_text())
        self.assertEqual(failure["stage"], "verify_after_flush")
        self.assertFalse((out / "manifest.json").exists())

    def test_file_that_differs_from_header_plus_source_is_caught(self):
        real = rc.file_digests

        def lying(path):
            _, payload_sha, offset = real(path)
            return "0" * 64, payload_sha, offset
        with mock.patch.object(rc, "file_digests", lying):
            with self.assertRaisesRegex(ValueError, "expected npy header"):
                self.f64.build(self.f64.selected[:1])

    def test_a_write_that_lands_wrong_bytes_is_caught(self):
        real_open = np.lib.format.open_memmap

        class Corrupting:
            def __init__(self, array):
                self.array = array

            def __setitem__(self, index, value):
                value = np.array(value, copy=True)
                value.flat[0] = value.flat[0] + 1
                self.array[index] = value

            def flush(self):
                self.array.flush()

            @property
            def shape(self):
                return self.array.shape

        def corrupting_open(path, **kwargs):
            array = real_open(path, **kwargs)
            return Corrupting(array) if Path(path).name == "thermal.npy" else array
        out = Path(self._tmp.name) / "corrupt_write"
        with mock.patch.object(rc.np.lib.format, "open_memmap", corrupting_open):
            with self.assertRaisesRegex(ValueError, "thermal.npy payload read back"):
                self.f64.build(self.f64.selected[:2], out=out)
        self.assertEqual(json.loads((out / "failure.json").read_text())["stage"],
                         "verify_after_flush")

    def test_expected_header_matches_what_open_memmap_writes(self):
        for name in rc.ARRAYS:
            entry = self.cache32.manifest["arrays"][name]
            header = rc._npy_header(np.dtype(entry["dtype"]), tuple(entry["shape"]))
            with open(self.cache32.out / entry["file"], "rb") as handle:
                self.assertEqual(handle.read(len(header)), header)
            self.assertEqual(len(header), entry["header_bytes"])

    def test_attach_round_trips_and_detects_a_shape_preserving_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.f64.build(self.f64.selected[:2], out=Path(tmp) / "c")
            attached = rc.RawNormalCache.attach(cache.out).open()
            for x, y in zip(cache.open().window(1), attached.window(1)):
                self.assertEqual(x.tobytes(), y.tobytes())
            del attached, cache
            data = np.load(Path(tmp) / "c" / "thermal.npy", mmap_mode="r+")
            data[1, 5, 10, 10] += 1.0
            data.flush()
            del data
            with self.assertRaisesRegex(ValueError, "thermal.npy content changed"):
                rc.RawNormalCache.attach(Path(tmp) / "c")

    def test_attach_refuses_an_edited_or_unfinished_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "c"
            self.f64.build(self.f64.selected[:1], out=out)
            path = out / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["windows"][0]["role"] = "calibration"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest content differs"):
                rc.RawNormalCache.attach(out)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "never finished"):
                rc.RawNormalCache.attach(out)

    def test_read_time_verification_catches_a_change_after_attach(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.f64.build(self.f64.selected[:1], out=Path(tmp) / "c").open()
            window = self.f64.selected[0]
            cache.reader(verify_each_read=True)(window)
            cache._data = None
            data = np.load(Path(tmp) / "c" / "sensor.npy", mmap_mode="r+")
            data[0, 0, 0] += 1.0
            data.flush()
            del data
            cache.open()
            with self.assertRaisesRegex(ValueError, "sensor of .* changed since"):
                cache.reader(verify_each_read=True)(window)

    def test_unverified_or_unopened_cache_is_refused(self):
        unverified = rc.RawNormalCache(self.cache64.out, self.cache64.manifest)
        with self.assertRaisesRegex(ValueError, "never verified"):
            unverified.open()
        with self.assertRaisesRegex(ValueError, "must be opened"):
            unverified.window(0)


class LeafTests(unittest.TestCase):
    def test_module_is_a_numpy_leaf(self):
        source = (ROOT / "src/data/anomaly_raw_cache.py").read_text()
        imports = [line.split()[1] for line in source.splitlines()
                   if line.startswith(("import ", "from "))]
        self.assertNotIn("torch", imports)
        self.assertFalse(any(name.startswith(("src", ".")) for name in imports))
        self.assertIn("numpy", imports)

    def test_no_import_anywhere_in_the_module_reaches_torch_or_src(self):
        """Every import statement at any depth, and no dynamic import by name."""
        tree = ast.parse((ROOT / "src/data/anomaly_raw_cache.py").read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "relative import")
                names.append(node.module)
            elif isinstance(node, ast.Call):
                func = node.func
                called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                self.assertNotIn(called, ("import_module", "__import__"), "dynamic import")
        roots = {name.split(".")[0] for name in names}
        self.assertIn("numpy", roots)
        self.assertLessEqual(roots, set(sys.stdlib_module_names) | {"numpy"})

    def test_loading_the_module_imports_neither_torch_nor_the_src_package(self):
        """Fresh interpreter with stub torch and src packages on the path, so an import of
        either, however guarded, would succeed and be recorded rather than fail silently."""
        with tempfile.TemporaryDirectory() as tmp:
            for stub in ("torch", "src", "src/data"):
                (Path(tmp) / stub).mkdir(parents=True, exist_ok=True)
                (Path(tmp) / stub / "__init__.py").write_text("")
            script = (
                "import importlib.util, json, sys\n"
                f"spec = importlib.util.spec_from_file_location('m', {str(ROOT / 'src/data/anomaly_raw_cache.py')!r})\n"
                "module = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(module)\n"
                "print(json.dumps(sorted(k for k in sys.modules\n"
                "                        if k.split('.')[0] in ('torch', 'src'))))\n")
            env = dict(os.environ, PYTHONPATH=tmp, PYTHONNOUSERSITE="1")
            result = subprocess.run([sys.executable, "-c", script], env=env, cwd=tmp,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [])

    def test_sampler_and_dev_weighting_are_not_decided_here(self):
        for name in ("sampler", "sample", "weight", "epoch_permutation", "synthetic_stream"):
            self.assertFalse(any(name in attr.lower() for attr in dir(rc)), name)

    def test_public_api_is_exactly_the_cache_and_its_checks(self):
        """A new public def, class or method, whatever its name, changes this set."""
        own = {name for name in dir(rc) if not name.startswith("_")
               and callable(getattr(rc, name))
               and getattr(getattr(rc, name), "__module__", None) == rc.__name__}
        self.assertEqual(own, {"RawNormalCache", "check_array", "check_windows",
                               "declared_layout", "file_digests", "plain_json",
                               "raw_sources", "required_bytes", "select_windows",
                               "window_identity"})
        methods = {name for name in vars(rc.RawNormalCache) if not name.startswith("_")}
        self.assertEqual(methods, {"attach", "build", "open", "reader", "window"})


if __name__ == "__main__":
    unittest.main()
