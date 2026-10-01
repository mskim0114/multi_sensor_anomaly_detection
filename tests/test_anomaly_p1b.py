"""P1b metadata gates and actual main paths on disposable artificial raw data.

All evidence fabricated inside these tests is explicitly a gate fixture. Only the
external verifier's exact-source test report is experimental preparation evidence.
The real-data training CLI has no CPU/short-epoch switch.
"""
import argparse
import ast
from collections import Counter
import contextlib
import copy
import functools
import hashlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def leaf(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p1b = leaf("_test_p1b_runner", "src/train_anomaly_p1b.py")
raw = leaf("_test_p1b_raw_fixture", "tests/test_anomaly_raw_cache.py")
MAIN_RESULTS = {}
FIXTURE_STEPS = 0


def write_json(path, value, signed=False):
    value = copy.deepcopy(value)
    if signed:
        value.pop("sha256", None)
        value["sha256"] = p1b.sa.digest_json(value)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return value


def expected_tests():
    return sum(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
               for n in ast.walk(ast.parse(Path(__file__).read_text())))


class ExperimentFixture:
    def __init__(self, directory):
        self.root = Path(directory)
        self.raw = raw.Fixture(self.root, np.float64, raw.F64)
        self.prepared, self.bank, self.cache_run = (self.root / n for n in ("prepared", "bank", "cache-run"))
        for path in (self.prepared, self.bank, self.cache_run):
            path.mkdir()
        normalizer = copy.deepcopy(raw.NORMALIZER)
        normalizer.pop("sha256")
        # Artificial scales keep every device/family/strength cell represented in
        # this tiny fixture; these are not estimated project-data statistics.
        normalizer["sensor_std"] = [0.02] * 8
        ids = sorted({i for w in self.raw.selected if raw.ROLES[w["device"]] == "fit" for i in w["raw_base_ids"]})
        normalizer.update(fit_raw_ids=ids, fit_ids_sha256=p1b.sa.digest_json(ids))
        normalizer["sha256"] = p1b.sa.digest_json(normalizer)
        write_json(self.prepared / "raw_manifest.json", self.raw.ledger)
        split = write_json(self.prepared / "split_manifest.json", {
            "protocol_id": p1b.sa.PROTOCOL, "data_gate": "PASS", "windows": self.raw.windows,
            "provenance": {"data_root": str(self.raw.data_root)},
            "source_manifest_sha256": self.raw.source_sha,
            "folds": {"0": {"roles": raw.ROLES, "normalizer": normalizer}}}, signed=True)
        identities = [p1b.rc.window_identity(w, raw.ROLES[w["device"]]) for w in self.raw.selected]
        layout = p1b.rc.declared_layout(raw.F64)
        binding = {"fold": 0,
                   "source_manifest_file_sha256": p1b.file_sha(self.prepared / "raw_manifest.json"),
                   "source_manifest_payload_sha256": self.raw.source_sha,
                   "split_manifest_file_sha256": p1b.file_sha(self.prepared / "split_manifest.json"),
                   "split_manifest_payload_sha256": split["sha256"],
                   "roles_sha256": p1b.sa.digest_json(raw.ROLES), "count": len(identities),
                   "role_counts": dict(Counter(i["role"] for i in identities)),
                   "base_order_sha256": p1b.sa.digest_json([i["base_window_id"] for i in identities]),
                   "role_order_sha256": p1b.sa.digest_json([i["role"] for i in identities]),
                   "window_order_sha256": p1b.sa.digest_json(identities),
                   "declared_dtypes": {n: layout[n].str for n in p1b.rc.ARRAYS},
                   "array_payload_bytes": p1b.rc.required_bytes(len(identities), layout)}
        provenance = {"environment_profile": "SYNTHETIC_FIXTURE", "source_files": p1b.cache_cli.source_hashes(),
                      "normalizer_sha256": normalizer["sha256"], "binding": binding}
        write_json(self.cache_run / "provenance.json", provenance)
        cache = p1b.rc.RawNormalCache.build(
            self.cache_run / "cache", self.raw.selected, raw.ROLES, self.raw.canonical,
            data_root=self.raw.data_root, raw_manifest=self.raw.ledger, source_manifest_sha256=self.raw.source_sha,
            dtypes=raw.F64, provenance=provenance).open()
        groups = len({(i["role"], i["device"]) for i in identities})
        parity = p1b.cache_cli.verify_parity(cache, {"windows": self.raw.selected, "roles": raw.ROLES,
                                                   "normalizer": normalizer,
                                                   "cost_plan": {"normal_tensor_pairs": 2 * len(identities),
                                                                 "synthetic_paired_attempts": groups * 18}},
                                             self.raw.canonical)
        write_json(self.cache_run / "result.json", {
            "schema": p1b.cache_cli.RESULT_SCHEMA, "gate": "PASS", "fold": 0, "optimizer_steps": 0,
            "source_files": p1b.cache_cli.source_hashes(), "binding": binding,
            "cache_manifest_sha256": cache.manifest["sha256"],
            "provenance_file_sha256": p1b.file_sha(self.cache_run / "provenance.json"), "parity": parity,
            "canonical_window_reads": 2 * len(identities),
            "canonical_role_reads": {r: 2 * count for r, count in binding["role_counts"].items()},
            "raw_roles_read": ["dev", "fit"], "full_P0_gate": "NOT_EVALUATED"}, signed=True)
        events = []
        for window in self.raw.selected:
            role = raw.ROLES[window["device"]]
            s, t, labels = cache.reader()(window)
            for family, strength in itertools.product(("sensor-only", "thermal-only", "coupled"), (1, 2, 4)):
                _, _, meta = p1b.sa.inject_blind(
                    s, t, tick_labels=labels, normalizer=normalizer, protocol=p1b.sa.PROTOCOL,
                    fold=0, role=role, base_window_id=window["base_window_id"], family=family,
                    strength=strength, generator_seed=42, ct_channel="CT1")
                events.append({**meta, "role": role, "device": window["device"],
                               "base_window_id": window["base_window_id"]})
        (self.bank / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        write_json(self.bank / "bank.json", {
            "protocol_id": p1b.sa.PROTOCOL, "bank_gate": "PASS", "fold": 0, "seed": 42, "ct_channel": "CT1",
            "normalizer_sha256": normalizer["sha256"], "event_manifest_sha256": p1b.file_sha(self.bank / "events.jsonl"),
            "heldout_read": False, "calibration_read": False, "gradient_training_runs": 0,
            "full_P0_gate": "NOT_EVALUATED", "coverage": {
                r: {"bank_gate": "PASS", "expected_cells": 9 * len({w["device"] for w in self.raw.selected
                                                                    if raw.ROLES[w["device"]] == r}),
                    "missing_accepted_cells": []} for r in ("fit", "dev")}}, signed=True)
        cache._data = None

    def clone(self, root):
        for name in ("prepared", "bank"):
            shutil.copytree(self.root / name, root / name)
        shutil.copytree(self.cache_run, root / "cache-run", ignore=shutil.ignore_patterns("*.npy"))
        for array in (self.cache_run / "cache").glob("*.npy"):
            (root / "cache-run/cache" / array.name).symlink_to(array)


class GateFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.fixture = ExperimentFixture(cls.shared.name)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixture.clone(self.root)
        self.prepared, self.bank, self.cache_run = (self.root / n for n in ("prepared", "bank", "cache-run"))
        self.out = self.root / "outputs/run"
        self.sampler_path, self.runner_path = (self.root / n for n in ("sampler-fixtures.json", "runner-fixtures.json"))
        split = json.loads((self.prepared / "split_manifest.json").read_text())
        events = [json.loads(line) for line in (self.bank / "events.jsonl").read_text().splitlines()]
        fold, norm = split["folds"]["0"], split["folds"]["0"]["normalizer"]
        indexes = {role: p1b.bs.build_event_index(
            [w for w in split["windows"] if raw.ROLES[w["device"]] == role], raw.ROLES,
            [e for e in events if e["role"] == role], role=role, fold=0, generator_seed=42, ct_channel="CT1",
            normalizer_sha256=norm["sha256"], fit_ids_sha256=norm["fit_ids_sha256"]) for role in ("fit", "dev")}
        live = p1b.source_hashes()
        self.sampler = {"status": "PASS", "unit_tests": dict(tests_run=29, skipped=0, errors=0, failures=0),
                        "purpose": "fabricated gate fixture; never external research evidence", "optimizer_steps_actual": 0,
                        "source_files": {p: live[p] for p in ("src/data/anomaly_bin_sampler.py", "tests/test_anomaly_bin_sampler.py")},
                        "indexes": {r: i.record() for r, i in indexes.items()},
                        "input_file_sha256": {"split_manifest.json": p1b.file_sha(self.prepared / "split_manifest.json"),
                                              "events.jsonl": p1b.file_sha(self.bank / "events.jsonl")}}
        self.runner = {"schema": p1b.FIXTURE_SCHEMA, "gate": "PASS", "source_files": live,
                       "unit_tests": dict(tests_run=expected_tests(), skipped=0, errors=0, failures=0),
                       "purpose": "fabricated gate fixture; never external research evidence", "git_commit_sha": "a" * 40,
                       "real_data_optimizer_steps": 0, "fixture_optimizer_steps": 1,
                       "main_completed_arms": sorted(f"{a}-{o}" for a, o in p1b.ARMS)}
        self.seal()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(p1b, "OUTPUT_PARENT", self.root / "outputs"))

    def seal(self):
        self.sampler = write_json(self.sampler_path, self.sampler, signed=True)
        self.runner = write_json(self.runner_path, self.runner, signed=True)

    def args(self, arm="S", objective="BIN"):
        return argparse.Namespace(prepared=self.prepared, bank_result=self.bank, cache_run=self.cache_run,
                                  cache_result_sha256=p1b.file_sha(self.cache_run / "result.json"),
                                  sampler_fixtures=self.sampler_path, sampler_fixtures_sha256=p1b.file_sha(self.sampler_path),
                                  runner_fixtures=self.runner_path, runner_fixtures_sha256=p1b.file_sha(self.runner_path),
                                  out=self.out, arm=arm, objective=objective, micro_batch=2, quiet=True)

    def argv(self, arm="S", objective="BIN"):
        return [part for key, value in vars(self.args(arm, objective)).items()
                for part in (("--quiet",) if key == "quiet" else ("--" + key.replace("_", "-"), str(value)))]

    def inputs(self):
        return p1b.validate_inputs(self.args())

    def cache(self):
        return p1b.rc.RawNormalCache.attach(self.cache_run / "cache").open()

    def fail_before_read(self, match):
        with mock.patch.object(p1b.rc.RawNormalCache, "attach") as attach, \
                mock.patch.object(p1b, "_training") as training:
            with self.assertRaisesRegex(ValueError, match):
                p1b.main(self.argv())
            attach.assert_not_called()
            training.assert_not_called()
        failure = json.loads((self.out / "failure.json").read_text())
        self.assertEqual(failure["stage"], "inputs")
        self.assertEqual(failure["context"]["optimizer_steps_completed"], 0)
        self.assertFalse(failure["run_usable"])
        self.assertFalse((self.out / "result.json").exists())
        return failure

    def edit_json(self, path, change):
        value = json.loads(path.read_text())
        change(value)
        return write_json(path, value, signed=True)

    def refresh_cache_links(self, *, change_provenance=None, change_manifest=None, change_result=None):
        provenance_path = self.cache_run / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        if change_provenance:
            change_provenance(provenance)
        write_json(provenance_path, provenance)
        manifest = json.loads((self.cache_run / "cache/manifest.json").read_text())
        manifest["provenance"] = provenance
        if change_manifest:
            change_manifest(manifest)
        manifest = write_json(self.cache_run / "cache/manifest.json", manifest, signed=True)
        result = json.loads((self.cache_run / "result.json").read_text())
        result.update(provenance_file_sha256=p1b.file_sha(provenance_path), cache_manifest_sha256=manifest["sha256"],
                      source_files=provenance["source_files"], binding=provenance["binding"])
        if change_result:
            change_result(result)
        write_json(self.cache_run / "result.json", result, signed=True)

    def negative_cases(self, cases):
        """Restore all metadata between single-field cases; data arrays stay untouched."""
        saved = {p: p.read_bytes() for directory in (self.prepared, self.bank, self.cache_run)
                 for p in directory.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl")}
        sampler, runner = copy.deepcopy(self.sampler), copy.deepcopy(self.runner)
        for i, (label, change, expected) in enumerate(cases):
            with self.subTest(case=label):
                for directory in (self.prepared, self.bank, self.cache_run):
                    for path in directory.rglob("*"):
                        if path.is_file() and path.suffix in (".json", ".jsonl") and path not in saved:
                            path.unlink()
                for path, content in saved.items():
                    path.write_bytes(content)
                self.sampler, self.runner = copy.deepcopy(sampler), copy.deepcopy(runner)
                self.out = self.root / "outputs" / f"case-{i}"
                change()
                self.seal()
                self.fail_before_read(expected)

    def rebind_normalizer_fixture(self, normalizer):
        """Update every independent pin, isolating the Normal fit-ID provenance gate."""
        split_path = self.prepared / "split_manifest.json"
        split = json.loads(split_path.read_text())
        split["folds"]["0"]["normalizer"] = normalizer
        split = write_json(split_path, split, signed=True)
        event_path = self.bank / "events.jsonl"
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        for event in events:
            event.update(normalizer_sha256=normalizer["sha256"], fit_ids_sha256=normalizer["fit_ids_sha256"])
        event_path.write_text("".join(json.dumps(e) + "\n" for e in events))
        self.edit_json(self.bank / "bank.json", lambda b: b.update(
            normalizer_sha256=normalizer["sha256"], event_manifest_sha256=p1b.file_sha(event_path)))
        for role in ("fit", "dev"):
            windows = [w for w in split["windows"] if raw.ROLES[w["device"]] == role]
            index = p1b.bs.build_event_index(
                windows, raw.ROLES, [e for e in events if e["role"] == role], role=role, fold=0,
                generator_seed=42, ct_channel="CT1", normalizer_sha256=normalizer["sha256"],
                fit_ids_sha256=normalizer["fit_ids_sha256"])
            self.sampler["indexes"][role] = index.record()
        self.sampler["input_file_sha256"] = {"split_manifest.json": p1b.file_sha(split_path),
                                               "events.jsonl": p1b.file_sha(event_path)}
        def provenance_change(p):
            p["normalizer_sha256"] = normalizer["sha256"]
            p["binding"].update(split_manifest_file_sha256=p1b.file_sha(split_path),
                                 split_manifest_payload_sha256=split["sha256"])
        self.refresh_cache_links(change_provenance=provenance_change)


class GateAndLoaderTests(GateFixture):
    def test_evidence_numeric_provenance_gates_each_have_a_negative_case(self):
        cases = []
        for key, value in (("real_data_optimizer_steps", 1), ("fixture_optimizer_steps", 0),
                           ("git_commit_sha", "a" * 39), ("git_commit_sha", "g" * 40)):
            cases.append((f"runner-{key}-{value}", lambda k=key, v=value: self.runner.update({k: v}),
                          "runner fixture|git revision"))
        for field in ("failures", "errors", "skipped"):
            cases.append(("sampler-" + field, lambda k=field: self.sampler["unit_tests"].update({k: 1}), "sampler evidence"))
        cases.append(("sampler-steps", lambda: self.sampler.update(optimizer_steps_actual=1), "sampler evidence"))
        cases.append(("sampler-input-pin", lambda: self.sampler["input_file_sha256"].update(
            {"events.jsonl": "f" * 64}), "sampler input file pins"))
        self.negative_cases(cases)

    def test_split_and_bank_semantic_gates_each_have_a_negative_case(self):
        cases = []
        for key, value in (("data_gate", "FAIL"), ("protocol_id", "foreign-protocol")):
            cases.append(("split-" + key, lambda k=key, v=value: self.edit_json(
                self.prepared / "split_manifest.json", lambda d: d.update({k: v})), "split is not"))
        for key, value in (("heldout_read", True), ("calibration_read", True), ("ct_channel", "CT2"),
                           ("gradient_training_runs", 1), ("normalizer_sha256", "f" * 64),
                           ("seed", 123), ("full_P0_gate", "PASS")):
            cases.append(("bank-" + key, lambda k=key, v=value: self.edit_json(
                self.bank / "bank.json", lambda d: d.update({k: v})), "bank namespace"))
        for key, value in (("expected_cells", 1), ("missing_accepted_cells", ["missing"]), ("bank_gate", "FAIL")):
            cases.append(("coverage-" + key, lambda k=key, v=value: self.edit_json(
                self.bank / "bank.json", lambda d: d["coverage"]["fit"].update({k: v})), "accepted-cell coverage"))
        self.negative_cases(cases)

    def test_normalizer_foreign_fit_ids_are_rejected_despite_consistent_downstream_pins(self):
        split = json.loads((self.prepared / "split_manifest.json").read_text())
        normalizer = split["folds"]["0"]["normalizer"]
        normalizer.pop("sha256")
        normalizer["fit_raw_ids"] = sorted(normalizer["fit_raw_ids"] + ["foreign-fit-id"])
        normalizer["fit_ids_sha256"] = p1b.sa.digest_json(normalizer["fit_raw_ids"])
        normalizer["sha256"] = p1b.sa.digest_json(normalizer)
        self.rebind_normalizer_fixture(normalizer)
        self.seal()
        self.fail_before_read("normalizer fit provenance")

    def test_cache_completion_semantic_gates_each_have_a_negative_case(self):
        cases = []
        for key, value in (("allowed_roles", ["fit", "dev", "heldout"]), ("verified_after_flush", False)):
            cases.append((key, lambda k=key, v=value: self.refresh_cache_links(
                change_manifest=lambda d: d.update({k: v})), "exact selected"))
        cases.append(("cache-array-file", lambda: self.refresh_cache_links(change_manifest=lambda d:
            d["arrays"]["thermal"].update(file="foreign.npy")), "canonical layout"))
        cases.append(("nested-cache-failure", lambda: (self.cache_run / "cache/failure.json").write_text("{}"), "failed cache"))
        cases.append(("provenance-pin", lambda: self.refresh_cache_links(change_result=lambda d:
            d.update(provenance_file_sha256="f" * 64)), "cache result/provenance"))
        cases.append(("stale-CLI-source-everywhere", lambda: self.refresh_cache_links(change_provenance=lambda d:
            d["source_files"].update({"src/build_anomaly_raw_cache.py": "f" * 64})), "cache result/provenance"))
        cases.append(("forged-binding-consistent-everywhere", lambda: self.refresh_cache_links(change_provenance=lambda d:
            d["binding"].update(source_manifest_file_sha256="f" * 64)), "cache manifest/order"))
        for key, value in (("synthetic_exhaustive", True), ("synthetic_sample_windows", 1),
                           ("raw_windows", 1), ("synthetic_paired_attempts", 55),
                           ("synthetic_accepted", -1), ("synthetic_rejected", -1)):
            cases.append(("parity-" + key, lambda k=key, v=value: self.refresh_cache_links(
                change_result=lambda d: d["parity"].update({k: v})), "parity/read coverage"))
        cases.append(("canonical-read-count", lambda: self.refresh_cache_links(change_result=lambda d:
            d.update(canonical_window_reads=1)), "parity/read coverage"))
        self.negative_cases(cases)

    def test_raw_manifest_payload_tamper_is_rejected_with_updated_file_pin(self):
        path = self.prepared / "raw_manifest.json"
        raw_manifest = json.loads(path.read_text())
        raw_manifest[0]["sha256"] = "f" * 64
        write_json(path, raw_manifest)
        self.refresh_cache_links(change_provenance=lambda p: p["binding"].update(
            source_manifest_file_sha256=p1b.file_sha(path)))
        self.fail_before_read("raw source manifest payload")

    def test_unsigned_payload_edit_is_rejected_even_with_current_file_pin(self):
        record = json.loads(self.runner_path.read_text())
        record["purpose"] = "unsigned edit"
        write_json(self.runner_path, record)
        self.fail_before_read("payload digest mismatch")

    def test_file_pin_mismatch_is_rejected_even_when_payload_is_resigned(self):
        argv = self.argv()
        self.runner["purpose"] = "valid digest but changed pinned bytes"
        self.seal()
        with mock.patch.object(p1b.rc.RawNormalCache, "attach") as attach, \
                mock.patch.object(p1b, "_training") as training, self.assertRaisesRegex(ValueError, "file pin mismatch"):
            p1b.main(argv)
        attach.assert_not_called()
        training.assert_not_called()

    def test_all_reinjection_identity_and_acceptance_fields_have_negative_cases(self):
        inputs, cache = self.inputs(), self.cache()
        self.addCleanup(setattr, cache, "_data", None)
        pair = p1b.bs.BasePairedSampler(inputs["indexes"]["fit"], training_seed=42).epoch_pairs(1)[0]
        ref = p1b.paired_examples((pair,))[1]
        original = p1b.sa.inject_blind
        changes = {"event_id": "f" * 64, "event_instance_id": "e" * 64,
                   "family": "foreign", "strength": 99, "ct_channel": "CT2",
                   "sensor_changed": not ref.sensor_changed, "thermal_changed": not ref.thermal_changed,
                   "accepted": False, "parameters": {"changed": True}}
        for field, value in changes.items():
            with self.subTest(field=field):
                loader = p1b.PairLoader(inputs, cache.reader())
                loader.context = {"epoch": 1, "batch": 0}
                def corrupt(*args, **kwargs):
                    s, t, meta = original(*args, **kwargs)
                    meta[field] = value
                    return s, t, meta
                with mock.patch.object(p1b.sa, "inject_blind", side_effect=corrupt), \
                        self.assertRaises(p1b.P1bFailure):
                    loader.load("fit", ref)
                self.assertEqual(loader.counts["fit_injections"], 0)

    def test_loader_ct1_inputs_match_independent_canonical_normalization(self):
        inputs, cache = self.inputs(), self.cache()
        self.addCleanup(setattr, cache, "_data", None)
        window = inputs["windows"]["fit"][0]
        s, t, _ = self.fixture.raw.canonical(window)
        expected = tuple(np.ascontiguousarray(x, dtype=np.float32) for x in
                         p1b.controls.normalized_pair(s, t, inputs["normalizer"], "CT1"))
        actual = p1b.PairLoader(inputs, cache.reader()).load("fit", p1b.normal_example(window, 0))
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        ct2_sensor = p1b.controls.normalized_pair(s, t, inputs["normalizer"], "CT2")[0]
        self.assertFalse(np.array_equal(actual[0], ct2_sensor.astype(np.float32)))

    def test_cache_io_and_hash_time_are_separate_measurements(self):
        inputs, cache = self.inputs(), self.cache()
        self.addCleanup(setattr, cache, "_data", None)
        read, real_hash, clock = cache.reader(verify_each_read=False), hashlib.sha256, [0.0]
        def timed_read(window):
            arrays = read(window)
            clock[0] += 2.0
            return arrays
        def timed_hash(*args, **kwargs):
            value = real_hash(*args, **kwargs)
            clock[0] += 3.0
            return value
        loader = p1b.PairLoader(inputs, timed_read)
        with mock.patch.object(p1b.time, "perf_counter", side_effect=lambda: clock[0]), \
                mock.patch.object(p1b.hashlib, "sha256", side_effect=timed_hash):
            loader.load("fit", p1b.normal_example(inputs["windows"]["fit"][0], 0))
        self.assertEqual(loader.seconds["cache_io"], 2.0)
        self.assertEqual(loader.seconds["array_hash"], 9.0)

    def test_t_ae_is_refused_before_output_or_model_creation(self):
        with mock.patch.object(p1b, "_training") as training, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            p1b.main(self.argv("T", "AE"))
        training.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_new_output_nested_in_each_input_directory_is_refused(self):
        with mock.patch.object(p1b, "OUTPUT_PARENT", self.root):
            for directory in (self.prepared, self.bank, self.cache_run):
                with self.subTest(directory=directory):
                    self.out = directory / "new-output"
                    with self.assertRaisesRegex(ValueError, "overlaps a source/input"):
                        p1b.main(self.argv())
                    self.assertFalse(self.out.exists())

    def test_metadata_gates_match_frozen_provenance_without_reader_or_torch(self):
        with mock.patch.object(p1b.rc.RawNormalCache, "attach") as attach, \
                mock.patch.object(p1b, "_training") as training:
            inputs = self.inputs()
        self.assertEqual([len(inputs["windows"][r]) for r in ("fit", "dev")], [4, 2])
        attach.assert_not_called()
        training.assert_not_called()

    def test_runner_source_change_is_refused_before_reader_and_optimizer(self):
        self.runner["source_files"]["src/train_anomaly_p1b.py"] = "f" * 64
        self.seal()
        self.fail_before_read("runner fixture")

    def test_runner_incomplete_test_evidence_is_refused(self):
        for changes in ({"skipped": 1}, {"tests_run": expected_tests() - 1}):
            with self.subTest(changes=changes):
                self.runner["unit_tests"] = dict(tests_run=expected_tests(), skipped=0, errors=0, failures=0)
                self.runner["unit_tests"].update(changes)
                self.seal()
                if self.out.exists():
                    self.out = self.out.with_name(self.out.name + "-next")
                self.fail_before_read("runner fixture")

    def test_runner_main_arm_and_optimizer_evidence_is_required(self):
        self.runner["main_completed_arms"].remove("F-AE")
        self.seal()
        self.fail_before_read("runner fixture")

    def test_sampler_binding_change_is_refused_before_reader(self):
        self.sampler["indexes"]["fit"]["sha256"] = "e" * 64
        self.seal()
        self.fail_before_read("sampler-verified")

    def test_cache_failure_marker_is_refused_before_reader(self):
        (self.cache_run / "failure.json").write_text("{}")
        self.fail_before_read("failed cache")

    def test_cache_source_binding_change_is_refused_before_reader(self):
        path = self.cache_run / "result.json"
        result = json.loads(path.read_text())
        result["source_files"]["src/train_anomaly_pilot.py"] = "e" * 64
        write_json(path, result, signed=True)
        self.fail_before_read("cache result/provenance")

    def test_incomplete_cache_parity_is_refused_before_reader(self):
        path = self.cache_run / "result.json"
        result = json.loads(path.read_text())
        result["parity"]["normal_tensor_pairs"] -= 1
        write_json(path, result, signed=True)
        self.fail_before_read("parity/read coverage")

    def test_cache_foreign_role_window_is_refused_before_reader(self):
        path = self.cache_run / "cache/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["windows"][0]["role"] = "heldout"
        manifest = write_json(path, manifest, signed=True)
        result = json.loads((self.cache_run / "result.json").read_text())
        result["cache_manifest_sha256"] = manifest["sha256"]
        write_json(self.cache_run / "result.json", result, signed=True)
        self.fail_before_read("exact selected")

    def test_bank_pin_change_is_refused_before_reader(self):
        with (self.bank / "events.jsonl").open("a") as stream:
            stream.write("{}\n")
        self.fail_before_read("bank namespace")

    def test_bank_foreign_event_role_is_refused_before_reader(self):
        path = self.bank / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[0]["role"] = "calibration"
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
        bank = json.loads((self.bank / "bank.json").read_text())
        bank["event_manifest_sha256"] = p1b.file_sha(path)
        write_json(self.bank / "bank.json", bank, signed=True)
        self.fail_before_read("foreign roles")

    def test_boolean_seed_is_not_a_numeric_fold_alias(self):
        path = self.bank / "bank.json"
        bank = json.loads(path.read_text())
        bank["fold"] = False
        write_json(path, bank, signed=True)
        self.fail_before_read("bank namespace")

    def test_paired_and_full_dev_reinjection_match_all_identity_fields(self):
        inputs, cache = self.inputs(), self.cache()
        self.addCleanup(setattr, cache, "_data", None)
        read = raw.CountingReader(cache.reader())
        loader = p1b.PairLoader(inputs, read)
        sampler = p1b.bs.BasePairedSampler(inputs["indexes"]["fit"], training_seed=42, planned_epochs=2)
        for batch in sampler.epoch_batches(1):
            for example in p1b.paired_examples(batch):
                s, t = loader.load("fit", example)
                self.assertEqual((s.dtype, t.dtype), (np.float32, np.float32))
                self.assertTrue(s.flags.c_contiguous and t.flags.c_contiguous)
        self.assertEqual(len(read.calls), 4)
        loader.begin_phase()
        refs = p1b.bs.dev_examples(inputs["indexes"]["dev"])
        for ref in refs:
            loader.load("dev", ref)
        self.assertEqual(len(read.calls), 6)
        self.assertEqual(loader.counts["dev_injections"], sum(ref.label for ref in refs))
        self.assertEqual(set(loader.seconds), {"cache_io", "array_hash", "injection", "normalization"})

    def test_reinjection_parameters_mismatch_preserves_context(self):
        inputs, cache = self.inputs(), self.cache()
        self.addCleanup(setattr, cache, "_data", None)
        event = next(e for e in inputs["events"] if e["role"] == "fit" and e["accepted"])
        event["parameters"] = {"wrong": True}
        pair = next(e for e in p1b.bs.BasePairedSampler(inputs["indexes"]["fit"], training_seed=42).epoch_pairs(1)
                    if e.base_window_id == event["base_window_id"])
        # Bind the corrupted frozen row to the event that is actually selected.
        selected = next(e for e in inputs["events"] if e["event_id"] == pair.event_id)
        selected["parameters"] = {"wrong": True}
        loader = p1b.PairLoader(inputs, cache.reader())
        loader.context = {"epoch": 1, "batch": 0}
        with self.assertRaisesRegex(p1b.P1bFailure, "parameters") as caught:
            loader.load("fit", p1b.paired_examples((pair,))[1])
        self.assertEqual(caught.exception.context["epoch"], 1)

    def test_per_base_array_hash_refuses_modified_payload(self):
        inputs, cache = self.inputs(), self.cache()
        self.addCleanup(setattr, cache, "_data", None)
        reader = cache.reader()
        def corrupt(window):
            arrays = reader(window)
            arrays[0][0, 0] += 0.125
            return arrays
        loader = p1b.PairLoader(inputs, corrupt)
        with self.assertRaisesRegex(ValueError, "cached sensor changed"):
            loader.load("fit", p1b.normal_example(inputs["windows"]["fit"][0], 0))

    def test_existing_or_source_output_is_refused_without_altering_it(self):
        self.out.parent.mkdir()
        self.out.mkdir()
        marker = self.out / "marker"
        marker.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "new directory"):
            p1b.main(self.argv())
        self.assertEqual(marker.read_text(), "preserve")

    def test_atomic_json_serialization_failure_preserves_original_and_temp(self):
        path = self.root / "atomic.json"
        path.write_text('{"old":true}\n')
        with self.assertRaises(ValueError):
            p1b._write_json(path, {"value": float("nan")})
        self.assertEqual(json.loads(path.read_text()), {"old": True})
        self.assertTrue(path.with_name(path.name + ".tmp").exists())

    def test_real_cli_exposes_no_short_epoch_or_cpu_mode(self):
        for extra in (("--epochs", "1"), ("--device", "cpu")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p1b.main(self.argv() + list(extra))
        self.assertFalse(self.out.exists())

    def test_failure_writer_failure_does_not_replace_original_interrupt(self):
        with mock.patch.object(p1b, "validate_inputs", side_effect=KeyboardInterrupt("original interruption")), \
                mock.patch.object(p1b, "_write_json", side_effect=OSError("disk full")), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(KeyboardInterrupt, "original interruption"):
                p1b.main(self.argv())


@raw.requires_torch
class TrainingAndMainTests(GateFixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        p1b._training()
        cls.threads = p1b.torch.get_num_threads()
        p1b.torch.set_num_threads(1)
        original_step = p1b.torch.optim.AdamW.step

        @functools.wraps(original_step)
        def count_step(optimizer, *args, **kwargs):
            global FIXTURE_STEPS
            result = original_step(optimizer, *args, **kwargs)
            FIXTURE_STEPS += 1
            return result

        cls.counter_patch = mock.patch.object(p1b.torch.optim.AdamW, "step", new=count_step)
        cls.counter_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.counter_patch.stop()
        p1b.torch.set_num_threads(cls.threads)
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.stack.enter_context(mock.patch.object(p1b, "REAL_EPOCHS", 2))
        self.stack.enter_context(mock.patch.object(p1b, "runtime_environment", return_value=(
            p1b.torch.device("cpu"), {"environment_profile": "SYNTHETIC_FIXTURE", "hostname": "fixture",
                                       "python_version": sys.version.split()[0], "numpy_version": np.__version__,
                                       "torch_version": p1b.torch.__version__, "jetpack_l4t": None,
                                       "sensor_driver_versions": {}}, "synthetic_fixture")))

    def tiny_models(self, *, constant=0.0, input_term=False, nan_eval=False):
        th = p1b.torch

        class TinyArm(th.nn.Module):
            def __init__(self, arm, objective):
                super().__init__()
                self.arm, self.objective = arm, objective
                self.offset = th.nn.Parameter(th.tensor(float(constant)))

            def forward(self, **tensors):
                if set(tensors) != set(p1b.detectors.ARM_MODALITIES[self.arm]):
                    raise AssertionError("arm consumed a wrong modality")
                value = next(iter(tensors.values()))
                if self.objective == "BIN":
                    logits = self.offset.expand(len(value))
                    if input_term:
                        logits = logits + value.flatten(1).mean(1)
                    if nan_eval and not self.training:
                        logits = logits + float("nan")
                    return {"logit": logits}
                return {name + "_reconstruction": th.zeros_like(x) + self.offset for name, x in tensors.items()}

        def factory(objective, *, seed):
            return {a: TinyArm(a, objective) for a in ("S", "T", "F")}, {"fixture_only": True, "seed": seed}

        return mock.patch.object(p1b.detectors, "build_arm_set", side_effect=factory)

    def run_main(self, arm="S", objective="BIN"):
        with contextlib.redirect_stdout(io.StringIO()):
            status = p1b.main(self.argv(arm, objective))
        self.assertEqual(status, 0)
        result = json.loads((self.out / "result.json").read_text())
        p1b._signed(result, "final result")
        return result

    def core(self, inputs, *, arm="S", objective="BIN", epochs=2, batch=16, micro=2):
        cache = self.cache()
        self.addCleanup(setattr, cache, "_data", None)
        loader = p1b.PairLoader(inputs, cache.reader())
        record, state = p1b.train_loop(inputs, loader, arm=arm, objective=objective,
                                      device=p1b.torch.device("cpu"), seed=42, scope="synthetic_fixture",
                                      epochs=epochs, nominal_batch=batch, micro_batch=micro)
        return record, state

    def failed_main(self, exception, message, arm="S", objective="BIN"):
        with self.assertRaisesRegex(exception, message):
            p1b.main(self.argv(arm, objective))
        failure = json.loads((self.out / "failure.json").read_text())
        self.assertFalse(failure["run_usable"])
        self.assertEqual(failure["exception"], exception.__name__)
        self.assertFalse((self.out / "result.json").exists())
        self.assertFalse(failure["retried"])
        return failure

    def test_actual_main_all_five_real_architectures_reach_final_json(self):
        results = {}
        for arm, objective in sorted(p1b.ARMS):
            with self.subTest(arm=arm, objective=objective):
                self.out = self.root / "outputs" / (arm + "-" + objective)
                result = self.run_main(arm, objective)
                results[(arm, objective)] = result
                MAIN_RESULTS[arm + "-" + objective] = copy.deepcopy(result)
                self.assertEqual(result["epochs_completed"], 2)
                self.assertEqual(result["optimizer_steps_total"], 2)
                self.assertEqual(result["fixture_optimizer_steps"], 2)
                self.assertEqual(result["real_data_optimizer_steps"], 0)
                self.assertFalse(result["real_30_epoch_budget_completed"])
                self.assertFalse(result["threshold_generated"])
                self.assertFalse(result["heldout_model_performance_computed"])
                self.assertEqual(result["source_dataset_windows_read"], 0)
                self.assertFalse((self.out / "result.json.tmp").exists())
                saved = p1b.torch.load(self.out / "checkpoint.pt", weights_only=True)
                self.assertEqual(p1b.detectors.state_tensor_sha256(saved["state_dict"]), result["selected_state_sha256"])
                self.assertEqual(p1b.file_sha(self.out / "checkpoint.pt"), result["checkpoint_file_sha256"])
                self.assertTrue(result["shared_initialization_manifest"]["copy_is_explicit_not_rng_order"])
                self.assertEqual(result["loader_counts"]["fit_cache_base_reads"], 8)
                self.assertEqual(result["loader_counts"]["dev_cache_base_reads"], 4)
                if objective == "AE":
                    self.assertNotIn("dev_weight_manifest", result)
                    self.assertNotIn("fit_injections", result["loader_counts"])
                else:
                    self.assertEqual(result["loader_counts"]["fit_injections"], 8)
        binaries = [results[(a, "BIN")] for a in ("S", "T", "F")]
        self.assertEqual(len({r["actual_fit_event_stream_sha256"] for r in binaries}), 1)
        self.assertEqual(len({tuple(e["fit_order_manifest_sha256"] for e in r["epochs"]) for r in binaries}), 1)
        ae_orders = [tuple(e["fit_order_manifest_sha256"] for e in results[(a, "AE")]["epochs"]) for a in ("S", "F")]
        self.assertEqual(ae_orders[0], ae_orders[1])
        for objective in ("AE", "BIN"):
            manifest = results[("F", objective)]["shared_initialization_manifest"]
            for hashes in manifest["shared_branch_sha256"].values():
                self.assertEqual(len(set(hashes.values())), 1)

    def test_final_json_serialization_failure_preserves_history_and_checkpoint(self):
        original = p1b._write_json
        def fail_final(path, value):
            if path.name == "result.json":
                return original(path, {**value, "deliberate_nonfinite_fixture": float("nan")})
            return original(path, value)
        with self.tiny_models(), mock.patch.object(p1b, "_write_json", side_effect=fail_final):
            failure = self.failed_main(ValueError, "Out of range float")
        self.assertEqual(failure["stage"], "final_record")
        self.assertEqual(failure["context"]["optimizer_steps_completed"], 2)
        self.assertEqual(len((self.out / "epochs.jsonl").read_text().splitlines()), 2)
        self.assertTrue((self.out / "checkpoint.pt").is_file())
        self.assertIn("result.json.tmp", failure["partial_files_preserved"])

    def test_keyboard_interrupt_during_second_epoch_records_position_and_partial_artifacts(self):
        original = p1b.PairLoader.load
        def interrupt(loader, role, example):
            if role == "fit" and loader.context.get("epoch") == 2:
                raise KeyboardInterrupt("fixture interrupted epoch2")
            return original(loader, role, example)
        with self.tiny_models(), mock.patch.object(p1b.PairLoader, "load", new=interrupt):
            failure = self.failed_main(KeyboardInterrupt, "interrupted epoch2")
        self.assertEqual(failure["stage"], "training")
        self.assertEqual(failure["context"]["epoch"], 2)
        self.assertEqual(failure["context"]["batch"], 0)
        self.assertEqual(failure["context"]["optimizer_steps_completed"], 1)
        self.assertEqual(len((self.out / "epochs.jsonl").read_text().splitlines()), 1)
        self.assertTrue((self.out / "checkpoint.pt").is_file())

    def test_system_exit_from_training_is_also_preserved(self):
        with self.tiny_models(), mock.patch.object(p1b.PairLoader, "load", side_effect=SystemExit("fixture exit")):
            failure = self.failed_main(SystemExit, "fixture exit")
        self.assertEqual(failure["stage"], "training")
        self.assertEqual(failure["context"]["optimizer_steps_completed"], 0)
        self.assertTrue((self.out / "epochs.jsonl").is_file())

    def test_reinjection_flag_mismatch_stops_before_first_optimizer_step(self):
        original = p1b.sa.inject_blind
        def bad_flag(*args, **kwargs):
            s, t, meta = original(*args, **kwargs)
            meta["thermal_changed"] = not meta["thermal_changed"]
            return s, t, meta
        with self.tiny_models(), mock.patch.object(p1b.sa, "inject_blind", side_effect=bad_flag):
            failure = self.failed_main(p1b.P1bFailure, "event identity")
        self.assertEqual(failure["context"]["field"], "thermal_changed")
        self.assertEqual(failure["context"]["optimizer_steps_completed"], 0)
        self.assertFalse((self.out / "checkpoint.pt").exists())

    def test_nonfinite_loss_is_recorded_with_training_context(self):
        with self.tiny_models(constant=float("nan")):
            failure = self.failed_main(p1b.P1bFailure, "nonfinite train loss")
        self.assertEqual(failure["context"]["epoch"], 1)
        self.assertEqual(failure["context"]["phase"], "train")
        self.assertEqual(failure["context"]["optimizer_steps_completed"], 0)

    def test_nonfinite_dev_loss_preserves_completed_optimizer_step(self):
        with self.tiny_models(nan_eval=True):
            failure = self.failed_main(p1b.P1bFailure, "nonfinite dev BCE")
        self.assertEqual(failure["context"]["phase"], "dev")
        self.assertEqual(failure["context"]["optimizer_steps_completed"], 1)

    def test_missing_and_nonfinite_gradients_fail_without_optimizer_step(self):
        for problem in ("missing", "nonfinite"):
            with self.subTest(problem=problem), self.tiny_models() as factory:
                self.out = self.root / "outputs" / problem
                original = factory.side_effect
                def bad_gradients(*args, **kwargs):
                    models, manifest = original(*args, **kwargs)
                    for model in models.values():
                        if problem == "missing":
                            model.unused = p1b.torch.nn.Parameter(p1b.torch.tensor(1.0))
                        else:
                            model.offset.register_hook(lambda grad: p1b.torch.full_like(grad, float("nan")))
                    return models, manifest
                factory.side_effect = bad_gradients
                failure = self.failed_main(p1b.P1bFailure, "missing/nonfinite gradient")
                self.assertEqual(failure["context"]["phase"], "gradient_check")
                self.assertEqual(failure["context"]["optimizer_steps_completed"], 0)

    def test_bin_dev_constant_zero_logit_equals_log_two_without_reaveraging(self):
        with self.tiny_models(), mock.patch.object(p1b.torch.optim.AdamW, "step", return_value=None):
            record, _ = self.core(self.inputs())
        self.assertAlmostEqual(record["dev_checkpoint_loss"], math.log(2), delta=1e-7)
        self.assertEqual(record["dev_checkpoint_epoch"], 1)
        self.assertEqual(record["dev_checkpoint_ties"], [2])
        self.assertGreater(record["example_exposures"]["dev_positive"], record["example_exposures"]["dev_Normal"])

    def test_bin_nonconstant_dev_matches_independent_weighted_bce_not_simple_mean(self):
        inputs = self.inputs()
        index = inputs["indexes"]["dev"]
        devices = {base.device for base in index.bases}
        bases_per_device = Counter(base.device for base in index.bases)
        events_per_base = {base.base_window_id: len(base.events) for base in index.bases}
        expected_terms, unweighted = [], []
        for ref in p1b.bs.dev_examples(index):
            window = inputs["windows"]["dev"][ref.window_index]
            s, t, labels = self.fixture.raw.canonical(window)
            if ref.label:
                s, t, _ = p1b.sa.inject_blind(
                    s, t, tick_labels=labels, normalizer=inputs["normalizer"], protocol=p1b.sa.PROTOCOL,
                    fold=0, role="dev", base_window_id=ref.base_window_id, family=ref.family,
                    strength=ref.strength, generator_seed=42, ct_channel="CT1")
            normalized = np.ascontiguousarray(p1b.controls.normalized_pair(s, t, inputs["normalizer"], "CT1")[0],
                                             dtype=np.float32)
            logit = float(p1b.torch.from_numpy(normalized).mean() + 0.75)
            bce = float(np.logaddexp(0, logit) - ref.label * logit)
            weight = 0.5 / (len(devices) * bases_per_device[ref.device])
            if ref.label:
                weight /= events_per_base[ref.base_window_id]
            expected_terms.append(weight * bce)
            unweighted.append(bce)
        with self.tiny_models(constant=0.75, input_term=True), \
                mock.patch.object(p1b.torch.optim.AdamW, "step", return_value=None):
            record, _ = self.core(inputs, epochs=1, micro=3)
        expected = math.fsum(expected_terms)
        self.assertAlmostEqual(record["dev_checkpoint_loss"], expected, delta=1e-7)
        self.assertGreater(abs(expected - sum(unweighted) / len(unweighted)), 0.01)

    def test_actual_main_disables_duplicate_reader_hashing_and_records_separate_costs(self):
        original = p1b.rc.RawNormalCache.reader
        with self.tiny_models(), mock.patch.object(p1b.rc.RawNormalCache, "reader", autospec=True,
                                                   side_effect=original) as factory:
            result = self.run_main()
        self.assertEqual(factory.call_count, 1)
        self.assertIs(factory.call_args.kwargs["verify_each_read"], False)
        self.assertGreater(result["phase_seconds"]["array_hash"], 0)
        self.assertGreater(result["phase_seconds"]["cache_io"], 0)

    def test_ae_dev_simple_normal_mean_uses_p1a_order_and_no_bin_weights(self):
        inputs = self.inputs()
        cache = self.cache()
        self.addCleanup(setattr, cache, "_data", None)
        loader = p1b.PairLoader(inputs, cache.reader())
        expected = []
        for i, window in enumerate(inputs["windows"]["dev"]):
            s, t = loader.load("dev", p1b.normal_example(window, i))
            expected.append(float(np.mean(s * s, dtype=np.float32)))
        with self.tiny_models(), mock.patch.object(p1b.torch.optim.AdamW, "step", return_value=None):
            record, _ = self.core(inputs, objective="AE", micro=1)
        self.assertAlmostEqual(record["dev_checkpoint_loss"], sum(expected) / len(expected), delta=1e-6)
        self.assertNotIn("dev_weight_manifest", record)
        for entry in record["epochs"]:
            order = p1b.pilot.epoch_permutation(4, seed=42, epoch=entry["epoch"])
            expected_order = [inputs["windows"]["fit"][int(i)]["base_window_id"] for i in order]
            self.assertEqual(entry["fit_order_manifest_sha256"], p1b.sa.digest_json(expected_order))
        self.assertEqual(record["example_exposures"], {"fit_Normal": 8, "dev_Normal": 4})

    def test_ae_dev_unequal_device_sizes_are_not_device_reweighted(self):
        inputs = {"windows": {"fit": [{"base_window_id": "fit", "device": "A"}],
                               "dev": [{"base_window_id": "one", "device": "A"},
                                       {"base_window_id": "three", "device": "B"},
                                       {"base_window_id": "five", "device": "B"}]}}
        class KnownInputs:
            def __init__(self):
                self.counts, self.seconds, self.context = Counter(), Counter(), {}
            def begin_phase(self):
                pass
            def load(self, role, example):
                value = {"fit": 2, "one": 1, "three": 3, "five": 5}[example.base_window_id]
                return np.full((30, 5), value, np.float32), np.zeros((30, 120, 160), np.float32)
        with self.tiny_models(), mock.patch.object(p1b.torch.optim.AdamW, "step", return_value=None):
            record, _ = p1b.train_loop(inputs, KnownInputs(), arm="S", objective="AE",
                                       device=p1b.torch.device("cpu"), seed=42, scope="synthetic_fixture",
                                       epochs=1, nominal_batch=2, micro_batch=2)
        self.assertAlmostEqual(record["dev_checkpoint_loss"], (1 + 9 + 25) / 3)
        self.assertNotAlmostEqual(record["dev_checkpoint_loss"], (1 + (9 + 25) / 2) / 2)

    def test_partial_pairs_and_uneven_microbatches_have_same_gradient_update(self):
        inputs = self.inputs()
        before_steps = []
        original_step = p1b.torch.optim.AdamW.step
        def capture_gradient(optimizer, *args, **kwargs):
            parameter = optimizer.param_groups[0]["params"][0]
            before_steps.append((float(parameter.detach()), float(parameter.grad)))
            return original_step(optimizer, *args, **kwargs)
        with self.tiny_models(constant=0.25, input_term=True), \
                mock.patch.object(p1b.torch.optim.AdamW, "step", new=capture_gradient):
            full, full_state = self.core(inputs, epochs=1, batch=6, micro=6)
            uneven, uneven_state = self.core(inputs, epochs=1, batch=6, micro=5)
        self.assertEqual(full["effective_batch_counts"], {"2": 1, "6": 1})
        self.assertEqual(full["optimizer_steps_total"], 2)
        self.assertEqual(full["example_exposures"]["fit_Normal"], 4)
        self.assertEqual(full["example_exposures"]["fit_positive"], 4)
        self.assertEqual(full["actual_fit_event_stream_sha256"], uneven["actual_fit_event_stream_sha256"])
        for key in full_state:
            p1b.torch.testing.assert_close(full_state[key], uneven_state[key], rtol=1e-6, atol=1e-7)
        cache = self.cache()
        self.addCleanup(setattr, cache, "_data", None)
        loader = p1b.PairLoader(inputs, cache.reader())
        batches = [p1b.paired_examples(b) for b in p1b.bs.BasePairedSampler(
            inputs["indexes"]["fit"], training_seed=42, paired_bases_per_batch=3, planned_epochs=1).epoch_batches(1)]
        self.assertEqual(len(before_steps), 4)
        for position, (offset, actual_gradient) in enumerate(before_steps):
            refs = batches[position % 2]
            gradients = []
            for ref in refs:
                s, _ = loader.load("fit", ref)
                logit = float(p1b.torch.from_numpy(s).mean() + offset)
                gradients.append(1 / (1 + math.exp(-logit)) - ref.label)
            self.assertAlmostEqual(actual_gradient, math.fsum(gradients) / len(refs), delta=1e-7)

    def test_cosine_is_stepped_once_per_epoch_with_used_lr_recorded(self):
        with self.tiny_models():
            record, _ = self.core(self.inputs(), epochs=3, batch=2, micro=2)
        self.assertEqual([e["optimizer_steps"] for e in record["epochs"]], [4, 4, 4])
        for i, entry in enumerate(record["epochs"]):
            self.assertAlmostEqual(entry["lr_used_this_epoch"], 0.001 * (1 + math.cos(math.pi * i / 3)) / 2)
        self.assertEqual(record["scheduler"], {"name": "CosineAnnealingLR", "interval": "epoch", "T_max": 3})

    def test_exact_tie_main_checkpoint_is_first_epoch_with_same_saved_state(self):
        with self.tiny_models(), mock.patch.object(p1b.torch.optim.AdamW, "step", return_value=None):
            result = self.run_main()
        saved = p1b.torch.load(self.out / "checkpoint.pt", weights_only=True)
        self.assertEqual(result["dev_checkpoint_ties"], [2])
        self.assertEqual((result["dev_checkpoint_epoch"], saved["epoch"]), (1, 1))

    def test_input_change_at_final_gate_preserves_completed_training(self):
        original = p1b.run_training
        def alter_after_training(*args, **kwargs):
            record = original(*args, **kwargs)
            self.runner["unit_tests"]["skipped"] = 1
            self.seal()
            return record
        with self.tiny_models(), mock.patch.object(p1b, "run_training", side_effect=alter_after_training):
            failure = self.failed_main(ValueError, "runner fixture")
        self.assertEqual(failure["stage"], "final_inputs")
        self.assertEqual(failure["context"]["optimizer_steps_completed"], 2)
        self.assertEqual(len((self.out / "epochs.jsonl").read_text().splitlines()), 2)
        self.assertTrue((self.out / "checkpoint.pt").exists())

    def test_corrupted_persisted_checkpoint_is_rejected(self):
        original = p1b._save_checkpoint
        def corrupt(path, state, epoch, dev_loss):
            original(path, state, epoch + 10, dev_loss)
        with self.tiny_models(), mock.patch.object(p1b, "_save_checkpoint", side_effect=corrupt):
            failure = self.failed_main(p1b.P1bFailure, "persisted checkpoint/history")
        self.assertEqual(failure["stage"], "checkpoint_verification")
        self.assertTrue((self.out / "checkpoint.pt").exists())

    def test_invalid_core_options_fail_before_model_and_optimizer_creation(self):
        inputs = self.inputs()
        cache = self.cache()
        self.addCleanup(setattr, cache, "_data", None)
        loader = p1b.PairLoader(inputs, cache.reader())
        with mock.patch.object(p1b.detectors, "build_arm_set") as factory, \
                mock.patch.object(p1b.torch.optim, "AdamW") as optimizer:
            with self.assertRaisesRegex(ValueError, "complete pairs"):
                p1b.train_loop(inputs, loader, arm="S", objective="BIN", device=p1b.torch.device("cpu"),
                               seed=42, scope="synthetic_fixture", nominal_batch=3)
            factory.assert_not_called()
            optimizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
