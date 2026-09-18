"""Local regression checks; raw datasets and hardware are never accessed.

Run from the checkout: python -m unittest discover -s tests -v
Torch tests run only in an environment where torch is already installed.
"""

import ast
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_leaf(name, relative):
    # src.data.__init__ eagerly imports torch. Load its independent leaves
    # directly so stdlib/YAML checks can run without installing a training stack.
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


index = load_leaf("portability_index", "src/data/session_index.py")
config = load_leaf("portability_config", "src/data/config.py")
from src.paths import PROJECT_ROOT, project_path


class IndexCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.labels = self.root / "labels"
        self.cache = self.root / "cache"
        self.source.mkdir()
        self.labels.mkdir()
        self.name = "agv01_0901_120000"
        self.add_sample(self.name, 0)

    def add_sample(self, name, state):
        (self.source / (name + ".csv")).write_text("NTC\n1\n")
        self.set_label(name, state)

    def set_label(self, name, state):
        path = self.labels / (name + ".json")
        previous = path.stat().st_mtime_ns if path.exists() else 0
        path.write_text(json.dumps({"annotations": [{"tagging": [{"state": state}]}]}))
        stamp = max(path.stat().st_mtime_ns, previous + 1_000_000)
        os.utime(path, ns=(stamp, stamp))

    def load(self, **kwargs):
        return index.load_or_build_index(
            str(self.source), str(self.labels), str(self.cache), split="train", **kwargs)

    def test_unchanged_inputs_use_cache(self):
        expected = self.load()
        with mock.patch.object(index, "build_session_index", side_effect=AssertionError("cache miss")):
            self.assertEqual(self.load(), expected)

    def test_label_edit_is_seen(self):
        self.load()
        self.set_label(self.name, 3)
        self.assertEqual(self.load().sessions[0].labels, [3])

    def test_csv_addition_and_removal_change_membership(self):
        self.load()
        name = "agv01_0901_120001"
        self.add_sample(name, 1)
        self.assertEqual(self.load().total_samples, 2)
        (self.source / (name + ".csv")).unlink()
        self.assertEqual(self.load().total_samples, 1)

    def test_gap_changes_session_boundaries(self):
        self.add_sample("agv01_0901_120010", 1)
        self.assertEqual(len(self.load(gap_threshold=120).sessions), 1)
        self.assertEqual(len(self.load(gap_threshold=5).sessions), 2)

    def test_different_source_root_cannot_reuse_cache(self):
        self.load()
        other = self.root / "other-source"
        shutil.copytree(self.source, other)
        self.source = other
        with mock.patch.object(index, "build_session_index", wraps=index.build_session_index) as build:
            self.load()
            build.assert_called_once()

    def test_different_label_root_cannot_reuse_cache(self):
        self.load()
        other = self.root / "other-labels"
        shutil.copytree(self.labels, other)
        self.labels = other
        self.set_label(self.name, 2)
        self.assertEqual(self.load().sessions[0].labels, [2])

    def test_split_identity_is_checked(self):
        self.load()
        shutil.copyfile(self.cache / "session_index_train.pkl", self.cache / "session_index_val.pkl")
        actual = index.load_or_build_index(str(self.source), str(self.labels), str(self.cache), split="val")
        self.assertEqual(actual.split, "val")

    def test_missing_label_does_not_hide_behind_cache(self):
        self.load()
        (self.labels / (self.name + ".json")).unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_empty_label_does_not_hide_behind_cache(self):
        self.load()
        (self.labels / (self.name + ".json")).write_text("")
        with self.assertRaises(RuntimeError):
            self.load()

    def test_missing_source_does_not_hide_behind_cache(self):
        self.load()
        self.source = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_legacy_cache_is_rebuilt(self):
        old = self.load()
        path = self.cache / "session_index_train.pkl"
        path.write_bytes(pickle.dumps(old))
        self.set_label(self.name, 2)
        self.assertEqual(self.load().sessions[0].labels, [2])

    def test_corrupt_cache_is_rebuilt(self):
        self.load()
        (self.cache / "session_index_train.pkl").write_bytes(b"truncated pickle")
        self.assertEqual(self.load().sessions[0].labels, [0])

    def test_builder_version_requires_rebuild(self):
        self.load()
        with mock.patch.object(index, "INDEX_CACHE_VERSION", index.INDEX_CACHE_VERSION + 1):
            with mock.patch.object(index, "build_session_index", wraps=index.build_session_index) as build:
                self.load()
                build.assert_called_once()

    def test_force_rebuild_still_works(self):
        self.load()
        with mock.patch.object(index, "build_session_index", wraps=index.build_session_index) as build:
            self.load(force_rebuild=True)
            build.assert_called_once()

    def test_source_content_is_not_index_integrity_validation(self):
        expected = self.load()
        (self.source / (self.name + ".csv")).write_text("")
        with mock.patch.object(index, "build_session_index", side_effect=AssertionError("cache miss")):
            self.assertEqual(self.load(), expected)

    def test_failed_atomic_replace_keeps_previous_cache(self):
        self.load()
        path = self.cache / "session_index_train.pkl"
        old = path.read_bytes()
        self.set_label(self.name, 3)
        with mock.patch.object(index.os, "replace", side_effect=OSError("synthetic failure")):
            with self.assertRaises(OSError):
                self.load()
        self.assertEqual(path.read_bytes(), old)
        self.assertEqual(list(self.cache.iterdir()), [path])

    def test_input_mutation_during_build_is_rejected(self):
        build = index.build_session_index
        def changing_build(*args, **kwargs):
            result = build(*args, **kwargs)
            self.set_label(self.name, 2)
            return result
        with mock.patch.object(index, "build_session_index", side_effect=changing_build):
            with self.assertRaisesRegex(RuntimeError, "changed during build"):
                self.load()
        self.assertFalse((self.cache / "session_index_train.pkl").exists())


class PathTests(unittest.TestCase):
    def test_default_paths_do_not_depend_on_cwd(self):
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                self.assertEqual(config.DataConfig().cache_dir, str(ROOT / "cache"))
                self.assertEqual(project_path("results/v2plus"), str(ROOT / "results/v2plus"))
            finally:
                os.chdir(old_cwd)

    def test_yaml_relative_paths_are_checkout_relative(self):
        cfg = config.DataConfig.from_yaml(ROOT / "configs/data_config.yaml")
        self.assertEqual(cfg.train_source_dir, str(ROOT / "data/aihub/datasets/extracted/Training/원천데이터"))
        self.assertEqual(cfg.cache_dir, str(ROOT / "cache"))

    def test_explicit_absolute_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config.DataConfig(cache_dir=tmp)
            self.assertEqual(cfg.cache_dir, str(Path(tmp).resolve()))

    def test_yaml_round_trip_keeps_explicit_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config.DataConfig(cache_dir=tmp)
            path = Path(tmp) / "config.yaml"
            cfg.to_yaml(path)
            self.assertEqual(config.DataConfig.from_yaml(path), cfg)

    def test_direct_script_bootstraps_work_after_relocation(self):
        # Execute only the real import bootstrap, in its original order. No
        # training libraries, model constructors, datasets or script main run.
        with tempfile.TemporaryDirectory(prefix="moved checkout ") as tmp:
            moved = Path(tmp) / "project"
            (moved / "src").mkdir(parents=True)
            for name in ("__init__.py", "paths.py"):
                shutil.copyfile(ROOT / "src" / name, moved / "src" / name)
            checked = 0
            for path in (ROOT / "src").rglob("*.py"):
                source = path.read_text()
                if "sys.path.insert(0, str(Path(__file__)" not in source:
                    continue
                nodes = []
                for node in ast.parse(source).body:
                    if isinstance(node, ast.Import) and any(x.name == "sys" for x in node.names):
                        nodes.append(node)
                    elif isinstance(node, ast.ImportFrom) and node.module == "pathlib":
                        nodes.append(node)
                    elif isinstance(node, ast.Expr) and ast.unparse(node).startswith("sys.path.insert("):
                        nodes.append(node)
                    elif isinstance(node, ast.ImportFrom) and node.module == "src.paths":
                        nodes.append(node)
                script = "__file__ = " + repr(str(moved / path.relative_to(ROOT))) + "\n"
                script += "\n".join(ast.unparse(node) for node in nodes)
                script += "\nfrom src.paths import PROJECT_ROOT\nprint(PROJECT_ROOT)\n"
                with self.subTest(path=str(path.relative_to(ROOT))):
                    proc = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp,
                                          capture_output=True, text=True, check=True)
                    self.assertEqual(proc.stdout.strip(), str(moved))
                checked += 1
            self.assertGreaterEqual(checked, 15)

    def test_offline_builder_reads_canonical_channel_definition(self):
        path = ROOT / "src/data/scripts/build_windows.py"
        script = ("import runpy; ns=runpy.run_path(" + repr(str(path)) + "); "
                  "print(ns['wb'].LEGACY_CHANNEL_SOURCE)")
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp,
                                  capture_output=True, text=True, check=True)
        self.assertEqual(proc.stdout.strip(), "src/data/config.py:SENSOR_CHANNELS")

    def test_legacy_environment_command_delegates_and_preserves_exit(self):
        with tempfile.TemporaryDirectory(prefix="legacy checker ") as tmp:
            deploy = Path(tmp) / "jetson_deploy"
            (deploy / "scripts").mkdir(parents=True)
            alias = deploy / "scripts/01_check_environment.py"
            shutil.copyfile(ROOT / "jetson_deploy/scripts/01_check_environment.py", alias)
            wrapper = deploy / "run_python.sh"
            wrapper.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit 37\n')
            wrapper.chmod(0o700)
            proc = subprocess.run([sys.executable, "-I", "-B", str(alias), "--marker"],
                                  cwd=tmp, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 37)
            self.assertEqual(proc.stdout.splitlines(), [str(deploy / "check_environment.py"), "--marker"])


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "PyTorch unavailable; run in audited SERVER-TRAINING environment")
class SamplerTests(unittest.TestCase):
    def test_training_cli_passes_requested_seed_to_data_module(self):
        from src import train_v2plus
        class StopBeforeData(Exception):
            pass
        captured = []
        def capture(cfg):
            captured.append(cfg)
            raise StopBeforeData()
        with mock.patch.object(sys, "argv", ["train_v2plus", "--seed", "123"]):
            with mock.patch.object(train_v2plus, "ManufacturingDataModule", side_effect=capture):
                with mock.patch.object(train_v2plus.torch.cuda, "is_available", return_value=False):
                    with self.assertRaises(StopBeforeData):
                        train_v2plus.main()
        self.assertEqual(captured[0].seed, 123)

    def sampler(self, seed):
        import numpy as np
        from src.data.sampler import build_weighted_sampler
        class Labels:
            def get_all_labels(self):
                return np.tile(np.arange(4), 128)
        return build_weighted_sampler(Labels(), seed=seed)

    def test_global_rng_consumption_does_not_change_sample_order(self):
        import torch
        torch.manual_seed(42)
        first = self.sampler(42)
        torch.rand(10)
        expected = list(first)
        torch.manual_seed(42)
        second = self.sampler(42)
        torch.rand(10000)
        self.assertEqual(list(second), expected)

    def test_sampler_advances_across_epochs_and_replays_across_runs(self):
        first = self.sampler(42)
        epoch1, epoch2 = list(first), list(first)
        self.assertNotEqual(epoch1, epoch2)
        second = self.sampler(42)
        self.assertEqual(list(second), epoch1)
        self.assertEqual(list(second), epoch2)

    def test_different_seeds_select_different_orders(self):
        self.assertNotEqual(list(self.sampler(42)), list(self.sampler(43)))


if __name__ == "__main__":
    unittest.main()
