#!/usr/bin/env python3
"""Build the EXP004 P0 manifest on SERVER-TRAINING; never train/evaluate.

Fixture-only checks (no dataset or torch needed):
  python -I src/prepare_synthetic_anomaly.py --self-test
Server data accounting:
  PYTHONNOUSERSITE=1 ~/venvs/factory_training/bin/python src/prepare_synthetic_anomaly.py \
    --data-root /mnt/data-ssd/keti_data/factory_safety/aihub_extracted \
    --out /home/keti/review_runs/EXP004_P0_NEW_RUN
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_leaf():
    name = "_synthetic_anomaly_p0"
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/data/synthetic_anomaly.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    # A run directory is unique; never silently replace another run's evidence.
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def git(*args, checkout=ROOT):
    return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True,
                          text=True, timeout=15).stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="synthetic fixture unit checks only")
    parser.add_argument("--data-root", type=Path, help="verified extracted AI Hub root on the server")
    parser.add_argument("--out", type=Path, help="new output directory outside raw data")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/experiments/synthetic_modality_plan.yaml")
    parser.add_argument("--base-checkout", type=Path, default=ROOT,
                        help="existing source checkout when executing an isolated uncommitted code bundle")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.data_root or args.out:
            parser.error("--self-test cannot be combined with data/output paths")
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_synthetic_anomaly.py")
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.data_root is None or args.out is None:
        parser.error("--data-root and --out are required for server accounting")
    if (platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}
            or Path("/etc/nv_tegra_release").exists()
            or Path(sys.prefix).resolve() != (Path.home() / "venvs/factory_training").resolve()
            or os.environ.get("PYTHONNOUSERSITE") != "1"):
        parser.error("dataset reads require Linux SERVER-TRAINING factory_training venv with PYTHONNOUSERSITE=1")
    data, out = args.data_root.expanduser().resolve(), args.out.expanduser().resolve()
    if out == data or data in out.parents or out in data.parents:
        parser.error("output must be separate from the raw tree")
    if out.exists():
        parser.error("output already exists; choose a new run directory")
    # This import is confined to the audited training environment; reuse the
    # exact existing equipment assignment function, not its training loop.
    base_checkout = args.base_checkout.expanduser().resolve()
    sys.path.insert(0, str(base_checkout))
    from src.train_kfold import assign_folds
    import numpy as np
    import yaml
    sa = load_leaf()
    plan_bytes = args.plan.read_bytes()
    plan = yaml.safe_load(plan_bytes)
    if (plan["protocol_id"] != sa.PROTOCOL or plan["data"]["window_ticks"] != 30
            or plan["data"]["stride_ticks"] != 10 or plan["split"]["outer_k"] != 4
            or plan["data"]["normal_filter"] != "all_30_tick_labels_equal_Normal"
            or plan["data"]["quality"]["policy_id"] != sa.QUALITY
            or plan["data"]["quality"]["soft_flag_policy"] != "include_and_stratify"
            or plan["inputs"]["normalizer"] != "run_local_fit_pure_Normal_unique_raw_ticks_only"):
        raise ValueError("plan is not supported by this P0 implementation")
    provenance = {"timestamp": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
                  "environment_profile": "SERVER-TRAINING", "python_executable": sys.executable,
                  "python_version": platform.python_version(), "numpy_version": np.__version__,
                  "git_commit_sha": git("rev-parse", "HEAD", checkout=base_checkout),
                  "git_dirty": ROOT != base_checkout or bool(git("status", "--porcelain", checkout=base_checkout)),
                  "base_checkout": str(base_checkout), "executed_code_root": str(ROOT),
                  "isolated_uncommitted_bundle": ROOT != base_checkout,
                  "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "data_root": str(data),
                  "source_code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (Path(__file__).resolve(), ROOT / "src/data/synthetic_anomaly.py", base_checkout / "src/train_kfold.py")},
                  "command": sys.argv if argv is None else list(argv), "purpose": "P0_data_accounting_only",
                  "training_executed": False, "heldout_model_evaluation": False,
                  "raw_stability_check": "content_SHA256_at_read_then_size_mtime_ctime_inode_guard"}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "provenance.json", provenance)
    try:
        def progress(split, n, total):
            print(f"{split}: {n}/{total} ticks", flush=True)
        ticks, ledger = sa.scan_split(data, "Training", progress=progress)
        devices = sorted({t.device for t in ticks})
        manifest = sa.prepare_manifests(ticks, assign_folds(devices, 4))
        # Metadata stability is separate from the content hashes just computed.
        for item in ledger:
            if sa.file_identity(item["path"]) != item["identity"]:
                raise RuntimeError(f"raw changed during preparation: {item['path']}")
        manifest.pop("sha256")
        manifest["source_manifest_sha256"] = sa.digest_json(ledger)
        manifest["provenance"] = provenance
        manifest["sha256"] = sa.digest_json(manifest)
        write_json(out / "raw_manifest.json", ledger)
        write_json(out / "split_manifest.json", manifest)
        print(json.dumps({"data_gate": manifest["data_gate"], "full_P0_gate": "NOT_EVALUATED",
                          "ticks": len(ticks), "windows": len(manifest["windows"]),
                          "output": str(out)}, ensure_ascii=False), flush=True)
        return 0 if manifest["data_gate"] == "PASS" else 2
    except Exception as exc:
        write_json(out / "failure.json", {"timestamp": datetime.now(timezone.utc).isoformat(),
                                          "exception": type(exc).__name__, "message": str(exc),
                                          "complete": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
