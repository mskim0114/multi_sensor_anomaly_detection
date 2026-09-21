#!/usr/bin/env python3
"""Device-grouped K-fold for V2+ on the AI Hub training machines (EXP-20260907-006).

Why: every model, lag and configuration choice in the paper was selected and reported on
the same four validation machines (agv17, agv18, oht17, oht18), so the reported spread
(seed std on one split) says nothing about machine-to-machine variation (CE-046). Here the
32 training machines are split into K device-disjoint folds; each fold trains on the other
machines and is scored on its held-out machines. The provider validation split is never
used for selection: it is scored once per fold model, for reporting only ("frozen").

Fold assignment is deterministic: AGV and OHT machines are sorted by id separately and the
i-th machine of each type goes to fold i % K, so every fold holds 16/K AGVs and 16/K OHTs.

Protocol is the G2e/EXP-003 protocol (V2+ lags [1,5,10], SE, SupCon 0.3, 30 epochs, AdamW
1e-3, plateau LR, best held-out macro-F1 checkpoint, fp32), with --deterministic on by
default (D-024). Normalisation constants are the fixed training-set constants in DataConfig,
so a small statistics leak across folds remains and is stated in the results.

Usage:
    python -m src.train_kfold --self-test
    python -m src.train_kfold --fold 0 --gpu 0            # one fold
    python -m src.train_kfold --fold all --gpu 0          # K folds sequentially
"""

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import project_path
from src.data import DataConfig
from src.data.dataset import ManufacturingDataset
from src.data.datamodule import _collate_fn, _worker_init_fn
from src.data.normalization import SensorNormalizer, ThermalNormalizer
from src.data.sampler import build_weighted_sampler, compute_class_weights
from src.data.session_index import DatasetIndex, load_or_build_index
from src.models.v2_plus import V2Plus, SupConLoss
from src.train_v2plus import (enable_deterministic, evaluate, run_provenance,
                              state_dict_sha256, train_one_epoch)

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-20260907-006"
RESULTS_BASE = "results/kfold_devices"
CLASSES = ["Normal", "Mild", "Moderate", "Severe"]


def assign_folds(device_ids, k: int) -> dict[str, int]:
    """Deterministic device -> fold map, balanced by device type."""
    out = {}
    for prefix in ("agv", "oht"):
        ids = sorted(d for d in device_ids if d.startswith(prefix))
        for i, d in enumerate(ids):
            out[d] = i % k
    missing = sorted(set(device_ids) - set(out))
    if missing:
        raise ValueError(f"devices with unknown type: {missing}")
    return out


def self_test() -> int:
    devs = [f"agv{i:02d}" for i in range(1, 17)] + [f"oht{i:02d}" for i in range(1, 17)]
    ok = True
    for k in (2, 4, 8):
        fm = assign_folds(devs, k)
        folds = {f: sorted(d for d, ff in fm.items() if ff == f) for f in range(k)}
        sizes = {f: (sum(d.startswith("agv") for d in ds), sum(d.startswith("oht") for d in ds)) for f, ds in folds.items()}
        cover = sorted(sum(folds.values(), [])) == sorted(devs)
        balanced = all(s == (16 // k, 16 // k) for s in sizes.values())
        ok &= cover and balanced
        print(f"K={k}: cover={cover} balanced={balanced} sizes={sizes}")
        if k == 4:
            for f in range(k):
                print(f"   fold {f}: {folds[f]}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def subset_index(index: DatasetIndex, keep, split: str) -> DatasetIndex:
    return DatasetIndex(sessions=[s for s in index.sessions if keep(s.device_id)], split=split)


def make_loader(ds, cfg, *, train: bool, seed: int) -> DataLoader:
    if train:
        sampler = build_weighted_sampler(ds, seed=seed) if cfg.use_weighted_sampler else None
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=sampler is None, sampler=sampler,
                          num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                          prefetch_factor=cfg.prefetch_factor, persistent_workers=cfg.num_workers > 0,
                          worker_init_fn=_worker_init_fn, generator=torch.Generator().manual_seed(seed),
                          collate_fn=_collate_fn)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                      pin_memory=cfg.pin_memory, prefetch_factor=cfg.prefetch_factor,
                      persistent_workers=cfg.num_workers > 0, worker_init_fn=_worker_init_fn,
                      collate_fn=_collate_fn)


def window_devices(ds: ManufacturingDataset) -> list[str]:
    """Device id of every window, in DataLoader (unshuffled) order."""
    windows = getattr(ds, "windows", None) or getattr(ds, "_windows", None)
    if windows is None:
        windows = ds._build_windows()
    return [ds.index.sessions[w.session_idx].device_id for w in windows]


def per_device_metrics(preds, labels, devices) -> dict:
    out = {}
    preds, labels, devices = np.asarray(preds), np.asarray(labels), np.asarray(devices)
    for d in sorted(set(devices)):
        m = devices == d
        y, p = labels[m], preds[m]
        present = sorted(set(y.tolist()) | set(p.tolist()))
        cm = confusion_matrix(y, p, labels=[0, 1, 2, 3])
        out[d] = {
            "windows": int(m.sum()),
            "macro_f1_present_classes": float(f1_score(y, p, average="macro", labels=present)),
            "macro_f1_4class": float(f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0)),
            "accuracy": float(accuracy_score(y, p)),
            "normal_mild_errors": int(cm[0][1] + cm[1][0]),
            "confusion_matrix": cm.tolist(),
            "classes_present_in_labels": sorted(set(y.tolist())),
        }
    return out


def run_fold(fold: int, args, cfg, train_index, val_index, fold_map, device) -> dict:
    results_dir = project_path(args.results_dir or f"{RESULTS_BASE}/fold{fold}_of{args.k}_seed{args.seed}")
    if os.path.exists(os.path.join(results_dir, "results.json")):
        logger.error(f"{results_dir}/results.json exists; refusing to overwrite")
        return {"fold": fold, "skipped": True}
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    heldout_devices = sorted(d for d, f in fold_map.items() if f == fold)
    tr_index = subset_index(train_index, lambda d: fold_map[d] != fold, f"train_fold{fold}")
    ho_index = subset_index(train_index, lambda d: fold_map[d] == fold, f"heldout_fold{fold}")

    sensor_norm = SensorNormalizer(cfg.sensor_stats)
    thermal_norm = ThermalNormalizer(cfg.thermal_stats, cfg.thermal_norm_mode)
    mk = lambda idx, src, lab: ManufacturingDataset(index=idx, source_dir=src, label_dir=lab, config=cfg,
                                                    sensor_normalizer=sensor_norm, thermal_normalizer=thermal_norm,
                                                    augmentation=None)
    tr_ds = mk(tr_index, cfg.train_source_dir, cfg.train_label_dir)
    ho_ds = mk(ho_index, cfg.train_source_dir, cfg.train_label_dir)
    fv_ds = mk(val_index, cfg.val_source_dir, cfg.val_label_dir)
    tr_loader = make_loader(tr_ds, cfg, train=True, seed=args.seed)
    ho_loader = make_loader(ho_ds, cfg, train=False, seed=args.seed)
    fv_loader = make_loader(fv_ds, cfg, train=False, seed=args.seed)
    class_weights = compute_class_weights(tr_ds)
    logger.info(f"[fold {fold}/{args.k}] held-out {heldout_devices}")
    logger.info(f"  train {len(tr_ds)} windows / {len(tr_index.sessions)} sessions / {len(set(s.device_id for s in tr_index.sessions))} devices"
                f" | held-out {len(ho_ds)} windows / {len(ho_index.sessions)} sessions | frozen val {len(fv_ds)} windows")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import random; random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = V2Plus(sensor_dim=8, hidden_dim=128, num_layers=3, num_classes=4, lags=[1, 5, 10]).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    init_sha = state_dict_sha256(model)
    logger.info(f"  params {param_count:,} init_sha256 {init_sha}")

    ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
    supcon = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    provenance = run_provenance(args)
    logger.info(f"  provenance commit={str(provenance['git_commit'])[:7]} dirty={provenance['git_dirty']}")

    history = {k: [] for k in ("train_loss", "train_f1", "heldout_loss", "heldout_f1", "frozen_val_f1", "lr")}
    best = 0.0
    logger.info(f"{'Ep':>3s} {'TrLoss':>8s} {'TrF1':>7s} {'HoLoss':>8s} {'HoF1':>9s} {'FvF1':>9s} {'LR':>10s} {'Time':>6s}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, _, tr_f1 = train_one_epoch(model, tr_loader, ce, supcon, optimizer, device, args.supcon_weight, scaler=None)
        ho_loss, _, ho_f1, _, _ = evaluate(model, ho_loader, ce, device, use_amp=False)
        _, _, fv_f1, _, _ = evaluate(model, fv_loader, ce, device, use_amp=False)
        scheduler.step(ho_loss)
        for k, v in (("train_loss", tr_loss), ("train_f1", tr_f1), ("heldout_loss", ho_loss),
                     ("heldout_f1", ho_f1), ("frozen_val_f1", fv_f1), ("lr", lr)):
            history[k].append(float(v))
        star = ""
        if ho_f1 > best:
            best = ho_f1
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "heldout_f1": ho_f1},
                       os.path.join(results_dir, "best_model.pt"))
            star = " ★"
        logger.info(f"{epoch:>3d} {tr_loss:>8.4f} {tr_f1:>7.4f} {ho_loss:>8.4f} {ho_f1:>9.6f} {fv_f1:>9.6f} {lr:>10.6f} {time.time()-t0:>5.1f}s{star}")

    ckpt = torch.load(os.path.join(results_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    ho_loss, ho_acc, ho_f1, ho_p, ho_y = evaluate(model, ho_loader, ce, device, use_amp=False)
    fv_loss, fv_acc, fv_f1, fv_p, fv_y = evaluate(model, fv_loader, ce, device, use_amp=False)
    ho_cm = confusion_matrix(ho_y, ho_p, labels=[0, 1, 2, 3]); fv_cm = confusion_matrix(fv_y, fv_p, labels=[0, 1, 2, 3])
    ho_dev = per_device_metrics(ho_p, ho_y, window_devices(ho_ds))
    fv_dev = per_device_metrics(fv_p, fv_y, window_devices(fv_ds))
    logger.info(f"  BEST epoch {ckpt['epoch']}  held-out F1 {ho_f1:.6f} acc {ho_acc:.4f} NM {int(ho_cm[0][1]+ho_cm[1][0])}"
                f"  | frozen val F1 {fv_f1:.6f} acc {fv_acc:.4f} NM {int(fv_cm[0][1]+fv_cm[1][0])}")
    for d, m in ho_dev.items():
        logger.info(f"    {d}: F1 {m['macro_f1_present_classes']:.4f} acc {m['accuracy']:.4f} NM {m['normal_mild_errors']} ({m['windows']} windows)")

    results = {
        "experiment_id": EXPERIMENT_ID, "k": args.k, "fold": fold, "seed": args.seed,
        "fold_assignment": fold_map, "heldout_devices": heldout_devices,
        "train_devices": sorted(d for d, f in fold_map.items() if f != fold),
        "counts": {"train_windows": len(tr_ds), "train_sessions": len(tr_index.sessions),
                   "heldout_windows": len(ho_ds), "heldout_sessions": len(ho_index.sessions),
                   "frozen_val_windows": len(fv_ds)},
        "class_weights_train_fold": class_weights.tolist(),
        "best_epoch": ckpt["epoch"], "checkpoint_rule": "best held-out macro-F1; frozen val never used for selection",
        "heldout": {"macro_f1": float(ho_f1), "accuracy": float(ho_acc), "loss": float(ho_loss),
                    "normal_mild_errors": int(ho_cm[0][1] + ho_cm[1][0]), "confusion_matrix": ho_cm.tolist(),
                    "per_device": ho_dev},
        "frozen_val": {"macro_f1": float(fv_f1), "accuracy": float(fv_acc), "loss": float(fv_loss),
                       "normal_mild_errors": int(fv_cm[0][1] + fv_cm[1][0]), "confusion_matrix": fv_cm.tolist(),
                       "per_device": fv_dev},
        "history": history,
        "protocol": {"model": "V2Plus lags [1,5,10] + SE + SupCon 0.3 (tau 0.07)", "epochs": args.epochs,
                     "optimizer": f"AdamW(lr={args.lr}, wd=0.01)", "scheduler": "ReduceLROnPlateau(min,0.5,3) on held-out loss",
                     "grad_clip_norm": 1.0, "precision": "fp32", "class_weighted_ce_and_weighted_sampler": True,
                     "normalisation": "fixed DataConfig training-set constants (shared across folds; small leak)"},
        "model_params": param_count, "seed_controls_init": True, "initial_state_sha256": init_sha,
        "args": vars(args),
        "provenance": {**provenance, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
        "deterministic": {
            "requested": bool(args.deterministic),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "non_deterministic_ops_warned": list(args._nondet) if args._nondet is not None else [],
        },
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  saved {results_dir}/results.json")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fold", default="all", help="fold index or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--supcon-weight", type=float, default=0.3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--results-dir", default=None, help=f"default {RESULTS_BASE}/fold<f>_of<K>_seed<seed> (single fold only)")
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                    help="disable the deterministic kernels that are on by default (D-024)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    if args.self_test:
        return self_test()
    if args.results_dir and args.fold == "all":
        ap.error("--results-dir only applies to a single --fold")
    args._nondet = enable_deterministic() if args.deterministic else None

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cfg = DataConfig(); cfg.batch_size = args.batch_size; cfg.seed = args.seed
    train_index = load_or_build_index(cfg.train_source_dir, cfg.train_label_dir, cfg.cache_dir, split="train",
                                      gap_threshold=cfg.session_gap_threshold)
    val_index = load_or_build_index(cfg.val_source_dir, cfg.val_label_dir, cfg.cache_dir, split="val",
                                    gap_threshold=cfg.session_gap_threshold)
    fold_map = assign_folds({s.device_id for s in train_index.sessions}, args.k)
    logger.info(f"[{EXPERIMENT_ID}] K={args.k} devices={len(fold_map)} frozen val devices="
                f"{sorted({s.device_id for s in val_index.sessions})} device={device}")
    folds = range(args.k) if args.fold == "all" else [int(args.fold)]
    for f in folds:
        run_fold(f, args, cfg, train_index, val_index, fold_map, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
