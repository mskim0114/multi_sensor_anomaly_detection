#!/usr/bin/env python3
"""How much data a five-channel model needs, measured by shrinking the training set.

The field sensors, optics, equipment and fault physics all differ from the AI Hub
testbed, so the field model will be trained from scratch on field data rather than
adapted from an AI Hub checkpoint (D-029). That makes one planning question urgent
before any field campaign starts: how many machines, and how many windows, does a
model of this shape need before it is worth deploying.

AI Hub cannot answer that for our equipment, but it is the only dataset of this kind
we have, and it can answer the shape of the curve: train the field-schema model on N
machines drawn from the 32 and score it on the four provider validation machines,
for N from 1 upward. What that yields is a planning estimate with a stated caveat,
not a prediction of field accuracy.

Draws are deterministic given --draw, and every draw keeps the AGV/OHT mix as even as
the count allows, so a small-N run is not accidentally all of one machine type.

Usage:
    python -m src.train_data_scaling --self-test
    python -m src.train_data_scaling --machines 4 --draw 0 --gpu 0
"""

import argparse
import datetime as dt
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import project_path
from src.data import DataConfig
from src.data.config import SENSOR_CHANNELS
from src.data.dataset import ManufacturingDataset
from src.data.normalization import SensorNormalizer, ThermalNormalizer
from src.data.sampler import compute_class_weights
from src.data.session_index import load_or_build_index
from src.models.v2_plus import SupConLoss
from src.train_channel_subset import V2PlusChannels, parse_channels
from src.train_kfold import make_loader, subset_index
from src.train_v2plus import (enable_deterministic, evaluate, run_provenance,
                              state_dict_sha256, train_one_epoch)

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-20260928-003"
RESULTS_BASE = "results/data_scaling"
FIELD_CHANNELS = "NTC,PM1.0,PM2.5,PM10,CT2"


def draw_machines(all_devices, n: int, draw: int) -> list[str]:
    """Pick n machines, keeping the AGV/OHT mix even and the choice reproducible."""
    agv = sorted(d for d in all_devices if d.startswith("agv"))
    oht = sorted(d for d in all_devices if d.startswith("oht"))
    rng = random.Random(1000 * draw + n)
    rng.shuffle(agv)
    rng.shuffle(oht)
    picked, i = [], 0
    while len(picked) < n:
        pool = agv if (i % 2 == 0) else oht
        other = oht if (i % 2 == 0) else agv
        src = pool if pool else other
        if not src:
            raise ValueError(f"cannot draw {n} machines from {len(all_devices)}")
        picked.append(src.pop())
        i += 1
    return sorted(picked)


def self_test() -> int:
    devices = [f"agv{i:02d}" for i in range(1, 17)] + [f"oht{i:02d}" for i in range(1, 17)]
    ok = True
    for n in (1, 2, 4, 8, 16, 24, 32):
        a = draw_machines(devices, n, 0)
        b = draw_machines(devices, n, 0)
        c = draw_machines(devices, n, 1)
        n_agv = sum(x.startswith("agv") for x in a)
        balanced = abs(n_agv - (n - n_agv)) <= 1
        ok &= len(a) == n and len(set(a)) == n and a == b and balanced
        if n < 32:
            ok &= a != c
        print(f"  n={n:>2d}  agv {n_agv} / oht {n - n_agv}  reproducible {a == b}  "
              f"differs across draws {a != c if n < 32 else 'n/a'}  {a if n <= 4 else ''}")
    # nesting is not required, but a larger draw should contain small-draw machines
    # often enough that the curve is not dominated by which machines were picked
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machines", type=int, required=False, default=4)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--channels", default=FIELD_CHANNELS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--supcon-weight", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    if args.self_test:
        return self_test()

    channels = parse_channels(args.channels)
    names = [SENSOR_CHANNELS[i] for i in channels]
    args._nondet = enable_deterministic() if args.deterministic else None
    results_dir = project_path(args.results_dir or
                               f"{RESULTS_BASE}/n{args.machines:02d}_draw{args.draw}_seed{args.seed}")
    if os.path.exists(os.path.join(results_dir, "results.json")):
        logger.error(f"{results_dir}/results.json exists; refusing to overwrite")
        return 2
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cfg = DataConfig(); cfg.batch_size = args.batch_size; cfg.seed = args.seed
    train_index = load_or_build_index(cfg.train_source_dir, cfg.train_label_dir, cfg.cache_dir,
                                      split="train", gap_threshold=cfg.session_gap_threshold)
    val_index = load_or_build_index(cfg.val_source_dir, cfg.val_label_dir, cfg.cache_dir,
                                    split="val", gap_threshold=cfg.session_gap_threshold)
    all_devices = {s.device_id for s in train_index.sessions}
    picked = draw_machines(all_devices, args.machines, args.draw)
    sub_index = subset_index(train_index, lambda d: d in set(picked), f"train_n{args.machines}")

    mk = lambda idx, src, lab: ManufacturingDataset(
        index=idx, source_dir=src, label_dir=lab, config=cfg,
        sensor_normalizer=SensorNormalizer(cfg.sensor_stats),
        thermal_normalizer=ThermalNormalizer(cfg.thermal_stats, cfg.thermal_norm_mode),
        augmentation=None)
    train_ds = mk(sub_index, cfg.train_source_dir, cfg.train_label_dir)
    val_ds = mk(val_index, cfg.val_source_dir, cfg.val_label_dir)
    train_loader = make_loader(train_ds, cfg, train=True, seed=args.seed)
    val_loader = make_loader(val_ds, cfg, train=False, seed=args.seed)
    class_counts = np.bincount(train_ds.get_all_labels(), minlength=4).tolist()
    logger.info(f"[{EXPERIMENT_ID}] {args.machines} machines (draw {args.draw}): {picked}")
    logger.info(f"  channels {names}")
    logger.info(f"  train {len(train_ds)} windows / {len(sub_index.sessions)} sessions, classes {class_counts}")
    logger.info(f"  val   {len(val_ds)} windows, four provider machines, untouched")
    if min(class_counts) == 0:
        logger.warning(f"  a class is absent from this draw: {class_counts}")

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = V2PlusChannels(channels, hidden_dim=128, num_layers=3, num_classes=4,
                           lags=[1, 5, 10]).to(device)
    init_sha = state_dict_sha256(model)
    ce = nn.CrossEntropyLoss(weight=compute_class_weights(train_ds).to(device))
    supcon = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    provenance = run_provenance(args)
    logger.info(f"  provenance commit={str(provenance['git_commit'])[:7]} dirty={provenance['git_dirty']}")

    history = {k: [] for k in ("train_loss", "train_f1", "val_loss", "val_acc", "val_f1", "lr")}
    best, t0 = 0.0, time.time()
    logger.info(f"{'Ep':>3s} {'TrLoss':>8s} {'TrF1':>7s} {'VaLoss':>8s} {'VaAcc':>7s} {'VaF1':>9s} {'LR':>10s} {'Time':>6s}")
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, _, tr_f1 = train_one_epoch(model, train_loader, ce, supcon, optimizer,
                                            device, args.supcon_weight, scaler=None)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, ce, device, use_amp=False)
        scheduler.step(va_loss)
        for k, v in (("train_loss", tr_loss), ("train_f1", tr_f1), ("val_loss", va_loss),
                     ("val_acc", va_acc), ("val_f1", va_f1), ("lr", lr)):
            history[k].append(float(v))
        star = ""
        if va_f1 > best:
            best = va_f1
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_f1": va_f1},
                       os.path.join(results_dir, "best_model.pt"))
            star = " *"
        logger.info(f"{epoch:>3d} {tr_loss:>8.4f} {tr_f1:>7.4f} {va_loss:>8.4f} {va_acc:>7.4f} "
                    f"{va_f1:>9.6f} {lr:>10.6f} {time.time() - te:>5.1f}s{star}")

    ckpt = torch.load(os.path.join(results_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    va_loss, va_acc, va_f1, preds, labels = evaluate(model, val_loader, ce, device, use_amp=False)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    logger.info(f"  BEST epoch {ckpt['epoch']}  F1 {va_f1:.6f}  acc {va_acc:.4f}  "
                f"Severe recall {cm[3][3]}/{cm[3].sum()}  ({len(train_ds)} training windows)")

    results = {
        "experiment_id": EXPERIMENT_ID, "machines": args.machines, "draw": args.draw,
        "picked_machines": picked, "channels": names, "channel_indices": channels,
        "counts": {"train_windows": len(train_ds), "train_sessions": len(sub_index.sessions),
                   "train_class_counts": class_counts, "val_windows": len(val_ds)},
        "best_epoch": ckpt["epoch"], "val_f1_macro": float(va_f1), "val_accuracy": float(va_acc),
        "normal_mild_errors": int(cm[0][1] + cm[1][0]),
        "severe_recall": float(cm[3][3] / cm[3].sum()) if cm[3].sum() else None,
        "confusion_matrix": cm.tolist(), "history": history,
        "model_params": sum(p.numel() for p in model.parameters()),
        "seed": args.seed, "seed_controls_init": True, "initial_state_sha256": init_sha,
        "args": vars(args), "wall_clock_s": round(time.time() - t0, 1),
        "provenance": {**provenance, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
        "deterministic": {
            "requested": bool(args.deterministic),
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "non_deterministic_ops_warned": list(args._nondet) if args._nondet is not None else [],
        },
        "limitations": [
            "AI Hub machines, sensors and faults are not the field's; this measures the shape of "
            "the curve for a problem of this kind, not field accuracy",
            "evaluated on the four provider validation machines, so the machine-level uncertainty "
            "of N23 (+/- 0.005) applies on top",
            "one seed per point",
        ],
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  saved {results_dir}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
