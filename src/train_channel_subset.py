#!/usr/bin/env python3
"""What V2+ costs when it only gets the sensor channels the field hardware has.

The deployment front-end carries one current transformer (operator, 2026-09-28), so a
field model can see NTC, PM1.0, PM2.5, PM10 and one CT: five of the eight channels that
src/data/config.py:SENSOR_CHANNELS defines. The protocol forbids zero-filling or copying
CT1 into CT2-4, and the measurement in 연구노트 #23 shows why that ban is not merely a
convention: CT1 is uncorrelated with CT2-4 (|r| < 0.05) while CT2-4 correlate with each
other, and a least-squares reconstruction of CT2-4 from the five available channels
leaves 16-25 A of residual against channel standard deviations of 20-46 A. There is
nothing to impute from.

That leaves one honest option: train the model on the channels that exist. This script
does that and prices it, by training the same architecture under the same protocol on a
chosen subset and comparing against the eight-channel model.

Nothing here modifies SENSOR_CHANNELS, the stored normalisation constants or any existing
checkpoint; the subset is applied inside the model, so the dataset still yields all eight
channels and the same per-channel statistics normalise them.

Usage:
    python -m src.train_channel_subset --self-test
    python -m src.train_channel_subset --channels NTC,PM1.0,PM2.5,PM10,CT1 --gpu 0
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
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import project_path
from src.data import DataConfig, ManufacturingDataModule
from src.data.config import SENSOR_CHANNELS
from src.models.v2_plus import V2Plus, SupConLoss
from src.train_v2plus import (enable_deterministic, evaluate, run_provenance,
                              state_dict_sha256, train_one_epoch)

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-20260928-002"
RESULTS_BASE = "results/channel_subset"
FIELD_CHANNELS = "NTC,PM1.0,PM2.5,PM10,CT1"


class V2PlusChannels(V2Plus):
    """V2+ that reads only some of the eight sensor channels.

    The selection happens on the input tensor rather than in the data pipeline, so the
    dataset, the cache and the normalisation constants are untouched and a run with all
    eight indices is the ordinary V2+.
    """

    def __init__(self, channels: list[int], **kwargs):
        super().__init__(sensor_dim=len(channels), **kwargs)
        self.register_buffer("channel_index", torch.tensor(channels, dtype=torch.long))

    def encode(self, sensor_data: torch.Tensor, thermal_data: torch.Tensor) -> torch.Tensor:
        return super().encode(sensor_data.index_select(-1, self.channel_index), thermal_data)


def parse_channels(spec: str) -> list[int]:
    out = []
    for name in [s.strip() for s in spec.split(",") if s.strip()]:
        if name not in SENSOR_CHANNELS:
            raise ValueError(f"unknown channel {name!r}; known: {', '.join(SENSOR_CHANNELS)}")
        idx = SENSOR_CHANNELS.index(name)
        if idx in out:
            raise ValueError(f"channel {name!r} given twice")
        out.append(idx)
    if not out:
        raise ValueError("no channels selected")
    return out


def self_test() -> int:
    ok = True
    idx = parse_channels(FIELD_CHANNELS)
    ok &= idx == [0, 1, 2, 3, 4]
    print(f"field channels {FIELD_CHANNELS} -> indices {idx}")

    torch.manual_seed(0)
    sensor = torch.randn(3, 30, 8)
    thermal = torch.randn(3, 30, 120, 160)

    torch.manual_seed(1)
    full = V2PlusChannels(list(range(8)), lags=[1, 5, 10])
    torch.manual_seed(1)
    plain = V2Plus(sensor_dim=8, lags=[1, 5, 10])
    full.eval(); plain.eval()
    with torch.no_grad():
        d = (full(sensor, thermal) - plain(sensor, thermal)).abs().max().item()
    same_params = sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in plain.parameters())
    ok &= d == 0.0 and same_params
    print(f"all eight indices reproduce V2+ exactly: max|dlogit| {d:.1e}, same parameter count {same_params}")

    torch.manual_seed(2)
    sub = V2PlusChannels(idx, lags=[1, 5, 10])
    sub.eval()
    with torch.no_grad():
        a = sub(sensor, thermal)
        perturbed = sensor.clone()
        perturbed[:, :, 5:] += 100.0          # CT2-4, which the subset must not read
        b = sub(perturbed, thermal)
        perturbed2 = sensor.clone()
        perturbed2[:, :, 4] += 100.0          # CT1, which it must read
        c = sub(perturbed2, thermal)
    ignores_dropped = torch.equal(a, b)
    uses_kept = not torch.allclose(a, c)
    ok &= ignores_dropped and uses_kept
    print(f"subset ignores the dropped channels: {ignores_dropped}; still uses CT1: {uses_kept}")

    n_sub = sum(p.numel() for p in sub.parameters())
    n_full = sum(p.numel() for p in plain.parameters())
    print(f"parameters {n_sub:,} (five channels) vs {n_full:,} (eight)")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", default=FIELD_CHANNELS,
                    help=f"comma-separated subset of {', '.join(SENSOR_CHANNELS)}")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--supcon-weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default=None, help="results directory suffix; default from the channel count")
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
    tag = args.tag or f"{len(channels)}ch"
    results_dir = project_path(args.results_dir or f"{RESULTS_BASE}/{tag}_seed{args.seed}")
    if os.path.exists(os.path.join(results_dir, "results.json")):
        logger.error(f"{results_dir}/results.json exists; refusing to overwrite")
        return 2
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cfg = DataConfig(); cfg.batch_size = args.batch_size; cfg.seed = args.seed
    dm = ManufacturingDataModule(cfg)
    dm.setup()
    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()
    logger.info(f"[{EXPERIMENT_ID}] channels {names} (indices {channels}) device {device}")
    logger.info(f"  train {len(dm.train_dataset)} windows / val {len(dm.val_dataset)} windows")

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = V2PlusChannels(channels, hidden_dim=128, num_layers=3, num_classes=4,
                           lags=[1, 5, 10]).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    init_sha = state_dict_sha256(model)
    logger.info(f"  params {param_count:,}  init_sha256 {init_sha}")

    ce = nn.CrossEntropyLoss(weight=dm.class_weights.to(device))
    supcon = SupConLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    provenance = run_provenance(args)
    logger.info(f"  provenance commit={str(provenance['git_commit'])[:7]} dirty={provenance['git_dirty']}")

    history = {k: [] for k in ("train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1", "lr")}
    best = 0.0
    t0 = time.time()
    logger.info(f"{'Ep':>3s} {'TrLoss':>8s} {'TrF1':>7s} {'VaLoss':>8s} {'VaAcc':>7s} {'VaF1':>9s} {'LR':>10s} {'Time':>6s}")
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, ce, supcon, optimizer,
                                                 device, args.supcon_weight, scaler=None)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, ce, device, use_amp=False)
        scheduler.step(va_loss)
        for k, v in (("train_loss", tr_loss), ("train_acc", tr_acc), ("train_f1", tr_f1),
                     ("val_loss", va_loss), ("val_acc", va_acc), ("val_f1", va_f1), ("lr", lr)):
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
                f"NM {int(cm[0][1] + cm[1][0])}  Severe recall {cm[3][3]}/{cm[3].sum()}")
    logger.info("\n" + classification_report(labels, preds,
                                             target_names=["Normal", "Mild", "Moderate", "Severe"]))

    results = {
        "experiment_id": EXPERIMENT_ID,
        "channels": names, "channel_indices": channels, "dropped_channels":
            [c for c in SENSOR_CHANNELS if c not in names],
        "best_epoch": ckpt["epoch"],
        "val_f1_macro": float(va_f1), "val_accuracy": float(va_acc),
        "normal_mild_errors": int(cm[0][1] + cm[1][0]),
        "severe_recall": float(cm[3][3] / cm[3].sum()),
        "confusion_matrix": cm.tolist(), "history": history,
        "model_params": param_count, "seed": args.seed, "seed_controls_init": True,
        "initial_state_sha256": init_sha, "args": vars(args),
        "protocol": {"model": "V2Plus lags [1,5,10] + SE + SupCon 0.3, channel subset applied at the input",
                     "epochs": args.epochs, "optimizer": f"AdamW(lr={args.lr}, wd=0.01)",
                     "scheduler": "ReduceLROnPlateau(min, 0.5, 3) on val loss",
                     "precision": "fp32", "normalisation": "unchanged per-channel statistics"},
        "wall_clock_s": round(time.time() - t0, 1),
        "provenance": {**provenance, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
        "deterministic": {
            "requested": bool(args.deterministic),
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "non_deterministic_ops_warned": list(args._nondet) if args._nondet is not None else [],
        },
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  saved {results_dir}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
