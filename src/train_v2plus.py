#!/usr/bin/env python3
"""Train V2+ model (Multi-Scale TempDiff + Channel Attention + SupCon Loss).

Usage:
    cd /home/keti/factory_safety
    python -m src.train_v2plus
    python -m src.train_v2plus --epochs 30 --supcon-weight 0.3
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.data import DataConfig, ManufacturingDataModule
from src.models.v2_plus import V2Plus, SupConLoss

logger = logging.getLogger(__name__)


def enable_fast_math():
    """Enable TF32 + cuDNN benchmark for faster training on Ampere+ GPUs.

    Both settings introduce non-determinism / minor numeric drift, so they
    are opt-in via --fast rather than always on. Reproducibility runs that
    must match paper numbers (F1 = 0.9557 +/- 0.0006) should leave them off.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logger.info("Fast math enabled: TF32 matmul, TF32 cudnn, cudnn benchmark=True")

RESULTS_DIR = "/home/keti/factory_safety/results/v2plus"


def train_one_epoch(model, loader, ce_criterion, supcon_criterion,
                    optimizer, device, supcon_weight=0.3, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    use_amp = scaler is not None

    for batch in loader:
        sensor = batch["sensor"].to(device, non_blocking=True)
        thermal = batch["thermal"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            # Mixed precision training (fp16 forward/loss, fp32 optimizer)
            with torch.amp.autocast("cuda"):
                embedding = model.encode(sensor, thermal)
                logits = model.classifier(model.dropout(embedding))
                ce_loss = ce_criterion(logits, labels)
                supcon_loss = supcon_criterion(embedding, labels)
                loss = (1 - supcon_weight) * ce_loss + supcon_weight * supcon_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Full precision training
            embedding = model.encode(sensor, thermal)
            logits = model.classifier(model.dropout(embedding))
            ce_loss = ce_criterion(logits, labels)
            supcon_loss = supcon_criterion(embedding, labels)
            loss = (1 - supcon_weight) * ce_loss + supcon_weight * supcon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, ce_criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        sensor = batch["sensor"].to(device, non_blocking=True)
        thermal = batch["thermal"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Use autocast for consistent mixed precision during eval
        # (harmless in fp32 mode; keeps AMP-trained + fp32-trained eval symmetric)
        with torch.amp.autocast("cuda"):
            logits = model(sensor, thermal)
            loss = ce_criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lags", type=str, default="1,5,10", help="Comma-separated lag values")
    parser.add_argument("--supcon-weight", type=float, default=0.3,
                        help="Weight for SupCon loss (0=CE only, 1=SupCon only)")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--amp", action="store_true",
                        help="Enable Automatic Mixed Precision for faster training")
    parser.add_argument("--fast", action="store_true",
                        help="Enable TF32 matmul + cudnn.benchmark. Introduces "
                             "non-determinism; do not use for paper-reproducibility runs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.fast:
        enable_fast_math()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    lags = [int(x) for x in args.lags.split(",")]

    # Data
    cfg = DataConfig()
    cfg.batch_size = args.batch_size
    dm = ManufacturingDataModule(cfg)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    logger.info(f"Train: {len(dm.train_dataset)} windows, {len(train_loader)} batches")
    logger.info(f"Val: {len(dm.val_dataset)} windows, {len(val_loader)} batches")

    # Model
    model = V2Plus(
        sensor_dim=8,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=4,
        lags=lags,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Temporal diff lags: {lags}")
    logger.info(f"SupCon weight: {args.supcon_weight}")

    # Loss functions
    ce_criterion = nn.CrossEntropyLoss(weight=dm.class_weights.to(device))
    supcon_criterion = SupConLoss(temperature=args.temperature)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    if args.amp:
        logger.info("Mixed Precision (AMP) enabled")

    # Training
    best_val_f1 = 0.0
    history = {"train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'Epoch':>5s} {'TrLoss':>8s} {'TrAcc':>7s} {'TrF1':>7s} "
                f"{'VaLoss':>8s} {'VaAcc':>7s} {'VaF1':>7s} {'LR':>10s} {'Time':>6s}")
    logger.info("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, ce_criterion, supcon_criterion,
            optimizer, device, args.supcon_weight, scaler=scaler,
        )

        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, ce_criterion, device,
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        logger.info(
            f"{epoch:>5d} {train_loss:>8.4f} {train_acc:>7.4f} {train_f1:>7.4f} "
            f"{val_loss:>8.4f} {val_acc:>7.4f} {val_f1:>7.4f} {current_lr:>10.6f} {elapsed:>5.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, os.path.join(RESULTS_DIR, "best_model.pt"))
            logger.info(f"  ★ New best model saved (F1={val_f1:.4f})")

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Final evaluation with best model")
    logger.info("=" * 60)

    ckpt = torch.load(os.path.join(RESULTS_DIR, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
        model, val_loader, ce_criterion, device,
    )

    logger.info(f"Best epoch: {ckpt['epoch']}")
    logger.info(f"Val Accuracy: {val_acc:.4f}")
    logger.info(f"Val F1 (macro): {val_f1:.4f}")
    logger.info(
        f"\nClassification Report:\n"
        f"{classification_report(val_labels, val_preds, target_names=['Normal', 'Mild', 'Moderate', 'Severe'])}"
    )

    cm = confusion_matrix(val_labels, val_preds)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Compare with V2
    v2_path = "/home/keti/factory_safety/results/ablation_v2/results.json"
    if os.path.exists(v2_path):
        with open(v2_path) as f:
            v2 = json.load(f)
        v2_cm = np.array(v2["confusion_matrix"])
        logger.info(f"\n--- V2 vs V2+ comparison ---")
        logger.info(f"V2 F1: {v2['val_f1_macro']:.4f} → V2+ F1: {val_f1:.4f} "
                     f"(Δ={val_f1 - v2['val_f1_macro']:+.4f})")
        old_nm = v2_cm[0][1] + v2_cm[1][0]
        new_nm = cm[0][1] + cm[1][0]
        logger.info(f"Normal↔Mild errors: {old_nm} → {new_nm} (Δ={new_nm - old_nm:+d})")

    results = {
        "best_epoch": ckpt["epoch"],
        "val_accuracy": float(val_acc),
        "val_f1_macro": float(val_f1),
        "history": history,
        "confusion_matrix": cm.tolist(),
        "args": vars(args),
        "model_params": param_count,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
