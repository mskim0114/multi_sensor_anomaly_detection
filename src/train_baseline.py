#!/usr/bin/env python3
"""Train the Multimodal LSTM baseline model.

Usage:
    cd /home/keti/factory_safety
    python -m src.train_baseline
    python -m src.train_baseline --epochs 20 --lr 0.001
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.data import DataConfig, ManufacturingDataModule
from src.models import MultimodalLSTM

logger = logging.getLogger(__name__)

RESULTS_DIR = "/home/keti/factory_safety/results/baseline"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        sensor = batch["sensor"].to(device)
        thermal = batch["thermal"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(sensor, thermal)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        sensor = batch["sensor"].to(device)
        thermal = batch["thermal"].to(device)
        labels = batch["label"].to(device)

        outputs = model(sensor, thermal)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = outputs.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=1, help="GPU index")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    cfg = DataConfig.from_yaml(args.config) if args.config else DataConfig()
    cfg.batch_size = args.batch_size
    dm = ManufacturingDataModule(cfg)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    logger.info(f"Train: {len(dm.train_dataset)} windows, {len(train_loader)} batches")
    logger.info(f"Val: {len(dm.val_dataset)} windows, {len(val_loader)} batches")

    # Model
    model = MultimodalLSTM(
        sensor_dim=8,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=4,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Optimizer, loss, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=dm.class_weights.to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True,
    )

    # Training loop
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

        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        # LR scheduler
        scheduler.step(val_loss)

        elapsed = time.time() - t0

        # Log
        logger.info(
            f"{epoch:>5d} {train_loss:>8.4f} {train_acc:>7.4f} {train_f1:>7.4f} "
            f"{val_loss:>8.4f} {val_acc:>7.4f} {val_f1:>7.4f} {current_lr:>10.6f} {elapsed:>5.1f}s"
        )

        # History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        # Save best model
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

    # Final evaluation with best model
    logger.info("\n" + "=" * 60)
    logger.info("Final evaluation with best model")
    logger.info("=" * 60)

    ckpt = torch.load(os.path.join(RESULTS_DIR, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
        model, val_loader, criterion, device
    )

    logger.info(f"Best epoch: {ckpt['epoch']}")
    logger.info(f"Val Accuracy: {val_acc:.4f}")
    logger.info(f"Val F1 (macro): {val_f1:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(val_labels, val_preds, target_names=['Normal', 'Mild', 'Moderate', 'Severe'])}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(val_labels, val_preds)}")

    # Save results
    results = {
        "best_epoch": ckpt["epoch"],
        "val_accuracy": float(val_acc),
        "val_f1_macro": float(val_f1),
        "history": history,
        "confusion_matrix": confusion_matrix(val_labels, val_preds).tolist(),
        "args": vars(args),
        "model_params": param_count,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
