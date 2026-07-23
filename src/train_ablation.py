#!/usr/bin/env python3
"""Run ablation study: train all variants and compare.

Usage:
    cd /home/keti/factory_safety
    python -m src.train_ablation
    python -m src.train_ablation --variant 2
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
from src.models.ablation_variants import (
    LSTMWithTemporalDiff,
    LSTMWithEfficientNet,
    CATFTNoCrossAttention,
)

logger = logging.getLogger(__name__)

VARIANTS = {
    2: ("LSTM+TemporalDiff", LSTMWithTemporalDiff, {"batch_size": 16, "lr": 1e-3, "epochs": 20}),
    3: ("LSTM+EfficientNet", LSTMWithEfficientNet, {"batch_size": 8, "lr": 5e-4, "epochs": 30}),
    4: ("CATFT-NoCrossAttn", CATFTNoCrossAttention, {"batch_size": 8, "lr": 5e-4, "epochs": 30}),
}


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")


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
        all_preds.extend(outputs.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


def train_variant(variant_id: int, gpu: int = 1):
    name, model_cls, hparams = VARIANTS[variant_id]
    results_dir = f"/home/keti/factory_safety/results/ablation_v{variant_id}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*60}")
    logger.info(f"Variant {variant_id}: {name}")
    logger.info(f"{'='*60}")

    # Data
    cfg = DataConfig()
    cfg.batch_size = hparams["batch_size"]
    dm = ManufacturingDataModule(cfg)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Model
    if variant_id == 4:
        model = model_cls(sensor_dim=8, d_model=256, nhead=4, num_encoder_layers=2, num_classes=4).to(device)
    else:
        model = model_cls(sensor_dim=8, hidden_dim=128, num_layers=3, num_classes=4).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: {param_count:,} (trainable: {trainable:,})")

    # Training setup
    lr = hparams["lr"]
    epochs = hparams["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=dm.class_weights.to(device))

    if epochs > 20:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Train
    best_val_f1 = 0.0
    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}

    logger.info(f"{'Epoch':>5s} {'TrLoss':>8s} {'TrF1':>7s} {'VaLoss':>8s} {'VaF1':>7s} {'Time':>6s}")
    logger.info("-" * 50)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        elapsed = time.time() - t0
        logger.info(f"{epoch:>5d} {train_loss:>8.4f} {train_f1:>7.4f} {val_loss:>8.4f} {val_f1:>7.4f} {elapsed:>5.1f}s")

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_f1": val_f1, "val_acc": val_acc}, os.path.join(results_dir, "best_model.pt"))
            logger.info(f"  ★ Best (F1={val_f1:.4f})")

    # Final eval
    ckpt = torch.load(os.path.join(results_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

    report = classification_report(val_labels, val_preds, target_names=["Normal", "Mild", "Moderate", "Severe"])
    cm = confusion_matrix(val_labels, val_preds)

    logger.info(f"\nBest epoch: {ckpt['epoch']}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")
    logger.info(f"\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")

    results = {
        "variant": variant_id, "name": name,
        "best_epoch": ckpt["epoch"],
        "val_accuracy": float(val_acc), "val_f1_macro": float(val_f1),
        "confusion_matrix": cm.tolist(),
        "history": history,
        "params": param_count, "trainable_params": trainable,
        "hparams": hparams,
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, default=0, help="0=all, 2/3/4=specific")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    variants_to_run = [args.variant] if args.variant > 0 else [2, 3, 4]
    all_results = []

    for v in variants_to_run:
        r = train_variant(v, args.gpu)
        all_results.append(r)

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("=" * 70)

    # Load baseline and CATFT results
    for path, label in [
        ("/home/keti/factory_safety/results/baseline/results.json", "V1: Baseline LSTM"),
        ("/home/keti/factory_safety/results/catft/results.json", "V5: Full CATFT"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            logger.info(f"  {label:30s}  F1={d['val_f1_macro']:.4f}  Acc={d['val_accuracy']:.4f}  Params={d['model_params']:,}")

    for r in all_results:
        logger.info(f"  V{r['variant']}: {r['name']:30s}  F1={r['val_f1_macro']:.4f}  Acc={r['val_accuracy']:.4f}  Params={r['params']:,}")


if __name__ == "__main__":
    main()
