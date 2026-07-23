#!/usr/bin/env python3
"""Train external SOTA baselines (TimesNet, PatchTST) for comparison.

Usage:
    cd /home/keti/factory_safety
    python -m src.train_external_baselines --all
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
from src.models.external_baselines import TimesNetClassifier, PatchTSTClassifier

logger = logging.getLogger(__name__)

RESULTS_BASE = "/home/keti/factory_safety/results/external_baselines"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        sensor = batch["sensor"].to(device)
        thermal = batch["thermal"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(sensor, thermal)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds), \
           f1_score(all_labels, all_preds, average="macro")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        sensor = batch["sensor"].to(device)
        thermal = batch["thermal"].to(device)
        labels = batch["label"].to(device)
        logits = model(sensor, thermal)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


def train_model(name, model, device, epochs=30, lr=1e-3, batch_size=16):
    results_dir = os.path.join(RESULTS_BASE, name)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    cfg = DataConfig()
    cfg.batch_size = batch_size
    dm = ManufacturingDataModule(cfg)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=dm.class_weights.to(device))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    logger.info(f"\n{'='*50}")
    logger.info(f"{name} | Params: {param_count:,}")
    logger.info(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 5 == 0 or val_f1 > best_val_f1:
            logger.info(f"  Ep {epoch:>2d}: TrF1={train_f1:.4f} VaF1={val_f1:.4f} ({elapsed:.0f}s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_f1": val_f1, "val_acc": val_acc},
                        os.path.join(results_dir, "best_model.pt"))

    # Final eval
    ckpt = torch.load(os.path.join(results_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    cm = confusion_matrix(val_labels, val_preds)

    logger.info(f"  BEST: Epoch {ckpt['epoch']}, F1={val_f1:.4f}, Acc={val_acc:.4f}")
    logger.info(f"\n{classification_report(val_labels, val_preds, target_names=['Normal','Mild','Moderate','Severe'])}")
    logger.info(f"Confusion Matrix:\n{cm}")

    results = {
        "model": name, "best_epoch": ckpt["epoch"],
        "val_f1_macro": float(val_f1), "val_accuracy": float(val_acc),
        "confusion_matrix": cm.tolist(),
        "normal_mild_errors": int(cm[0][1] + cm[1][0]),
        "params": param_count,
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", type=str, default="", help="timesnet or patchtst")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    models_to_run = []
    if args.all or args.model == "timesnet":
        models_to_run.append(("TimesNet", TimesNetClassifier(d_model=64, d_ff=128, num_layers=2)))
    if args.all or args.model == "patchtst":
        models_to_run.append(("PatchTST", PatchTSTClassifier(d_model=64, nhead=4, num_layers=2)))

    if not models_to_run:
        print("Specify --all or --model timesnet/patchtst")
        return

    all_results = []
    for name, model in models_to_run:
        r = train_model(name, model, device, epochs=30, lr=1e-3, batch_size=16)
        all_results.append(r)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("EXTERNAL BASELINES SUMMARY")
    logger.info(f"{'='*60}")
    for r in all_results:
        logger.info(f"  {r['model']:<15s} F1={r['val_f1_macro']:.4f} Acc={r['val_accuracy']:.4f} "
                     f"NM_err={r['normal_mild_errors']} Params={r['params']:,}")


if __name__ == "__main__":
    main()
