#!/usr/bin/env python3
"""Run all additional experiments for paper revision.

Experiments:
  Major 1 - V2+ ablation separation:
    v2a: V2 + Multi-Scale Diff only (lag=1,5,10, no SE, no SupCon)
    v2b: V2 + SE Channel Attention only (lag=1, +SE, no SupCon)
    v2c: V2 + SupCon Loss only (lag=1, no SE, +SupCon)
  Major 2 - Thermal contribution:
    sensor_only: V2+ without thermal branch
  Major 3 - Repeated runs:
    seed123, seed456: V2+ with different seeds
  Minor 6 - Lag sensitivity:
    lag_1_3_7, lag_1_10_20: Different lag combinations

Usage:
    cd /home/keti/factory_safety
    python -m src.train_paper_experiments --all
    python -m src.train_paper_experiments --exp v2a
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
from src.models.v2_plus import V2Plus, SupConLoss, SEBlock
from src.models.ablation_variants import LSTMWithTemporalDiff

logger = logging.getLogger(__name__)

RESULTS_BASE = "/home/keti/factory_safety/results/paper_experiments"


# ========== Model Variants ==========

class V2MultiScaleOnly(nn.Module):
    """V2a: Multi-scale diff (lag=1,5,10), NO SE, NO SupCon."""
    def __init__(self, sensor_dim=8, hidden_dim=128, num_layers=3,
                 num_classes=4, lags=None):
        super().__init__()
        self.lags = lags or [1, 5, 10]
        input_dim = sensor_dim * (1 + len(self.lags))
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc_sensor = nn.Linear(hidden_dim, hidden_dim)
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_thermal = nn.Linear(64 * 15 * 20, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, sensor_data, thermal_data):
        B, T, C = sensor_data.shape
        diffs = []
        for lag in self.lags:
            d = torch.zeros_like(sensor_data)
            d[:, lag:, :] = sensor_data[:, lag:, :] - sensor_data[:, :-lag, :]
            diffs.append(d)
        sensor_input = torch.cat([sensor_data] + diffs, dim=-1)
        lstm_out, _ = self.lstm(sensor_input)
        sensor_out = self.fc_sensor(lstm_out[:, -1, :])
        B, T, H, W = thermal_data.shape
        x = thermal_data.view(B * T, 1, H, W)
        x = self.thermal_conv(x)
        x = self.fc_thermal(x.view(B * T, -1))
        thermal_out = x.view(B, T, -1)
        sensor_out = sensor_out.unsqueeze(1).expand_as(thermal_out)
        combined = torch.cat([sensor_out, thermal_out], dim=-1)
        return self.classifier(combined.mean(dim=1))


class V2SEOnly(nn.Module):
    """V2b: Single lag (lag=1) + SE block, NO SupCon."""
    def __init__(self, sensor_dim=8, hidden_dim=128, num_layers=3, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(sensor_dim * 2, hidden_dim, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc_sensor = nn.Linear(hidden_dim, hidden_dim)
        self.se = SEBlock(hidden_dim)
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_thermal = nn.Linear(64 * 15 * 20, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, sensor_data, thermal_data):
        B, T, C = sensor_data.shape
        diff = torch.zeros_like(sensor_data)
        diff[:, 1:, :] = sensor_data[:, 1:, :] - sensor_data[:, :-1, :]
        sensor_input = torch.cat([sensor_data, diff], dim=-1)
        lstm_out, _ = self.lstm(sensor_input)
        sensor_out = self.se(self.fc_sensor(lstm_out[:, -1, :]))
        B, T, H, W = thermal_data.shape
        x = thermal_data.view(B * T, 1, H, W)
        x = self.thermal_conv(x)
        x = self.fc_thermal(x.view(B * T, -1))
        thermal_out = x.view(B, T, -1)
        sensor_out = sensor_out.unsqueeze(1).expand_as(thermal_out)
        combined = torch.cat([sensor_out, thermal_out], dim=-1)
        return self.classifier(combined.mean(dim=1))


class V2SupConOnly(nn.Module):
    """V2c: Single lag (lag=1), NO SE, WITH SupCon compatible encoding."""
    def __init__(self, sensor_dim=8, hidden_dim=128, num_layers=3, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(sensor_dim * 2, hidden_dim, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc_sensor = nn.Linear(hidden_dim, hidden_dim)
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_thermal = nn.Linear(64 * 15 * 20, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def encode(self, sensor_data, thermal_data):
        B, T, C = sensor_data.shape
        diff = torch.zeros_like(sensor_data)
        diff[:, 1:, :] = sensor_data[:, 1:, :] - sensor_data[:, :-1, :]
        sensor_input = torch.cat([sensor_data, diff], dim=-1)
        lstm_out, _ = self.lstm(sensor_input)
        sensor_out = self.fc_sensor(lstm_out[:, -1, :])
        B, T, H, W = thermal_data.shape
        x = thermal_data.view(B * T, 1, H, W)
        x = self.thermal_conv(x)
        x = self.fc_thermal(x.view(B * T, -1))
        thermal_out = x.view(B, T, -1)
        sensor_out = sensor_out.unsqueeze(1).expand_as(thermal_out)
        combined = torch.cat([sensor_out, thermal_out], dim=-1)
        return combined.mean(dim=1)

    def forward(self, sensor_data, thermal_data):
        embedding = self.encode(sensor_data, thermal_data)
        return self.classifier(self.dropout(embedding))


class V2PlusSensorOnly(nn.Module):
    """V2+ sensor-only (no thermal branch)."""
    def __init__(self, sensor_dim=8, hidden_dim=128, num_layers=3,
                 num_classes=4, lags=None):
        super().__init__()
        self.lags = lags or [1, 5, 10]
        input_dim = sensor_dim * (1 + len(self.lags))
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc_sensor = nn.Linear(hidden_dim, hidden_dim)
        self.se = SEBlock(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def encode(self, sensor_data, thermal_data):
        B, T, C = sensor_data.shape
        diffs = []
        for lag in self.lags:
            d = torch.zeros_like(sensor_data)
            d[:, lag:, :] = sensor_data[:, lag:, :] - sensor_data[:, :-lag, :]
            diffs.append(d)
        sensor_input = torch.cat([sensor_data] + diffs, dim=-1)
        lstm_out, _ = self.lstm(sensor_input)
        return self.se(self.fc_sensor(lstm_out[:, -1, :]))

    def forward(self, sensor_data, thermal_data):
        embedding = self.encode(sensor_data, thermal_data)
        return self.classifier(self.dropout(embedding))


# ========== Training Functions ==========

def train_one_epoch(model, loader, criterion, optimizer, device,
                    supcon_criterion=None, supcon_weight=0.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        sensor = batch["sensor"].to(device)
        thermal = batch["thermal"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        if supcon_criterion and supcon_weight > 0 and hasattr(model, 'encode'):
            embedding = model.encode(sensor, thermal)
            logits = model.classifier(model.dropout(embedding))
            ce_loss = criterion(logits, labels)
            sc_loss = supcon_criterion(embedding, labels)
            loss = (1 - supcon_weight) * ce_loss + supcon_weight * sc_loss
        else:
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


def run_experiment(exp_name, model, device, epochs, lr, batch_size,
                   use_supcon=False, supcon_weight=0.3, seed=42):
    results_dir = os.path.join(RESULTS_BASE, exp_name)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = DataConfig()
    cfg.batch_size = batch_size
    cfg.seed = seed
    dm = ManufacturingDataModule(cfg)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=dm.class_weights.to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    supcon_criterion = SupConLoss() if use_supcon else None

    best_val_f1 = 0.0
    logger.info(f"\n{'='*50}")
    logger.info(f"Experiment: {exp_name} | Params: {param_count:,} | Seed: {seed}")
    logger.info(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            supcon_criterion, supcon_weight if use_supcon else 0.0)
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device)
        scheduler.step(val_loss)
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
    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
        model, val_loader, criterion, device)
    cm = confusion_matrix(val_labels, val_preds)
    nm_errors = int(cm[0][1] + cm[1][0])

    logger.info(f"  BEST: Epoch {ckpt['epoch']}, F1={val_f1:.4f}, Acc={val_acc:.4f}, NM_errors={nm_errors}")

    results = {
        "experiment": exp_name, "best_epoch": ckpt["epoch"],
        "val_f1_macro": float(val_f1), "val_accuracy": float(val_acc),
        "confusion_matrix": cm.tolist(), "normal_mild_errors": nm_errors,
        "params": param_count, "seed": seed,
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ========== Experiment Definitions ==========

EXPERIMENTS = {
    # Major 1: V2+ ablation separation
    "v2a": lambda dev: run_experiment(
        "v2a_multiscale_only",
        V2MultiScaleOnly(lags=[1, 5, 10]),
        dev, epochs=20, lr=1e-3, batch_size=16),
    "v2b": lambda dev: run_experiment(
        "v2b_se_only",
        V2SEOnly(),
        dev, epochs=20, lr=1e-3, batch_size=16),
    "v2c": lambda dev: run_experiment(
        "v2c_supcon_only",
        V2SupConOnly(),
        dev, epochs=20, lr=1e-3, batch_size=16, use_supcon=True, supcon_weight=0.3),

    # Major 2: Sensor-only
    "sensor_only": lambda dev: run_experiment(
        "v2plus_sensor_only",
        V2PlusSensorOnly(lags=[1, 5, 10]),
        dev, epochs=30, lr=1e-3, batch_size=16, use_supcon=True, supcon_weight=0.3),

    # Major 3: Repeated runs
    "seed123": lambda dev: run_experiment(
        "v2plus_seed123",
        V2Plus(lags=[1, 5, 10]),
        dev, epochs=30, lr=1e-3, batch_size=16, use_supcon=True, supcon_weight=0.3, seed=123),
    "seed456": lambda dev: run_experiment(
        "v2plus_seed456",
        V2Plus(lags=[1, 5, 10]),
        dev, epochs=30, lr=1e-3, batch_size=16, use_supcon=True, supcon_weight=0.3, seed=456),

    # Minor 6: Lag sensitivity
    "lag_1_3_7": lambda dev: run_experiment(
        "lag_1_3_7",
        V2Plus(lags=[1, 3, 7]),
        dev, epochs=30, lr=1e-3, batch_size=16, use_supcon=True, supcon_weight=0.3),
    "lag_1_10_20": lambda dev: run_experiment(
        "lag_1_10_20",
        V2Plus(lags=[1, 10, 20]),
        dev, epochs=30, lr=1e-3, batch_size=16, use_supcon=True, supcon_weight=0.3),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="", help="Specific experiment to run")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    Path(RESULTS_BASE).mkdir(parents=True, exist_ok=True)

    if args.all:
        exps_to_run = list(EXPERIMENTS.keys())
    elif args.exp:
        exps_to_run = [e.strip() for e in args.exp.split(",")]
    else:
        print("Specify --all or --exp <name>. Available:", ", ".join(EXPERIMENTS.keys()))
        return

    all_results = []
    for exp_name in exps_to_run:
        if exp_name not in EXPERIMENTS:
            logger.warning(f"Unknown experiment: {exp_name}, skipping")
            continue
        result = EXPERIMENTS[exp_name](device)
        all_results.append(result)

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Experiment':<25s} {'F1':>8s} {'Acc':>8s} {'NM_err':>8s} {'Params':>10s}")
    logger.info("-" * 65)
    for r in all_results:
        logger.info(f"{r['experiment']:<25s} {r['val_f1_macro']:>8.4f} {r['val_accuracy']:>8.4f} "
                     f"{r['normal_mild_errors']:>8d} {r['params']:>10,}")

    with open(os.path.join(RESULTS_BASE, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSummary saved to {RESULTS_BASE}/summary.json")


if __name__ == "__main__":
    main()
