#!/usr/bin/env python3
"""2^3 factorial ablation of V2+ under one training protocol (EXP-20260907-003).

Factors (each on/off):
  multiscale  temporal-difference lags [1, 5, 10] (on) vs [1] (off)
  se          SE channel attention on the sensor encoder output
  supcon      supervised contrastive loss, weight 0.3, temperature 0.07

The paper's Table 8 compares V2 (all off) and the three single-factor variants trained
for 20 epochs against V2+ (all on) trained for 30 epochs, each with one seed and with the
initial weights not controlled by the seed (연구노트 #16 §3.3, F03; claim CE-010). Under a
plateau LR scheduler a 30-epoch run cut at epoch 20 is not a 20-epoch run, so the +1.20 %p
could not be attributed to the factors. This script removes those confounds: every one
of the 8 conditions gets the same 30-epoch budget, the same optimiser and scheduler, the
same best-val-F1 checkpoint rule, the same seed-controlled initialisation, and records
the LR trajectory, the initial-weight hash and the run provenance.

Architectural equivalence with the legacy classes is checked by `--self-test`:
  ms0_se0 == LSTMWithTemporalDiff (V2)      ms1_se0 == V2MultiScaleOnly (V2a)
  ms0_se1 == V2SEOnly (V2b)                 ms1_se1 == V2Plus (V2+)
(state_dict keys and shapes identical, legacy weights load strictly, eval outputs equal).
One deliberate difference from the legacy V2/V2a/V2b runs: dropout(0.1) is applied before
the classifier in every condition, as V2Plus does, so that dropout is not a hidden factor.

Usage:
    python -m src.train_factorial_ablation --self-test
    python -m src.train_factorial_ablation --plan            # print the 24 commands
    python -m src.train_factorial_ablation --multiscale 1 --se 0 --supcon 1 --seed 42 --gpu 0
"""

import argparse
import datetime as dt
import itertools
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
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import project_path
from src.data import DataConfig, ManufacturingDataModule
from src.models.v2_plus import SEBlock, SupConLoss
from src.train_v2plus import evaluate, run_provenance, state_dict_sha256

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-20260907-003"
RESULTS_BASE = "results/factorial_ablation"
LAGS_MULTI = [1, 5, 10]
LAGS_SINGLE = [1]
SEEDS = [42, 123, 456]


class V2Factorial(nn.Module):
    """V2Plus with the multiscale and SE factors switchable.

    Module creation order is identical to `V2Plus.__init__` so that, for the same seed,
    ms1_se1 has bit-identical initial weights to V2Plus (checked by --self-test).
    """

    def __init__(self, multiscale: bool, se: bool, sensor_dim: int = 8,
                 hidden_dim: int = 128, num_layers: int = 3, num_classes: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.multiscale = bool(multiscale)
        self.use_se = bool(se)
        self.lags = list(LAGS_MULTI if multiscale else LAGS_SINGLE)
        input_dim = sensor_dim * (1 + len(self.lags))

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc_sensor = nn.Linear(hidden_dim, hidden_dim)
        self.se = SEBlock(hidden_dim) if se else nn.Identity()
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_thermal = nn.Linear(64 * 15 * 20, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def encode(self, sensor_data: torch.Tensor, thermal_data: torch.Tensor) -> torch.Tensor:
        diffs = []
        for lag in self.lags:
            d = torch.zeros_like(sensor_data)
            d[:, lag:, :] = sensor_data[:, lag:, :] - sensor_data[:, :-lag, :]
            diffs.append(d)
        sensor_input = torch.cat([sensor_data] + diffs, dim=-1)
        lstm_out, _ = self.lstm(sensor_input)
        sensor_out = self.se(self.fc_sensor(lstm_out[:, -1, :]))

        B, T, H, W = thermal_data.shape
        x = self.thermal_conv(thermal_data.view(B * T, 1, H, W))
        thermal_out = self.fc_thermal(x.view(B * T, -1)).view(B, T, -1)

        sensor_out = sensor_out.unsqueeze(1).expand_as(thermal_out)
        return torch.cat([sensor_out, thermal_out], dim=-1).mean(dim=1)

    def forward(self, sensor_data: torch.Tensor, thermal_data: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.encode(sensor_data, thermal_data)))


def condition_name(multiscale: int, se: int, supcon: int) -> str:
    return f"ms{int(multiscale)}_se{int(se)}_sc{int(supcon)}"


def default_results_dir(multiscale: int, se: int, supcon: int, seed: int) -> str:
    return f"{RESULTS_BASE}/{condition_name(multiscale, se, supcon)}_seed{seed}"


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- self-test

def self_test() -> int:
    """Prove the factorial model reproduces the four legacy architectures exactly."""
    from src.models.v2_plus import V2Plus
    from src.models.ablation_variants import LSTMWithTemporalDiff
    from src.train_paper_experiments import V2MultiScaleOnly, V2SEOnly

    legacy = {
        (0, 0): ("LSTMWithTemporalDiff (V2)", lambda: LSTMWithTemporalDiff()),
        (1, 0): ("V2MultiScaleOnly (V2a)", lambda: V2MultiScaleOnly(lags=LAGS_MULTI)),
        (0, 1): ("V2SEOnly (V2b)", lambda: V2SEOnly()),
        (1, 1): ("V2Plus (V2+)", lambda: V2Plus(lags=LAGS_MULTI)),
    }
    torch.manual_seed(0)
    sensor = torch.randn(4, 30, 8)
    thermal = torch.randn(4, 30, 120, 160)
    failures = 0
    for (ms, se), (name, factory) in legacy.items():
        seed_everything(42)
        ours = V2Factorial(multiscale=ms, se=se)
        seed_everything(42)
        ref = factory()
        ours_sd, ref_sd = ours.state_dict(), ref.state_dict()
        same_keys = list(ours_sd.keys()) == list(ref_sd.keys())
        same_shapes = same_keys and all(ours_sd[k].shape == ref_sd[k].shape for k in ours_sd)
        same_init = same_shapes and all(torch.equal(ours_sd[k], ref_sd[k]) for k in ours_sd)
        ours.load_state_dict(ref_sd, strict=True)
        ours.eval(); ref.eval()
        with torch.no_grad():
            diff = (ours(sensor, thermal) - ref(sensor, thermal)).abs().max().item()
        n_ours = sum(p.numel() for p in ours.parameters())
        n_ref = sum(p.numel() for p in ref.parameters())
        ok = same_shapes and n_ours == n_ref and diff == 0.0
        failures += 0 if ok else 1
        print(f"{condition_name(ms, se, 0)[:-4]:<8s} vs {name:<28s} params {n_ours:>9,} / {n_ref:>9,}"
              f"  keys+shapes={'OK' if same_shapes else 'MISMATCH'}"
              f"  same_init@seed42={'yes' if same_init else 'no'}"
              f"  max|Δlogit| after weight copy={diff:.1e}  -> {'PASS' if ok else 'FAIL'}")
    print("SELF-TEST", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


def print_plan() -> None:
    for seed in SEEDS:
        for ms, se, sc in itertools.product((0, 1), repeat=3):
            print(f"python -m src.train_factorial_ablation --multiscale {ms} --se {se} "
                  f"--supcon {sc} --seed {seed} --gpu $GPU")


# ---------------------------------------------------------------- training

def train_one_epoch(model, loader, ce_criterion, supcon_criterion, optimizer, device,
                    use_supcon: bool, supcon_weight: float):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        sensor = batch["sensor"].to(device, non_blocking=True)
        thermal = batch["thermal"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        embedding = model.encode(sensor, thermal)
        logits = model.classifier(model.dropout(embedding))
        ce_loss = ce_criterion(logits, labels)
        if use_supcon:
            loss = (1 - supcon_weight) * ce_loss + supcon_weight * supcon_criterion(embedding, labels)
        else:
            loss = ce_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    preds, labs = np.array(all_preds), np.array(all_labels)
    from sklearn.metrics import accuracy_score, f1_score
    return (total_loss / len(loader), accuracy_score(labs, preds),
            f1_score(labs, preds, average="macro"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multiscale", type=int, choices=(0, 1))
    parser.add_argument("--se", type=int, choices=(0, 1))
    parser.add_argument("--supcon", type=int, choices=(0, 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--supcon-weight", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results-dir", default=None,
                        help=f"default {RESULTS_BASE}/<condition>_seed<seed>")
    parser.add_argument("--self-test", action="store_true", help="architecture equivalence check (CPU, no data)")
    parser.add_argument("--plan", action="store_true", help="print the 24 run commands and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    if args.self_test:
        return self_test()
    if args.plan:
        print_plan()
        return 0
    if None in (args.multiscale, args.se, args.supcon):
        parser.error("--multiscale, --se and --supcon are required for a training run")

    cond = condition_name(args.multiscale, args.se, args.supcon)
    results_dir = project_path(args.results_dir or default_results_dir(args.multiscale, args.se, args.supcon, args.seed))
    if os.path.exists(os.path.join(results_dir, "results.json")):
        logger.error(f"{results_dir}/results.json exists; refusing to overwrite a completed run")
        return 2
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{EXPERIMENT_ID}] condition={cond} seed={args.seed} device={device}")

    # Same order as train_v2plus: data module first (its sampler has its own seeded
    # generator), then seed everything, then build the model.
    cfg = DataConfig()
    cfg.batch_size = args.batch_size
    cfg.seed = args.seed
    dm = ManufacturingDataModule(cfg)
    dm.setup()
    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()
    logger.info(f"Train: {len(dm.train_dataset)} windows / Val: {len(dm.val_dataset)} windows")

    seed_everything(args.seed)
    model = V2Factorial(multiscale=args.multiscale, se=args.se, hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    init_state_sha256 = state_dict_sha256(model)
    logger.info(f"params={param_count:,} lags={model.lags} se={model.use_se} supcon={bool(args.supcon)} "
                f"init_sha256={init_state_sha256}")

    ce_criterion = nn.CrossEntropyLoss(weight=dm.class_weights.to(device))
    supcon_criterion = SupConLoss(temperature=args.temperature) if args.supcon else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    provenance = run_provenance(args)
    logger.info(f"Provenance: commit={str(provenance['git_commit'])[:7]} dirty={provenance['git_dirty']} "
                f"host={provenance['hostname']} torch={provenance['torch_version']}")

    history = {k: [] for k in ("train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1", "lr")}
    best_val_f1 = 0.0
    logger.info(f"{'Ep':>3s} {'TrLoss':>8s} {'TrF1':>7s} {'VaLoss':>8s} {'VaAcc':>7s} {'VaF1':>8s} {'LR':>10s} {'Time':>6s}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, ce_criterion, supcon_criterion, optimizer, device,
            use_supcon=bool(args.supcon), supcon_weight=args.supcon_weight)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, ce_criterion, device, use_amp=False)
        scheduler.step(val_loss)
        for k, v in (("train_loss", train_loss), ("train_acc", train_acc), ("train_f1", train_f1),
                     ("val_loss", val_loss), ("val_acc", val_acc), ("val_f1", val_f1), ("lr", current_lr)):
            history[k].append(float(v))
        star = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_f1": val_f1, "val_acc": val_acc},
                       os.path.join(results_dir, "best_model.pt"))
            star = " ★"
        logger.info(f"{epoch:>3d} {train_loss:>8.4f} {train_f1:>7.4f} {val_loss:>8.4f} {val_acc:>7.4f} "
                    f"{val_f1:>8.6f} {current_lr:>10.6f} {time.time() - t0:>5.1f}s{star}")

    ckpt = torch.load(os.path.join(results_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, ce_criterion, device, use_amp=False)
    cm = confusion_matrix(val_labels, val_preds)
    nm_errors = int(cm[0][1] + cm[1][0])
    logger.info(f"BEST epoch {ckpt['epoch']}  F1={val_f1:.6f}  Acc={val_acc:.4f}  NM={nm_errors}")
    logger.info("\n" + classification_report(val_labels, val_preds,
                                              target_names=["Normal", "Mild", "Moderate", "Severe"]))

    results = {
        "experiment_id": EXPERIMENT_ID,
        "condition": cond,
        "factors": {"multiscale": int(args.multiscale), "se": int(args.se), "supcon": int(args.supcon)},
        "lags": model.lags,
        "best_epoch": ckpt["epoch"],
        "val_accuracy": float(val_acc),
        "val_f1_macro": float(val_f1),
        "normal_mild_errors": nm_errors,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "args": vars(args),
        "protocol": {
            "epochs": args.epochs,
            "optimizer": f"AdamW(lr={args.lr}, weight_decay=0.01)",
            "scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=3) on val_loss",
            "checkpoint_rule": "best val_macro_f1 over the fixed epoch budget; no early stopping",
            "grad_clip_norm": 1.0,
            "precision": "fp32 train and eval",
            "classifier_dropout": 0.1,
            "class_weighted_ce_and_weighted_sampler": True,
        },
        "model_params": param_count,
        "seed": args.seed,
        "seed_controls_init": True,
        "initial_state_sha256": init_state_sha256,
        "provenance": {**provenance, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
