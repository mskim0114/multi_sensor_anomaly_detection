#!/usr/bin/env python3
"""Train V2+ model (Multi-Scale TempDiff + Channel Attention + SupCon Loss).

Usage:
    cd <repository-root>
    python -m src.train_v2plus
    python -m src.train_v2plus --epochs 30 --supcon-weight 0.3
"""

import sys
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import random
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PROJECT_ROOT, project_path

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

def enable_deterministic():
    """Ask PyTorch for deterministic kernels and record which ops cannot comply.

    Same-seed re-runs reproduce the initial weights but not the trajectory (연구노트 #19
    §5, N17). This switches on every determinism control PyTorch offers; ops that have no
    deterministic CUDA kernel raise a UserWarning under warn_only=True, and those warnings
    are collected so the run's results.json states exactly where determinism was not
    available instead of implying it was.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    seen: list[str] = []

    def _capture(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if "deterministic" in text and text not in seen:
            seen.append(text)
            logger.warning(f"non-deterministic op: {text}")
    warnings.showwarning = _capture
    logger.info("Deterministic mode: cudnn.deterministic=True, benchmark=False, "
                "use_deterministic_algorithms(True, warn_only=True), "
                f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}")
    return seen


# Default output location. Pass --results-dir to write a new run elsewhere; the paper
# run lives here and must not be overwritten by later experiments.
DEFAULT_RESULTS_DIR = "results/v2plus"


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


def run_provenance(args) -> dict:
    """Facts a later reader needs to trust or reproduce this run.

    §11 of the research brief requires git commit, dirty state, environment and the exact
    command for every training run; no earlier run recorded them, which is why the legacy
    results carry `no_record` in the experiment index. A dirty tree is recorded as such,
    never hidden: AGENTS §5 forbids citing results from a dirty revision.
    """
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(PROJECT_ROOT), *a], capture_output=True,
                                  text=True, timeout=10, check=True).stdout.strip()
        except Exception:
            return None
    porcelain = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if porcelain is None else bool(porcelain),
        "hostname": socket.gethostname(),
        "environment_profile": "SERVER-TRAINING",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else None,
        "command": " ".join(sys.argv),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def state_dict_sha256(model) -> str:
    """Fingerprint of the parameters as currently initialised.

    Recorded so a later run can prove it started from the same weights; §11 of the
    research brief requires the initial-weight hash and no run had it before.
    """
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, loader, ce_criterion, device, use_amp=False):
    """Validation pass. `use_amp` must mirror the training precision.

    This used to enter CUDA autocast unconditionally, which made `--amp` a no-op for
    evaluation and left fp32 training paired with fp16 evaluation. Near-tie argmaxes on
    the Normal/Mild boundary can flip under fp16, so evaluation precision now follows the
    flag that the run was launched with.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        sensor = batch["sensor"].to(device, non_blocking=True)
        thermal = batch["thermal"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with (torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()):
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
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Output directory, relative to the checkout unless absolute. "
                             f"Default {DEFAULT_RESULTS_DIR} is the paper run - use a new "
                             "directory for any re-run so it is never overwritten.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed. Set BEFORE the model is built, so it "
                             "controls the initial weights as well as the sample order.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable Automatic Mixed Precision for faster training")
    parser.add_argument("--fast", action="store_true",
                        help="Enable TF32 matmul + cudnn.benchmark. Introduces "
                             "non-determinism; do not use for paper-reproducibility runs.")
    parser.add_argument("--deterministic", action="store_true",
                        help="torch.use_deterministic_algorithms(True, warn_only=True) + "
                             "cudnn.deterministic + CUBLAS_WORKSPACE_CONFIG. Ops without a "
                             "deterministic kernel are logged, not silently ignored. "
                             "Mutually exclusive with --fast.")
    args = parser.parse_args()
    if args.deterministic and args.fast:
        parser.error("--deterministic and --fast are mutually exclusive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.fast:
        enable_fast_math()
    nondeterministic_ops = enable_deterministic() if args.deterministic else None

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    lags = [int(x) for x in args.lags.split(",")]

    # Data
    cfg = DataConfig()
    cfg.batch_size = args.batch_size
    cfg.seed = args.seed
    dm = ManufacturingDataModule(cfg)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    logger.info(f"Train: {len(dm.train_dataset)} windows, {len(train_loader)} batches")
    logger.info(f"Val: {len(dm.val_dataset)} windows, {len(val_loader)} batches")

    # Seeds are set here, before any parameter is allocated. Setting them later would
    # leave the initial weights on the process-default RNG (see 연구노트 #16, F03).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Model
    model = V2Plus(
        sensor_dim=8,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=4,
        lags=lags,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    init_state_sha256 = state_dict_sha256(model)
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Initial weight sha256: {init_state_sha256}")
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

    results_dir = project_path(args.results_dir)
    if os.path.exists(os.path.join(results_dir, "results.json")):
        logger.warning(f"{results_dir}/results.json already exists and will be overwritten")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    provenance = run_provenance(args)
    logger.info(f"Provenance: commit={str(provenance['git_commit'])[:7]} dirty={provenance['git_dirty']} "
                f"host={provenance['hostname']} torch={provenance['torch_version']}")

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
            model, val_loader, ce_criterion, device, use_amp=args.amp,
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
            }, os.path.join(results_dir, "best_model.pt"))
            logger.info(f"  ★ New best model saved (F1={val_f1:.4f})")

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Final evaluation with best model")
    logger.info("=" * 60)

    ckpt = torch.load(os.path.join(results_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
        model, val_loader, ce_criterion, device, use_amp=args.amp,
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
    v2_path = project_path("results/ablation_v2/results.json")
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
        "seed": args.seed,
        "seed_controls_init": True,
        "initial_state_sha256": init_state_sha256,
        "provenance": {**provenance, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
        "deterministic": {
            "requested": bool(args.deterministic),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "non_deterministic_ops_warned": nondeterministic_ops if nondeterministic_ops is not None else [],
        },
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
