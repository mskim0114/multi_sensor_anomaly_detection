#!/usr/bin/env python3
"""What happens to V2+ when it is fine-tuned on normal-only data from a new domain.

Why this exists. The data we can collect right now is all Normal: an office bench with
no machine under observation and no fault to induce (RESEARCH_STATUS N26). That is not
training data for a 4-class degradation model, but it is exactly the shape of the data a
field deployment produces first, so the question "what would fine-tuning on it do to the
model" can be answered before any field data exists — and answered with ground truth,
which the office data will never have.

The rehearsal uses machines instead of rooms. A K-fold checkpoint has never seen its own
held-out machines, so those machines are a genuine new domain for it. We fine-tune it on
their *Normal windows only*, which is what normal-only field adaptation would do, and
watch three things as the steps go by:

  1. where the weights go   - relative L2 displacement from the starting point, per
                              parameter group, plus the cosine between the total update
                              and the current step, i.e. whether the drift has a direction
  2. what is forgotten      - macro F1 on the provider validation machines, which carry
                              all four classes and are untouched by this procedure
  3. what is gained         - macro F1 on the target machines' own full four-class data;
                              if normal-only adaptation cannot help even here, it cannot
                              help anywhere

Nothing here produces a deployable model. The output is a budget: how many steps at which
learning rate a normal-only adaptation can take before the fault classes are gone.

Usage:
    python -m src.train_adaptation_trajectory --self-test
    python -m src.train_adaptation_trajectory --fold 0 --lr 1e-5 --steps 400
"""

import argparse
import copy
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import project_path
from src.data import DataConfig
from src.data.dataset import ManufacturingDataset
from src.data.datamodule import _collate_fn, _worker_init_fn
from src.data.normalization import SensorNormalizer, ThermalNormalizer
from src.data.session_index import DatasetIndex, load_or_build_index
from src.models.v2_plus import V2Plus
from src.train_kfold import assign_folds
from src.train_v2plus import enable_deterministic, run_provenance, state_dict_sha256

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-20260928-001"
RESULTS_BASE = "results/adaptation_trajectory"
NORMAL_CLASS = 0
CLASSES = ["Normal", "Mild", "Moderate", "Severe"]


# ---------------------------------------------------------------- trajectory

def parameter_groups(model) -> dict[str, list[str]]:
    """Group parameters by the branch a reader reasons about, not by tensor."""
    groups: dict[str, list[str]] = {"lstm": [], "fc_sensor": [], "se": [],
                                    "thermal_conv": [], "fc_thermal": [], "classifier": []}
    for name, _ in model.named_parameters():
        head = name.split(".")[0]
        groups.setdefault(head, []).append(name)
    return {k: v for k, v in groups.items() if v}


def displacement(model, reference: dict, groups: dict) -> dict:
    """Relative L2 movement from the reference weights, overall and per group."""
    cur = {k: v.detach() for k, v in model.state_dict().items()}
    out = {}
    num_all = den_all = 0.0
    for gname, names in groups.items():
        num = den = 0.0
        for n in names:
            d = (cur[n].float() - reference[n].float())
            num += float((d * d).sum())
            den += float((reference[n].float() ** 2).sum())
        out[gname] = (num ** 0.5) / (den ** 0.5) if den > 0 else 0.0
        num_all += num
        den_all += den
    out["all"] = (num_all ** 0.5) / (den_all ** 0.5) if den_all > 0 else 0.0
    return out


def flat_delta(model, reference: dict, names: list[str]) -> torch.Tensor:
    cur = model.state_dict()
    return torch.cat([(cur[n].float() - reference[n].float()).flatten() for n in names])


# ---------------------------------------------------------------- data

def subset_index(index: DatasetIndex, keep, split: str) -> DatasetIndex:
    return DatasetIndex(sessions=[s for s in index.sessions if keep(s.device_id)], split=split)


def make_dataset(index, source_dir, label_dir, cfg):
    return ManufacturingDataset(
        index=index, source_dir=source_dir, label_dir=label_dir, config=cfg,
        sensor_normalizer=SensorNormalizer(cfg.sensor_stats),
        thermal_normalizer=ThermalNormalizer(cfg.thermal_stats, cfg.thermal_norm_mode),
        augmentation=None)


def loader(ds, cfg, *, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                      prefetch_factor=cfg.prefetch_factor,
                      persistent_workers=cfg.num_workers > 0,
                      worker_init_fn=_worker_init_fn, collate_fn=_collate_fn,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)


@torch.no_grad()
def score(model, dl, device) -> dict:
    model.eval()
    preds, labels = [], []
    for batch in dl:
        logits = model(batch["sensor"].to(device), batch["thermal"].to(device))
        preds.extend(logits.argmax(-1).cpu().numpy())
        labels.extend(batch["label"].numpy())
    p, y = np.array(preds), np.array(labels)
    cm = confusion_matrix(y, p, labels=[0, 1, 2, 3])
    return {
        "macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0)),
        "accuracy": float(accuracy_score(y, p)),
        "per_class_f1": [float(x) for x in f1_score(y, p, average=None, labels=[0, 1, 2, 3], zero_division=0)],
        "predicted_class_fraction": [float((p == c).mean()) for c in range(4)],
        "confusion_matrix": cm.tolist(),
        "severe_recall": float(cm[3][3] / cm[3].sum()) if cm[3].sum() else None,
    }


def measurement_steps(total: int) -> list[int]:
    """Dense early, sparse later: the collapse happens in the first few dozen steps."""
    pts = {0, total}
    s = 1
    while s < total:
        pts.add(s)
        s = max(s + 1, int(round(s * 1.5)))
    return sorted(pts)


# ---------------------------------------------------------------- self-test

def self_test() -> int:
    ok = True
    steps = measurement_steps(400)
    ok &= steps[0] == 0 and steps[-1] == 400 and len(steps) < 30
    print(f"measurement steps ({len(steps)}): {steps}")

    torch.manual_seed(0)
    m = V2Plus(lags=[1, 5, 10])
    ref = copy.deepcopy(m.state_dict())
    groups = parameter_groups(m)
    d0 = displacement(m, ref, groups)
    ok &= abs(d0["all"]) < 1e-12
    print(f"groups: {sorted(groups)}")
    print(f"displacement against itself: all={d0['all']:.3e}")

    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.startswith("classifier"):
                p.add_(torch.ones_like(p) * 0.01)
    d1 = displacement(m, ref, groups)
    moved = [g for g, v in d1.items() if g != "all" and v > 1e-9]
    ok &= moved == ["classifier"]
    print(f"after perturbing classifier only, groups moved: {moved}  all={d1['all']:.4f}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--checkpoint", default=None,
                    help="default results/kfold_devices/fold<f>_of<K>_seed42/best_model.pt")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    if args.self_test:
        return self_test()
    args._nondet = enable_deterministic() if args.deterministic else None

    results_dir = project_path(args.results_dir or f"{RESULTS_BASE}/fold{args.fold}_lr{args.lr:g}_seed{args.seed}")
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
    fold_map = assign_folds({s.device_id for s in train_index.sessions}, args.k)
    target_devices = sorted(d for d, f in fold_map.items() if f == args.fold)

    target_index = subset_index(train_index, lambda d: fold_map[d] == args.fold, "target")
    target_ds = make_dataset(target_index, cfg.train_source_dir, cfg.train_label_dir, cfg)
    val_ds = make_dataset(val_index, cfg.val_source_dir, cfg.val_label_dir, cfg)

    labels = target_ds.get_all_labels()
    normal_idx = [i for i, y in enumerate(labels) if y == NORMAL_CLASS]
    adapt_ds = Subset(target_ds, normal_idx)
    logger.info(f"[{EXPERIMENT_ID}] fold {args.fold} target machines {target_devices}")
    logger.info(f"  adaptation set: {len(adapt_ds)} Normal windows of {len(target_ds)} "
                f"(class counts {np.bincount(labels, minlength=4).tolist()})")
    logger.info(f"  forgetting set: provider validation, {len(val_ds)} windows, all four classes")
    logger.info(f"  target-domain set: same machines, all four classes, {len(target_ds)} windows")

    adapt_dl = loader(adapt_ds, cfg, shuffle=True, seed=args.seed)
    val_dl = loader(val_ds, cfg, shuffle=False, seed=args.seed)
    target_dl = loader(target_ds, cfg, shuffle=False, seed=args.seed)

    ckpt_path = args.checkpoint or project_path(
        f"results/kfold_devices/fold{args.fold}_of{args.k}_seed42/best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = V2Plus(lags=[1, 5, 10]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    start_sha = state_dict_sha256(model)
    reference = {k: v.detach().clone() for k, v in model.state_dict().items()}
    groups = parameter_groups(model)
    logger.info(f"  checkpoint {ckpt_path} (epoch {ckpt.get('epoch')}) sha256 {start_sha}")

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Plain cross-entropy with no class weighting: the adaptation set has one class, so a
    # weighted loss would be undefined. This is deliberately the naive procedure.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    provenance = run_provenance(args)
    logger.info(f"  provenance commit={str(provenance['git_commit'])[:7]} dirty={provenance['git_dirty']} lr={args.lr}")

    marks = measurement_steps(args.steps)
    trace, step, prev_delta = [], 0, None
    it = iter(adapt_dl)

    def measure(step, loss_value):
        nonlocal prev_delta
        disp = displacement(model, reference, groups)
        names = [n for n, _ in model.named_parameters()]
        delta = flat_delta(model, reference, names)
        cos = None
        if prev_delta is not None and float(delta.norm()) > 0 and float(prev_delta.norm()) > 0:
            cos = float(torch.dot(delta, prev_delta) / (delta.norm() * prev_delta.norm()))
        prev_delta = delta
        v, t = score(model, val_dl, device), score(model, target_dl, device)
        row = {"step": step, "train_loss": loss_value, "displacement": disp,
               "cosine_with_previous_total_update": cos,
               "forgetting_set": v, "target_domain_set": t}
        trace.append(row)
        logger.info(f"  step {step:>5d}  loss {('%.4f' % loss_value) if loss_value is not None else '   -  '}"
                    f"  |Δθ|/|θ| {disp['all']:.2e}  val F1 {v['macro_f1']:.4f}"
                    f"  target F1 {t['macro_f1']:.4f}  P(Normal) {t['predicted_class_fraction'][0]:.3f}")
        return row

    t0 = time.time()
    measure(0, None)
    last_loss = None
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(adapt_dl)
            batch = next(it)
        model.train()
        optimizer.zero_grad()
        logits = model(batch["sensor"].to(device), batch["thermal"].to(device))
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.item())
        if step in marks:
            measure(step, last_loss)

    base = trace[0]
    final = trace[-1]
    collapse_step = next((r["step"] for r in trace
                          if r["target_domain_set"]["predicted_class_fraction"][0] > 0.99), None)
    half_step = next((r["step"] for r in trace
                      if r["forgetting_set"]["macro_f1"] < base["forgetting_set"]["macro_f1"] / 2), None)
    results = {
        "experiment_id": EXPERIMENT_ID,
        "question": ("what normal-only fine-tuning does to a trained V2+: where the weights go, "
                     "what is forgotten, and whether it helps on the adapted domain"),
        "fold": args.fold, "k": args.k, "target_devices": target_devices,
        "checkpoint": str(ckpt_path), "checkpoint_epoch": ckpt.get("epoch"),
        "start_state_sha256": start_sha, "final_state_sha256": state_dict_sha256(model),
        "counts": {"adaptation_normal_windows": len(adapt_ds),
                   "target_all_class_windows": len(target_ds),
                   "forgetting_set_windows": len(val_ds),
                   "target_class_counts": np.bincount(labels, minlength=4).tolist()},
        "optimiser": f"AdamW(lr={args.lr}, wd=0.01), unweighted CE, grad clip 1.0",
        "steps": args.steps, "batch_size": args.batch_size, "seed": args.seed,
        "summary": {
            "baseline_forgetting_f1": base["forgetting_set"]["macro_f1"],
            "final_forgetting_f1": final["forgetting_set"]["macro_f1"],
            "baseline_target_f1": base["target_domain_set"]["macro_f1"],
            "final_target_f1": final["target_domain_set"]["macro_f1"],
            "final_relative_displacement": final["displacement"]["all"],
            "step_where_target_predictions_are_99pct_normal": collapse_step,
            "step_where_forgetting_f1_halves": half_step,
            "wall_clock_s": round(time.time() - t0, 1),
        },
        "trace": trace,
        "provenance": {**provenance, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
        "deterministic": {
            "requested": bool(args.deterministic),
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "non_deterministic_ops_warned": list(args._nondet) if args._nondet is not None else [],
        },
        "limitations": [
            "machines stand in for a new environment; an office bench differs from an unseen "
            "machine in ways this cannot measure",
            "one seed, one fold, one checkpoint",
            "the adaptation set is Normal windows of machines that also supply the target-domain "
            "evaluation, so the target-domain score is optimistic for the adapted class",
        ],
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  saved {results_dir}/results.json")
    logger.info(f"  forgetting F1 {base['forgetting_set']['macro_f1']:.4f} -> {final['forgetting_set']['macro_f1']:.4f}"
                f" | target F1 {base['target_domain_set']['macro_f1']:.4f} -> {final['target_domain_set']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
