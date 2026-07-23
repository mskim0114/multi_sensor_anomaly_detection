#!/usr/bin/env python3
"""Generate reference inputs and predictions for Jetson validation.

Saves validation set inputs (already normalized) + PyTorch/ONNX predictions
to a portable .npz file. The Jetson then runs ONNX on the same inputs and
checks prediction match rate.

Usage:
    cd /home/keti/factory_safety
    python -m src.deploy.generate_reference_data
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "/home/keti/factory_safety")

from src.data import DataConfig, ManufacturingDataModule
from src.models.v2_plus import V2Plus

CKPT_PATH = "/home/keti/factory_safety/results/v2plus/best_model.pt"
ONNX_PATH = "/home/keti/factory_safety/jetson_deploy/model/model_v2plus.onnx"
OUT_DIR = "/home/keti/factory_safety/jetson_deploy/reference"

# How many samples to include. Full val set is ~1862 windows.
# Keep all so we can reproduce F1; 30 floats x 8 + 30 x 120 x 160 per sample.
N_SAMPLES = None  # None = all


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = V2Plus(sensor_dim=8, hidden_dim=128, num_layers=3, num_classes=4, lags=[1, 5, 10])
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"Loaded V2+ (epoch {ckpt['epoch']}, F1={ckpt['val_f1']:.4f})")

    cfg = DataConfig()
    cfg.batch_size = 1
    cfg.num_workers = 0
    dm = ManufacturingDataModule(cfg)
    dm.setup()
    n_total = len(dm.val_dataset)
    n = n_total if N_SAMPLES is None else min(N_SAMPLES, n_total)
    print(f"Generating reference for {n}/{n_total} validation samples")

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    sensors = np.zeros((n, 30, 8), dtype=np.float32)
    thermals = np.zeros((n, 30, 120, 160), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    pt_logits = np.zeros((n, 4), dtype=np.float32)
    onnx_logits = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        sample = dm.val_dataset[i]
        s = sample["sensor"].numpy().astype(np.float32)
        t = sample["thermal"].numpy().astype(np.float32)
        sensors[i] = s
        thermals[i] = t
        labels[i] = sample["label"].item()

        s_in = torch.from_numpy(s).unsqueeze(0).to(device)
        t_in = torch.from_numpy(t).unsqueeze(0).to(device)
        with torch.no_grad():
            pt_logits[i] = model(s_in, t_in).cpu().numpy()[0]
        onnx_logits[i] = sess.run(None, {"sensor": s[None], "thermal": t[None]})[0][0]

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n}")

    pt_preds = pt_logits.argmax(-1)
    onnx_preds = onnx_logits.argmax(-1)
    match = float((pt_preds == onnx_preds).mean())
    pt_acc = float((pt_preds == labels).mean())
    print(f"PT vs ONNX match: {match:.4f}  |  PT val acc: {pt_acc:.4f}")

    out_path = os.path.join(OUT_DIR, "val_reference.npz")
    np.savez_compressed(
        out_path,
        sensors=sensors,
        thermals=thermals,
        labels=labels,
        pt_logits=pt_logits,
        onnx_logits=onnx_logits,
        pt_preds=pt_preds,
        onnx_preds=onnx_preds,
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Saved {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
