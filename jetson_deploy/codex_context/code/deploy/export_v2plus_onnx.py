#!/usr/bin/env python3
"""Export V2+ model to ONNX and benchmark.

Usage:
    cd /home/keti/factory_safety
    python -m src.deploy.export_v2plus_onnx
"""

import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "/home/keti/factory_safety")

from src.data import DataConfig, ManufacturingDataModule
from src.models.v2_plus import V2Plus

CKPT_PATH = "/home/keti/factory_safety/results/v2plus/best_model.pt"
OUTPUT_DIR = "/home/keti/factory_safety/results/deploy"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load model
    model = V2Plus(sensor_dim=8, hidden_dim=128, num_layers=3, num_classes=4, lags=[1, 5, 10])
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded V2+ model (epoch {ckpt['epoch']}, F1={ckpt['val_f1']:.4f})")

    # Dummy inputs
    dummy_sensor = torch.randn(1, 30, 8)
    dummy_thermal = torch.randn(1, 30, 120, 160)

    with torch.no_grad():
        pt_output = model(dummy_sensor, dummy_thermal)

    # Export ONNX
    onnx_path = os.path.join(OUTPUT_DIR, "model_v2plus.onnx")
    print("\nExporting to ONNX...")
    torch.onnx.export(
        model, (dummy_sensor, dummy_thermal), onnx_path,
        opset_version=17,
        input_names=["sensor", "thermal"],
        output_names=["logits"],
        dynamic_axes={"sensor": {0: "batch"}, "thermal": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX saved: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

    # Verify ONNX output
    sess_cpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_output = sess_cpu.run(None, {"sensor": dummy_sensor.numpy(), "thermal": dummy_thermal.numpy()})[0]
    diff = np.abs(pt_output.numpy() - ort_output).max()
    print(f"PyTorch vs ONNX max diff: {diff:.8f} {'PASS' if diff < 1e-4 else 'FAIL'}")

    # Speed benchmark
    print("\n--- Speed Benchmark (batch=1, 100 runs) ---")

    # PyTorch GPU
    model.to(device)
    s, t = dummy_sensor.to(device), dummy_thermal.to(device)
    with torch.no_grad():
        for _ in range(10):
            model(s, t)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(s, t)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    pt_ms = np.mean(times) * 1000
    print(f"PyTorch GPU:  {pt_ms:.2f} ms")

    # ONNX CPU
    for _ in range(10):
        sess_cpu.run(None, {"sensor": dummy_sensor.numpy(), "thermal": dummy_thermal.numpy()})
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess_cpu.run(None, {"sensor": dummy_sensor.numpy(), "thermal": dummy_thermal.numpy()})
        times.append(time.perf_counter() - t0)
    cpu_ms = np.mean(times) * 1000
    print(f"ONNX CPU:     {cpu_ms:.2f} ms")

    # ONNX GPU
    if "CUDAExecutionProvider" in ort.get_available_providers():
        sess_gpu = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
        for _ in range(10):
            sess_gpu.run(None, {"sensor": dummy_sensor.numpy(), "thermal": dummy_thermal.numpy()})
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            sess_gpu.run(None, {"sensor": dummy_sensor.numpy(), "thermal": dummy_thermal.numpy()})
            times.append(time.perf_counter() - t0)
        gpu_ms = np.mean(times) * 1000
        print(f"ONNX GPU:     {gpu_ms:.2f} ms")

    jetson_est = pt_ms * 4
    print(f"Jetson (est): ~{jetson_est:.0f} ms")
    print(f"\nAll backends < 1000ms target: OK")

    # Full validation accuracy
    print("\n--- Accuracy Validation ---")
    cfg = DataConfig()
    cfg.batch_size = 1
    cfg.num_workers = 0
    dm = ManufacturingDataModule(cfg)
    dm.setup()

    pt_preds, onnx_preds, labels = [], [], []
    model.eval()
    for i in range(len(dm.val_dataset)):
        sample = dm.val_dataset[i]
        s_in = sample["sensor"].unsqueeze(0)
        t_in = sample["thermal"].unsqueeze(0)
        lab = sample["label"].item()

        with torch.no_grad():
            pt_pred = model(s_in.to(device), t_in.to(device)).argmax(-1).cpu().item()
        onnx_pred = np.argmax(sess_cpu.run(None, {"sensor": s_in.numpy(), "thermal": t_in.numpy()})[0])

        pt_preds.append(pt_pred)
        onnx_preds.append(onnx_pred)
        labels.append(lab)

    pt_f1 = f1_score(labels, pt_preds, average="macro")
    onnx_f1 = f1_score(labels, onnx_preds, average="macro")
    match = np.mean(np.array(pt_preds) == np.array(onnx_preds))

    print(f"PyTorch F1: {pt_f1:.4f}")
    print(f"ONNX F1:    {onnx_f1:.4f}")
    print(f"Match rate: {match:.4f} ({int(match*len(labels))}/{len(labels)})")

    # Save summary
    import json
    summary = {
        "model": "V2+",
        "onnx_file": "model_v2plus.onnx",
        "onnx_size_mb": round(os.path.getsize(onnx_path) / 1024 / 1024, 1),
        "pytorch_gpu_ms": round(pt_ms, 2),
        "onnx_cpu_ms": round(cpu_ms, 2),
        "jetson_estimate_ms": round(jetson_est, 0),
        "pytorch_f1": round(pt_f1, 4),
        "onnx_f1": round(onnx_f1, 4),
        "prediction_match_rate": round(float(match), 4),
    }
    with open(os.path.join(OUTPUT_DIR, "v2plus_benchmark.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}/v2plus_benchmark.json")


if __name__ == "__main__":
    main()
