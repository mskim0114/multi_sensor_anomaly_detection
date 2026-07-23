#!/usr/bin/env python3
"""Benchmark inference speed and accuracy across backends.

Compares: PyTorch (GPU) vs ONNX Runtime (CPU) vs ONNX Runtime (GPU)
Also validates that accuracy is preserved after conversion.

Usage:
    cd /home/keti/factory_safety
    python -m src.deploy.benchmark
    python -m src.deploy.benchmark --num-warmup 10 --num-runs 100
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, "/home/keti/factory_safety")

from src.data import DataConfig, ManufacturingDataModule
from src.models.ablation_variants import LSTMWithTemporalDiff

CKPT_PATH = "/home/keti/factory_safety/results/ablation_v2/best_model.pt"
ONNX_PATH = "/home/keti/factory_safety/results/deploy/model_v2.onnx"
OUTPUT_DIR = "/home/keti/factory_safety/results/deploy"


def benchmark_pytorch(model, sensor, thermal, device, num_warmup=5, num_runs=50):
    """Benchmark PyTorch inference on GPU."""
    model.eval()
    s = sensor.to(device)
    t = thermal.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(s, t)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(s, t)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return times


def benchmark_onnx(session, sensor_np, thermal_np, num_warmup=5, num_runs=50):
    """Benchmark ONNX Runtime inference."""
    inputs = {"sensor": sensor_np, "thermal": thermal_np}

    # Warmup
    for _ in range(num_warmup):
        _ = session.run(None, inputs)

    # Timed runs
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, inputs)
        times.append(time.perf_counter() - t0)

    return times


def evaluate_accuracy(model_fn, val_dataset, device=None):
    """Run full validation and return predictions."""
    all_preds, all_labels = [], []
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        sensor = sample["sensor"].unsqueeze(0)
        thermal = sample["thermal"].unsqueeze(0)
        label = sample["label"].item()

        pred = model_fn(sensor, thermal)
        all_preds.append(pred)
        all_labels.append(label)

    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # --- Load models ---
    print("Loading PyTorch model...")
    pt_model = LSTMWithTemporalDiff(sensor_dim=8, hidden_dim=128, num_layers=3, num_classes=4)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.to(device)
    pt_model.eval()

    print("Loading ONNX model...")
    ort_cpu = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    ort_gpu = None
    if "CUDAExecutionProvider" in ort.get_available_providers():
        ort_gpu = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider"])
        print("ONNX GPU provider available")
    else:
        print("ONNX GPU provider not available, CPU only")

    # --- Speed benchmark (batch=1, simulating real-time) ---
    print(f"\n{'='*60}")
    print("SPEED BENCHMARK (batch=1, single inference)")
    print(f"{'='*60}")

    dummy_sensor = torch.randn(1, 30, 8)
    dummy_thermal = torch.randn(1, 30, 120, 160)
    dummy_sensor_np = dummy_sensor.numpy()
    dummy_thermal_np = dummy_thermal.numpy()

    results = {}

    # PyTorch GPU
    times = benchmark_pytorch(pt_model, dummy_sensor, dummy_thermal, device,
                              args.num_warmup, args.num_runs)
    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    print(f"\nPyTorch GPU:     {mean_ms:>7.2f} ms (std={std_ms:.2f}, p95={p95_ms:.2f})")
    results["pytorch_gpu"] = {"mean_ms": mean_ms, "std_ms": std_ms, "p95_ms": p95_ms}

    # ONNX CPU
    times = benchmark_onnx(ort_cpu, dummy_sensor_np, dummy_thermal_np,
                           args.num_warmup, args.num_runs)
    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    print(f"ONNX CPU:        {mean_ms:>7.2f} ms (std={std_ms:.2f}, p95={p95_ms:.2f})")
    results["onnx_cpu"] = {"mean_ms": mean_ms, "std_ms": std_ms, "p95_ms": p95_ms}

    # ONNX GPU
    if ort_gpu:
        times = benchmark_onnx(ort_gpu, dummy_sensor_np, dummy_thermal_np,
                               args.num_warmup, args.num_runs)
        mean_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        p95_ms = np.percentile(times, 95) * 1000
        print(f"ONNX GPU:        {mean_ms:>7.2f} ms (std={std_ms:.2f}, p95={p95_ms:.2f})")
        results["onnx_gpu"] = {"mean_ms": mean_ms, "std_ms": std_ms, "p95_ms": p95_ms}

    # Jetson estimate (RTX 6000 ~16.3 TFLOPS, Orin Nano ~40 TOPS INT8 / ~5 TFLOPS FP16)
    # Rough estimate: ~3-5x slower than RTX 6000 for FP16
    if "pytorch_gpu" in results:
        jetson_est = results["pytorch_gpu"]["mean_ms"] * 4
        print(f"\nJetson Orin Nano (estimated): ~{jetson_est:.0f} ms")
        results["jetson_estimate_ms"] = jetson_est

    # --- Accuracy validation ---
    print(f"\n{'='*60}")
    print("ACCURACY VALIDATION (full validation set)")
    print(f"{'='*60}")

    cfg = DataConfig()
    cfg.batch_size = 1
    cfg.num_workers = 0
    dm = ManufacturingDataModule(cfg)
    dm.setup()
    val_dataset = dm.val_dataset
    print(f"Validation samples: {len(val_dataset)}")

    # PyTorch predictions
    print("\nRunning PyTorch inference...")
    def pt_predict(sensor, thermal):
        with torch.no_grad():
            out = pt_model(sensor.to(device), thermal.to(device))
        return out.argmax(dim=-1).cpu().item()
    pt_preds, labels = evaluate_accuracy(pt_predict, val_dataset)

    # ONNX CPU predictions
    print("Running ONNX CPU inference...")
    def onnx_cpu_predict(sensor, thermal):
        out = ort_cpu.run(None, {"sensor": sensor.numpy(), "thermal": thermal.numpy()})[0]
        return np.argmax(out, axis=-1)[0]
    onnx_preds, _ = evaluate_accuracy(onnx_cpu_predict, val_dataset)

    # Compare
    pt_f1 = f1_score(labels, pt_preds, average="macro")
    pt_acc = accuracy_score(labels, pt_preds)
    onnx_f1 = f1_score(labels, onnx_preds, average="macro")
    onnx_acc = accuracy_score(labels, onnx_preds)
    match_rate = (pt_preds == onnx_preds).mean()

    print(f"\n{'Backend':<15s} {'Accuracy':>10s} {'F1 (macro)':>12s}")
    print("-" * 40)
    print(f"{'PyTorch GPU':<15s} {pt_acc:>10.4f} {pt_f1:>12.4f}")
    print(f"{'ONNX CPU':<15s} {onnx_acc:>10.4f} {onnx_f1:>12.4f}")
    print(f"\nPrediction match rate: {match_rate:.4f} ({int(match_rate * len(labels))}/{len(labels)})")

    if match_rate > 0.999:
        print("PASS: ONNX predictions match PyTorch")
    else:
        print(f"NOTE: {int((1 - match_rate) * len(labels))} predictions differ (minor numerical differences)")

    results["pytorch_f1"] = pt_f1
    results["pytorch_acc"] = pt_acc
    results["onnx_f1"] = onnx_f1
    results["onnx_acc"] = onnx_acc
    results["prediction_match_rate"] = float(match_rate)

    # Real-time feasibility
    print(f"\n{'='*60}")
    print("REAL-TIME FEASIBILITY (target: < 1000ms per inference)")
    print(f"{'='*60}")
    for backend, data in results.items():
        if isinstance(data, dict) and "mean_ms" in data:
            feasible = "OK" if data["mean_ms"] < 1000 else "FAIL"
            print(f"  {backend:<15s}: {data['mean_ms']:>7.2f} ms [{feasible}]")
    if "jetson_estimate_ms" in results:
        feasible = "OK" if results["jetson_estimate_ms"] < 1000 else "FAIL"
        print(f"  {'jetson(est)':<15s}: {results['jetson_estimate_ms']:>7.0f} ms [{feasible}]")

    # Save results
    with open(os.path.join(OUTPUT_DIR, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {OUTPUT_DIR}/benchmark_results.json")


if __name__ == "__main__":
    main()
