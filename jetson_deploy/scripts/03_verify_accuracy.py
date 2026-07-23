#!/usr/bin/env python3
"""Verify that Jetson ONNX predictions match PC reference predictions.

Loads pre-saved (input, PC prediction) pairs from val_reference[_small].npz,
runs them through ONNX on Jetson, and reports:
- Prediction match rate vs PC ONNX
- Logit max abs diff vs PC ONNX
- F1/accuracy on the same samples

Usage:
    python3 03_verify_accuracy.py [--small] [--provider auto|cpu|cuda|tensorrt]
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ONNX_PATH = os.path.join(ROOT, "model", "model_v2plus.onnx")
REF_FULL = os.path.join(ROOT, "reference", "val_reference.npz")
REF_SMALL = os.path.join(ROOT, "reference", "val_reference_small.npz")
RESULTS_DIR = os.path.join(ROOT, "results")
CLASS_NAMES = ["Normal", "Mild", "Moderate", "Severe"]


def pick_provider(arg):
    available = ort.get_available_providers()
    if arg == "auto":
        for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
            if p in available:
                return p
    mapping = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
               "tensorrt": "TensorrtExecutionProvider"}
    name = mapping.get(arg, arg)
    if name not in available:
        raise SystemExit(f"Provider {name} not available. Available: {available}")
    return name


def macro_f1(y_true, y_pred, n_classes=4):
    f1s = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), f1s


def confusion(y_true, y_pred, n_classes=4):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true", help="Use stratified 100-sample subset")
    p.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda", "tensorrt"])
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    ref_path = REF_SMALL if args.small else REF_FULL
    if not os.path.exists(ref_path):
        raise SystemExit(f"Reference not found: {ref_path}")
    print(f"Loading reference: {ref_path}")
    d = np.load(ref_path)
    sensors = d["sensors"].astype(np.float32)
    thermals = d["thermals"].astype(np.float32)
    labels = d["labels"]
    pc_onnx_logits = d["onnx_logits"]
    pc_onnx_preds = d["onnx_preds"]
    n = len(labels)
    print(f"Samples: {n}")
    print(f"PC ONNX preds dist: {dict(zip(*np.unique(pc_onnx_preds, return_counts=True)))}")

    provider = pick_provider(args.provider)
    print(f"Provider: {provider}")
    sess = ort.InferenceSession(ONNX_PATH, providers=[provider])

    # Warmup
    sess.run(None, {"sensor": sensors[:1], "thermal": thermals[:1]})

    jet_logits = np.zeros_like(pc_onnx_logits)
    t0 = time.perf_counter()
    for i in range(n):
        out = sess.run(None, {"sensor": sensors[i:i+1], "thermal": thermals[i:i+1]})[0][0]
        jet_logits[i] = out
    elapsed = time.perf_counter() - t0

    jet_preds = jet_logits.argmax(-1)
    match = float((jet_preds == pc_onnx_preds).mean())
    logit_max_diff = float(np.abs(jet_logits - pc_onnx_logits).max())
    logit_mean_diff = float(np.abs(jet_logits - pc_onnx_logits).mean())

    acc = float((jet_preds == labels).mean())
    f1_macro, f1_per_class = macro_f1(labels, jet_preds)
    cm = confusion(labels, jet_preds)

    print()
    print(f"Inference: {n} samples in {elapsed:.1f}s ({elapsed/n*1000:.2f} ms/sample)")
    print(f"Pred match rate (Jetson vs PC ONNX): {match*100:.2f}%  ({int(match*n)}/{n})")
    print(f"Logit max abs diff:  {logit_max_diff:.6f}")
    print(f"Logit mean abs diff: {logit_mean_diff:.6f}")
    print(f"\nJetson accuracy: {acc*100:.2f}%")
    print(f"Jetson macro-F1: {f1_macro:.4f}")
    print("Per-class F1:")
    for name, f in zip(CLASS_NAMES, f1_per_class):
        print(f"  {name:<9} F1={f:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    suffix = "_small" if args.small else ""
    out_path = os.path.join(RESULTS_DIR, f"jetson_accuracy{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({
            "provider": provider,
            "n_samples": n,
            "match_rate_vs_pc_onnx": match,
            "logit_max_abs_diff": logit_max_diff,
            "logit_mean_abs_diff": logit_mean_diff,
            "accuracy": acc,
            "macro_f1": f1_macro,
            "per_class_f1": dict(zip(CLASS_NAMES, [float(x) for x in f1_per_class])),
            "confusion_matrix": cm.tolist(),
            "inference_total_s": elapsed,
            "inference_per_sample_ms": elapsed / n * 1000,
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    if match >= 0.99:
        print("\n[OK] Predictions match PC reference within 1%.")
    else:
        print("\n[!!] Prediction mismatch >1%. Check ONNX version / EP determinism.")


if __name__ == "__main__":
    main()
