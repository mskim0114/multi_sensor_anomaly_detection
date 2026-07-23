#!/usr/bin/env python3
"""Simulate the real-time streaming inference pipeline on Jetson.

The deployed system needs ~30 sensor samples (and matched thermal frames)
to form a window, sliding every 10 samples. This script:
  1. Reads N samples from val_reference.npz as if they were arriving live
  2. Maintains rolling buffers of size 30
  3. Triggers inference whenever a full window is ready (every 'stride' steps)
  4. Reports per-window latency, total throughput, and predicted class
     transitions over time.

Usage:
    python3 04_realtime_pipeline.py [--small] [--stride 10] [--n 300]
                                    [--provider auto|cpu|cuda|tensorrt]
"""

import argparse
import collections
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
WINDOW = 30


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--n", type=int, default=300,
                   help="Number of streaming 'frames' to simulate")
    p.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda", "tensorrt"])
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    ref_path = REF_SMALL if args.small else REF_FULL
    if not os.path.exists(ref_path):
        raise SystemExit(f"Reference not found: {ref_path}")
    d = np.load(ref_path)

    # Treat each window-sample's first time step as a single "frame" for the stream.
    # This is a synthetic stream; we just want timing realism.
    # We'll concatenate sensor windows into a long stream by taking the
    # last time step of each window in order.
    sensors_stream = d["sensors"][:, -1, :].astype(np.float32)  # (N, 8)
    thermals_stream = d["thermals"][:, -1, :, :].astype(np.float32)  # (N, 120, 160)
    n_stream = min(args.n, len(sensors_stream))
    print(f"Streaming {n_stream} synthetic frames, window={WINDOW}, stride={args.stride}")

    provider = pick_provider(args.provider)
    print(f"Provider: {provider}")
    sess = ort.InferenceSession(ONNX_PATH, providers=[provider])
    # Warmup
    sess.run(None, {
        "sensor": np.zeros((1, WINDOW, 8), dtype=np.float32),
        "thermal": np.zeros((1, WINDOW, 120, 160), dtype=np.float32),
    })

    s_buf = collections.deque(maxlen=WINDOW)
    t_buf = collections.deque(maxlen=WINDOW)

    latencies_ms = []
    preds_timeline = []  # (frame_idx, pred_label)
    confidences = []

    t_start = time.perf_counter()
    frames_since_last_infer = 0
    for i in range(n_stream):
        s_buf.append(sensors_stream[i])
        t_buf.append(thermals_stream[i])
        frames_since_last_infer += 1

        if len(s_buf) < WINDOW:
            continue
        if frames_since_last_infer < args.stride:
            continue
        frames_since_last_infer = 0

        s_win = np.stack(list(s_buf))[None].astype(np.float32)
        t_win = np.stack(list(t_buf))[None].astype(np.float32)
        t0 = time.perf_counter()
        logits = sess.run(None, {"sensor": s_win, "thermal": t_win})[0][0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        pred = int(probs.argmax())
        preds_timeline.append((i, pred))
        confidences.append(float(probs.max()))

    total_s = time.perf_counter() - t_start
    n_infer = len(latencies_ms)
    lat = np.array(latencies_ms)

    print()
    print(f"Frames processed: {n_stream} in {total_s:.2f}s ({n_stream/total_s:.1f} fps)")
    print(f"Inferences: {n_infer} (every {args.stride} frames)")
    if n_infer > 0:
        print(f"Inference latency: mean={lat.mean():.2f}ms  median={np.median(lat):.2f}ms"
              f"  p95={np.percentile(lat, 95):.2f}ms  max={lat.max():.2f}ms")

    # Class transition summary
    if preds_timeline:
        labels_seq = [p for _, p in preds_timeline]
        unique, counts = np.unique(labels_seq, return_counts=True)
        print("\nPredicted class distribution over stream:")
        for u, c in zip(unique, counts):
            print(f"  {CLASS_NAMES[u]:<9}: {c} ({c/len(labels_seq)*100:.1f}%)")

        transitions = sum(1 for a, b in zip(labels_seq[:-1], labels_seq[1:]) if a != b)
        print(f"Class transitions: {transitions}")
        if confidences:
            cf = np.array(confidences)
            print(f"Confidence: mean={cf.mean():.3f}  min={cf.min():.3f}  max={cf.max():.3f}")

    out_path = os.path.join(RESULTS_DIR, "jetson_realtime.json")
    with open(out_path, "w") as f:
        json.dump({
            "provider": provider,
            "stride": args.stride,
            "window": WINDOW,
            "n_frames": int(n_stream),
            "n_inferences": int(n_infer),
            "total_seconds": float(total_s),
            "fps": float(n_stream / total_s),
            "latency_ms_mean": float(lat.mean()) if n_infer else None,
            "latency_ms_median": float(np.median(lat)) if n_infer else None,
            "latency_ms_p95": float(np.percentile(lat, 95)) if n_infer else None,
            "latency_ms_max": float(lat.max()) if n_infer else None,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
