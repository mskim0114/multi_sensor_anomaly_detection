#!/usr/bin/env python3
"""Measure inference latency on Jetson for V2+ ONNX model.

Runs warmup + 200 timed runs at batch=1 for each available execution
provider, then prints mean/median/p95/p99 in milliseconds and saves a
JSON report.

Usage:
    python3 02_benchmark_latency.py [--runs 200]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ONNX_PATH = os.path.join(ROOT, "model", "model_v2plus.onnx")
RESULTS_DIR = os.path.join(ROOT, "results")


TENSORRT_WARNING = (
    "WARNING: TensorRT EP is experimental on this Jetson runtime.\n"
    "         Known issue: ORT TensorRT EP crashes with SIGSEGV on the current\n"
    "         V2+ model. Benchmark results for the other providers are lost when\n"
    "         that happens. Details: docs/JETSON_ENVIRONMENT.md section 10."
)


def bench_provider(provider, runs, warmup=20):
    sess = ort.InferenceSession(ONNX_PATH, providers=[provider])
    sensor = np.random.randn(1, 30, 8).astype(np.float32)
    thermal = np.random.randn(1, 30, 120, 160).astype(np.float32)
    inputs = {"sensor": sensor, "thermal": thermal}

    for _ in range(warmup):
        sess.run(None, inputs)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return {
        "provider": provider,
        "runs": runs,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--include-tensorrt", action="store_true",
                   help="Also benchmark TensorrtExecutionProvider. Off by default: "
                        "it segfaults on this runtime and a segfault kills the whole "
                        "process, taking the other results with it.")
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Model: {ONNX_PATH}")
    print(f"Size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.1f} MB\n")

    available = ort.get_available_providers()
    print(f"Available providers: {available}")

    # TensorRT is excluded by default. See the provider policy note in
    # 03_verify_accuracy.py and docs/JETSON_ENVIRONMENT.md section 10: the ORT
    # TensorRT EP dies with SIGSEGV on the first sess.run() with this model, and
    # a segfault cannot be caught by the `except Exception` below - it would kill
    # this process and discard the CUDA and CPU results too.
    candidates = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if args.include_tensorrt:
        print(TENSORRT_WARNING, file=sys.stderr)
        candidates = ["TensorrtExecutionProvider"] + candidates
    to_run = [p for p in candidates if p in available]
    skipped = [p for p in available if p not in to_run]
    if skipped:
        print(f"Skipped (not benchmarked): {skipped}")

    results = []
    for prov in to_run:
        print(f"\nBenchmarking {prov}...")
        try:
            r = bench_provider(prov, args.runs)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        print(f"  mean: {r['mean_ms']:.2f} ms  | median: {r['median_ms']:.2f} ms"
              f"  | p95: {r['p95_ms']:.2f} ms  | p99: {r['p99_ms']:.2f} ms")
        results.append(r)

    out_path = os.path.join(RESULTS_DIR, "jetson_latency.json")
    with open(out_path, "w") as f:
        json.dump({"runs": args.runs, "results": results}, f, indent=2)
    print(f"\nSaved {out_path}")

    # Target check
    print("\n--- Target: < 5ms per inference (single sample) ---")
    for r in results:
        ok = "[OK]" if r["mean_ms"] < 5.0 else "[!!]"
        print(f"  {ok} {r['provider']}: mean={r['mean_ms']:.2f}ms")


if __name__ == "__main__":
    main()
