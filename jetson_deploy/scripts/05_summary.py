#!/usr/bin/env python3
"""Aggregate all Jetson result JSONs into one summary report.

Usage:
    python3 05_summary.py
"""

import json
import os
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(ROOT, "results")


def load(name):
    p = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    summary = {
        "latency": load("jetson_latency.json"),
        "accuracy_small": load("jetson_accuracy_small.json"),
        "accuracy_full": load("jetson_accuracy.json"),
        "realtime": load("jetson_realtime.json"),
    }

    out_path = os.path.join(RESULTS_DIR, "jetson_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("JETSON ORIN NANO DEPLOYMENT VERIFICATION SUMMARY")
    print("=" * 60)

    lat = summary["latency"]
    if lat:
        print("\n[Latency benchmark]")
        for r in lat["results"]:
            print(f"  {r['provider']:<28}  mean={r['mean_ms']:>6.2f}ms  "
                  f"p95={r['p95_ms']:>6.2f}ms  p99={r['p99_ms']:>6.2f}ms")

    acc = summary["accuracy_full"] or summary["accuracy_small"]
    if acc:
        scope = "FULL" if summary["accuracy_full"] else "SMALL"
        print(f"\n[Accuracy verification - {scope}]")
        print(f"  Match vs PC ONNX:    {acc['match_rate_vs_pc_onnx']*100:.2f}%")
        print(f"  Logit max abs diff:  {acc['logit_max_abs_diff']:.6f}")
        print(f"  Accuracy:            {acc['accuracy']*100:.2f}%")
        print(f"  Macro-F1:            {acc['macro_f1']:.4f}")

    rt = summary["realtime"]
    if rt:
        print("\n[Real-time pipeline]")
        print(f"  Provider: {rt['provider']}")
        print(f"  Frames: {rt['n_frames']}, Inferences: {rt['n_inferences']}")
        print(f"  FPS: {rt['fps']:.1f}")
        if rt.get("latency_ms_mean") is not None:
            print(f"  Latency: mean={rt['latency_ms_mean']:.2f}ms"
                  f"  p95={rt['latency_ms_p95']:.2f}ms")

    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
