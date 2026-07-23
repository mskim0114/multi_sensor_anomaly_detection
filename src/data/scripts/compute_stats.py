#!/usr/bin/env python3
"""Compute normalization statistics from the training set.

Usage:
    python -m src.data.scripts.compute_stats
    python -m src.data.scripts.compute_stats --sample-rate 0.1
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, "/home/keti/factory_safety")

from src.data.config import DataConfig


def main():
    parser = argparse.ArgumentParser(description="Compute normalization stats")
    parser.add_argument("--sample-rate", type=float, default=0.1,
                        help="Fraction of files to sample (default: 0.1)")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = DataConfig.from_yaml(args.config) if args.config else DataConfig()
    source_dir = cfg.train_source_dir

    csv_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".csv")])
    bin_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".bin")])

    # Sample
    rng = np.random.RandomState(42)
    csv_sample = rng.choice(csv_files, size=int(len(csv_files) * args.sample_rate), replace=False)
    bin_sample = rng.choice(bin_files, size=int(len(bin_files) * args.sample_rate), replace=False)

    # Sensor stats
    print(f"Computing sensor stats from {len(csv_sample)} CSV files...")
    sensor_data = []
    for f in csv_sample:
        path = os.path.join(source_dir, f)
        row = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
        sensor_data.append(row)
    sensor_arr = np.stack(sensor_data)

    channels = ["NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4"]
    print(f"\n{'Channel':>8s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    means, stds = [], []
    for i, ch in enumerate(channels):
        col = sensor_arr[:, i]
        m, s = col.mean(), col.std()
        means.append(round(float(m), 2))
        stds.append(round(float(s), 2))
        print(f"{ch:>8s} {m:>10.2f} {s:>10.2f} {col.min():>10.2f} {col.max():>10.2f}")

    print(f"\nSensorStats(mean={means}, std={stds})")

    # Thermal stats
    print(f"\nComputing thermal stats from {len(bin_sample)} BIN files...")
    global_min, global_max = np.inf, -np.inf
    for f in bin_sample:
        path = os.path.join(source_dir, f)
        img = np.load(path, mmap_mode="r")
        global_min = min(global_min, img.min())
        global_max = max(global_max, img.max())

    print(f"ThermalStats(global_min={global_min:.2f}, global_max={global_max:.2f})")


if __name__ == "__main__":
    main()
