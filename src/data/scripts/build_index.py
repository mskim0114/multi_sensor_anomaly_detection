#!/usr/bin/env python3
"""Build session index for the manufacturing dataset.

Usage:
    python -m src.data.scripts.build_index
    python -m src.data.scripts.build_index --force
"""

import argparse
import logging
import sys
from collections import Counter

sys.path.insert(0, "/home/keti/factory_safety")

from src.data.config import DataConfig
from src.data.session_index import load_or_build_index


def main():
    parser = argparse.ArgumentParser(description="Build session index")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = DataConfig.from_yaml(args.config) if args.config else DataConfig()

    for split, src_dir, lbl_dir in [
        ("train", cfg.train_source_dir, cfg.train_label_dir),
        ("val", cfg.val_source_dir, cfg.val_label_dir),
    ]:
        print(f"\n{'='*60}")
        print(f"Building {split} index...")
        print(f"  Source: {src_dir}")
        print(f"  Labels: {lbl_dir}")
        print(f"{'='*60}")

        index = load_or_build_index(
            src_dir, lbl_dir, cfg.cache_dir,
            split=split,
            gap_threshold=cfg.session_gap_threshold,
            force_rebuild=args.force,
        )

        # Print summary
        print(f"\n{index.summary()}")

        # Session details
        print(f"\nSessions by device type:")
        device_types = Counter(s.device_type for s in index.sessions)
        for dt, count in sorted(device_types.items()):
            print(f"  {dt}: {count} sessions")

        # Label distribution
        all_labels = []
        for s in index.sessions:
            all_labels.extend(s.labels)
        label_counts = Counter(all_labels)
        total = sum(label_counts.values())
        print(f"\nLabel distribution:")
        for state in sorted(label_counts.keys()):
            count = label_counts[state]
            print(f"  State {state}: {count:>6d} ({count/total*100:.1f}%)")
        print(f"  Total:  {total:>6d}")

        # Session length stats
        lengths = [len(s) for s in index.sessions]
        print(f"\nSession lengths: min={min(lengths)}, max={max(lengths)}, "
              f"mean={sum(lengths)/len(lengths):.0f}")


if __name__ == "__main__":
    main()
