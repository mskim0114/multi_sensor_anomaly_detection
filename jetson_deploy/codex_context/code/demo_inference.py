#!/usr/bin/env python3
"""Interactive inference demo with visualization.

Loads the best model (V2: LSTM+TemporalDiff) and runs inference on
validation samples, displaying sensor plots, thermal images, and predictions.

Usage (run from terminal, not from Claude):
    cd /home/keti/factory_safety
    python src/demo_inference.py
    python src/demo_inference.py --num-samples 10
    python src/demo_inference.py --save-dir results/demo  # save images instead of display
"""

import argparse
import os
import sys

import matplotlib
# Try GUI backend first, fall back to file saving
GUI_AVAILABLE = False
if os.environ.get("DISPLAY"):
    for backend in ["Qt5Agg", "TkAgg", "GTK3Agg"]:
        try:
            matplotlib.use(backend)
            GUI_AVAILABLE = True
            break
        except Exception:
            continue
if not GUI_AVAILABLE:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, "/home/keti/factory_safety")

from src.data import DataConfig, ManufacturingDataModule
from src.data.normalization import SensorNormalizer, ThermalNormalizer
from src.models.ablation_variants import LSTMWithTemporalDiff

STATE_NAMES = ["Normal", "Mild", "Moderate", "Severe"]
STATE_COLORS = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
SENSOR_NAMES = ["NTC (°C)", "PM1.0", "PM2.5", "PM10", "CT1 (A)", "CT2 (A)", "CT3 (A)", "CT4 (A)"]


def load_model(device):
    model = LSTMWithTemporalDiff(sensor_dim=8, hidden_dim=128, num_layers=3, num_classes=4)
    ckpt_path = "/home/keti/factory_safety/results/ablation_v2/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (F1={ckpt['val_f1']:.4f})")
    return model


def denormalize_sensor(sensor_norm, normalizer):
    """Convert normalized sensor back to original scale."""
    return normalizer.inverse(sensor_norm.numpy())


def denormalize_thermal(thermal_norm, normalizer):
    """Convert normalized thermal back to °C."""
    return normalizer.inverse(thermal_norm.numpy())


def visualize_sample(sensor_raw, thermal_raw, true_label, pred_label, pred_probs,
                     metadata, sample_idx, save_path=None):
    """Create a comprehensive visualization for one sample."""

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Sample #{sample_idx}  |  Session: {metadata['session_id']}  |  "
        f"Window start: {metadata['window_start']}",
        fontsize=14, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

    # --- Top row: Sensor time series (4 plots) ---
    sensor_groups = [
        (0, "NTC Temperature (°C)"),
        (slice(1, 4), "Dust (PM1.0, PM2.5, PM10)"),
        (slice(4, 6), "Current (CT1, CT2)"),
        (slice(6, 8), "Current (CT3, CT4)"),
    ]
    for col, (idx, title) in enumerate(sensor_groups):
        ax = fig.add_subplot(gs[0, col])
        t = np.arange(sensor_raw.shape[0])
        if isinstance(idx, int):
            ax.plot(t, sensor_raw[:, idx], color="#3498db", linewidth=1.5)
        else:
            channels = range(idx.start, idx.stop)
            for ch in channels:
                ax.plot(t, sensor_raw[:, ch], linewidth=1.5, label=SENSOR_NAMES[ch])
            ax.legend(fontsize=7, loc="upper left")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.3)

    # --- Middle row: Thermal images (4 frames from the window) ---
    frame_indices = [0, 9, 19, 29]  # first, 1/3, 2/3, last
    for col, fi in enumerate(frame_indices):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(thermal_raw[fi], cmap="inferno", aspect="auto")
        ax.set_title(f"Thermal t={fi} ({thermal_raw[fi].min():.1f}~{thermal_raw[fi].max():.1f}°C)",
                      fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Bottom left: Prediction result ---
    ax = fig.add_subplot(gs[2, 0:2])
    bars = ax.barh(STATE_NAMES, pred_probs, color=STATE_COLORS)
    # Highlight predicted
    bars[pred_label].set_edgecolor("black")
    bars[pred_label].set_linewidth(3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")

    # Correct/incorrect indicator
    is_correct = true_label == pred_label
    result_text = "CORRECT" if is_correct else "INCORRECT"
    result_color = "#2ecc71" if is_correct else "#e74c3c"
    ax.set_title(
        f"Prediction: {STATE_NAMES[pred_label]} ({pred_probs[pred_label]:.1%})  |  "
        f"True: {STATE_NAMES[true_label]}  |  [{result_text}]",
        fontsize=12, fontweight="bold", color=result_color,
    )

    # --- Bottom right: Sensor change rate (temporal diff) ---
    ax = fig.add_subplot(gs[2, 2:4])
    diff = np.diff(sensor_raw, axis=0)  # (29, 8)
    # Show key sensors
    for ch, name, color in [(0, "NTC", "#e74c3c"), (4, "CT1", "#3498db"), (5, "CT2", "#2ecc71")]:
        ax.plot(diff[:, ch], label=f"Δ{name}", linewidth=1.5, color=color)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Sensor Rate of Change (Temporal Diff)", fontsize=10)
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inference Demo")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save images to directory instead of displaying")
    parser.add_argument("--show-errors-only", action="store_true",
                        help="Only show incorrectly predicted samples")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Determine output mode
    save_mode = args.save_dir is not None or not GUI_AVAILABLE
    if not GUI_AVAILABLE and args.save_dir is None:
        args.save_dir = "/home/keti/factory_safety/results/demo"
        print(f"GUI not available. Saving images to {args.save_dir}/")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model = load_model(device)

    # Load data
    cfg = DataConfig()
    cfg.batch_size = 1
    cfg.num_workers = 0
    dm = ManufacturingDataModule(cfg)
    dm.setup()

    # Normalizers for denormalization (visualization in original scale)
    sensor_norm = SensorNormalizer(cfg.sensor_stats)
    thermal_norm = ThermalNormalizer(cfg.thermal_stats, cfg.thermal_norm_mode)

    val_dataset = dm.val_dataset
    print(f"Validation set: {len(val_dataset)} windows")

    # Select samples
    if args.show_errors_only:
        print("Finding misclassified samples...")
        indices = []
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            with torch.no_grad():
                out = model(sample["sensor"].unsqueeze(0).to(device),
                           sample["thermal"].unsqueeze(0).to(device))
            pred = out.argmax(dim=-1).item()
            if pred != sample["label"].item():
                indices.append(i)
            if len(indices) >= args.num_samples:
                break
        print(f"Found {len(indices)} errors")
    else:
        # Evenly sample across dataset
        indices = np.linspace(0, len(val_dataset) - 1, args.num_samples, dtype=int).tolist()

    # Run inference and visualize
    print(f"\nRunning inference on {len(indices)} samples...\n")

    for count, idx in enumerate(indices):
        sample = val_dataset[idx]
        sensor = sample["sensor"].unsqueeze(0).to(device)
        thermal = sample["thermal"].unsqueeze(0).to(device)
        true_label = sample["label"].item()
        metadata = sample["metadata"]

        with torch.no_grad():
            logits = model(sensor, thermal)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_label = logits.argmax(dim=-1).item()

        # Denormalize for visualization
        sensor_raw = denormalize_sensor(sample["sensor"], sensor_norm)
        thermal_raw = denormalize_thermal(sample["thermal"], thermal_norm)

        is_correct = "O" if pred_label == true_label else "X"
        print(f"[{is_correct}] Sample {idx}: True={STATE_NAMES[true_label]}, "
              f"Pred={STATE_NAMES[pred_label]} ({probs[pred_label]:.1%}), "
              f"Session={metadata['session_id']}")

        save_path = os.path.join(args.save_dir, f"sample_{idx:04d}.png") if save_mode else None
        visualize_sample(sensor_raw, thermal_raw, true_label, pred_label, probs,
                         metadata, idx, save_path)

    if save_mode:
        print(f"\nAll images saved to {args.save_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
