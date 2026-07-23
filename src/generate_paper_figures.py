#!/usr/bin/env python3
"""Generate all figures for the paper.

Usage:
    cd /home/keti/factory_safety
    python src/generate_paper_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

sys.path.insert(0, "/home/keti/factory_safety")

OUT_DIR = "/home/keti/factory_safety/docs/논문/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig1_model_architecture():
    """Fig 1: Model architecture diagram (text-based, clean)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    arch_text = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     Proposed V2+ Architecture                      │
    │                                                                     │
    │  Sensor Input (B, 30, 8)                                            │
    │       │                                                             │
    │       ├── Original: s(t)                    Thermal Input            │
    │       ├── Δ₁: s(t) - s(t-1)                (B, 30, 120, 160)       │
    │       ├── Δ₅: s(t) - s(t-5)                      │                 │
    │       └── Δ₁₀: s(t) - s(t-10)               3-layer CNN            │
    │              │                             (1→16→32→64)             │
    │         Concat → (B, 30, 32)                      │                 │
    │              │                              FC → (B, 30, 128)       │
    │         3-layer LSTM (128)                        │                 │
    │              │                                     │                 │
    │         FC → (B, 128)                              │                 │
    │              │                                     │                 │
    │         SE-block (Channel Attention)                │                 │
    │              │                                     │                 │
    │              └──── Broadcast + Concat ─────────────┘                 │
    │                          │                                           │
    │                    Mean Pool → (B, 256)                              │
    │                          │                                           │
    │                    FC → (B, 4)  [Logits]                            │
    │                          │                                           │
    │              Loss = 0.7 × CE + 0.3 × SupCon                        │
    └─────────────────────────────────────────────────────────────────────┘
    """
    ax.text(0.5, 0.5, arch_text, transform=ax.transAxes, fontsize=9,
            fontfamily="monospace", verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    fig.savefig(os.path.join(OUT_DIR, "fig1_architecture.png"))
    plt.close()
    print("Fig 1: Architecture saved")


def fig2_sensor_patterns():
    """Fig 2: Sensor behavior by degradation state."""
    states = ["Normal\n(State 0)", "Mild\n(State 1)", "Moderate\n(State 2)", "Severe\n(State 3)"]
    x = np.arange(len(states))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Temperature
    ntc = [26.78, 30.95, 40.65, 50.71]
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    bars = axes[0].bar(x, ntc, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("(a) Temperature (NTC)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(states)
    for bar, val in zip(bars, ntc):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", fontsize=9)
    axes[0].annotate("Only 4°C gap", xy=(0.5, 30.95), xytext=(1.5, 22),
                     arrowprops=dict(arrowstyle="->", color="red"),
                     fontsize=9, color="red", fontweight="bold")

    # Particulate Matter
    pm10 = [31.60, 37.59, 43.47, 43.28]
    pm25 = [18.38, 21.31, 24.53, 24.44]
    pm1 = [13.66, 16.18, 18.83, 18.76]
    w = 0.25
    axes[1].bar(x - w, pm10, w, label="PM10", color="#3498db", edgecolor="black", linewidth=0.5)
    axes[1].bar(x, pm25, w, label="PM2.5", color="#2980b9", edgecolor="black", linewidth=0.5)
    axes[1].bar(x + w, pm1, w, label="PM1.0", color="#1abc9c", edgecolor="black", linewidth=0.5)
    axes[1].set_title("(b) Particulate Matter")
    axes[1].set_ylabel("Concentration (µg/m³)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(states)
    axes[1].legend()
    axes[1].axhline(y=43, color="red", linestyle="--", alpha=0.5)
    axes[1].text(2.5, 44, "Saturation", fontsize=8, color="red")

    # Current
    ct1 = [2.16, 3.21, 10.16, 29.85]
    ct2 = [33.15, 30.97, 50.43, 115.38]
    axes[2].bar(x - 0.15, ct1, 0.3, label="CT1", color="#e74c3c", edgecolor="black", linewidth=0.5)
    axes[2].bar(x + 0.15, ct2, 0.3, label="CT2", color="#c0392b", edgecolor="black", linewidth=0.5)
    axes[2].set_title("(c) Motor Current")
    axes[2].set_ylabel("Current (A)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(states)
    axes[2].legend()
    axes[2].annotate("14× jump", xy=(3, 29.85), xytext=(2, 50),
                     arrowprops=dict(arrowstyle="->", color="red"),
                     fontsize=9, color="red", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_sensor_patterns.png"))
    plt.close()
    print("Fig 2: Sensor patterns saved")


def fig3_confusion_matrices():
    """Fig 3: Confusion matrix comparison (V1 baseline vs V2+ proposed)."""
    labels = ["Normal", "Mild", "Moderate", "Severe"]

    cm_v1 = np.array([[470, 39, 10, 1], [18, 256, 12, 0], [0, 9, 272, 4], [0, 0, 1, 65]])
    cm_v2plus = np.array([[497, 12, 9, 2], [12, 258, 16, 0], [0, 5, 280, 0], [0, 0, 0, 66]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_v1, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax1, cbar=False, linewidths=0.5)
    ax1.set_title("(a) V1: Baseline LSTM (F1 = 0.9235)", fontweight="bold")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    sns.heatmap(cm_v2plus, annot=True, fmt="d", cmap="Greens", xticklabels=labels,
                yticklabels=labels, ax=ax2, cbar=False, linewidths=0.5)
    ax2.set_title("(b) V2+: Proposed (F1 = 0.9557)", fontweight="bold")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    # Highlight Normal-Mild errors
    for ax, cm in [(ax1, cm_v1), (ax2, cm_v2plus)]:
        ax.add_patch(plt.Rectangle((1, 0), 1, 1, fill=False, edgecolor="red", linewidth=2))
        ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor="red", linewidth=2))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_confusion_matrices.png"))
    plt.close()
    print("Fig 3: Confusion matrices saved")


def fig4_ablation_chart():
    """Fig 4: Ablation study bar chart."""
    models = [
        "TimesNet\n[25]", "PatchTST\n[26]", "V1\nBaseline", "V2\n+TempDiff",
        "V3\n+EffNet", "V4\nCATFT\n-CrossAttn", "V5\nFull\nCATFT", "V2+\n(Proposed)"
    ]
    f1_scores = [0.9189, 0.9311, 0.9235, 0.9430, 0.9242, 0.9112, 0.9252, 0.9557]
    params = [1.55, 1.36, 2.83, 2.84, 4.52, 7.63, 10.79, 2.85]

    colors = ["#95a5a6", "#95a5a6", "#3498db", "#2980b9", "#3498db",
              "#e67e22", "#e67e22", "#e74c3c"]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    bars = ax1.bar(range(len(models)), f1_scores, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Macro F1-Score")
    ax1.set_ylim(0.90, 0.96)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=9)

    # F1 values on bars
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{f1:.4f}", ha="center", fontsize=8, fontweight="bold")

    # Params as secondary axis
    ax2 = ax1.twinx()
    ax2.plot(range(len(models)), params, "ko--", markersize=6, linewidth=1.5, label="Params (M)")
    ax2.set_ylabel("Parameters (M)")
    ax2.set_ylim(0, 12)
    ax2.legend(loc="upper left")

    # Baseline reference line
    ax1.axhline(y=0.9235, color="gray", linestyle=":", alpha=0.5)
    ax1.text(7.5, 0.9240, "Baseline", fontsize=8, color="gray")

    ax1.set_title("Model Performance Comparison", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_ablation_chart.png"))
    plt.close()
    print("Fig 4: Ablation chart saved")


def fig5_component_synergy():
    """Fig 5: V2+ component ablation (individual vs combined)."""
    labels = ["V2\n(base)", "+Multi-Scale\nDiff only", "+SE\nonly", "+SupCon\nonly", "V2+\n(all combined)"]
    f1 = [0.9430, 0.9432, 0.9319, 0.9422, 0.9557]
    nm_errors = [45, 45, 53, 46, 24]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = ["#3498db", "#f39c12", "#f39c12", "#f39c12", "#e74c3c"]
    bars1 = ax1.bar(range(len(labels)), f1, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Macro F1-Score")
    ax1.set_ylim(0.92, 0.96)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("(a) F1-Score", fontweight="bold")
    for bar, val in zip(bars1, f1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f"{val:.4f}", ha="center", fontsize=8)
    ax1.axhline(y=0.9430, color="gray", linestyle=":", alpha=0.5)

    bars2 = ax2.bar(range(len(labels)), nm_errors, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Normal↔Mild Errors")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("(b) Normal-Mild Boundary Errors", fontweight="bold")
    for bar, val in zip(bars2, nm_errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha="center", fontsize=9)
    ax2.annotate("-47%", xy=(4, 24), xytext=(3, 35),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2),
                 fontsize=12, color="red", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_component_synergy.png"))
    plt.close()
    print("Fig 5: Component synergy saved")


def fig6_lag_sensitivity():
    """Fig 6: Lag sensitivity analysis."""
    lags_labels = ["[1]\n(single)", "[1, 3, 7]", "[1, 5, 10]\n(proposed)", "[1, 10, 20]"]
    f1 = [0.9430, 0.9498, 0.9557, 0.9520]
    nm = [45, 35, 24, 31]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = ["#95a5a6", "#f39c12", "#e74c3c", "#f39c12"]
    ax1.bar(range(len(lags_labels)), f1, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Macro F1-Score")
    ax1.set_ylim(0.935, 0.96)
    ax1.set_xticks(range(len(lags_labels)))
    ax1.set_xticklabels(lags_labels, fontsize=9)
    ax1.set_title("(a) F1-Score by Lag Combination", fontweight="bold")
    for i, val in enumerate(f1):
        ax1.text(i, val + 0.0003, f"{val:.4f}", ha="center", fontsize=9)

    ax2.bar(range(len(lags_labels)), nm, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Normal↔Mild Errors")
    ax2.set_xticks(range(len(lags_labels)))
    ax2.set_xticklabels(lags_labels, fontsize=9)
    ax2.set_title("(b) Boundary Errors by Lag Combination", fontweight="bold")
    for i, val in enumerate(nm):
        ax2.text(i, val + 0.5, str(val), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_lag_sensitivity.png"))
    plt.close()
    print("Fig 6: Lag sensitivity saved")


if __name__ == "__main__":
    fig1_model_architecture()
    fig2_sensor_patterns()
    fig3_confusion_matrices()
    fig4_ablation_chart()
    fig5_component_synergy()
    fig6_lag_sensitivity()
    print(f"\nAll figures saved to {OUT_DIR}/")
