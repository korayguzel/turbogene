"""Generate paper figures for TurboGene."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

out_dir = Path("paper/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C_ORIG = "#d62728"     # red
C_SDPA = "#ff7f0e"     # orange
C_TURBO = "#2ca02c"    # green
C_KIVI = "#9467bd"     # purple
C_OOM = "#cccccc"      # gray
C_FP16 = "#1f77b4"     # blue

GPU_TOTAL = 11864  # MB


# ═══════════════════════════════════════════════════════════
# Figure 1: VRAM bar chart
# ═══════════════════════════════════════════════════════════
def fig1_vram():
    batches = [1, 4, 8, 12, 16]

    orig_vals =  [2005, 5534, None, None, None]
    sdpa_vals =  [2067, 4647, None, None, None]
    turbo_vals = [1446, 2160, 3113, 4067, 5019]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x = np.arange(len(batches))
    w = 0.25

    # Plot bars, using gray+hatch for OOM
    for i, (vals, color, label) in enumerate([
        (orig_vals, C_ORIG, "Original (FlexAttn)"),
        (sdpa_vals, C_SDPA, "SDPA (no chunking)"),
        (turbo_vals, C_TURBO, "TurboGene (ours)"),
    ]):
        for j, v in enumerate(vals):
            pos = x[j] + (i - 1) * w
            if v is not None:
                bar = ax.bar(pos, v / 1024, w * 0.9, color=color,
                             label=label if j == 0 else None, edgecolor="white", linewidth=0.5)
            else:
                bar = ax.bar(pos, GPU_TOTAL / 1024, w * 0.9, color=C_OOM,
                             edgecolor=color, linewidth=1.5, linestyle="--",
                             label=None, hatch="//", alpha=0.4)
                ax.text(pos, GPU_TOTAL / 1024 / 2, "OOM", ha="center", va="center",
                        fontsize=7, fontweight="bold", color=color)

    # GPU limit line
    ax.axhline(y=GPU_TOTAL / 1024, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(len(batches) - 0.5, GPU_TOTAL / 1024 + 0.15, "RTX 4070 (12 GB)",
            ha="right", va="bottom", fontsize=8, alpha=0.6)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(batches)
    ax.set_ylim(0, 14)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Peak VRAM Usage on RTX 4070 (12 GB)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_dir / "fig1_vram.png")
    fig.savefig(out_dir / "fig1_vram.pdf")
    plt.close(fig)
    print("  fig1_vram saved")


# ═══════════════════════════════════════════════════════════
# Figure 2: Per-layer KIVI vs TurboGene
# ═══════════════════════════════════════════════════════════
def fig2_kivi_comparison():
    layers = list(range(12))

    kivi_vals = [0.7990, 0.9293, 0.9273, 0.9286, 0.9202, 0.9310,
                 0.9349, 0.9576, 0.9431, 0.9589, 0.8957, 0.8821]
    tg_vals =   [0.9757, 0.9816, 0.9658, 0.9828, 0.9830, 0.8544,
                 0.9899, 0.9932, 0.9919, 0.9933, 0.9925, 0.9924]

    fig, ax = plt.subplots(figsize=(6, 3.2))

    x = np.arange(len(layers))
    w = 0.35

    ax.bar(x - w/2, kivi_vals, w, color=C_KIVI, label="KIVI 2-bit", edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, tg_vals, w, color=C_TURBO, label="TurboGene (ours)", edgecolor="white", linewidth=0.5)

    # FP16 reference line
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

    # Highlight improvement
    for i in range(len(layers)):
        delta = tg_vals[i] - kivi_vals[i]
        ax.annotate(f"+{delta:.2f}", xy=(x[i], max(tg_vals[i], kivi_vals[i])),
                    fontsize=5.5, ha="center", va="bottom", color="#333333")

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Attention Cosine Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(0.75, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Per-Layer Decode Quality: TurboGene vs KIVI")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_dir / "fig2_kivi_comparison.png")
    fig.savefig(out_dir / "fig2_kivi_comparison.pdf")
    plt.close(fig)
    print("  fig2_kivi_comparison saved")


# ═══════════════════════════════════════════════════════════
# Figure 3: Ablation bar chart
# ═══════════════════════════════════════════════════════════
def fig3_ablation():
    configs = [
        "FP16\n(baseline)",
        "Rotation\nonly",
        "+ Lloyd-Max\n3-bit",
        "+ QJL 1-bit\n(full)",
        "SVD rot\n(ours)",
        "Random rot",
        "Shared\nLUT",
        "Dual LUT\n(ours)",
    ]
    values = [1.000, 1.000, 0.979, 0.975, 0.975, 0.985, 0.967, 0.975]

    # Group colors
    colors = [
        C_FP16,      # FP16
        "#aec7e8",   # Rot only
        "#ffbb78",   # +LM
        C_TURBO,     # +QJL (full)
        C_TURBO,     # SVD (ours)
        "#aec7e8",   # Random
        "#ffbb78",   # Shared
        C_TURBO,     # Dual (ours)
    ]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    x = np.arange(len(configs))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels
    for i, (xi, v) in enumerate(zip(x, values)):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if colors[i] == C_TURBO else "normal")

    # Group separators
    for sep in [1.5, 3.5, 5.5]:
        ax.axvline(x=sep, color="#cccccc", linestyle="-", linewidth=0.5, alpha=0.5)

    # Group labels
    groups = [(0.5, "Baseline"), (2, "Pipeline"), (4.5, "Rotation"), (6.5, "LUT type")]
    for gx, label in groups:
        ax.text(gx, 0.94, label, ha="center", va="top", fontsize=7.5,
                fontstyle="italic", color="#666666")

    ax.set_ylabel("Attention Cosine Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=7.5)
    ax.set_ylim(0.83, 1.02)
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
    ax.set_title("Ablation Study: Component Contributions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_dir / "fig3_ablation.png")
    fig.savefig(out_dir / "fig3_ablation.pdf")
    plt.close(fig)
    print("  fig3_ablation saved")


# ═══════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_vram()
    fig2_kivi_comparison()
    fig3_ablation()
    print(f"All figures saved to {out_dir}/")
