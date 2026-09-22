#!/usr/bin/env python3
"""Regenerate the publication-quality benchmark summary from repository measurements."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "benchmark-summary"
PALETTE = {
    "blue": "#0F4D92", "blue2": "#3775BA", "green": "#8BCF8B",
    "green2": "#AADCA9", "red": "#B64342", "pink": "#E9A6A1",
    "gray": "#767676", "light": "#CFCECE", "dark": "#272727",
}
plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12,
    "axes.linewidth": 1.8, "axes.spines.right": False,
    "axes.spines.top": False, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

def label_bars(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h), (bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=10, fontweight="bold")

def finish(fig):
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

data = json.loads((ROOT / "bench_results.json").read_text())
rows = data["results"]
c = [x["concurrency"] for x in rows]
agg = [x["aggregate_tok_per_s"] for x in rows]
stream = [x["avg_tok_per_s_per_stream"] for x in rows]
ttft = [x["avg_ttft_s"] for x in rows]

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
fig.suptitle("Qwen3.6 35B-A3B NVFP4 — measured on one DGX Spark", fontsize=18, fontweight="bold")
colors = [PALETTE["light"], PALETTE["green"], PALETTE["blue"]]
bars = axes[0].bar(c, agg, color=colors, edgecolor="black", linewidth=1.2)
label_bars(axes[0], bars)
axes[0].set(title="Aggregate throughput", xlabel="Concurrent requests", ylabel="tokens/s")
axes[0].set_xticks(c)
bars = axes[1].bar(c, stream, color=colors, edgecolor="black", linewidth=1.2)
label_bars(axes[1], bars)
axes[1].set(title="Average per stream", xlabel="Concurrent requests", ylabel="tokens/s/stream")
axes[1].set_xticks(c)
bars = axes[2].bar(c, ttft, color=colors, edgecolor="black", linewidth=1.2)
label_bars(axes[2], bars, "{:.2f}")
axes[2].set(title="Time to first token", xlabel="Concurrent requests", ylabel="seconds")
axes[2].set_xticks(c)
for ax in axes: ax.grid(axis="y", alpha=.18, linewidth=.8)
fig.text(.5, .005, "Source: bench_results.json · successful requests only", ha="center", color=PALETTE["gray"], fontsize=9)
finish(fig)
