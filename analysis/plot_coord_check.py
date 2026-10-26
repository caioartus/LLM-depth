"""
Plots the coordinate check (jobs/depth/coord_check.sbatch). Run after analysis/collect_results.py,
which copies the runs to results/coord_check/:

    uv run --frozen python analysis/plot_coord_check.py

Writes results/coord_check/depth.png (width 256, L = 2..64) and width.png (L = 2, d = 256..1024).
Each panel: mean |output| of one layer type vs depth (or width), one line per training step,
averaged over seeds. Pass: the CompleteP lines are roughly flat (no systematic growth or decay
with L or d) at every step. SP is expected to drift with depth.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DIR = Path("results/coord_check")
LAYERS = [("token_embedding", "token embedding"), ("attn", "attention branch"), ("mlp", "MLP branch"),
          ("last_layer", "last block (residual stream)"), ("lm_head", "lm_head (logits)")]

# values[(param, d, L)][layer][step] -> list over seeds
values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for path in sorted(DIR.glob("*.csv")):
    param, shape, _seed = path.stem.split("-")
    d, L = map(int, shape.split("x"))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key, _ in LAYERS:
                if row.get(f"{key}_act_abs_mean"):  # empty at iter 0 (nothing trained yet)
                    values[(param, d, L)][key][int(row["iter"])].append(float(row[f"{key}_act_abs_mean"]))
if not values:
    raise SystemExit(f"no coordinate-check runs in {DIR}, run analysis/collect_results.py first")


def plot_sweep(filename, x_name, shapes):
    params = ["sp", "completep"]
    shapes = [s for s in shapes if any((p, *s) in values for p in params)]  # only shapes that were run
    if len(shapes) < 2:
        print(f"skipping {filename}: fewer than 2 of its shapes have results")
        return
    fig, axes = plt.subplots(len(params), len(LAYERS), figsize=(3.2 * len(LAYERS), 3.0 * len(params)),
                             sharex=True, squeeze=False)
    steps = sorted({s for v in values.values() for layer in v.values() for s in layer})
    colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(steps)))  # one hue, light (early) -> dark (late)
    for i, param in enumerate(params):
        for j, (key, title) in enumerate(LAYERS):
            ax = axes[i, j]
            for step, color in zip(steps, colors):
                xs, ys = [], []
                for d, L in shapes:
                    seeds = values.get((param, d, L), {}).get(key, {}).get(step)
                    if seeds:
                        xs.append(d if x_name == "width d" else L)
                        ys.append(np.mean(seeds))
                ax.plot(xs, ys, color=color, linewidth=1.5, marker="o", markersize=3, label=f"step {step}")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="major", color="0.9", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{'CompleteP' if param == 'completep' else 'SP'}\nmean |output|")
            if i == len(params) - 1:
                ax.set_xlabel(x_name)
    axes[0, -1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"Coordinate check vs {x_name} (mean over seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(DIR / filename, dpi=150, bbox_inches="tight")
    print(f"wrote {DIR / filename}")


plot_sweep("depth.png", "depth L", [(256, L) for L in [2, 4, 8, 16, 32, 64]])
plot_sweep("width.png", "width d", [(d, 2) for d in [256, 512, 1024]])
