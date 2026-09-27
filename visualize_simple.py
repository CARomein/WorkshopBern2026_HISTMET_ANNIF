"""
visualize_simple.py — collect whichever Annif eval result files exist (fold 0
only, no original/optimal split) and plot backend performance, 2020 vs 2024.

Usage (e.g. in Colab):
    from visualize_simple import vis_results
    vis_results("micro")   # or vis_results("macro")

Expects result files written by:
    annif eval ... --results-file results/trans_<year>/trans_<year>_<level>_<model>_fold0.json
e.g. results/trans_2020/trans_2020_level2_omikuji_parabel_fold0.json

Missing models/levels are simply skipped — nothing is flagged, nothing is saved.
"""
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR = "results"
LEVELS = ["level2", "level3", "level4"]
LEVEL_SHADE = {"level2": 0.55, "level3": 0.27, "level4": 0.0}

DISPLAY = {
    "tfidf": "TF-IDF", "mllm": "MLLM", "yake": "YAKE", "svc": "SVC",
    "omikuji_parabel": "Parabel", "omikuji_bonsai": "Bonsai",
    "nn_ensemble": "NN Ensemble",
}
MODEL_COLORS = {
    "tfidf": "#4878A8", "svc": "#E0697A", "omikuji_parabel": "#46A24C",
    "omikuji_bonsai": "#5FC4E8", "nn_ensemble": "#B0468C",
    "mllm": "#9E9E9E", "yake": "#BDBDBD",
}


def _shade(hex_color, f):
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(c + (1.0 - c) * f)


def _f1_from_file(path, metric):
    """metric='micro' -> aggregate TP/FP/FN across labels, derive F1.
    metric='macro' -> mean of per-label F1 values."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)
    if not rows:
        return np.nan

    if metric == "micro":
        tp = sum(int(r["True_positives"]) for r in rows)
        fp = sum(int(r["False_positives"]) for r in rows)
        fn = sum(int(r["False_negatives"]) for r in rows)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r_ / (p + r_) if (p + r_) else 0.0
    elif metric == "macro":
        return sum(float(r["F1_score"]) for r in rows) / len(rows)
    raise ValueError("metric must be 'micro' or 'macro'")


def _collect(year, metric):
    """Return {level: {model: F1}} for whichever result files exist."""
    scores = {lv: {} for lv in LEVELS}
    folder = os.path.join(RESULTS_DIR, f"trans_{year}")
    for path in glob.glob(os.path.join(folder, "*.json")):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        try:
            li = next(i for i, p in enumerate(parts) if p.startswith("level"))
        except StopIteration:
            continue
        level = parts[li]
        model = "_".join(parts[li + 1:-1])  # drop trailing "fold0"
        if level in scores:
            scores[level][model] = _f1_from_file(path, metric)
    return scores


def vis_results(metric):
    """Plot backend performance, 2020 vs 2024, side by side.
    metric: 'micro' or 'macro'."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    bar_w = 0.8 / 3

    for ax, year in zip(axes, ("2020", "2024")):
        scores = _collect(year, metric)
        models = sorted({m for lv in scores.values() for m in lv})
        x = np.arange(len(models))

        for k, lv in enumerate(LEVELS):
            vals = [scores[lv].get(m, np.nan) for m in models]
            colors = [_shade(MODEL_COLORS.get(m, "#888888"), LEVEL_SHADE[lv]) for m in models]
            offset = (k - 1) * bar_w
            ax.bar(x + offset,
                   [0 if np.isnan(v) else v for v in vals],
                   bar_w * 0.92, color=colors, edgecolor="#777777", linewidth=0.4)
            for xi, v in zip(x, vals):
                if not np.isnan(v):
                    label = "0" if v == 0 else f"{v:.3f}".lstrip("0")
                    ax.text(xi + offset, v + 0.008, label,
                            ha="center", va="bottom", fontsize=9, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY.get(m, m) for m in models], rotation=20, ha="right")
        ax.set_title(year, fontweight="bold", fontsize=13)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#eeeeee")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel(f"F1 score ({metric}-avg)")
    handles = [Patch(facecolor=_shade("#9E5BB0", LEVEL_SHADE[lv]), label=lv) for lv in LEVELS]
    axes[1].legend(handles=handles, loc="upper right", frameon=True)
    fig.suptitle(f"Backend performance: 2020 vs 2024 ({metric}-avg)", fontweight="bold", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
