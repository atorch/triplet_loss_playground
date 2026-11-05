"""Summarize results/*/metrics.json into a markdown table and a bar chart.

    uv run src/compare.py
"""

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data

RESULTS_DIR = data.REPO_ROOT / "results"


def condition_name(config):
    parts = []
    if config["augment"]:
        parts.append("augment")
    if config["outlier_exposure"]:
        parts.append("outlier exposure")
    return " + ".join(parts) or "baseline"


def main():
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*/metrics.json"))]
    order = ["baseline", "augment", "outlier exposure", "augment + outlier exposure"]

    # condition -> OOD-set kind -> list of AUROCs (one per held-out digit / seed)
    aurocs = defaultdict(lambda: defaultdict(list))
    runs.sort(key=lambda r: (r["config"]["holdout_digit"], order.index(condition_name(r["config"])), r["config"]["seed"]))
    rows = []
    for run in runs:
        cfg = run["config"]
        cond = condition_name(cfg)
        aurocs[cond]["kNN accuracy"].append(run["knn_accuracy_in_dist"])
        ood = run["ood"]
        held_out = next((v for k, v in ood.items() if k.startswith("held-out")), None)
        if held_out:
            aurocs[cond]["held-out digit"].append(held_out["auroc"])
        aurocs[cond]["synthetic Xs"].append(ood["synthetic Xs"]["auroc"])
        aurocs[cond]["Fashion-MNIST"].append(ood["Fashion-MNIST"]["auroc"])
        xs = run["hand_drawn"]["my hand-drawn Xs"]["percentile_vs_mnist_test"]
        eights = run["hand_drawn"]["my hand-drawn 8s"]["percentile_vs_mnist_test"]
        aurocs[cond]["my Xs: percentile"].append(np.mean(xs))
        aurocs[cond]["my 8s: percentile"].append(np.mean(eights))
        rows.append(
            f"| {cfg['holdout_digit']} | {cond} | {cfg['seed']} | {run['knn_accuracy_in_dist']:.3f} "
            f"| {held_out['auroc'] if held_out else float('nan'):.3f} "
            f"| {ood['synthetic Xs']['auroc']:.3f} | {ood['Fashion-MNIST']['auroc']:.3f} "
            f"| {', '.join(f'{p:.0f}' for p in xs)} | {', '.join(f'{p:.0f}' for p in eights)} |"
        )

    columns = ["kNN accuracy", "held-out digit", "synthetic Xs", "Fashion-MNIST", "my Xs: percentile", "my 8s: percentile"]
    mean_rows = []
    for cond in [c for c in order if c in aurocs]:
        cells = []
        for col in columns:
            vals = np.array(aurocs[cond][col])
            fmt = ".0f" if "percentile" in col else ".3f"
            cells.append(f"{vals.mean():{fmt}} ± {vals.std(ddof=1):{fmt}}" if len(vals) > 1 else f"{vals.mean():{fmt}}")
        mean_rows.append(f"| {cond} | {len(aurocs[cond]['synthetic Xs'])} | " + " | ".join(cells) + " |")
    means_table = [
        "Mean ± sd over runs (all held-out digits and seeds). AUROC columns unless noted; "
        "percentiles are averaged over the three hand-drawn images.",
        "",
        "| condition | runs | kNN acc (in-dist) | AUROC held-out digit | AUROC synthetic Xs | AUROC Fashion-MNIST "
        "| my Xs: percentile | my 8s: percentile |",
        "|---|---|---|---|---|---|---|---|",
        *mean_rows,
        "",
        "All runs:",
        "",
    ]

    table = [
        *means_table,
        "| held-out digit | condition | seed | kNN acc (in-dist) | AUROC held-out digit | AUROC synthetic Xs "
        "| AUROC Fashion-MNIST | my Xs: percentile | my 8s: percentile |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
    ]
    summary = "\n".join(table)
    (RESULTS_DIR / "summary.md").write_text(summary + "\n")
    print(summary)

    conditions = [c for c in order if c in aurocs]
    kinds = ["held-out digit", "synthetic Xs", "Fashion-MNIST"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(conditions)
    colors = ["#8C8C8C", "#4C72B0", "#DD8452", "#55A868"]
    for i, cond in enumerate(conditions):
        means = [np.mean(aurocs[cond][k]) if aurocs[cond][k] else np.nan for k in kinds]
        x = np.arange(len(kinds)) + (i - (len(conditions) - 1) / 2) * width
        ax.bar(x, means, width, label=cond, color=colors[i % len(colors)])
        for k_idx, k in enumerate(kinds):  # individual runs as dots
            vals = aurocs[cond][k]
            ax.scatter([x[k_idx]] * len(vals), vals, color="black", s=8, zorder=3)
    ax.set_xticks(np.arange(len(kinds)), kinds)
    ax.set_ylabel("AUROC (kNN distance score)")
    ax.set_ylim(0.9, 1.0)  # zoomed in: every run is far above chance (0.5)
    ax.set_title("Do OOD inputs land farther from the training digits? (bars: mean, dots: runs)", fontsize=9)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(conditions), frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "auroc_summary.png", dpi=120)
    print(f"Wrote {RESULTS_DIR / 'summary.md'} and {RESULTS_DIR / 'auroc_summary.png'}")


if __name__ == "__main__":
    main()
