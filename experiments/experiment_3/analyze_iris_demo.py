from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import PROJECT_ROOT


RUN_DIR = PROJECT_ROOT / "runs" / "iris-uci"
THRESHOLD = 0.5


def main() -> None:
    with (RUN_DIR / "history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    evaluations = [
        row
        for row in history
        if row.get("evaluated", "").lower() == "true" and row.get("silhouette") not in (None, "")
    ]
    if not evaluations:
        raise SystemExit(f"No recorded silhouette evaluations found in {RUN_DIR / 'history.csv'}")

    epochs = np.asarray([int(row["epoch"]) for row in evaluations])
    scores = np.asarray([float(row["silhouette"]) for row in evaluations])
    triggered = np.asarray([row.get("gate_triggered", "").lower() == "true" for row in evaluations])
    adjustment_by_epoch = {
        int(row["epoch"]): row.get("adjustment_applied", "").lower() == "true"
        for row in history
    }
    adjustments_applied = sum(adjustment_by_epoch.values())
    adjustments_next = np.asarray([adjustment_by_epoch.get(int(epoch) + 1, False) for epoch in epochs])
    first_trigger = int(np.flatnonzero(triggered)[0]) if triggered.any() else None

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.6, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
    )
    score_axis = axes[0]
    score_axis.plot(epochs, scores, color="#2864a5", linewidth=1.8, zorder=1)
    score_axis.scatter(epochs[~triggered], scores[~triggered], s=38, color="#777777", zorder=3, label="Gate closed")
    score_axis.scatter(epochs[triggered], scores[triggered], s=42, color="#d94841", zorder=3, label="Gate activated")
    score_axis.axhline(THRESHOLD, color="#222222", linestyle="--", linewidth=1.1, label=f"Threshold = {THRESHOLD:g}")
    if first_trigger is not None:
        score_axis.annotate(
            f"first activation: epoch {epochs[first_trigger]}\nscore {scores[first_trigger]:.3f}",
            xy=(epochs[first_trigger], scores[first_trigger]),
            xytext=(epochs[first_trigger] + 18, scores[first_trigger] - 0.18),
            arrowprops={"arrowstyle": "->", "color": "#444444"},
            fontsize=9,
        )
    score_axis.annotate(f"final: {scores[-1]:.3f}", xy=(epochs[-1], scores[-1]), xytext=(-42, 12), textcoords="offset points", fontsize=9)
    score_axis.set(
        ylabel="Reference silhouette",
        ylim=(-1, 1),
        title="Saved Iris UCI demo run: silhouette scores and gate decisions",
    )
    score_axis.grid(alpha=0.25)
    score_axis.legend(frameon=False, ncol=3, fontsize=8, loc="lower right")

    gate_axis = axes[1]
    gate_axis.step(epochs, triggered.astype(int), where="mid", color="#d94841", linewidth=1.8)
    gate_axis.scatter(epochs, triggered.astype(int), s=24, color=np.where(triggered, "#d94841", "#777777"), zorder=3)
    gate_axis.set_yticks([0, 1], ["closed", "open"])
    gate_axis.set_ylim(-0.35, 1.35)
    gate_axis.set(xlabel="Training epoch", ylabel="Gate")
    gate_axis.grid(axis="x", alpha=0.2)
    fig.savefig(RUN_DIR / "silhouette_gate_activation.png", dpi=300)
    plt.close(fig)

    with (RUN_DIR / "silhouette_evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "reference_silhouette", "threshold", "gate_triggered", "adjustment_applied_next_epoch"))
        writer.writerows(
            (
                int(row["epoch"]),
                float(row["silhouette"]),
                THRESHOLD,
                row.get("gate_triggered", "").lower() == "true",
                bool(next_adjustment),
            )
            for row, next_adjustment in zip(evaluations, adjustments_next, strict=True)
        )

    with (RUN_DIR / "metrics.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    peak_index = int(np.argmax(scores))
    report = [
        "# Existing Iris UCI demo run: silhouette and gate activation",
        "",
        "This graph uses only the saved `runs/iris-uci/history.csv` and `metrics.json`; no model was trained or rerun.",
        "",
        f"- {len(evaluations)} scheduled silhouette evaluations.",
        f"- Gate activated at {int(triggered.sum())}/{len(evaluations)} evaluations; {adjustments_applied} adjustments were applied in subsequent training epochs.",
        "- First gate activation: not recorded." if first_trigger is None else f"- First gate activation: epoch {int(epochs[first_trigger])}, reference silhouette {scores[first_trigger]:.4f}.",
        f"- Peak recorded silhouette: {scores[peak_index]:.4f} at epoch {int(epochs[peak_index])}; final score: {scores[-1]:.4f}.",
        f"- Final external metrics: ACC {summary['external_metrics']['ACC']:.3f}, NMI {summary['external_metrics']['NMI']:.3f}, ARI {summary['external_metrics']['ARI']:.3f}.",
        "",
        "This demo run saved only the reference silhouette series (`silhouette`). It did not save sampled or simplified silhouette histories, so this run alone cannot show those other two trajectories.",
        "",
    ]
    (RUN_DIR / "silhouette_gate_analysis.md").write_text("\n".join(report), encoding="utf-8")
    print(
        f"Saved analysis from existing run: evaluations={len(evaluations)}, "
        f"gate_triggers={int(triggered.sum())}, adjustments_applied={adjustments_applied}, "
        f"final_silhouette={scores[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
