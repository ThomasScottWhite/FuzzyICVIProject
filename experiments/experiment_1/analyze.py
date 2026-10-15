from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "runs" / "preliminary" / "experiment_1"
SCHEDULE_ORDER = (
    "every_1",
    "every_5",
    "every_10",
    "every_20",
    "front_loaded",
    "back_loaded",
)
FINAL_METRICS = ("final_silhouette", "ACC", "NMI", "ARI")
PAIRED_METRICS = (*FINAL_METRICS, "wall_seconds", "compute_seconds", "evaluations")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize completed Experiment 1 runs and save CSV/JSON tables and plots. "
            "Incomplete or missing runs are skipped."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Experiment run directory.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Analysis output directory (default: <input>/analysis).",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Raster plot resolution (default: 180).")
    return parser


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for name, value in row.items():
                if name == "evaluated":
                    parsed[name] = str(value).strip().lower() in {"true", "1", "yes"}
                elif name in {"gate_triggered", "adjustment_applied"}:
                    parsed[name] = str(value).strip().lower() in {"true", "1", "yes"}
                elif name == "epoch":
                    parsed[name] = (
                        int(float(value)) if value not in (None, "") else None
                    )
                else:
                    parsed[name] = _number(value)
            rows.append(parsed)
        return rows


def _load_runs(input_dir: Path) -> tuple[list[dict[str, Any]], int, list[str]]:
    runs: list[dict[str, Any]] = []
    incomplete = 0
    problems: list[str] = []
    for summary_path in sorted(input_dir.rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "complete":
                incomplete += 1
                continue
            run_dir = summary_path.parent
            history_path = run_dir / summary.get("history_file", "history.csv")
            if not history_path.is_file():
                incomplete += 1
                problems.append(f"{summary_path}: history CSV is missing")
                continue
            metadata = summary.get("experiment_metadata") or {}
            schedule = metadata.get("schedule") or run_dir.parent.name
            dataset = summary.get("dataset") or run_dir.parent.parent.name
            history = _read_history(history_path)
            evaluations = [row for row in history if row.get("evaluated")]
            timings = summary.get("timing_seconds") or {}
            external = summary.get("external_metrics") or {}
            runs.append(
                {
                    "dataset": str(dataset),
                    "schedule": str(schedule),
                    "seed": int(summary["seed"]),
                    "run_key": summary.get("run_key", str(run_dir)),
                    "n_samples": summary.get("n_samples"),
                    "n_features": summary.get("n_features"),
                    "n_clusters": summary.get("n_clusters"),
                    "device": summary.get("device"),
                    "evaluations": len(evaluations),
                    "final_silhouette": _number(summary.get("final_silhouette")),
                    "ACC": _number(external.get("ACC")),
                    "NMI": _number(external.get("NMI")),
                    "ARI": _number(external.get("ARI")),
                    "gate_triggers": _number(summary.get("gate_triggers")),
                    "adjustments_applied": _number(summary.get("adjustments_applied")),
                    "wall_seconds": _number(timings.get("wall")),
                    "compute_seconds": _number(timings.get("compute")),
                    "autoencoder_seconds": _number(timings.get("autoencoder_training")),
                    "fcm_seconds": _number(timings.get("fcm")),
                    "validation_seconds": _number(timings.get("validation")),
                    "io_seconds": _number(timings.get("io")),
                    "history": history,
                    "summary_path": str(summary_path),
                }
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            incomplete += 1
            problems.append(f"{summary_path}: {exc}")
    return runs, incomplete, problems


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if not array.size:
        return None, None
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else None


def _sort_key(run: dict[str, Any]) -> tuple[Any, ...]:
    try:
        schedule_order = SCHEDULE_ORDER.index(run["schedule"])
    except ValueError:
        schedule_order = len(SCHEDULE_ORDER)
    return run["dataset"], schedule_order, run["schedule"], run["seed"]


def _display_dataset(dataset: str) -> str:
    return {"covertype": "Covertype", "fashion_mnist": "Fashion-MNIST"}.get(
        dataset, dataset.replace("_", " ").title()
    )


def _display_schedule(schedule: str) -> str:
    return schedule.replace("_", " ").title()


def _make_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["dataset"], run["schedule"])].append(run)

    rows: list[dict[str, Any]] = []
    for (dataset, schedule), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            SCHEDULE_ORDER.index(item[0][1]) if item[0][1] in SCHEDULE_ORDER else len(SCHEDULE_ORDER),
            item[0][1],
        ),
    ):
        row: dict[str, Any] = {
            "dataset": dataset,
            "schedule": schedule,
            "n_runs": len(group),
            "seeds": ",".join(str(run["seed"]) for run in sorted(group, key=lambda run: run["seed"])),
        }
        for metric in (
            *FINAL_METRICS,
            "evaluations",
            "gate_triggers",
            "adjustments_applied",
            "wall_seconds",
            "compute_seconds",
            "autoencoder_seconds",
            "fcm_seconds",
            "validation_seconds",
            "io_seconds",
        ):
            mean, std = _mean_std([run.get(metric) for run in group])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return rows


def _make_paired_deltas(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(run["dataset"], run["schedule"], run["seed"]): run for run in runs}
    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=_sort_key):
        baseline = by_key.get((run["dataset"], "every_10", run["seed"]))
        if baseline is None:
            continue
        for metric in PAIRED_METRICS:
            candidate = run.get(metric)
            reference = baseline.get(metric)
            if candidate is None or reference is None:
                continue
            rows.append(
                {
                    "dataset": run["dataset"],
                    "schedule": run["schedule"],
                    "seed": run["seed"],
                    "baseline_schedule": "every_10",
                    "metric": metric,
                    "run_value": candidate,
                    "baseline_value": reference,
                    "delta_vs_every_10": candidate - reference,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _paired_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["schedule"], row["metric"])].append(
            float(row["delta_vs_every_10"])
        )
    result = []
    for (dataset, schedule, metric), values in sorted(grouped.items()):
        mean, std = _mean_std(values)
        result.append(
            {
                "dataset": dataset,
                "schedule": schedule,
                "metric": metric,
                "n_pairs": len(values),
                "delta_mean": mean,
                "delta_std": std,
            }
        )
    return result


def _plot_final_metrics(
    rows: list[dict[str, Any]], runs: list[dict[str, Any]], figure_dir: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    metrics = (
        ("final_silhouette", "Final reference silhouette", (-1, 1)),
        ("ACC", "Clustering accuracy (ACC)", (0, 1)),
        ("NMI", "Normalized mutual information (NMI)", (0, 1)),
        ("ARI", "Adjusted Rand index (ARI)", (0, 1)),
    )
    datasets = sorted({row["dataset"] for row in rows})
    colors = {dataset: plt.get_cmap("tab10")(index) for index, dataset in enumerate(datasets)}
    configs = sorted(
        rows,
        key=lambda row: (
            row["dataset"],
            SCHEDULE_ORDER.index(row["schedule"]) if row["schedule"] in SCHEDULE_ORDER else len(SCHEDULE_ORDER),
            row["schedule"],
        ),
    )
    labels = [
        f"{_display_dataset(row['dataset'])}\n{_display_schedule(row['schedule'])}"
        for row in configs
    ]
    x = np.arange(len(configs))
    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(configs) * 0.85), 8))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.26, top=0.87, wspace=0.26, hspace=0.52)
    for axis, (metric, title, limits) in zip(axes.flat, metrics):
        means = [row.get(f"{metric}_mean") for row in configs]
        stds = [row.get(f"{metric}_std") or 0 for row in configs]
        axis.bar(
            x,
            [value if value is not None else 0 for value in means],
            color=[colors[row["dataset"]] for row in configs],
            alpha=0.78,
            edgecolor="black",
            linewidth=0.45,
        )
        axis.errorbar(
            x,
            [np.nan if value is None else value for value in means],
            yerr=stds,
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
        for position, row in enumerate(configs):
            values = [
                run.get(metric)
                for run in runs
                if run["dataset"] == row["dataset"] and run["schedule"] == row["schedule"]
            ]
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0]
            axis.scatter(
                [position + offset for offset, value in zip(jitter, values) if value is not None],
                [value for value in values if value is not None],
                s=16,
                color="black",
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Experiment 1: final results (bars show means; dots are individual runs)", y=0.98)
    fig.savefig(figure_dir / "final_metrics.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_runtime(rows: list[dict[str, Any]], figure_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    datasets = sorted({row["dataset"] for row in rows})
    colors = {dataset: plt.get_cmap("tab10")(index) for index, dataset in enumerate(datasets)}
    configs = sorted(
        rows,
        key=lambda row: (
            row["dataset"],
            SCHEDULE_ORDER.index(row["schedule"]) if row["schedule"] in SCHEDULE_ORDER else len(SCHEDULE_ORDER),
            row["schedule"],
        ),
    )
    labels = [
        f"{_display_dataset(row['dataset'])}\n{_display_schedule(row['schedule'])}"
        for row in configs
    ]
    x = np.arange(len(configs))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(configs) * 0.8), 4.8))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.30, top=0.84, wspace=0.28)
    for axis, metric, title, divisor in (
        (axes[0], "wall_seconds", "Total wall time (hours)", 3600),
        (axes[1], "validation_seconds", "Validation time (minutes)", 60),
    ):
        means = [row.get(f"{metric}_mean") for row in configs]
        stds = [row.get(f"{metric}_std") or 0 for row in configs]
        axis.bar(
            x,
            [value / divisor if value is not None else 0 for value in means],
            color=[colors[row["dataset"]] for row in configs],
            alpha=0.78,
            edgecolor="black",
            linewidth=0.45,
        )
        axis.errorbar(
            x,
            [value / divisor if value is not None else 0 for value in means],
            yerr=[value / divisor for value in stds],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Experiment 1: runtime by dataset and schedule", y=0.98)
    fig.savefig(figure_dir / "runtime.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_silhouette_history(runs: list[dict[str, Any]], figure_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    datasets = sorted({run["dataset"] for run in runs})
    fig, axes = plt.subplots(
        len(datasets), 1, figsize=(10, max(4, 3.8 * len(datasets))), squeeze=False, constrained_layout=True
    )
    cmap = plt.get_cmap("tab10")
    schedules_present = sorted(
        {run["schedule"] for run in runs},
        key=lambda name: SCHEDULE_ORDER.index(name) if name in SCHEDULE_ORDER else len(SCHEDULE_ORDER),
    )
    schedule_colors = {schedule: cmap(index % 10) for index, schedule in enumerate(schedules_present)}
    for axis, dataset in zip(axes[:, 0], datasets):
        for schedule in schedules_present:
            group = [run for run in runs if run["dataset"] == dataset and run["schedule"] == schedule]
            by_epoch: dict[int, list[float]] = defaultdict(list)
            for run in group:
                for item in run["history"]:
                    if not item.get("evaluated"):
                        continue
                    score = item.get("silhouette_reference")
                    if score is None:
                        score = item.get("silhouette")
                    epoch = item.get("epoch")
                    if score is not None and epoch is not None:
                        by_epoch[int(epoch)].append(float(score))
            if not by_epoch:
                continue
            epochs = np.asarray(sorted(by_epoch), dtype=int)
            means = np.asarray([np.mean(by_epoch[epoch]) for epoch in epochs])
            stds = np.asarray(
                [np.std(by_epoch[epoch], ddof=1) if len(by_epoch[epoch]) > 1 else 0 for epoch in epochs]
            )
            color = schedule_colors[schedule]
            axis.plot(
                epochs,
                means,
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                label=_display_schedule(schedule),
                color=color,
            )
            if np.any(stds > 0):
                axis.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)
        axis.set_title(_display_dataset(dataset))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Reference silhouette")
        axis.set_ylim(-1, 1)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", frameon=False)
    fig.suptitle("Experiment 1: reference silhouette at scheduled evaluations", y=1.02)
    fig.savefig(figure_dir / "silhouette_over_epochs.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_deltas(rows: list[dict[str, Any]], figure_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    metrics = FINAL_METRICS
    datasets = sorted({row["dataset"] for row in rows})
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["metric"] in metrics:
            grouped[(row["dataset"], row["schedule"], row["metric"])].append(
                float(row["delta_vs_every_10"])
            )
    schedules_present = sorted(
        {row["schedule"] for row in rows if row["schedule"] != "every_10"},
        key=lambda name: SCHEDULE_ORDER.index(name) if name in SCHEDULE_ORDER else len(SCHEDULE_ORDER),
    )
    if not schedules_present:
        return
    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(schedules_present) * 1.5), 7.5), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics):
        offset = 0
        all_ticks: list[str] = []
        all_x: list[int] = []
        for dataset in datasets:
            selected_schedules = [
                schedule
                for schedule in schedules_present
                if grouped.get((dataset, schedule, metric))
            ]
            positions = np.arange(offset, offset + len(selected_schedules))
            means = [np.mean(grouped[(dataset, schedule, metric)]) for schedule in selected_schedules]
            stds = [
                np.std(grouped[(dataset, schedule, metric)], ddof=1)
                if len(grouped[(dataset, schedule, metric)]) > 1
                else 0
                for schedule in selected_schedules
            ]
            if len(positions):
                axis.errorbar(
                    positions,
                    means,
                    yerr=stds,
                    fmt="o",
                    capsize=3,
                    label=dataset,
                    color=plt.get_cmap("tab10")(datasets.index(dataset)),
                )
                all_x.extend(positions.tolist())
                all_ticks.extend(
                    f"{_display_dataset(dataset)}\n{_display_schedule(name)}"
                    for name in selected_schedules
                )
            offset += len(selected_schedules) + 1
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.7)
        metric_title = {
            "final_silhouette": "final silhouette",
            "ACC": "ACC",
            "NMI": "NMI",
            "ARI": "ARI",
        }[metric]
        axis.set_title(f"Paired delta in {metric_title} vs Every 10")
        axis.set_xticks(all_x, all_ticks, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Experiment 1: paired differences from every_10", y=1.03)
    fig.savefig(figure_dir / "paired_deltas_vs_every_10.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path,
    runs: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    incomplete_count: int,
    problems: list[str],
) -> None:
    lines = [
        "# Experiment 1 preliminary analysis",
        "",
        f"Completed runs included: **{len(runs)}**. Incomplete or unreadable runs skipped: **{incomplete_count}**.",
        "",
        "All values below are descriptive summaries of completed runs. Schedule comparisons use paired seeds "
        "against `every_10` when that baseline is available.",
        "",
    ]
    if not runs:
        lines.extend(["No completed runs with history files were found.", ""])
    else:
        lines.extend(
            [
                "| Dataset | Schedule | Runs | Seeds | Final silhouette | ACC | NMI | ARI | Wall hours |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows:
            def fmt(metric: str, digits: int = 3) -> str:
                value = row.get(f"{metric}_mean")
                spread = row.get(f"{metric}_std")
                if value is None:
                    return "—"
                if spread is None:
                    return f"{value:.{digits}f} (n=1)"
                return f"{value:.{digits}f} ± {spread:.{digits}f}"

            wall_hours = row.get("wall_seconds_mean")
            wall = "—" if wall_hours is None else f"{wall_hours / 3600:.2f}"
            lines.append(
                f"| {row['dataset']} | {row['schedule']} | {row['n_runs']} | {row['seeds']} | "
                f"{fmt('final_silhouette')} | {fmt('ACC')} | {fmt('NMI')} | {fmt('ARI')} | {wall} |"
            )
        datasets = sorted({run["dataset"] for run in runs})
        missing_baselines = [
            dataset
            for dataset in datasets
            if not any(run["dataset"] == dataset and run["schedule"] == "every_10" for run in runs)
        ]
        lines.append("")
        if missing_baselines:
            lines.append(
                "Paired schedule comparisons are not yet available for: "
                + ", ".join(missing_baselines)
                + ". Run matching-seed `every_10` configurations to enable those comparisons."
            )
            lines.append("")
        else:
            lines.append(
                "The `every_10` reference schedule is present for every dataset, so paired delta files were generated."
            )
            lines.append("")
        if any(row["n_runs"] == 1 for row in summary_rows):
            lines.extend(
                [
                    "At least one dataset/schedule has one completed seed, so its run-to-run variability cannot be estimated yet.",
                    "",
                ]
            )
    if problems:
        lines.extend(["## Files skipped with issues", ""])
        lines.extend(f"- `{problem}`" for problem in problems)
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            "- `analysis_runs.csv`: one row per completed run.",
            "- `analysis_summary.csv` and `.json`: mean and standard deviation by dataset and schedule.",
            "- `paired_deltas.csv`: per-seed differences from `every_10` where matching runs exist.",
            "- `paired_delta_summary.csv`: aggregate paired differences.",
            "- `figures/`: final metrics, runtime, evaluation trajectory, and paired comparison plots when available.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = (args.output or (input_dir / "analysis")).expanduser().resolve()
    if args.dpi < 50:
        raise SystemExit("--dpi must be at least 50")
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required to create the plots. Install project dependencies with `uv sync`."
        ) from exc

    runs, incomplete_count, problems = _load_runs(input_dir)
    runs.sort(key=_sort_key)
    plot_runs = runs
    flat_runs = [{key: value for key, value in run.items() if key != "history"} for run in runs]

    summary_rows = _make_summary(runs)
    paired_rows = _make_paired_deltas(runs)
    paired_summary = _paired_summary(paired_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    run_fields = [
        "dataset", "schedule", "seed", "run_key", "n_samples", "n_features", "n_clusters", "device",
        "evaluations", *FINAL_METRICS, "gate_triggers", "adjustments_applied", "wall_seconds",
        "compute_seconds", "autoencoder_seconds", "fcm_seconds", "validation_seconds", "io_seconds", "summary_path",
    ]
    _write_csv(output_dir / "analysis_runs.csv", flat_runs, run_fields)
    _write_csv(output_dir / "analysis_summary.csv", summary_rows)
    _write_csv(output_dir / "paired_deltas.csv", paired_rows, [
        "dataset", "schedule", "seed", "baseline_schedule", "metric", "run_value", "baseline_value", "delta_vs_every_10"
    ])
    _write_csv(output_dir / "paired_delta_summary.csv", paired_summary, [
        "dataset", "schedule", "metric", "n_pairs", "delta_mean", "delta_std"
    ])
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "completed_run_count": len(runs),
                "incomplete_or_unreadable_count": incomplete_count,
                "schedule_summary": summary_rows,
                "paired_delta_summary": paired_summary,
                "problems": problems,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if summary_rows:
        _plot_final_metrics(summary_rows, plot_runs, figure_dir, args.dpi)
        _plot_runtime(summary_rows, figure_dir, args.dpi)
        _plot_silhouette_history(plot_runs, figure_dir, args.dpi)
        _plot_paired_deltas(paired_rows, figure_dir, args.dpi)
    _write_report(output_dir / "analysis_report.md", runs, summary_rows, incomplete_count, problems)
    print(f"Analyzed {len(runs)} completed Experiment 1 runs; skipped {incomplete_count} incomplete/unreadable runs.")
    print(f"Analysis files saved to {output_dir}")
    if not runs:
        print("No plots were produced because there are no completed runs with histories yet.")


if __name__ == "__main__":
    main()
