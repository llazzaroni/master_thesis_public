#!/usr/bin/env python3
"""Plot the linear baseline and one closed-form penalty's target-test AUROC.

With no task or target filters, also produce averages by task, by target, and
over every available task-target pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
)
CORRUPTIONS = (
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter",
    "speckle_noise", "zoom_blur",
)
DEFAULT_DATA_ROOT = Path("results")


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


METRICS = {
    "auroc": ("target_test_auroc", "Target-test AUROC", True),
    "mse": ("target_test_mse", "Target-test MSE", False),
}


def set_budget_ticks(
    axis: object, budgets: list[float] | tuple[float, ...], labels: list[str] | None = None,
) -> None:
    """Show the actual effective target counts, including the resolved -1 budget."""
    axis.set_xticks(budgets)
    axis.set_xticklabels(labels or [str(budget) for budget in budgets])
    axis.minorticks_off()


def read_metric(path: Path, metric_key: str) -> float:
    payload = read_json(path)
    try:
        value = float(payload[metric_key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid {metric_key} in {path}") from error
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {metric_key} in {path}")
    return value


def read_variant_metric_points(
    run_root: Path, metric_key: str, aggregate_by_requested_budget: bool = False,
) -> list[tuple[int, float]]:
    if not run_root.is_dir():
        return []
    points: dict[int, float] = {}
    for budget_dir in run_root.iterdir():
        if not budget_dir.is_dir():
            continue
        try:
            int(budget_dir.name)
        except ValueError:
            continue
        metrics_path = budget_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            payload = read_json(metrics_path)
            effective_budget = int(payload["effective_target_unlabeled_count"])
            requested_budget = int(payload.get("budget", budget_dir.name))
            budget = requested_budget if aggregate_by_requested_budget else effective_budget
            value = float(payload[metric_key])
            if effective_budget <= 0 or not np.isfinite(value):
                raise ValueError("invalid effective budget or metric")
        except (KeyError, TypeError, ValueError) as error:
            print(f"Warning: invalid {metrics_path}: {error}; skipping", file=sys.stderr)
            continue
        points[budget] = value
    return sorted(points.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=CLASS_NAMES,
        help="Task to plot; omit to plot all ten tasks.",
    )
    parser.add_argument(
        "--target",
        choices=CORRUPTIONS,
        help="Target to plot; omit to plot all 19 targets.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--baseline-dir",
        default="disjoint_source_baseline",
        help="Baseline result subdirectory under DATA_ROOT.",
    )
    parser.add_argument(
        "--penalty-dir",
        default="disjoint_anti_causal_penalty",
        help=(
            "Result subdirectory under DATA_ROOT to compare with the linear baseline. "
            "Default: disjoint_anti_causal_penalty."
        ),
    )
    parser.add_argument(
        "--penalty-label",
        help="Legend label for the selected penalty. Defaults to a readable penalty-dir name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output path; defaults to DATA_ROOT/plots/PENALTY_DIR/"
            "TASK_vs_rest/TARGET_auroc.png."
        ),
    )
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--metric", choices=tuple(METRICS), default="auroc")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip individual task-target curves; write only aggregate curves and the heatmap.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def plot_one(
    *,
    task: str,
    target: str,
    data_root: Path,
    baseline_dir: str,
    penalty_dir: str,
    penalty_label: str,
    output_path: Path,
    dpi: int,
    linear_x: bool,
    metric: str,
    show: bool,
    pyplot: object,
) -> bool:
    task_dir = f"{task}_vs_rest"
    baseline_path = (
        data_root / baseline_dir / task_dir / target / "metrics.json"
    )
    if not baseline_path.is_file():
        print(
            f"Warning: linear-baseline metrics not found; skipping: {baseline_path}",
            file=sys.stderr,
        )
        return False
    metric_key, metric_label, higher_is_better = METRICS[metric]
    baseline_value = read_metric(baseline_path, metric_key)
    penalty_root = data_root / penalty_dir / task_dir / target
    points = read_variant_metric_points(penalty_root, metric_key)
    if not points:
        print(
            f"Warning: no {penalty_label} results; skipping: {penalty_root}",
            file=sys.stderr,
        )
        return False

    budgets, values = zip(*points)
    figure, axis = pyplot.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    axis.axhline(
        baseline_value,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Source baseline ({baseline_value:.6f})",
    )
    axis.plot(
        budgets,
        values,
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        label=penalty_label,
    )
    if not linear_x:
        axis.set_xscale("log")
    set_budget_ticks(axis, budgets)
    axis.set_xlabel("Unlabeled target budget")
    axis.set_ylabel(metric_label)
    axis.set_title(f"{task} vs rest — target: {target}")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    print(f"Saved plot: {output_path}")
    if show:
        pyplot.show()
    pyplot.close(figure)
    return True


def load_pair(
    data_root: Path, baseline_dir: str, penalty_dir: str, task: str, target: str, metric: str,
) -> tuple[float, dict[int, float]] | None:
    task_dir = f"{task}_vs_rest"
    baseline_path = (
        data_root / baseline_dir / task_dir / target / "metrics.json"
    )
    penalty_root = data_root / penalty_dir / task_dir / target
    if not baseline_path.is_file():
        return None
    metric_key, _metric_label, _higher_is_better = METRICS[metric]
    # Aggregate curves must treat the requested unlimited budget (-1) as one
    # common "all" condition even when environments have 440 versus 450 rows.
    points = read_variant_metric_points(
        penalty_root, metric_key, aggregate_by_requested_budget=True
    )
    if not points:
        return None
    return read_metric(baseline_path, metric_key), dict(points)


def plot_average(
    *,
    pairs: list[tuple[float, dict[int, float]]],
    penalty_label: str,
    title: str,
    output_path: Path,
    dpi: int,
    linear_x: bool,
    metric: str,
    show: bool,
    pyplot: object,
) -> bool:
    if not pairs:
        print(f"Warning: no results available for aggregate plot: {title}", file=sys.stderr)
        return False
    budgets = sorted(
        {budget for _baseline, points in pairs for budget in points},
        key=lambda budget: (budget == -1, budget),
    )
    if not budgets:
        return False
    mean_baseline = float(np.mean([baseline for baseline, _points in pairs]))
    mean_penalty = [
        float(
            np.mean(
                [points[budget] for _baseline, points in pairs if budget in points]
            )
        )
        for budget in budgets
    ]

    positive_budgets = [budget for budget in budgets if budget > 0]
    all_position = (max(positive_budgets) * 1.5) if positive_budgets else 1.0
    plotted_budgets = [all_position if budget == -1 else float(budget) for budget in budgets]
    budget_labels = ["all" if budget == -1 else str(budget) for budget in budgets]

    figure, axis = pyplot.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    axis.axhline(
        mean_baseline,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean linear baseline ({mean_baseline:.4f})",
    )
    axis.plot(
        plotted_budgets,
        mean_penalty,
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        label=f"Mean {penalty_label}",
    )
    if not linear_x:
        axis.set_xscale("log")
    set_budget_ticks(axis, plotted_budgets, budget_labels)
    axis.set_xlabel("Unlabeled target budget")
    axis.set_ylabel(f"Mean {METRICS[metric][1]}")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    print(f"Saved aggregate plot ({len(pairs)} runs): {output_path}")
    if show:
        pyplot.show()
    pyplot.close(figure)
    return True


def plot_task_target_delta_heatmap(
    *, data_root: Path, baseline_dir: str, penalty_dir: str, penalty_label: str,
    output_path: Path, csv_path: Path, dpi: int, show: bool, pyplot: object, metric: str,
) -> bool:
    """Plot mean paired metric improvements over each available target budget."""
    values = np.full((len(CLASS_NAMES), len(CORRUPTIONS)), np.nan, dtype=np.float64)
    budget_counts = np.zeros(values.shape, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(CLASS_NAMES):
        for target_index, target in enumerate(CORRUPTIONS):
            pair = load_pair(data_root, baseline_dir, penalty_dir, task, target, metric)
            if pair is None:
                continue
            baseline_auroc, points = pair
            _metric_key, metric_label, higher_is_better = METRICS[metric]
            deltas = [
                (value - baseline_auroc) if higher_is_better else (baseline_auroc - value)
                for value in points.values()
            ]
            values[task_index, target_index] = float(np.mean(deltas))
            budget_counts[task_index, target_index] = len(deltas)
            rows.append({
                "task": task,
                "target": target,
                "baseline_metric": baseline_auroc,
                "mean_penalty_improvement": values[task_index, target_index],
                "budget_count": len(deltas),
                "effective_budgets": ",".join(map(str, sorted(points))),
            })
    if not rows:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scale = 1e4 if metric == "auroc" else 1.0
    scaled = values * scale
    finite = scaled[np.isfinite(scaled)]
    limit = float(np.max(np.abs(finite)))
    # Keep zero visually meaningful even if all observed effects are tiny.
    limit = max(limit, 0.01)
    figure, axis = pyplot.subplots(figsize=(18, 8), constrained_layout=True)
    cmap = pyplot.colormaps["RdYlGn"].copy()
    cmap.set_bad(color="#D9D9D9")
    image = axis.imshow(
        np.ma.masked_invalid(scaled), cmap=cmap, vmin=-limit, vmax=limit,
        aspect="auto",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=.015)
    unit = "x 1e-4" if metric == "auroc" else "absolute MSE"
    colorbar.set_label(f"Mean penalty improvement in {metric_label} from source ({unit})")
    axis.set_xticks(range(len(CORRUPTIONS)), CORRUPTIONS, rotation=55, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Held-out target corruption")
    axis.set_ylabel("One-vs-rest task")
    axis.set_title(f"{penalty_label}: mean {metric_label} improvement from source across budgets")
    for task_index in range(len(CLASS_NAMES)):
        for target_index in range(len(CORRUPTIONS)):
            if not np.isfinite(scaled[task_index, target_index]):
                axis.text(target_index, task_index, "missing", ha="center", va="center", fontsize=6)
                continue
            value = scaled[task_index, target_index]
            text_color = "white" if abs(value) > .62 * limit else "black"
            axis.text(
                target_index, task_index,
                f"{value:+.2f}",
                ha="center", va="center", fontsize=6.5, color=text_color,
            )
    figure.savefig(output_path, dpi=dpi)
    print(f"Saved task-target AUROC-delta heatmap: {output_path}")
    print(f"Saved heatmap values: {csv_path}")
    if show:
        pyplot.show()
    pyplot.close(figure)
    return True


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    penalty_dir_path = Path(args.penalty_dir)
    if args.penalty_dir in {"", "."} or penalty_dir_path.name != args.penalty_dir:
        raise ValueError("--penalty-dir must be one result-directory name, not a path")
    baseline_dir_path = Path(args.baseline_dir)
    if args.baseline_dir in {"", "."} or baseline_dir_path.name != args.baseline_dir:
        raise ValueError("--baseline-dir must be one result-directory name, not a path")
    penalty_label = args.penalty_label or args.penalty_dir.replace("_", " ").title()

    tasks = (args.task,) if args.task is not None else CLASS_NAMES
    targets = (args.target,) if args.target is not None else CORRUPTIONS
    if args.output is not None and (len(tasks) != 1 or len(targets) != 1):
        raise ValueError("--output can only be used with one --task and one --target")
    if args.aggregate_only and (args.task is not None or args.target is not None or args.output is not None):
        raise ValueError("--aggregate-only cannot be combined with --task, --target, or --output")

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_root = args.data_root.expanduser().resolve()
    metric_suffix = args.metric
    plotted = 0
    if not args.aggregate_only:
        for task in tasks:
            for target in targets:
                output_path = (
                    args.output.expanduser().resolve()
                    if args.output is not None
                    else data_root
                    / "plots"
                    / args.penalty_dir
                    / f"{task}_vs_rest"
                    / f"{target}_{metric_suffix}.{args.format}"
                )
                plotted += int(
                    plot_one(
                        task=task,
                        target=target,
                        data_root=data_root,
                        baseline_dir=args.baseline_dir,
                        penalty_dir=args.penalty_dir,
                        penalty_label=penalty_label,
                        output_path=output_path,
                        dpi=args.dpi,
                        linear_x=args.linear_x,
                        metric=args.metric,
                        show=args.show,
                        pyplot=plt,
                    )
                )
    if args.task is None and args.target is None:
        available = {
            (task, target): pair
            for task in CLASS_NAMES
            for target in CORRUPTIONS
            if (pair := load_pair(data_root, args.baseline_dir, args.penalty_dir, task, target, args.metric)) is not None
        }
        aggregate_root = data_root / "plots" / args.penalty_dir / "averages"
        for task in CLASS_NAMES:
            plotted += int(
                plot_average(
                    pairs=[
                        available[(task, target)]
                        for target in CORRUPTIONS
                        if (task, target) in available
                    ],
                    penalty_label=penalty_label,
                    title=f"{task} vs rest — mean across targets",
                    output_path=(
                        aggregate_root / "by_task" / f"{task}_{metric_suffix}.{args.format}"
                    ),
                    dpi=args.dpi,
                    linear_x=args.linear_x,
                    metric=args.metric,
                    show=args.show,
                    pyplot=plt,
                )
            )
        for target in CORRUPTIONS:
            plotted += int(
                plot_average(
                    pairs=[
                        available[(task, target)]
                        for task in CLASS_NAMES
                        if (task, target) in available
                    ],
                    penalty_label=penalty_label,
                    title=f"Target: {target} — mean across tasks",
                    output_path=(
                        aggregate_root / "by_target" / f"{target}_{metric_suffix}.{args.format}"
                    ),
                    dpi=args.dpi,
                    linear_x=args.linear_x,
                    metric=args.metric,
                    show=args.show,
                    pyplot=plt,
                )
            )
        plotted += int(
            plot_average(
                pairs=list(available.values()),
                penalty_label=penalty_label,
                title="Mean across all tasks and targets",
                output_path=aggregate_root / f"overall_{metric_suffix}.{args.format}",
                dpi=args.dpi,
                linear_x=args.linear_x,
                metric=args.metric,
                show=args.show,
                pyplot=plt,
            )
        )
        plotted += int(
            plot_task_target_delta_heatmap(
                data_root=data_root,
                baseline_dir=args.baseline_dir,
                penalty_dir=args.penalty_dir,
                penalty_label=penalty_label,
                output_path=aggregate_root / f"task_target_mean_improvement_{metric_suffix}.{args.format}",
                csv_path=aggregate_root / f"task_target_mean_improvement_{metric_suffix}.csv",
                dpi=args.dpi,
                show=args.show,
                pyplot=plt,
                metric=args.metric,
            )
        )
    if plotted == 0:
        raise FileNotFoundError("No complete baseline/penalty result pairs were found")
    print(f"Created {plotted} plot(s).")


if __name__ == "__main__":
    main()
