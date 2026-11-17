from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from synthetic_anti_causal.run_experiment import build_argument_parser, run_configuration


TARGET_AWARE_KEYS = {
    "plain_inverse": "plain_inverse_mse",
    "pooled_inverse": "pooled_inverse_mse",
    "pooled_target_vir": "pooled_target_vir_mse",
    "target_aware_penalized": "target_aware_penalized_mse",
    "target_aware_inverse": "target_aware_inverse_mse",
    "target_aware_inverse_vir": "target_aware_inverse_vir_mse",
    "projector_inverse": "projector_inverse_mse",
    "projector_inverse_vir": "projector_inverse_vir_mse",
}

SOURCE_ONLY_KEYS = {
    "ols": "ols_mse",
    "mir": "mir_mse",
    "vir": "vir_mse",
    "projector": "projector_mse",
}

ALL_ESTIMATORS = set(TARGET_AWARE_KEYS) | set(SOURCE_ONLY_KEYS)

ESTIMATOR_ALIASES = {
    "target_aware_penalized": "target_aware",
}


METHOD_MSE_KEYS = {**SOURCE_ONLY_KEYS, **TARGET_AWARE_KEYS}
ORACLE_MSE_KEYS = {
    "target_inverse_baseline": "target_inverse_baseline_mse",
    "full_target_oracle": "full_target_oracle_mse",
}


def save_worst_case_budget_plot(
    output_path: Path,
    *,
    rows: list[dict[str, object]],
    field: str,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    budgets = [int(row["unlabeled_samples_per_test_env"]) for row in rows]
    method_names = list(rows[0][field].keys())
    # `plain_inverse` and the explicit target-inverse baseline are the same
    # estimator. Keep the descriptive baseline label when both are present.
    if "target_inverse_baseline" in method_names:
        method_names = [name for name in method_names if name != "plain_inverse"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method_name in method_names:
        values = [float(row[field][method_name]) for row in rows]
        ax.plot(budgets, values, marker="o", label=method_name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Target Unlabeled Samples")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compute_worst_case_rows(results: dict[str, object]) -> dict[str, float]:
    per_env = results["test_summary"]["per_test_environment"]
    included = set(results["included_estimators"])
    worst_case = {}
    for name, key in SOURCE_ONLY_KEYS.items():
        if name in included:
            worst_case[name] = float(max(row[key] for row in per_env))
    for name, key in TARGET_AWARE_KEYS.items():
        canonical_name = ESTIMATOR_ALIASES.get(name, name)
        if canonical_name in included:
            worst_case[name] = float(max(row[key] for row in per_env))
    worst_case["target_inverse_baseline"] = float(
        max(row["target_inverse_baseline_mse"] for row in per_env)
    )
    return worst_case


def compute_average_rows(results: dict[str, object]) -> dict[str, float]:
    summary = results["test_summary"]
    included = set(results["included_estimators"])
    averages = {}
    if "ols" in included:
        averages["ols"] = float(summary["mean_test_ols_mse"])
    if "plain_inverse" in included:
        averages["plain_inverse"] = float(summary["mean_test_plain_inverse_mse"])
    if "pooled_inverse" in included:
        averages["pooled_inverse"] = float(summary["mean_test_pooled_inverse_mse"])
    if "pooled_target_vir" in included:
        averages["pooled_target_vir"] = float(
            summary["mean_test_pooled_target_vir_mse"]
        )
    if "target_aware" in included:
        averages["target_aware_penalized"] = float(summary["mean_test_target_aware_penalized_mse"])
    if "target_aware_inverse" in included:
        averages["target_aware_inverse"] = float(summary["mean_test_target_aware_inverse_mse"])
    if "target_aware_inverse_vir" in included:
        averages["target_aware_inverse_vir"] = float(summary["mean_test_target_aware_inverse_vir_mse"])
    if "mir" in included:
        averages["mir"] = float(summary["mean_test_mir_mse"])
    if "vir" in included:
        averages["vir"] = float(summary["mean_test_vir_mse"])
    if "projector" in included:
        averages["projector"] = float(summary["mean_test_projector_mse"])
    if "projector_inverse" in included:
        averages["projector_inverse"] = float(summary["mean_test_projector_inverse_mse"])
    if "projector_inverse_vir" in included:
        averages["projector_inverse_vir"] = float(summary["mean_test_projector_inverse_vir_mse"])
    averages["target_inverse_baseline"] = float(
        summary["mean_test_target_inverse_baseline_mse"]
    )
    return averages


def compute_percent_reduction_vs_source_ols(
    results: dict[str, object],
    *,
    source_ols_mse_by_env: dict[int, float],
) -> dict[str, list[dict[str, float]]]:
    """Return per-target-environment percentage MSE reductions vs source OLS."""
    included = set(results["included_estimators"])
    values: dict[str, list[dict[str, float]]] = {}
    for method, mse_key in METHOD_MSE_KEYS.items():
        canonical_method = ESTIMATOR_ALIASES.get(method, method)
        if canonical_method not in included:
            continue
        method_values = []
        for row in results["test_summary"]["per_test_environment"]:
            env_id = int(row["env_id"])
            source_mse = float(source_ols_mse_by_env[env_id])
            method_mse = float(row[mse_key])
            method_values.append(
                {
                    "env_id": env_id,
                    "percent_mse_reduction": 100.0 * (source_mse - method_mse) / source_mse,
                }
            )
        values[method] = method_values
    return values


def compute_percent_reduction_vs_oracle_best(
    results: dict[str, object],
) -> dict[str, list[dict[str, float]]]:
    """Return per-environment MSE reductions vs the test-set source/target oracle.

    The comparison reference is the lower test MSE of source OLS and the
    unlabeled target-inverse baseline, M_target^-1 m_source. This is an oracle:
    it intentionally uses test performance only for evaluation.
    """
    included = set(results["included_estimators"])
    values: dict[str, list[dict[str, float]]] = {}
    all_keys = {**METHOD_MSE_KEYS, **ORACLE_MSE_KEYS}
    for method, mse_key in all_keys.items():
        canonical_method = ESTIMATOR_ALIASES.get(method, method)
        if method not in ORACLE_MSE_KEYS and canonical_method not in included:
            continue
        method_values = []
        for row in results["test_summary"]["per_test_environment"]:
            reference_mse = float(row["oracle_best_source_target_inverse_mse"])
            method_mse = float(row[mse_key])
            method_values.append(
                {
                    "env_id": int(row["env_id"]),
                    "reference_mse": reference_mse,
                    "method_mse": method_mse,
                    "percent_mse_reduction": 100.0 * (reference_mse - method_mse) / reference_mse,
                }
            )
        values[method] = method_values
    return values


def summarize_reduction_trajectories(
    rows: list[dict[str, object]],
) -> tuple[dict[str, dict[str, list[float]]], list[dict[str, object]]]:
    """Summarize and score gain trajectories relative to the oracle-best line."""
    methods = sorted(
        {
            method
            for row in rows
            for method in row["percent_reduction_vs_oracle_best"]
        }
    )
    budgets = np.asarray(
        [int(row["unlabeled_samples_per_test_env"]) for row in rows],
        dtype=np.float64,
    )
    log_budgets = np.log(budgets)
    trajectories: dict[str, dict[str, list[float]]] = {}
    for method in methods:
        distributions = [
            [
                float(point["percent_mse_reduction"])
                for point in row["percent_reduction_vs_oracle_best"].get(method, [])
            ]
            for row in rows
        ]
        trajectories[method] = {
            "mean": [float(np.mean(values)) if values else float("nan") for values in distributions],
            "median": [float(np.median(values)) if values else float("nan") for values in distributions],
            "p25": [float(np.percentile(values, 25)) if values else float("nan") for values in distributions],
        }

    oracle = trajectories.get("full_target_oracle")
    if oracle is None:
        return trajectories, []
    integrate = getattr(np, "trapezoid", None) or np.trapz
    oracle_mean_integral = float(integrate(oracle["mean"], x=log_budgets))
    oracle_median_integral = float(integrate(oracle["median"], x=log_budgets))
    oracle_p25_integral = float(integrate(oracle["p25"], x=log_budgets))
    log_budget_range = float(log_budgets[-1] - log_budgets[0])
    scores = []
    for method, trajectory in trajectories.items():
        if method in {"ols", "target_inverse_baseline", "full_target_oracle"}:
            continue
        mean_integral = float(integrate(trajectory["mean"], x=log_budgets))
        median_integral = float(integrate(trajectory["median"], x=log_budgets))
        p25_integral = float(integrate(trajectory["p25"], x=log_budgets))
        scores.append(
            {
                "method": method,
                "budgets": [int(value) for value in budgets],
                "integration_x_axis": "log_budget",
                "full_target_oracle_mean_gain_integral": oracle_mean_integral,
                "full_target_oracle_median_gain_integral": oracle_median_integral,
                "full_target_oracle_p25_gain_integral": oracle_p25_integral,
                "full_target_oracle_mean_log_budget_average": oracle_mean_integral / log_budget_range,
                "full_target_oracle_median_log_budget_average": oracle_median_integral / log_budget_range,
                "full_target_oracle_p25_log_budget_average": oracle_p25_integral / log_budget_range,
                "estimator_mean_gain_integral": mean_integral,
                "estimator_median_gain_integral": median_integral,
                "estimator_p25_gain_integral": p25_integral,
                "mean_integral_fraction_of_oracle": (
                    mean_integral / oracle_mean_integral
                    if abs(oracle_mean_integral) > 1e-12
                    else None
                ),
                "median_integral_fraction_of_oracle": (
                    median_integral / oracle_median_integral
                    if abs(oracle_median_integral) > 1e-12
                    else None
                ),
                "p25_integral_fraction_of_oracle": (
                    p25_integral / oracle_p25_integral
                    if abs(oracle_p25_integral) > 1e-12
                    else None
                ),
            }
        )
    return trajectories, scores


def save_oracle_best_boxplot(
    output_path: Path,
    *,
    rows: list[dict[str, object]],
) -> None:
    """Box plots of each adaptation method against the source/target oracle."""
    import matplotlib.pyplot as plt

    methods = sorted(
        {
            method
            for row in rows
            for method in row["percent_reduction_vs_oracle_best"]
            if method not in {"ols", "target_inverse_baseline", "full_target_oracle"}
        }
    )
    if not methods:
        print("Skipping oracle-best box plot: no adaptation methods available")
        return

    budgets = [int(row["unlabeled_samples_per_test_env"]) for row in rows]
    figure, axes = plt.subplots(
        len(methods),
        1,
        figsize=(9, max(3.2, 2.9 * len(methods))),
        sharex=True,
        constrained_layout=True,
    )
    if len(methods) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    positions = np.arange(len(budgets), dtype=float)
    for axis, method in zip(axes, methods):
        distributions = [
            [float(point["percent_mse_reduction"]) for point in row["percent_reduction_vs_oracle_best"].get(method, [])]
            for row in rows
        ]
        axis.boxplot(
            distributions,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            boxprops={"facecolor": "#82b6a2", "alpha": 0.75},
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for position, values in zip(positions, distributions):
            if values:
                jitter = rng.uniform(-0.16, 0.16, size=len(values))
                axis.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=14,
                    alpha=0.35,
                    color="#365f56",
                    edgecolors="none",
                )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_ylabel(f"{method}\n% MSE reduction")
        axis.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([str(budget) for budget in budgets])
    axes[-1].set_xlabel("Target unlabeled budget")
    figure.suptitle(
        "Improvement over the test-set best of source OLS and target inverse",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_oracle_best_trajectory_plot(
    output_path: Path,
    *,
    budgets: list[int],
    trajectories: dict[str, dict[str, list[float]]],
) -> None:
    """Plot mean and median gains, including the full-target oracle trajectory."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for axis, statistic in zip(axes, ("mean", "median")):
        for method, trajectory in sorted(trajectories.items()):
            if method == "ols":
                continue
            if method == "full_target_oracle":
                style = {"linewidth": 3.0, "color": "black"}
                label = "full target oracle"
            elif method == "target_inverse_baseline":
                style = {"linewidth": 2.5, "linestyle": "--", "color": "#6c757d"}
                label = "target inverse $M_t^{-1}m_s$"
            else:
                style = {}
                label = method
            axis.plot(budgets, trajectory[statistic], marker="o", label=label, **style)
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_ylabel(f"{statistic.title()} % MSE reduction")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Target unlabeled budget")
    figure.suptitle(
        "Gains over the test-set best of source OLS and target inverse",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_source_ols_improvement_boxplot(
    output_path: Path,
    *,
    rows: list[dict[str, object]],
) -> None:
    """Plot per-target-environment MSE changes against source OLS by budget."""
    import matplotlib.pyplot as plt

    methods = sorted(
        {
            method
            for row in rows
            for method in row["percent_reduction_vs_source_ols"]
            if method != "ols"
        }
    )
    if not methods:
        print("Skipping source-OLS box plot: no non-OLS methods available")
        return

    budgets = [int(row["unlabeled_samples_per_test_env"]) for row in rows]
    figure, axes = plt.subplots(
        len(methods),
        1,
        figsize=(9, max(3.2, 2.9 * len(methods))),
        sharex=True,
        constrained_layout=True,
    )
    if len(methods) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    positions = np.arange(len(budgets), dtype=float)
    for axis, method in zip(axes, methods):
        distributions = []
        for row in rows:
            points = row["percent_reduction_vs_source_ols"].get(method, [])
            distributions.append([float(point["percent_mse_reduction"]) for point in points])
        axis.boxplot(
            distributions,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            boxprops={"facecolor": "#82b6a2", "alpha": 0.75},
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for position, values in zip(positions, distributions):
            if values:
                jitter = rng.uniform(-0.16, 0.16, size=len(values))
                axis.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=14,
                    alpha=0.35,
                    color="#365f56",
                    edgecolors="none",
                )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_ylabel(f"{method}\n% MSE reduction")
        axis.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([str(budget) for budget in budgets])
    axes[-1].set_xlabel("Target unlabeled budget")
    figure.suptitle("Per-target-environment improvement over source OLS", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parent = build_argument_parser()
    parser = argparse.ArgumentParser(parents=[parent], conflict_handler="resolve")
    parser.add_argument(
        "--unlabeled-samples-per-test-env-grid",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--output-plot",
        type=str,
        default=str(Path(__file__).resolve().parent / "synthetic_env_worst_case_budget_curve.png"),
    )
    parser.add_argument(
        "--output-average-plot",
        type=str,
        default=str(Path(__file__).resolve().parent / "synthetic_env_average_budget_curve.png"),
    )
    parser.add_argument(
        "--output-source-ols-box-plot",
        type=str,
        default=None,
        help=(
            "Box plots of per-target-environment percentage MSE reduction "
            "relative to source OLS. Retained for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--output-oracle-best-box-plot",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "synthetic_env_oracle_best_improvement_boxplot.png"
        ),
        help=(
            "Box plots of per-target-environment percentage MSE reduction "
            "against the test-set best of source OLS and target inverse."
        ),
    )
    parser.add_argument(
        "--output-oracle-best-trajectory-plot",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "synthetic_env_oracle_best_trajectory.png"
        ),
        help="Mean/median gain trajectories, including the full-target oracle.",
    )
    parser.add_argument(
        "--output-oracle-scores-json",
        type=str,
        default=None,
        help=(
            "Where to write integral scores relative to the full-target oracle. "
            "Defaults beside --output-json when that flag is supplied."
        ),
    )
    parser.add_argument(
        "--box-plots-only",
        action="store_true",
        help="Skip the aggregate worst-case and mean-MSE line plots.",
    )
    parser.add_argument(
        "--skip-full-single-run-output",
        action="store_true",
        help="Do not save the per-budget single-run JSON/PNG artifacts; save only the sweep outputs.",
    )
    return parser


def main(args) -> None:
    budgets = sorted({int(x) for x in args.unlabeled_samples_per_test_env_grid})
    if not budgets:
        raise ValueError("Need at least one target unlabeled budget.")
    if args.oracle_labeled_samples_per_test_env <= 0:
        raise ValueError(
            "--oracle-labeled-samples-per-test-env must be positive when "
            "producing oracle-headroom scores."
        )

    print(f"[sweep] budgets={budgets}")
    baseline_budget = budgets[0]
    print(
        "[sweep] running baseline budget for source-only estimators and first target-aware point "
        f"budget={baseline_budget}"
    )
    baseline_args = copy.deepcopy(args)
    baseline_args.unlabeled_samples_per_test_env = baseline_budget
    if args.skip_full_single_run_output:
        baseline_args.output_json = None
        baseline_args.output_plot = None
        baseline_args.output_vector_plot = None
    baseline_results = run_configuration(baseline_args, emit_progress=True)
    baseline_worst = compute_worst_case_rows(baseline_results)
    baseline_average = compute_average_rows(baseline_results)
    source_ols_mse_by_env = {
        int(row["env_id"]): float(row["ols_mse"])
        for row in baseline_results["test_summary"]["per_test_environment"]
    }
    baseline_reductions = compute_percent_reduction_vs_source_ols(
        baseline_results,
        source_ols_mse_by_env=source_ols_mse_by_env,
    )
    baseline_oracle_reductions = compute_percent_reduction_vs_oracle_best(
        baseline_results,
    )

    sweep_rows: list[dict[str, object]] = [
        {
            "unlabeled_samples_per_test_env": baseline_budget,
            "worst_case_mse": baseline_worst,
            "average_mse": baseline_average,
            "percent_reduction_vs_source_ols": baseline_reductions,
            "percent_reduction_vs_oracle_best": baseline_oracle_reductions,
        }
    ]

    source_only_reference = {
        name: baseline_worst[name] for name in SOURCE_ONLY_KEYS if name in baseline_worst
    }
    source_only_average_reference = {
        name: baseline_average[name] for name in SOURCE_ONLY_KEYS if name in baseline_average
    }
    source_only_reduction_reference = {
        name: baseline_reductions[name]
        for name in SOURCE_ONLY_KEYS
        if name in baseline_reductions
    }

    for budget in budgets[1:]:
        print(f"[sweep] running target-aware budget={budget}")
        run_args = copy.deepcopy(args)
        run_args.unlabeled_samples_per_test_env = budget
        if args.skip_full_single_run_output:
            run_args.output_json = None
            run_args.output_plot = None
            run_args.output_vector_plot = None
        run_results = run_configuration(run_args, emit_progress=True)
        run_worst = compute_worst_case_rows(run_results)
        run_average = compute_average_rows(run_results)
        run_reductions = compute_percent_reduction_vs_source_ols(
            run_results,
            source_ols_mse_by_env=source_ols_mse_by_env,
        )
        run_oracle_reductions = compute_percent_reduction_vs_oracle_best(run_results)
        merged = dict(source_only_reference)
        merged_average = dict(source_only_average_reference)
        merged_reductions = dict(source_only_reduction_reference)
        for name in {*TARGET_AWARE_KEYS, "target_inverse_baseline"}:
            if name in run_worst:
                merged[name] = run_worst[name]
            if name in run_average:
                merged_average[name] = run_average[name]
            if name in run_reductions:
                merged_reductions[name] = run_reductions[name]
        sweep_rows.append(
            {
                "unlabeled_samples_per_test_env": budget,
                "worst_case_mse": merged,
                "average_mse": merged_average,
                "percent_reduction_vs_source_ols": merged_reductions,
                # Source-only estimator MSE is invariant, but its gain is not:
                # the budget-target OLS part of the oracle changes by budget.
                "percent_reduction_vs_oracle_best": run_oracle_reductions,
            }
        )

    trajectories, oracle_scores = summarize_reduction_trajectories(sweep_rows)

    output = {
        "budgets": budgets,
        "rows": sweep_rows,
        "source_only_methods": sorted(name for name in SOURCE_ONLY_KEYS if name not in args.exclude_estimators),
        "target_aware_methods": sorted(
            name
            for name in TARGET_AWARE_KEYS
            if ESTIMATOR_ALIASES.get(name, name) not in args.exclude_estimators
        ),
        "oracle_definition": {
            "comparison_reference": (
                "Per test environment, min(test MSE(source OLS), test MSE("
                "M_target^-1 m_source using the same source and target-unlabeled "
                "samples available to the estimators))."
            ),
            "full_target_oracle": (
                "Target OLS fitted on all independent reserved oracle labels. "
                "These labels are unavailable to every estimator."
            ),
            "reserved_labeled_samples_per_test_env": args.oracle_labeled_samples_per_test_env,
        },
        "oracle_best_trajectories": trajectories,
        "oracle_best_integral_scores": oracle_scores,
    }

    print(json.dumps(output, indent=2))

    if args.output_json is not None:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    oracle_scores_path = args.output_oracle_scores_json
    if oracle_scores_path is None and args.output_json is not None:
        output_json_path = Path(args.output_json)
        oracle_scores_path = str(output_json_path.with_name(f"{output_json_path.stem}_oracle_scores.json"))
    if oracle_scores_path is not None:
        score_path = Path(oracle_scores_path)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(oracle_scores, indent=2), encoding="utf-8")
        print(f"Saved oracle integral scores to {score_path}")

    if not args.box_plots_only and args.output_plot is not None:
        plot_path = Path(args.output_plot)
        save_worst_case_budget_plot(
            plot_path,
            rows=sweep_rows,
            field="worst_case_mse",
            title="Worst-Case Test MSE vs Target Unlabeled Budget",
            ylabel="Worst Test-Environment MSE",
        )
        print(f"Saved plot to {plot_path}")
    if not args.box_plots_only and args.output_average_plot is not None:
        avg_plot_path = Path(args.output_average_plot)
        save_worst_case_budget_plot(
            avg_plot_path,
            rows=sweep_rows,
            field="average_mse",
            title="Average Test MSE vs Target Unlabeled Budget",
            ylabel="Average Test-Environment MSE",
        )
        print(f"Saved plot to {avg_plot_path}")
    if args.output_source_ols_box_plot is not None:
        box_plot_path = Path(args.output_source_ols_box_plot)
        save_source_ols_improvement_boxplot(box_plot_path, rows=sweep_rows)
        print(f"Saved source-OLS improvement box plot to {box_plot_path}")
    if args.output_oracle_best_box_plot is not None:
        oracle_box_plot_path = Path(args.output_oracle_best_box_plot)
        save_oracle_best_boxplot(oracle_box_plot_path, rows=sweep_rows)
        print(f"Saved oracle-best improvement box plot to {oracle_box_plot_path}")
    if args.output_oracle_best_trajectory_plot is not None:
        oracle_trajectory_path = Path(args.output_oracle_best_trajectory_plot)
        save_oracle_best_trajectory_plot(
            oracle_trajectory_path,
            budgets=budgets,
            trajectories=trajectories,
        )
        print(f"Saved oracle-best trajectory plot to {oracle_trajectory_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
