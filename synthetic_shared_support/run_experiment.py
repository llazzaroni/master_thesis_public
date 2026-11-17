from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from synthetic_shared_support.data import EnvironmentData, SyntheticDataset, generate_dataset
from synthetic_shared_support.estimators import (
    fit_mixed_estimator,
    fit_mixed_estimator_with_alpha,
    fit_ols,
    fit_pseudo_lasso,
    mse,
    select_mixed_alpha_by_target_cv,
    stack_environments,
)


METHOD_COLORS = {
    "source_only_ols": "#0072B2",
    "target_only_ols": "#D55E00",
    "mixed_estimator": "#CC79A7",
    "pseudo_lasso": "#009E73",
}


def method_color(method: str, index: int = 0) -> str:
    return METHOD_COLORS.get(method, f"C{index % 10}")


def save_test_error_plot(
    output_path: Path,
    *,
    method_rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["method"]) for row in method_rows]
    errors = [float(row["target_test_mse"]) for row in method_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(labels), dtype=float), errors, color="tab:blue")
    ax.set_xticks(np.arange(len(labels), dtype=float))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Target Test MSE")
    ax.set_title("Pseudo-Lasso Synthetic Comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_budget_curve_plot(
    output_path: Path,
    *,
    budget_rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if not budget_rows:
        return

    method_names = list(budget_rows[0]["methods"].keys())
    budgets = [int(row["target_train_samples"]) for row in budget_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method_name in method_names:
        values = [float(row["methods"][method_name]["target_test_mse"]) for row in budget_rows]
        ax.plot(budgets, values, marker="o", label=method_name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Preserve the log scale while avoiding scientific tick labels such as 10^0.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Target Train Samples")
    ax.set_ylabel("Target Test MSE")
    ax.set_title("Performance vs Target Sample Budget")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_oracle_improvement_boxplot(
    output_path: Path,
    *,
    budgets: list[int],
    gains: dict[str, list[list[float]]],
    oracle_mean: list[float],
    oracle_median: list[float],
) -> list[Path]:
    """Draw one oracle-relative budget boxplot per estimator."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    positions = np.arange(len(budgets), dtype=float)
    outputs = []
    for method_index, method in enumerate(gains):
        values = gains[method]
        color = method_color(method, method_index)
        figure, axis = plt.subplots(figsize=(9.5, 5.5))
        boxes = axis.boxplot(
            values,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.7},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.45)

        # One point is one independent structural draw at this target budget.
        jitter_rng = np.random.default_rng(10_000 + method_index)
        for position, draw_values in zip(positions, values):
            jitter = jitter_rng.uniform(-0.16, 0.16, size=len(draw_values))
            axis.scatter(position + jitter, draw_values, s=22, color=color,
                         alpha=0.55, edgecolors="none", zorder=3)

        axis.plot(positions, [float(np.mean(value)) for value in values], "D-",
                  color=color, linewidth=2.0, markersize=6, zorder=4)
        axis.plot(positions, [float(np.median(value)) for value in values], "o-",
                  color="black", linewidth=1.5, markersize=5, zorder=5)
        axis.plot(positions, oracle_mean, "D--", color="#7570B3", linewidth=1.8, zorder=4)
        axis.plot(positions, oracle_median, "s--", color="#A6761D", linewidth=1.8, zorder=4)
        axis.axhline(0.0, color="black", linewidth=1.0)
        # Keep a single pathological draw from flattening the central boxplot.
        # These are the same 1.5-IQR whisker limits represented by the boxes.
        whisker_bounds = []
        for draw_values in values:
            q1, q3 = np.percentile(draw_values, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            inlier_values = np.asarray(draw_values)[(np.asarray(draw_values) >= lower) & (np.asarray(draw_values) <= upper)]
            if len(inlier_values):
                whisker_bounds.extend([float(np.min(inlier_values)), float(np.max(inlier_values))])
        # Retain the zero reference and both full-target oracle summaries.
        # Those reference trajectories are intentional comparison quantities,
        # not draw-level outliers to be hidden.
        visible_references = [0.0, *oracle_mean, *oracle_median]
        lower, upper = min(whisker_bounds + visible_references), max(whisker_bounds + visible_references)
        padding = max((upper - lower) * 0.08, 1e-3)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xticks(positions, [str(budget) for budget in budgets])
        axis.set_xlabel("Target training-sample budget")
        axis.set_ylabel("MSE improvement (%)")
        axis.set_title(method.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(handles=[
            Patch(facecolor=color, alpha=0.45, label="Distribution across draws"),
            Line2D([], [], color=color, marker="D", label=f"{method.replace('_', ' ')} mean"),
            Line2D([], [], color="black", marker="o", label=f"{method.replace('_', ' ')} median"),
            Line2D([], [], color="#7570B3", marker="D", linestyle="--", label="Full-target OLS oracle mean"),
            Line2D([], [], color="#A6761D", marker="s", linestyle="--", label="Full-target OLS oracle median"),
        ], loc="best")
        figure.tight_layout()
        method_path = output_path.with_name(f"{output_path.stem}_{method}{output_path.suffix}")
        method_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(method_path, dpi=180)
        plt.close(figure)
        outputs.append(method_path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-source-envs", type=int, default=8)
    parser.add_argument("--source-samples-per-env", type=int, default=1000)
    parser.add_argument("--target-train-samples", type=int, default=800)
    parser.add_argument(
        "--target-train-samples-grid",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30, 50, 100, 200, 400, 800],
    )
    parser.add_argument("--target-test-samples", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=40)
    parser.add_argument("--shared-support-size", type=int, default=20)
    parser.add_argument("--source-specific-support-size", type=int, default=5)
    parser.add_argument("--target-specific-support-size", type=int, default=5)
    parser.add_argument("--coefficient-low", type=float, default=0.8)
    parser.add_argument("--coefficient-high", type=float, default=1.2)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument(
        "--mixed-alpha-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    parser.add_argument("--screen-z-threshold", type=float, default=0.0)
    parser.add_argument("--screen-stability-threshold", type=float, default=0.35)
    parser.add_argument("--max-invariant-features", type=int, default=None)
    parser.add_argument(
        "--lasso-lambda-grid",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    )
    parser.add_argument("--lasso-cv-folds", type=int, default=5)
    parser.add_argument("--num-draws", type=int, default=50)
    parser.add_argument("--headroom-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--integration-min-budget",
        type=int,
        default=None,
        help="Use only budgets greater than or equal to this value when computing integral scores.",
    )
    parser.add_argument(
        "--oracle-target-samples",
        type=int,
        default=2000,
        help="Independent labeled target samples used only for the full-target OLS oracle.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--output-plot",
        type=str,
        default=str(Path(__file__).resolve().parent / "synthetic_shared_support_test_errors.png"),
    )
    parser.add_argument(
        "--output-budget-plot",
        type=str,
        default=str(Path(__file__).resolve().parent / "synthetic_shared_support_budget_curve.png"),
    )
    parser.add_argument(
        "--output-oracle-boxplot",
        type=str,
        default=str(Path(__file__).resolve().parent / "synthetic_shared_support_oracle_improvement_boxplot.png"),
    )
    parser.add_argument(
        "--output-integral-scores",
        type=str,
        default=str(Path(__file__).resolve().parent / "synthetic_shared_support_oracle_integral_scores.csv"),
    )
    return parser


def _subset_target_train(env: EnvironmentData, target_train_samples: int) -> EnvironmentData:
    if target_train_samples > env.num_samples:
        raise ValueError(
            f"Requested target_train_samples={target_train_samples} but only {env.num_samples} are available."
        )
    return EnvironmentData(
        env_id=env.env_id,
        x=env.x[:target_train_samples].copy(),
        y=env.y[:target_train_samples].copy(),
        beta_star=env.beta_star.copy(),
        support_specific=list(env.support_specific),
    )


def _build_budget_dataset(base_dataset: SyntheticDataset, target_train_samples: int) -> SyntheticDataset:
    return SyntheticDataset(
        source_envs=base_dataset.source_envs,
        target_train=_subset_target_train(base_dataset.target_train, target_train_samples),
        target_test=base_dataset.target_test,
        shared_support=list(base_dataset.shared_support),
        shared_beta=base_dataset.shared_beta.copy(),
    )


def _compute_support_recovery_diagnostics(dataset: SyntheticDataset, recovered_support: list[int]) -> dict[str, object]:
    recovered = set(recovered_support)
    shared = set(dataset.shared_support)
    target_specific = set(dataset.target_train.support_specific)
    source_specific_union = set().union(*(set(env.support_specific) for env in dataset.source_envs))
    null_support = set(range(dataset.target_train.x.shape[1])) - shared - target_specific - source_specific_union

    true_positives = sorted(recovered & shared)
    false_negatives = sorted(shared - recovered)
    false_positives_target = sorted(recovered & target_specific)
    false_positives_other = sorted(recovered - shared - target_specific)

    false_positives_source_specific = sorted(set(false_positives_other) & source_specific_union)
    false_positives_null = sorted(set(false_positives_other) & null_support)

    return {
        "true_positives": len(true_positives),
        "false_negatives": len(false_negatives),
        "false_positives_target": len(false_positives_target),
        "false_positives_other": len(false_positives_other),
        "true_positive_features": true_positives,
        "false_negative_features": false_negatives,
        "false_positive_target_features": false_positives_target,
        "false_positive_other_features": false_positives_other,
        "false_positive_source_specific_features": false_positives_source_specific,
        "false_positive_null_features": false_positives_null,
    }


def run_single_configuration(
    args,
    *,
    dataset: SyntheticDataset,
    target_train_samples: int,
    structural_draw: int,
    draw_seed: int,
    full_target_oracle_mse: float,
) -> dict[str, object]:
    print(f"[run] target_train_samples={target_train_samples}")
    dataset = _build_budget_dataset(dataset, target_train_samples)

    x_source, y_source = stack_environments(dataset.source_envs)
    print("[run] fitting source-only OLS")
    source_fit = fit_ols(x_source, y_source)
    print("[run] fitting target-only OLS")
    target_only_fit = fit_ols(dataset.target_train.x, dataset.target_train.y)
    print("[run] fitting analytic mixed estimator")
    mixed_analytic_fit = fit_mixed_estimator(dataset.source_envs, dataset.target_train)
    print("[run] selecting mixed alpha by target-subset CV")
    mixed_cv = select_mixed_alpha_by_target_cv(
        dataset.source_envs,
        target_train=dataset.target_train,
        alpha_grid=[float(x) for x in args.mixed_alpha_grid],
        cv_folds=args.lasso_cv_folds,
        seed=draw_seed,
    )
    print(f"[run] mixed alpha selected: {mixed_cv.best_alpha}")
    print("[run] refitting mixed estimator with selected alpha")
    mixed_fit = fit_mixed_estimator_with_alpha(
        dataset.source_envs,
        dataset.target_train,
        alpha=mixed_cv.best_alpha,
    )
    print("[run] fitting pseudo-lasso")
    pseudo_lasso_fit = fit_pseudo_lasso(
        dataset.source_envs,
        dataset.target_train,
        z_threshold=args.screen_z_threshold,
        stability_threshold=args.screen_stability_threshold,
        max_invariant_features=args.max_invariant_features,
        lambda_grid=[float(x) for x in args.lasso_lambda_grid],
        cv_folds=args.lasso_cv_folds,
        seed=draw_seed,
    )
    print(
        f"[run] pseudo-lasso selected lambda={pseudo_lasso_fit.lambda_lasso} "
        f"invariant_features={len(pseudo_lasso_fit.invariant_support)}"
    )
    methods = [
        {"method": "source_only_ols", "beta": source_fit.beta},
        {"method": "target_only_ols", "beta": target_only_fit.beta},
        {"method": "mixed_estimator", "beta": mixed_fit.beta},
        {"method": "pseudo_lasso", "beta": pseudo_lasso_fit.beta},
    ]

    method_rows: list[dict[str, object]] = []
    for row in methods:
        beta = np.asarray(row["beta"], dtype=np.float64)
        method_rows.append(
            {
                "method": row["method"],
                "target_test_mse": mse(dataset.target_test.x, dataset.target_test.y, beta),
                "target_train_mse": mse(dataset.target_train.x, dataset.target_train.y, beta),
                "beta_error_to_target": float(np.sum((beta - dataset.target_test.beta_star) ** 2)),
            }
        )

    invariant_overlap = sorted(set(pseudo_lasso_fit.invariant_support).intersection(set(dataset.shared_support)))
    support_recovery = _compute_support_recovery_diagnostics(dataset, pseudo_lasso_fit.invariant_support)
    print(
        "[run] support recovery "
        f"tp={support_recovery['true_positives']} "
        f"fn={support_recovery['false_negatives']} "
        f"fp_target={support_recovery['false_positives_target']} "
        f"fp_other={support_recovery['false_positives_other']}"
    )
    print("[run] evaluating methods on fixed target test set")

    results = {
        "seed": draw_seed,
        "structural_draw": structural_draw,
        "num_source_envs": args.num_source_envs,
        "source_samples_per_env": args.source_samples_per_env,
        "target_train_samples": target_train_samples,
        "target_test_samples": args.target_test_samples,
        "num_features": args.num_features,
        "shared_support_size": args.shared_support_size,
        "source_specific_support_size": args.source_specific_support_size,
        "target_specific_support_size": args.target_specific_support_size,
        "noise_std": args.noise_std,
        "full_target_oracle_mse": full_target_oracle_mse,
        "shared_support": dataset.shared_support,
        "target_specific_support": dataset.target_train.support_specific,
        "pseudo_lasso_invariant_support": pseudo_lasso_fit.invariant_support,
        "pseudo_lasso_invariant_overlap_with_shared": invariant_overlap,
        "pseudo_lasso_support_recovery": support_recovery,
        "mixed_estimator_alpha": mixed_fit.alpha,
        "mixed_estimator_alpha_analytic": mixed_analytic_fit.alpha,
        "mixed_alpha_cv_effective_train_size": mixed_cv.effective_train_size,
        "mixed_alpha_selection": "target_subset_cv",
        "mixed_alpha_cv_folds": min(args.lasso_cv_folds, dataset.target_train.num_samples),
        "mixed_alpha_cv_rows": mixed_cv.rows,
        "mixed_estimator_a_hat": mixed_fit.a_hat,
        "mixed_estimator_b_hat": mixed_fit.b_hat,
        "mixed_estimator_rho2_hat": mixed_fit.rho2_hat,
        "pseudo_lasso_lambda": pseudo_lasso_fit.lambda_lasso,
        "method_results": method_rows,
        "screening_details": pseudo_lasso_fit.screening_details,
    }
    return results


def _generate_draw_dataset(args, *, seed: int, max_target_train_samples: int) -> SyntheticDataset:
    return generate_dataset(
        seed=seed,
        num_source_envs=args.num_source_envs,
        source_samples_per_env=args.source_samples_per_env,
        target_train_samples=max_target_train_samples,
        target_test_samples=args.target_test_samples,
        num_features=args.num_features,
        shared_support_size=args.shared_support_size,
        source_specific_support_size=args.source_specific_support_size,
        target_specific_support_size=args.target_specific_support_size,
        coefficient_low=args.coefficient_low,
        coefficient_high=args.coefficient_high,
        noise_std=args.noise_std,
    )


def _full_target_oracle_mse(args, dataset: SyntheticDataset, *, draw_seed: int) -> float:
    """Fit OLS on independent target data that no reported estimator can use."""
    rng = np.random.default_rng(np.random.SeedSequence([draw_seed, 91_117]))
    x = rng.normal(size=(args.oracle_target_samples, args.num_features)).astype(np.float64)
    y = x @ dataset.target_train.beta_star + rng.normal(
        scale=args.noise_std,
        size=args.oracle_target_samples,
    )
    fit = fit_ols(x, y)
    return mse(dataset.target_test.x, dataset.target_test.y, fit.beta)


def _trapezoid(values: list[float], budgets: list[int]) -> float:
    """Integrate over log target budget, not the raw sample count."""
    x_values = np.log(np.asarray(budgets, dtype=np.float64))
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(np.asarray(values, dtype=np.float64), x=x_values))


def _log_budget_average(values: list[float], budgets: list[int]) -> float:
    """Return the area under a curve divided by its log-budget range."""
    x_values = np.log(np.asarray(budgets, dtype=np.float64))
    if len(x_values) < 2 or x_values[-1] <= x_values[0]:
        return float("nan")
    return _trapezoid(values, budgets) / float(x_values[-1] - x_values[0])


def filter_reporting_budgets(budgets, gains, full_oracle_gains, *, integration_min_budget=None):
    """Restrict all reported curves and score calculations to a budget suffix."""
    keep = [index for index, budget in enumerate(budgets)
            if integration_min_budget is None or budget >= integration_min_budget]
    if len(keep) < 2:
        raise ValueError("Reporting requires at least two budgets after --integration-min-budget filtering.")
    return (
        [budgets[index] for index in keep],
        {method: [values[index] for index in keep] for method, values in gains.items()},
        [full_oracle_gains[index] for index in keep],
        keep,
    )


def build_oracle_score_rows(
    budgets: list[int],
    gains: dict[str, list[list[float]]],
    full_oracle_gains: list[list[float]],
    *,
    num_draws: int,
    integration_min_budget: int | None = None,
) -> tuple[list[dict[str, object]], list[float], list[float]]:
    """Summarize method gains and full-target OLS headroom from saved draws."""
    plot_oracle_mean = [float(np.mean(values)) for values in full_oracle_gains]
    plot_oracle_median = [float(np.median(values)) for values in full_oracle_gains]
    budgets, gains, full_oracle_gains, _ = filter_reporting_budgets(
        budgets, gains, full_oracle_gains, integration_min_budget=integration_min_budget
    )
    oracle_mean = [float(np.mean(values)) for values in full_oracle_gains]
    oracle_median = [float(np.median(values)) for values in full_oracle_gains]
    oracle_p25 = [float(np.percentile(values, 25)) for values in full_oracle_gains]
    mean_headroom = _trapezoid(oracle_mean, budgets)
    median_headroom = _trapezoid(oracle_median, budgets)
    p25_headroom = _trapezoid(oracle_p25, budgets)
    score_rows = []
    for method, method_gains in gains.items():
        mean_curve = [float(np.mean(values)) for values in method_gains]
        median_curve = [float(np.median(values)) for values in method_gains]
        p25_curve = [float(np.percentile(values, 25)) for values in method_gains]
        mean_integral = _trapezoid(mean_curve, budgets)
        median_integral = _trapezoid(median_curve, budgets)
        p25_integral = _trapezoid(p25_curve, budgets)
        score_rows.append(
            {
                "method": method,
                "num_draws": num_draws,
                "budgets": ",".join(map(str, budgets)),
                "integration_x_axis": "log_budget",
                "integration_min_budget": integration_min_budget,
                "full_target_ols_headroom_mean_integral": mean_headroom,
                "full_target_ols_headroom_median_integral": median_headroom,
                "full_target_ols_headroom_p25_integral": p25_headroom,
                "full_target_ols_headroom_mean_log_budget_average": _log_budget_average(oracle_mean, budgets),
                "full_target_ols_headroom_median_log_budget_average": _log_budget_average(oracle_median, budgets),
                "full_target_ols_headroom_p25_log_budget_average": _log_budget_average(oracle_p25, budgets),
                "estimator_mean_gain_integral": mean_integral,
                "estimator_median_gain_integral": median_integral,
                "estimator_p25_gain_integral": p25_integral,
                "mean_integral_fraction_of_mean_headroom": mean_integral / mean_headroom if mean_headroom > 0 else float("nan"),
                "median_integral_fraction_of_median_headroom": median_integral / median_headroom if median_headroom > 0 else float("nan"),
                "p25_integral_fraction_of_p25_headroom": p25_integral / p25_headroom if p25_headroom > 0 else float("nan"),
            }
        )
    return score_rows, plot_oracle_mean, plot_oracle_median


def build_task_first_scores(budgets, gains, full_oracle_gains, *, epsilon=1e-6,
                            integration_min_budget=None):
    """Evaluate every structural draw before aggregating over draws.

    ``gains`` are percentage MSE reductions relative to the budget-specific
    source/target oracle.  Convert them back to the framework's log ratios.
    """
    budgets, gains, full_oracle_gains, _ = filter_reporting_budgets(
        budgets, gains, full_oracle_gains, integration_min_budget=integration_min_budget
    )
    methods = list(gains)
    log_budgets = np.log(np.asarray(budgets, dtype=float))
    budget_range = float(log_budgets[-1] - log_budgets[0])
    task_rows = []
    for draw_index in range(len(full_oracle_gains[0])):
        headroom_curve = np.asarray([
            -np.log1p(-float(full_oracle_gains[index][draw_index]) / 100.0)
            for index in range(len(budgets))
        ])
        integrate = getattr(np, "trapezoid", None) or np.trapz
        headroom = float(integrate(headroom_curve, x=log_budgets) / budget_range)
        for method in methods:
            gain_curve = np.asarray([
                -np.log1p(-float(gains[method][index][draw_index]) / 100.0)
                for index in range(len(budgets))
            ])
            integrated_gain = float(integrate(gain_curve, x=log_budgets) / budget_range)
            task_rows.append({
                "task": draw_index,
                "method": method,
                "integrated_log_gain": integrated_gain,
                "headroom": headroom,
                "score": integrated_gain / headroom if headroom > epsilon else float("nan"),
                "eligible": headroom > epsilon,
                "integration_min_budget": integration_min_budget,
                "integration_budgets": ",".join(map(str, budgets)),
            })
    return task_rows


def save_task_score_reports(output_dir, task_rows):
    """Write the task-first score table and its two requested visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "adaptation_task_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    methods = list(dict.fromkeys(row["method"] for row in task_rows))
    headrooms = [row["headroom"] for row in task_rows if row["method"] == methods[0]]
    eligible = {row["task"] for row in task_rows if row["eligible"]}
    summary_rows = []
    for method in methods:
        scores = [row["score"] for row in task_rows if row["method"] == method and row["task"] in eligible]
        summary_rows.append({
            "method": method,
            "tasks": len(headrooms),
            "eligible_tasks": len(eligible),
            "eligible_fraction": len(eligible) / len(headrooms),
            "headroom_mean": float(np.mean(headrooms)),
            "headroom_median": float(np.median(headrooms)),
            "headroom_p25": float(np.percentile(headrooms, 25)),
            "score_mean": float(np.mean(scores)),
            "score_median": float(np.median(scores)),
            "score_p25": float(np.percentile(scores, 25)),
        })
    summary_path = output_dir / "adaptation_score_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 5))
    values = [[row["score"] for row in task_rows if row["method"] == method and row["task"] in eligible] for method in methods]
    colors = [method_color(method, index) for index, method in enumerate(methods)]
    boxes = ax.boxplot(values, tick_labels=methods, showfliers=False, showmeans=True, patch_artist=True,
                       meanprops={"marker": "D", "markerfacecolor": "#D95F02", "markeredgecolor": "#D95F02"})
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.42)
    jitter_rng = np.random.default_rng(20_000)
    for index, (values_for_method, color) in enumerate(zip(values, colors), start=1):
        jitter = jitter_rng.uniform(-0.10, 0.10, size=len(values_for_method))
        ax.scatter(index + jitter, values_for_method, s=20, color=color, alpha=0.6,
                   edgecolors="none", zorder=3)
    for index, values_for_method in enumerate(values, start=1):
        ax.scatter(index, np.percentile(values_for_method, 25), marker="v", color="#5c3c92", zorder=4, label="p25" if index == 1 else None)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Task-level normalized adaptation score")
    ax.set_title("Distribution of task-level adaptation scores")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(handles=[
        Line2D([], [], color="#D95F02", marker="D", linestyle="None", label="Mean"),
        Line2D([], [], color="black", linewidth=1.7, label="Median"),
        Line2D([], [], color="#5C3C92", marker="v", linestyle="None", label="25th percentile"),
    ])
    fig.tight_layout()
    fig.savefig(output_dir / "adaptation_score_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for index, method in enumerate(methods):
        points = [row for row in task_rows if row["method"] == method]
        ax.scatter([row["headroom"] for row in points], [row["integrated_log_gain"] for row in points],
                   label=method, color=method_color(method, index), alpha=0.75)
    limits = ax.get_xlim()
    ax.plot(limits, limits, "k--", linewidth=1, label="full-target reference")
    ax.set_xlabel("Full-target headroom (integrated log gain)")
    ax.set_ylabel("Adaptation integrated log gain")
    ax.set_title("Adaptation gain versus available headroom")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "adaptation_gain_vs_headroom.png", dpi=180)
    plt.close(fig)
    return score_path, summary_path


def main(args) -> None:
    target_train_sizes = (
        [int(args.target_train_samples)]
        if args.target_train_samples_grid is None
        else sorted({int(x) for x in args.target_train_samples_grid})
    )
    if not target_train_sizes or target_train_sizes[0] <= 0:
        raise ValueError("Target training budgets must be positive.")
    if args.num_draws <= 0 or args.oracle_target_samples <= 0:
        raise ValueError("num_draws and oracle_target_samples must be positive.")

    max_target_train_samples = max(target_train_sizes)
    all_results: list[dict[str, object]] = []
    draw_iterator = tqdm(range(args.num_draws), desc="structural draws", unit="draw")
    for structural_draw in draw_iterator:
        draw_seed = args.seed + structural_draw
        draw_iterator.set_postfix(seed=draw_seed)
        dataset = _generate_draw_dataset(args, seed=draw_seed, max_target_train_samples=max_target_train_samples)
        full_oracle_mse = _full_target_oracle_mse(args, dataset, draw_seed=draw_seed)
        for target_train_samples in target_train_sizes:
            all_results.append(
                run_single_configuration(
                    args,
                    dataset=dataset,
                    target_train_samples=target_train_samples,
                    structural_draw=structural_draw,
                    draw_seed=draw_seed,
                    full_target_oracle_mse=full_oracle_mse,
                )
            )

    methods = ["source_only_ols", "target_only_ols", "mixed_estimator", "pseudo_lasso"]
    budget_rows: list[dict[str, object]] = []
    gains: dict[str, list[list[float]]] = {method: [] for method in methods}
    full_oracle_gains: list[list[float]] = []
    for budget in target_train_sizes:
        budget_results = [result for result in all_results if result["target_train_samples"] == budget]
        method_values = {
            method: [
                next(float(row["target_test_mse"]) for row in result["method_results"] if row["method"] == method)
                for result in budget_results
            ]
            for method in methods
        }
        oracle_best = np.minimum(method_values["source_only_ols"], method_values["target_only_ols"])
        for method in methods:
            gains[method].append(
                (100.0 * (oracle_best - np.asarray(method_values[method])) / oracle_best).tolist()
            )
        oracle_values = np.asarray(
            [float(result["full_target_oracle_mse"]) for result in budget_results]
        )
        full_oracle_gains.append((100.0 * (oracle_best - oracle_values) / oracle_best).tolist())
        budget_rows.append(
            {
                "target_train_samples": budget,
                "methods": {
                    method: {"target_test_mse": float(np.mean(values))}
                    for method, values in method_values.items()
                },
            }
        )

    score_rows, oracle_mean, oracle_median = build_oracle_score_rows(
        target_train_sizes,
        gains,
        full_oracle_gains,
        num_draws=args.num_draws,
        integration_min_budget=args.integration_min_budget,
    )
    task_score_rows = build_task_first_scores(
        target_train_sizes, gains, full_oracle_gains, epsilon=args.headroom_epsilon,
        integration_min_budget=args.integration_min_budget,
    )
    report_budgets, report_gains, report_oracle_gains, report_indices = filter_reporting_budgets(
        target_train_sizes, gains, full_oracle_gains, integration_min_budget=args.integration_min_budget
    )
    report_budget_rows = [budget_rows[index] for index in report_indices]
    report_oracle_mean = [float(np.mean(values)) for values in report_oracle_gains]
    report_oracle_median = [float(np.median(values)) for values in report_oracle_gains]

    output_payload = {
        "seed": args.seed,
        "num_draws": args.num_draws,
        "num_source_envs": args.num_source_envs,
        "source_samples_per_env": args.source_samples_per_env,
        "target_test_samples": args.target_test_samples,
        "oracle_target_samples": args.oracle_target_samples,
        "num_features": args.num_features,
        "shared_support_size": args.shared_support_size,
        "source_specific_support_size": args.source_specific_support_size,
        "target_specific_support_size": args.target_specific_support_size,
        "noise_std": args.noise_std,
        "target_train_samples_grid": target_train_sizes,
        "integration_min_budget": args.integration_min_budget,
        "draw_budget_results": all_results,
        "budget_curve": budget_rows,
        "oracle_relative_gains_percent": {
            "methods": gains,
            "full_target_ols_oracle": full_oracle_gains,
        },
        "oracle_integral_scores": score_rows,
        "task_first_scores": task_score_rows,
    }

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    if args.output_budget_plot is not None:
        budget_plot_path = Path(args.output_budget_plot)
        save_budget_curve_plot(budget_plot_path, budget_rows=report_budget_rows)
        print(f"Saved mean budget plot to {budget_plot_path}")
    if args.output_oracle_boxplot is not None:
        boxplot_path = Path(args.output_oracle_boxplot)
        boxplot_paths = save_oracle_improvement_boxplot(
            boxplot_path,
            budgets=report_budgets,
            gains=report_gains,
            oracle_mean=report_oracle_mean,
            oracle_median=report_oracle_median,
        )
        print("Saved oracle-relative boxplots:")
        for path in boxplot_paths:
            print(f"  {path}")
    if args.output_integral_scores is not None:
        score_path = Path(args.output_integral_scores)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with score_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
            writer.writeheader()
            writer.writerows(score_rows)
        print(f"Saved oracle integral scores to {score_path}")
        task_score_path, summary_path = save_task_score_reports(score_path.parent, task_score_rows)
        print(f"Saved task-level scores to {task_score_path}")
        print(f"Saved task-level score summary to {summary_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
