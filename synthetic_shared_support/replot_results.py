#!/usr/bin/env python3
"""Recreate pseudo-lasso synthetic reports from a completed results JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from synthetic_shared_support.run_experiment import (
    build_oracle_score_rows,
    build_task_first_scores,
    filter_reporting_budgets,
    save_task_score_reports,
    save_budget_curve_plot,
    save_oracle_improvement_boxplot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate pseudo-lasso synthetic plots and scores without rerunning experiments."
    )
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-plot-name", default="budget_curve.png")
    parser.add_argument("--oracle-boxplot-name", default="oracle_improvement_boxplot.png")
    parser.add_argument("--integral-scores-name", default="oracle_integral_scores.csv")
    parser.add_argument("--headroom-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--integration-min-budget",
        type=int,
        default=None,
        help="Use only budgets greater than or equal to this value when computing integral scores.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    with args.results_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    budgets = [int(value) for value in payload["target_train_samples_grid"]]
    gains_payload = payload["oracle_relative_gains_percent"]
    gains = {
        method: [[float(value) for value in values] for values in budget_values]
        for method, budget_values in gains_payload["methods"].items()
    }
    full_oracle_gains = [
        [float(value) for value in values]
        for values in gains_payload["full_target_ols_oracle"]
    ]
    if any(len(values) != len(budgets) for values in gains.values()) or len(full_oracle_gains) != len(budgets):
        raise ValueError("Results JSON has inconsistent budget and gain dimensions.")

    score_rows, oracle_mean, oracle_median = build_oracle_score_rows(
        budgets,
        gains,
        full_oracle_gains,
        num_draws=int(payload.get("num_draws", len(full_oracle_gains[0]) if full_oracle_gains else 0)),
        integration_min_budget=args.integration_min_budget,
    )
    report_budgets, report_gains, report_oracle_gains, report_indices = filter_reporting_budgets(
        budgets, gains, full_oracle_gains, integration_min_budget=args.integration_min_budget
    )
    report_budget_rows = [payload["budget_curve"][index] for index in report_indices]
    report_oracle_mean = [sum(values) / len(values) for values in report_oracle_gains]
    report_oracle_median = [sorted(values)[len(values) // 2] if len(values) % 2 else
                            (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2
                            for values in report_oracle_gains]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget_path = args.output_dir / args.budget_plot_name
    boxplot_path = args.output_dir / args.oracle_boxplot_name
    score_path = args.output_dir / args.integral_scores_name
    save_budget_curve_plot(budget_path, budget_rows=report_budget_rows)
    boxplot_paths = save_oracle_improvement_boxplot(
        boxplot_path,
        budgets=report_budgets,
        gains=report_gains,
        oracle_mean=report_oracle_mean,
        oracle_median=report_oracle_median,
    )
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    task_rows = build_task_first_scores(
        budgets, gains, full_oracle_gains, epsilon=args.headroom_epsilon,
        integration_min_budget=args.integration_min_budget,
    )
    task_score_path, summary_path = save_task_score_reports(args.output_dir, task_rows)
    print(f"Wrote {budget_path}")
    for path in boxplot_paths:
        print(f"Wrote {path}")
    print(f"Wrote {score_path}")
    print(f"Wrote {task_score_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main(parse_args())
