"""Plot saved target-test MSE against labeled target budget; never refit models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from causal_chamber.methods import pseudo_method_info
from causal_chamber.results import load_result_document


LABELS = {
    "source_only_ols": "Source-only OLS",
    "target_only_ols": "Target-only OLS",
    "target_only_lasso": "Target-only lasso",
    "pseudo_lasso": "Pseudo-lasso",
    "pseudo_lasso_centered": "Centered pseudo-lasso",
    "mixed_estimator": "Mixed estimator",
    "mixed_loss_estimator": "Mixed loss estimator",
}
COLORS = {"source_only_ols": "#0072B2", "target_only_ols": "#D55E00", "target_only_lasso": "#333333", "pseudo_lasso": "#009E73", "pseudo_lasso_centered": "#E69F00", "mixed_estimator": "#CC79A7", "mixed_loss_estimator": "#56B4E9"}
STYLES = {"source_only_ols": ("--", "s"), "target_only_ols": ("-", "o"), "target_only_lasso": ("-.", "x"), "pseudo_lasso": ("-.", "^"), "pseudo_lasso_centered": ("-", "v"), "mixed_estimator": (":", "D"), "mixed_loss_estimator": ("--", "P")}
ORACLE_SOURCE_TARGET = "oracle_source_or_target"


def method_label(method):
    if method == ORACLE_SOURCE_TARGET:
        return "Oracle: best of source and target"
    info = pseudo_method_info(method)
    if info and info[1] is not None:
        return f"{LABELS[info[0]]}, K={info[1]}"
    return LABELS.get(method, method)


def method_color(method):
    info = pseudo_method_info(method)
    if info and info[1] is not None:
        palette = {2: "#0072B2", 5: "#D55E00", 10: "#009E73", 15: "#CC79A7", 20: "#E69F00"}
        return palette.get(info[1], COLORS[info[0]])
    return COLORS.get(method)


def method_order(method):
    info = pseudo_method_info(method)
    base = info[0] if info else method
    return (base not in LABELS, list(LABELS).index(base) if base in LABELS else len(LABELS),
            info[1] if info and info[1] is not None else -1, method)


def set_boxplot_distribution_limits(axis, values):
    """Limit a boxplot to its Tukey-whisker range, rather than one extreme dot."""
    bounds = []
    for distribution in values:
        distribution = np.asarray(distribution, dtype=float)
        q1, q3 = np.percentile(distribution, [25, 75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        inliers = distribution[(distribution >= lower) & (distribution <= upper)]
        if len(inliers):
            bounds.extend([float(np.min(inliers)), float(np.max(inliers))])
    if not bounds:
        return
    lower, upper = min(bounds + [0.0]), max(bounds + [0.0])
    padding = max(0.08 * (upper - lower), 1e-3)
    axis.set_ylim(lower - padding, upper + padding)


def aggregate_results(result: dict) -> dict:
    """Each seed contributes once to each target/method/budget mean."""
    groups = defaultdict(list)
    seen = set()
    for row in result["results"]:
        target, method = str(row["target"]), str(row["method"])
        budget = row["target_budget"]
        error = float(row["target_test_mse"])
        if type(budget) is not int or budget < 1 or not math.isfinite(error) or error < 0:
            raise ValueError("Results require positive integer budgets and finite nonnegative test MSE")
        seed = row.get("subset_seed", "fixed")
        key = (target, method, budget)
        if (*key, seed) in seen:
            raise ValueError(f"Duplicate seed result: {key}, seed={seed}")
        seen.add((*key, seed))
        groups[key].append(error)
    if not groups:
        raise ValueError("No saved results to plot")
    curves = defaultdict(lambda: defaultdict(list))
    for (target, method, budget), errors in sorted(groups.items()):
        curves[target][method].append({
            "budget": budget, "mean": float(np.mean(errors)),
            "num_seeds": len(errors),
        })
    return dict(curves)


def with_source_target_oracle(result: dict) -> dict:
    """Add a seed-wise source/target MSE oracle for comparison-only plots.

    For each exact target, budget, and subset seed, the oracle MSE is the
    lower of source-only and target-only OLS. It is intentionally not added to
    the experiment's estimator list, so it cannot appear as a fitted method.
    """
    source_target_rows = {}
    for row in result["results"]:
        method = row.get("method")
        if method not in {"source_only_ols", "target_only_ols"}:
            continue
        key = (
            str(row["target"]),
            int(row["target_budget"]),
            row.get("subset_seed", "fixed"),
        )
        if key in source_target_rows and method in source_target_rows[key]:
            raise ValueError(f"Duplicate oracle baseline record: {key}, method={method}")
        source_target_rows.setdefault(key, {})[method] = row

    oracle_rows = []
    for key, baselines in source_target_rows.items():
        source = baselines.get("source_only_ols")
        target = baselines.get("target_only_ols")
        if source is None or target is None:
            continue
        source_mse = float(source["target_test_mse"])
        target_mse = float(target["target_test_mse"])
        if not (math.isfinite(source_mse) and math.isfinite(target_mse)):
            continue
        oracle_rows.append({
            "target": key[0],
            "method": ORACLE_SOURCE_TARGET,
            "subset_seed": key[2],
            "target_budget": key[1],
            "target_test_mse": min(source_mse, target_mse),
        })
    return {**result, "results": [*result["results"], *oracle_rows]}


def noise_level_headroom(result: dict) -> dict[tuple[str, int], float]:
    """Return the seed-wise full-target OLS headroom for each target/budget.

    The reference is target-only OLS at that target's largest completed budget.
    At every lower budget, the comparison baseline remains the existing oracle
    ``min(source OLS, target OLS)``.  Both are calculated per seed before
    aggregation, matching the oracle used in the box plots.
    """
    rows = defaultdict(dict)
    for row in result["results"]:
        method = row.get("method")
        if method not in {"source_only_ols", "target_only_ols"}:
            continue
        key = (str(row["target"]), int(row["target_budget"]), row.get("subset_seed", "fixed"))
        if method in rows[key]:
            raise ValueError(f"Duplicate noise-level baseline record: {key}, method={method}")
        rows[key][method] = float(row["target_test_mse"])

    target_seed_budgets = defaultdict(list)
    for (target, budget, seed), methods in rows.items():
        if "target_only_ols" in methods:
            target_seed_budgets[(target, seed)].append(budget)
    full_target = {
        key: rows[(key[0], max(budgets), key[1])]["target_only_ols"]
        for key, budgets in target_seed_budgets.items()
    }

    values = defaultdict(list)
    for (target, budget, seed), methods in rows.items():
        if "source_only_ols" not in methods or "target_only_ols" not in methods:
            continue
        reference = full_target.get((target, seed))
        oracle = min(methods["source_only_ols"], methods["target_only_ols"])
        if reference is None or not (math.isfinite(reference) and math.isfinite(oracle)) or oracle <= 0:
            continue
        values[(target, budget)].append((oracle, reference))

    return {
        key: float(100 * (np.mean([oracle for oracle, _ in pairs]) - np.mean([reference for _, reference in pairs]))
                   / np.mean([oracle for oracle, _ in pairs]))
        for key, pairs in values.items()
    }


def filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"
    if cleaned != value:
        cleaned += "_" + hashlib.sha256(value.encode()).hexdigest()[:8]
    return cleaned


def combine_experiments(directories: list[Path]) -> dict:
    """Join distinct estimator results only when their task and data agree."""
    paths = []
    for directory in directories:
        direct = directory / "results.json"
        found = [direct] if direct.is_file() else sorted(directory.glob("*/results.json"))
        if not found:
            raise ValueError(f"No method results found in {directory}")
        paths.extend(found)
    documents = [load_result_document(path) for path in paths]
    inputs = [{"path": str(path), "status": document.get("status", "complete"),
               "methods": document.get("methods", sorted({row["method"] for row in document["results"]})),
               "num_rows": len(document["results"]),
               "num_targets": len({row["target"] for row in document["results"]})}
              for path, document in zip(paths, documents)]
    first = documents[0]
    if len(documents) == 1:
        if first['results']:
            aggregate_results(first)
        return {**first, "input_files": inputs}
    required = ("manifest_sha256", "response", "features", "fit_intercept")
    optional = ("excluded_source_environments", "source_environments_by_target", "subset_seeds", "target_budget_pools")
    for directory, document in zip(paths, documents):
        for key in required:
            if key not in document or document[key] != first.get(key):
                raise ValueError(f"Cannot combine {directory}: {key} is missing or differs")
        for key in optional:
            if key in first and key in document and document[key] != first[key]:
                raise ValueError(f"Cannot combine {directory}: {key} differs")
    combined = {**first, "results": [row for document in documents for row in document["results"]]}
    combined['input_files'] = inputs
    combined["methods"] = sorted({row["method"] for row in combined["results"]})
    combined['status'] = 'partial' if any(doc.get('status','complete') != 'complete' for doc in documents) else 'complete'
    for key in ("source_screens", "mixed_source_means"):
        combined[key] = {target: value for document in documents for target, value in document.get(key, {}).items()}
    # Also detect duplicate estimator/seed records instead of double-counting runs.
    if combined['results']:
        aggregate_results(combined)
    return combined


def report_coverage(result, methods=None):
    """Report actual loaded records, independently of comparison filtering."""
    print('Loaded result files:')
    for entry in result.get('input_files', []):
        print(f"  {entry['path']}: status={entry['status']}, "
              f"{entry['num_targets']} environments, {entry['num_rows']} rows")
    groups = defaultdict(lambda: defaultdict(set))
    for row in result['results']:
        if methods is None or row['method'] in methods:
            groups[(row['method'],row['target_budget'])][row['target']].add(row.get('subset_seed','fixed'))
    print('Saved coverage by method and budget (before plotting filters):')
    for (method,budget), targets in sorted(groups.items(),key=lambda item: (method_order(item[0][0]),item[0][1])):
        counts=[len(seeds) for seeds in targets.values()]
        print(f"  {method}, budget={budget}: {len(targets)} environments, seeds per environment={min(counts)}–{max(counts)}")
    if not groups:
        print('  No completed fits for the requested methods.')


def mean_environment_curves(curves: dict, targets: list[str], methods: list[str]) -> dict:
    """Arithmetic mean of seed means, using the same cohort at every budget."""
    budget_sets = []
    for target in targets:
        for method in methods:
            if method not in curves[target]:
                raise ValueError(f"Method {method} missing for {target}")
            budget_sets.append({point["budget"] for point in curves[target][method]})
    common_budgets = sorted(set.intersection(*budget_sets))
    if not common_budgets:
        raise ValueError("No common target budgets across environments and estimators for summary plots")
    summary = {}
    for method in methods:
        values = {target: {point["budget"]: point["mean"] for point in curves[target][method]} for target in targets}
        summary[method] = [{"budget": budget, "mean": float(np.mean([values[target][budget] for target in targets]))}
                           for budget in common_budgets]
    return summary


def worst_source_targets(curves: dict, targets: list[str]) -> list[str]:
    """Fix the cohort once, ranking source-only seed-mean test MSE descending."""
    scores = {}
    for target in targets:
        points = curves[target].get("source_only_ols")
        if not points:
            raise ValueError(f"Source-only OLS results needed to rank worst environments: {target}")
        values = [point["mean"] for point in points]
        if not np.allclose(values, values[0], rtol=1e-8, atol=1e-12):
            raise ValueError(f"Source-only OLS MSE changes across budgets for {target}; cannot rank a fixed baseline")
        scores[target] = values[0]
    count = max(1, math.ceil(.1 * len(targets)))
    return sorted(targets, key=lambda target: (-scores[target], target))[:count]


def improvement_distribution(curves: dict, targets: list[str], method: str, baseline: str) -> list[dict]:
    """Use every available estimator/baseline environment pair at each budget."""
    values = {target: {name: {p["budget"]: p["mean"] for p in curves[target][name]}
                       for name in (method, baseline)} for target in targets
              if method in curves[target] and baseline in curves[target]}
    budgets = sorted({budget for value in values.values() for budget in set(value[method]) & set(value[baseline])})
    if not budgets:
        return []
    rows = []
    for budget in budgets:
        available = [target for target in targets if target in values and
                     budget in values[target][method] and budget in values[target][baseline]]
        cohort = [target for target in available if values[target][baseline][budget] > 0]
        if not cohort:
            continue
        base = np.array([values[target][baseline][budget] for target in cohort])
        candidate = np.array([values[target][method][budget] for target in cohort])
        absolute = base - candidate
        percent = 100 * absolute / base
        rows.append({"budget": budget, "targets": cohort,
                     "excluded_targets": [t for t in available if t not in cohort],
                     "pending_targets": [t for t in targets if t not in available],
                     "percent": percent, "absolute": absolute, "mean_percent": float(np.mean(percent)),
                     "median_percent": float(np.median(percent)), "wins": int(np.sum(absolute > 0)),
                     "aggregate_reduction_percent": float(100*np.sum(absolute)/np.sum(base))})
    return rows


def write_oracle_integral_scores(curves: dict, targets: list[str], methods: list[str], output_dir: Path, *,
                                 pooled_tasks: bool, raw_result: dict | None = None) -> Path | None:
    """Score relative-MSE-reduction curves against full-target OLS headroom.

    At each budget, the numerator curve is the estimator's MSE reduction
    against the source/target oracle. The denominator curve is the reduction
    achieved by target-only OLS at the largest available budget, against that
    same budget-specific oracle. The reported score is their integral ratio.
    A negative score is allowed: it means an estimator is, on average, worse
    than the source/target oracle over the requested budget range.
    """
    adaptive = [method for method in methods if pseudo_method_info(method) or method in {
        "mixed_estimator", "mixed_loss_estimator"
    }]
    headroom = noise_level_headroom(raw_result) if raw_result is not None else {}
    if not headroom:
        return None
    rows = []
    for method in adaptive:
        cohort = [target for target in targets if method in curves[target] and ORACLE_SOURCE_TARGET in curves[target]]
        if not cohort:
            continue
        budget_sets = []
        for target in cohort:
            estimator_budgets = {point["budget"] for point in curves[target][method]}
            oracle_budgets = {point["budget"] for point in curves[target][ORACLE_SOURCE_TARGET]}
            headroom_budgets = {budget for candidate, budget in headroom if candidate == target}
            budget_sets.append(estimator_budgets & oracle_budgets & headroom_budgets)
        budgets = sorted(set.intersection(*budget_sets)) if budget_sets else []
        if len(budgets) < 2:
            print(f"Skipping integral score for {method}: fewer than two shared budgets", flush=True)
            continue

        oracle_by_target = {
            target: {point["budget"]: point["mean"] for point in curves[target][ORACLE_SOURCE_TARGET]}
            for target in cohort
        }
        estimator_by_target = {
            target: {point["budget"]: point["mean"] for point in curves[target][method]}
            for target in cohort
        }
        # Score trajectories over log budget so equal multiplicative changes
        # receive equal weight (for example, 10 -> 20 equals 100 -> 200).
        x = np.log(np.asarray(budgets, dtype=float))
        headroom_mean_curve = np.asarray([
            np.mean([headroom[(target, budget)] for target in cohort]) for budget in budgets
        ])
        headroom_median_curve = np.asarray([
            np.median([headroom[(target, budget)] for target in cohort]) for budget in budgets
        ])
        headroom_p25_curve = np.asarray([
            np.percentile([headroom[(target, budget)] for target in cohort], 25) for budget in budgets
        ])
        estimator_gains = {
            (target, budget): 100 * (oracle_by_target[target][budget] - estimator_by_target[target][budget])
            / oracle_by_target[target][budget]
            for target in cohort for budget in budgets
        }
        estimator_mean_curve = np.asarray([
            np.mean([estimator_gains[(target, budget)] for target in cohort]) for budget in budgets
        ])
        estimator_median_curve = np.asarray([
            np.median([estimator_gains[(target, budget)] for target in cohort]) for budget in budgets
        ])
        estimator_p25_curve = np.asarray([
            np.percentile([estimator_gains[(target, budget)] for target in cohort], 25) for budget in budgets
        ])
        headroom_mean_integral = float(np.sum(
            np.diff(x) * (headroom_mean_curve[:-1] + headroom_mean_curve[1:]) / 2
        ))
        headroom_median_integral = float(np.sum(
            np.diff(x) * (headroom_median_curve[:-1] + headroom_median_curve[1:]) / 2
        ))
        headroom_p25_integral = float(np.sum(
            np.diff(x) * (headroom_p25_curve[:-1] + headroom_p25_curve[1:]) / 2
        ))
        mean_integral = float(np.sum(np.diff(x) * (estimator_mean_curve[:-1] + estimator_mean_curve[1:]) / 2))
        median_integral = float(np.sum(np.diff(x) * (estimator_median_curve[:-1] + estimator_median_curve[1:]) / 2))
        p25_integral = float(np.sum(np.diff(x) * (estimator_p25_curve[:-1] + estimator_p25_curve[1:]) / 2))
        if (
            not math.isfinite(headroom_mean_integral) or headroom_mean_integral <= 0
            or not math.isfinite(headroom_median_integral) or headroom_median_integral <= 0
        ):
            print(f"Skipping integral score for {method}: nonpositive full-target OLS headroom integral", flush=True)
            continue
        rows.append({
            "method": method,
            "cohort_size": len(cohort),
            "cohort_unit": "environment_task" if pooled_tasks else "environment",
            "budgets": ",".join(map(str, budgets)),
            "integration_x_axis": "log_budget",
            "full_target_ols_headroom_mean_integral": headroom_mean_integral,
            "full_target_ols_headroom_median_integral": headroom_median_integral,
            "full_target_ols_headroom_p25_integral": headroom_p25_integral,
            "full_target_ols_headroom_mean_log_budget_average": headroom_mean_integral / (x[-1] - x[0]),
            "full_target_ols_headroom_median_log_budget_average": headroom_median_integral / (x[-1] - x[0]),
            "full_target_ols_headroom_p25_log_budget_average": headroom_p25_integral / (x[-1] - x[0]),
            "estimator_mean_gain_integral": mean_integral,
            "estimator_median_gain_integral": median_integral,
            "estimator_p25_gain_integral": p25_integral,
            "mean_integral_fraction_of_mean_headroom": mean_integral / headroom_mean_integral,
            "median_integral_fraction_of_median_headroom": median_integral / headroom_median_integral,
            "p25_integral_fraction_of_p25_headroom": p25_integral / headroom_p25_integral if headroom_p25_integral > 0 else float("nan"),
        })

    if not rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "oracle_integral_scores.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['method']} fraction of full-target OLS headroom: "
            f"mean_score={row['mean_integral_fraction_of_mean_headroom']:.6f} "
            f"median_score={row['median_integral_fraction_of_median_headroom']:.6f}",
            flush=True,
        )
    print(f"Wrote {destination}", flush=True)
    return destination


def task_first_eligibility(curves, targets, methods, *, epsilon=1e-6,
                           integration_min_budget=None, max_task_score=None):
    """Return task-level gains and the common eligible cohort for every plot."""
    task_values = {}
    for target in targets:
        available = curves.get(target, {})
        if ORACLE_SOURCE_TARGET not in available or "target_only_ols" not in available:
            continue
        oracle = {point["budget"]: point["mean"] for point in available[ORACLE_SOURCE_TARGET]}
        target_only = {point["budget"]: point["mean"] for point in available["target_only_ols"]}
        finite_budgets = sorted(set(oracle) & set(target_only))
        if integration_min_budget is not None:
            finite_budgets = [budget for budget in finite_budgets if budget >= integration_min_budget]
        if len(finite_budgets) < 2:
            continue
        full_target = target_only[max(finite_budgets)]
        x = np.log(np.asarray(finite_budgets, dtype=float))
        if full_target <= 0:
            continue
        headroom_curve = [math.log(oracle[budget] / full_target) for budget in finite_budgets]
        integrate = getattr(np, "trapezoid", None) or np.trapz
        headroom = float(integrate(headroom_curve, x=x) / (x[-1] - x[0]))
        integrated = {}
        for method in methods:
            values = {point["budget"]: point["mean"] for point in available.get(method, [])}
            budgets = sorted(set(finite_budgets) & set(values))
            if len(budgets) != len(finite_budgets) or any(values[budget] <= 0 for budget in budgets):
                continue
            integrated[method] = float(integrate(
                [math.log(oracle[budget] / values[budget]) for budget in budgets], x=x
            ) / (x[-1] - x[0]))
        task_values[target] = (headroom, integrated)

    if max_task_score is not None and max_task_score <= 0:
        raise ValueError("max_task_score must be positive when provided")
    headroom_eligible_tasks = [
        target for target, (headroom, integrated) in task_values.items()
        if headroom > epsilon and all(method in integrated for method in methods)
    ]
    score_cap_excluded = {
        target for target in headroom_eligible_tasks
        if any(integrated[method] / headroom > max_task_score
               for method in methods)
    } if max_task_score is not None else set()
    complete_tasks = [target for target in headroom_eligible_tasks if target not in score_cap_excluded]
    return task_values, set(complete_tasks), score_cap_excluded


def write_task_first_score_reports(
    curves, targets, methods, output_dir: Path, *, title: str, epsilon=1e-6,
    integration_min_budget=None, max_task_score=None,
):
    """Implement the task-first log-gain framework for causal-chamber data."""
    task_values, complete_tasks, score_cap_excluded = task_first_eligibility(
        curves, targets, methods, epsilon=epsilon,
        integration_min_budget=integration_min_budget, max_task_score=max_task_score,
    )
    # The score summaries must describe the same eligible cohort as the score
    # distributions and improvement plots. Keep the all-task values only as
    # explicitly named diagnostics for auditing exclusions.
    all_headrooms = [headroom for headroom, _ in task_values.values()]
    eligible_headrooms = [task_values[target][0] for target in complete_tasks]
    task_rows = []
    for target, (headroom, integrated) in task_values.items():
        for method, gain in integrated.items():
            raw_score = gain / headroom if headroom != 0 else float("nan")
            if target in complete_tasks:
                exclusion_reason = "eligible"
            elif target in score_cap_excluded:
                exclusion_reason = "score_above_cap"
            elif headroom <= epsilon:
                exclusion_reason = "headroom_at_or_below_epsilon"
            else:
                exclusion_reason = "incomplete_method_coverage"
            task_rows.append({
                "task": target, "method": method,
                "integrated_log_gain": gain, "headroom": headroom,
                "integration_min_budget": integration_min_budget,
                "max_task_score": max_task_score,
                "eligible": target in complete_tasks,
                "exclusion_reason": exclusion_reason,
                "raw_score": raw_score,
                "score": raw_score if target in complete_tasks else float("nan"),
            })
    if not task_rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "adaptation_task_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader(); writer.writerows(task_rows)
    summary_rows = []
    for method in methods:
        scores = [row["score"] for row in task_rows if row["method"] == method and row["eligible"]]
        if not scores:
            continue
        summary_rows.append({
            "method": method, "tasks": len(all_headrooms), "eligible_tasks": len(complete_tasks),
            "integration_min_budget": integration_min_budget,
            "max_task_score": max_task_score,
            "score_cap_excluded_tasks": len(score_cap_excluded),
            "eligible_fraction": len(complete_tasks) / len(all_headrooms),
            "headroom_mean": float(np.mean(eligible_headrooms)),
            "headroom_median": float(np.median(eligible_headrooms)),
            "headroom_p25": float(np.percentile(eligible_headrooms, 25)),
            "all_tasks_headroom_mean": float(np.mean(all_headrooms)),
            "all_tasks_headroom_median": float(np.median(all_headrooms)),
            "all_tasks_headroom_p25": float(np.percentile(all_headrooms, 25)),
            "score_mean": float(np.mean(scores)), "score_median": float(np.median(scores)),
            "score_p25": float(np.percentile(scores, 25)),
        })
    summary_fields = [
        "method", "tasks", "eligible_tasks", "integration_min_budget", "max_task_score",
        "score_cap_excluded_tasks", "eligible_fraction", "headroom_mean", "headroom_median",
        "headroom_p25", "all_tasks_headroom_mean", "all_tasks_headroom_median",
        "all_tasks_headroom_p25", "score_mean", "score_median", "score_p25",
    ]
    with (output_dir / "adaptation_score_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader(); writer.writerows(summary_rows)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # The reporting cohort is applied consistently to all generated figures.
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(30, max(8, int(np.sqrt(len(eligible_headrooms)))))
    ax.hist(eligible_headrooms, bins=bins, color="#7570B3", alpha=0.72, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.2, label="No available headroom")
    ax.axvline(np.mean(eligible_headrooms), color="#D95F02", linestyle="--", linewidth=1.8, label="Mean")
    ax.axvline(np.median(eligible_headrooms), color="#1F1F1F", linestyle="-", linewidth=1.8, label="Median")
    ax.axvline(np.percentile(eligible_headrooms, 25), color="#5C3C92", linestyle=":", linewidth=2, label="25th percentile")
    ax.set_xlabel("Available headroom (integrated log gain)")
    ax.set_ylabel("Number of environment-task pairs")
    ax.set_title(f"Available adaptation headroom, eligible tasks only: {title}")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "available_headroom_distribution.png", dpi=180)
    plt.close(fig)

    baseline_methods = {"source_only_ols", "target_only_ols", "target_only_lasso"}

    def score_boxplot(items, destination, heading):
        if not items:
            return
        methods_for_plot = [item["method"] for item in items]
        labels = [method_label(method) for method in methods_for_plot]
        values = [[row["score"] for row in task_rows if row["method"] == method and row["eligible"]]
                  for method in methods_for_plot]
        fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(labels)), 5))
        boxes = ax.boxplot(
            values, tick_labels=labels, showfliers=False, showmeans=True, patch_artist=True,
            medianprops={"color": "#1f1f1f", "linewidth": 1.8},
            meanprops={"marker": "D", "markerfacecolor": "#d95f02", "markeredgecolor": "white", "markersize": 6},
        )
        for box, method in zip(boxes["boxes"], methods_for_plot):
            box.set_facecolor(method_color(method)); box.set_alpha(0.42)
        for median in boxes["medians"]:
            median.set_zorder(5)
        for mean in boxes["means"]:
            mean.set_zorder(6)
        jitter_rng = np.random.default_rng(40_000)
        for index, (values_for_method, method) in enumerate(zip(values, methods_for_plot), 1):
            jitter = jitter_rng.uniform(-0.10, 0.10, size=len(values_for_method))
            ax.scatter(index + jitter, values_for_method, s=20, color=method_color(method),
                       alpha=0.60, edgecolors="none", zorder=3)
            ax.scatter(index, np.percentile(values_for_method, 25), marker="v", color="#5c3c92",
                       zorder=7, label="p25" if index == 1 else None)
        legend_handles = [
            Line2D([0], [0], marker="D", linestyle="None", markerfacecolor="#d95f02", markeredgecolor="white", markersize=7, label="mean"),
            Line2D([0], [0], color="#1f1f1f", linewidth=1.8, label="median"),
            Line2D([0], [0], marker="v", linestyle="None", color="#5c3c92", markersize=7, label="p25"),
        ]
        set_boxplot_distribution_limits(ax, values)
        ax.axhline(0, color="black", linewidth=1); ax.set_ylabel("Task-level normalized adaptation score")
        ax.set_title(f"{heading}: {title}"); ax.tick_params(axis="x", rotation=25); ax.legend(handles=legend_handles); fig.tight_layout()
        fig.savefig(output_dir / destination, dpi=180); plt.close(fig)

    baseline_rows = [row for row in summary_rows if row["method"] in baseline_methods]
    adaptation_rows = [row for row in summary_rows if row["method"] not in baseline_methods]
    score_boxplot(baseline_rows, "adaptation_score_distribution_baselines.png", "Baseline task-level scores")
    score_boxplot(adaptation_rows, "adaptation_score_distribution_adaptation_methods.png", "Adaptation task-level scores")
    fig, ax = plt.subplots(figsize=(7, 5))
    gain_headroom_methods = [
        method for method in ("mixed_loss_estimator", "pseudo_lasso_centered_ranked_k10")
        if any(row["method"] == method for row in adaptation_rows)
    ]
    # This diagnostic benefits from stronger separation than the shared plot palette.
    headroom_colors = {
        "mixed_loss_estimator": "#D81B60",
        "pseudo_lasso_centered_ranked_k10": "#00897B",
    }
    for method in gain_headroom_methods:
        # Keep this diagnostic on the exact same task universe as the score
        # distributions and summaries.  Ineligible near-zero-headroom tasks
        # should not reappear merely because this is a scatter plot.
        points = [
            row for row in task_rows
            if row["method"] == method and row["eligible"]
        ]
        x_values = np.asarray([row["headroom"] for row in points], dtype=float)
        y_values = np.asarray([row["integrated_log_gain"] for row in points], dtype=float)
        color = headroom_colors.get(method, method_color(method))
        ax.scatter(x_values, y_values,
                   label=method_label(method), color=color, alpha=.45)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_fit, y_fit = x_values[valid], y_values[valid]
        if len(x_fit) >= 2 and np.ptp(x_fit) > 0:
            slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
            fitted = slope * x_fit + intercept
            total_variation = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r_squared = (
                1.0 - np.sum((y_fit - fitted) ** 2) / total_variation
                if total_variation > 0 else float("nan")
            )
            line_x = np.array([np.min(x_fit), np.max(x_fit)])
            fit_label = f"{method_label(method)} fit"
            if np.isfinite(r_squared):
                fit_label += f" ($R^2={r_squared:.2f}$)"
            ax.plot(
                line_x,
                slope * line_x + intercept,
                color=color,
                linewidth=1.8,
                label=fit_label,
            )
    ax.axhline(0, color="#404040", linestyle=":", linewidth=1.1, label="zero gain")
    limits = ax.get_xlim(); ax.plot(limits, limits, "k--", linewidth=1, label="full-target reference")
    ax.set_xlabel("Full-target headroom (integrated log gain)"); ax.set_ylabel("Adaptation integrated log gain")
    ax.set_title(f"Adaptation gain versus headroom: {title}"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(output_dir / "adaptation_gain_vs_headroom.png", dpi=180); plt.close(fig)
    return output_dir / "adaptation_score_summary.csv"


def draw_improvement_distributions(curves, targets, methods, response, output_dir, *,
                                   baselines=("source_only_ols", "target_only_ols", "target_only_lasso", ORACLE_SOURCE_TARGET),
                                   output_prefix="improvement_distribution", pooled_tasks=False,
                                   raw_result=None, min_budget=None) -> list[Path]:
    """Show per-environment relative gains and win counts for each budget."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    adaptive = [method for method in methods if pseudo_method_info(method) or method in {"mixed_estimator", "mixed_loss_estimator"}]
    saved = []
    for baseline in baselines:
        baseline_targets = [target for target in targets if baseline in curves[target]]
        if not adaptive or not baseline_targets:
            continue
        comparisons = [(method, improvement_distribution(curves, baseline_targets, method, baseline))
                       for method in adaptive if method != baseline]
        if min_budget is not None:
            comparisons = [(method, [row for row in rows if row["budget"] >= min_budget])
                           for method, rows in comparisons]
        comparisons = [(method, rows) for method, rows in comparisons if rows]
        if not comparisons:
            print(f"Skipping improvement distribution vs {baseline}: no positive baseline MSE")
            continue
        fig, axes = plt.subplots(len(comparisons), 1, figsize=(9, 4.7*len(comparisons)), squeeze=False)
        try:
            headroom = noise_level_headroom(raw_result) if (
                baseline == ORACLE_SOURCE_TARGET and raw_result is not None
            ) else {}
            for row_index, (method, rows) in enumerate(comparisons):
                left = axes[row_index, 0]
                positions = np.arange(1, len(rows)+1)
                color = method_color(method)
                left.boxplot([row["percent"] for row in rows], positions=positions, widths=.5,
                             whis=(10, 90), showfliers=False, showmeans=True,
                             meanprops={"marker": "D", "markerfacecolor": "#D55E00", "markeredgecolor": "#D55E00", "markersize": 5},
                             medianprops={"color": "black", "linewidth": 1.8})
                for position, row in zip(positions, rows):
                    # Deterministic offsets; every dot is a target environment.
                    jitter = np.linspace(-.17, .17, len(row["percent"]))
                    left.scatter(position+jitter, row["percent"], color=color, s=12, alpha=.4)
                if headroom:
                    headroom_means = []
                    headroom_medians = []
                    for row in rows:
                        values = [headroom[(target, row["budget"])] for target in row["targets"]
                                  if (target, row["budget"]) in headroom]
                        headroom_means.append(float(np.mean(values)) if values else float("nan"))
                        headroom_medians.append(float(np.median(values)) if values else float("nan"))
                    valid = np.isfinite(headroom_means)
                    if np.any(valid):
                        left.plot(
                            positions[valid], np.asarray(headroom_means)[valid],
                            color="#6A3D9A", linestyle="--", marker="*", markersize=8,
                            linewidth=1.8, label="Full-target OLS headroom (mean)",
                        )
                    median_valid = np.isfinite(headroom_medians)
                    if np.any(median_valid):
                        left.plot(
                            positions[median_valid], np.asarray(headroom_medians)[median_valid],
                            color="#B15928", linestyle=":", marker="X", markersize=6,
                            linewidth=1.8, label="Full-target OLS headroom (median)",
                        )
                left.axhline(0, color="black", linestyle="--", linewidth=1)
                left.set_yscale("symlog", linthresh=10)
                left.set_xticks(positions)
                left.set_xticklabels([f"{row['budget']}\n{row['wins']}/{len(row['targets'])} wins" for row in rows], fontsize=8)
                left.set_xlabel("Target training-sample budget")
                left.set_ylabel("MSE reduction (%) — positive is better\nSymmetric log scale outside ±10%")
                left.set_title(method_label(method))
                legend_handles = [
                    Line2D([], [], color="black", label="Median"),
                    Line2D([], [], linestyle="none", marker="D", color="#D55E00", label="Mean"),
                ]
                if headroom:
                    legend_handles.append(Line2D([], [], color="#6A3D9A", linestyle="--", marker="*",
                                                 label="Full-target OLS headroom (mean)"))
                    legend_handles.append(Line2D([], [], color="#B15928", linestyle=":", marker="X",
                                                 label="Full-target OLS headroom (median)"))
                left.legend(handles=legend_handles, fontsize=8)
                for row in rows:
                    print(f"{method} vs {baseline}, budget={row['budget']}: {row['wins']}/{len(row['targets'])} wins, "
                          f"mean gain={row['mean_percent']:.2f}%, median gain={row['median_percent']:.2f}%, "
                          f"aggregate MSE reduction={row['aggregate_reduction_percent']:.2f}%")
                    if row["excluded_targets"]:
                        print(f"Excluded zero-baseline environments vs {baseline}, budget={row['budget']}: "
                              f"{', '.join(row['excluded_targets'])}")
                left.grid(alpha=.2)
            fig.tight_layout()
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / f"{output_prefix}_vs_{baseline}.png"
            fig.savefig(destination, dpi=180)
            saved.append(destination)
            # Remove only superseded per-estimator images for comparisons saved above.
            for method, _ in comparisons:
                (output_dir / f"improvement_distribution_{filename(method)}_vs_{baseline}.png").unlink(missing_ok=True)
        finally:
            plt.close(fig)
    return saved


def plot_pooled_distributions(
    directories, output_dir, *, methods=None, targets=None, epsilon=1e-6,
    integration_min_budget=None, pseudo_lasso_methods=None, max_task_score=None,
):
    """Pool normalized MSE gains across tasks, preserving each environment/task pair."""
    by_response = defaultdict(list)
    for directory in directories:
        direct = directory / 'results.json'
        paths = [direct] if direct.is_file() else sorted(directory.glob('*/results.json'))
        if not paths:
            raise ValueError(f'No method results found in {directory}')
        for path in paths:
            document = load_result_document(path)
            by_response[document['response']].append(path.parent)
    pooled = []
    partial = False
    for response, folders in sorted(by_response.items()):
        document = combine_experiments(list(dict.fromkeys(folders)))
        report_coverage(document, methods)
        partial |= document.get('status', 'complete') != 'complete'
        for row in document['results']:
            if targets and row['target'] not in targets:
                continue
            if methods and row['method'] not in methods:
                continue
            # Task identity prevents identical environment names from being merged.
            pooled.append({**row, 'target': json.dumps([response, row['target']])})
    if not pooled:
        print('No completed results available for pooled distributions.')
        return []
    curves = aggregate_results(with_source_target_oracle({'results': pooled}))
    selected = sorted({row['method'] for row in pooled}, key=method_order)
    if pseudo_lasso_methods is not None:
        requested = set(pseudo_lasso_methods)
        available_pseudo = {method for method in selected if pseudo_method_info(method)}
        unknown = requested - available_pseudo
        if unknown:
            raise ValueError(f"Requested pseudo-lasso methods are unavailable: {', '.join(sorted(unknown))}")
        selected = [method for method in selected if not pseudo_method_info(method) or method in requested]
    pseudo = [method for method in selected if pseudo_method_info(method)]
    mixed_methods = [
        method for method in ("mixed_loss_estimator", "mixed_estimator")
        if method in selected
    ]
    _, eligible_targets, _ = task_first_eligibility(
        curves, sorted(curves), selected, epsilon=epsilon,
        integration_min_budget=integration_min_budget, max_task_score=max_task_score,
    )
    plot_targets = sorted(eligible_targets)
    if not plot_targets:
        raise ValueError("No tasks remain after task-level eligibility filtering.")
    label = ', '.join(sorted(by_response)) + (' (partial)' if partial else '')
    saved = []
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove figures made by the former multi-baseline pooled-reporting path.
    for baseline in ('source_only_ols', 'target_only_ols', 'target_only_lasso', 'mixed_estimator', 'mixed_loss_estimator'):
        for path in output_dir.glob(f"*_vs_{baseline}.png"):
            path.unlink()
    (output_dir / "pseudo_lassos_vs_oracle_source_or_target.png").unlink(missing_ok=True)
    for pseudo_method in pseudo:
        saved.extend(draw_improvement_distributions(
            curves, plot_targets, [pseudo_method], label, output_dir,
            baselines=(ORACLE_SOURCE_TARGET,),
            output_prefix=filename(pseudo_method), pooled_tasks=True, raw_result={'results': pooled},
            min_budget=integration_min_budget))
    for mixed in mixed_methods:
        saved.extend(draw_improvement_distributions(
            curves, plot_targets, [mixed], label, output_dir,
            baselines=(ORACLE_SOURCE_TARGET,),
            output_prefix=("mixed_risk" if mixed == "mixed_loss_estimator" else f"mixed_risk_{mixed}"),
            pooled_tasks=True, raw_result={'results': pooled}, min_budget=integration_min_budget))
    write_task_first_score_reports(
        curves, sorted(curves), selected, output_dir,
        title=label, epsilon=epsilon, integration_min_budget=integration_min_budget,
        max_task_score=max_task_score,
    )
    print(f'Saved {len(saved)} pooled distribution figures in {output_dir} using {len(plot_targets)} eligible tasks')
    return saved


def draw_curves(available, methods, title, footer, output_dir, stem, *, x_scale, y_scale, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if y_scale == "log" and any(point["mean"] <= 0 for method in methods for point in available[method]):
        raise ValueError(f"Zero mean MSE for {title}; use --y-scale linear")
    fig, ax = plt.subplots(figsize=(10, 6) if len(methods) > 6 else (8, 5))
    try:
        for method in methods:
            points = available[method]
            budgets = [point["budget"] for point in points]
            means = [point["mean"] for point in points]
            linestyle, marker = STYLES.get(method, ("-", "o"))
            info = pseudo_method_info(method)
            if info and info[1] is not None:
                linestyle, marker = ("--", "o") if info[0] == "pseudo_lasso" else ("-", "v")
            ax.plot(budgets, means, marker=marker, linestyle=linestyle, linewidth=1.8,
                    label=method_label(method), color=method_color(method))
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ticks = sorted({point["budget"] for method in methods for point in available[method]})
        if len(ticks) <= 15:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(tick) for tick in ticks])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if y_scale == "linear":
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Labeled target budget (train + validation)")
        ax.set_ylabel("Mean target test MSE (lower is better)")
        ax.set_title(title)
        ax.grid(alpha=.25, which="both")
        ax.legend(ncol=2 if len(methods) > 6 else 1, fontsize=8 if len(methods) > 6 else 10)
        fig.text(.5, .02, footer, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .045, 1, 1))
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for extension in formats:
            destination = output_dir / f"{stem}.{extension}"
            fig.savefig(destination, dpi=180)
            saved.append(destination)
        return saved
    finally:
        plt.close(fig)


def plot_experiment(experiment_dir: Path, *, formats=("png",), x_scale="log", y_scale="log",
                    targets=None, methods=None, result=None, output_root=None, epsilon=1e-6,
                    integration_min_budget=None, max_task_score=None) -> list[Path]:
    if set(formats) != {"png"}:
        raise ValueError("Plotting exports PNG files only")
    if result is None:
        result = combine_experiments([experiment_dir])
    report_coverage(result, methods)
    if not result['results']:
        print(f'{experiment_dir.name}: no completed fits yet; rerun plotting after the first fit finishes.')
        return []
    curves = aggregate_results(with_source_target_oracle(result))
    response = str(result.get("response", "unknown_response"))
    selected_targets = sorted(curves)
    if targets:
        selected_targets = []
        for requested in targets:
            known_targets = set(curves) | set(result.get('target_budget_pools',{}))
            matches = [target for target in known_targets if target == requested or Path(target).name == requested or Path(target).stem == requested]
            if len(matches) != 1:
                raise ValueError(f"Unknown or ambiguous target {requested!r} in {experiment_dir}")
            if matches[0] not in curves:
                print(f'{matches[0]}: no completed fits yet; skipping.')
            elif matches[0] not in selected_targets:
                selected_targets.append(matches[0])
    if not selected_targets:
        return []
    selected_methods = methods or sorted(
        {
            method
            for target in selected_targets
            for method in curves[target]
            if method != ORACLE_SOURCE_TARGET
        },
        key=method_order,
    )
    _, eligible_targets, _ = task_first_eligibility(
        curves, selected_targets, selected_methods, epsilon=epsilon,
        integration_min_budget=integration_min_budget, max_task_score=max_task_score,
    )
    plot_targets = [target for target in selected_targets if target in eligible_targets]
    if not plot_targets:
        raise ValueError("No tasks remain after task-level eligibility filtering.")
    # Validate and calculate summaries before writing figures.
    partial = result.get('status','complete') != 'complete'
    if partial:
        print('Partial results: means use available seeds; cohorts and curves can change as fits finish.')
    summary_targets = [target for target in plot_targets if all(method in curves[target] for method in selected_methods)]
    all_mean = None
    if summary_targets:
        try:
            all_mean = mean_environment_curves(curves, summary_targets, selected_methods)
        except ValueError:
            if not partial:
                raise
    if all_mean is None:
        print('Skipping comparison summaries: no shared environment/budget coverage yet.')
    worst = None
    worst_mean = None
    if all_mean is not None and all("source_only_ols" in curves[target] for target in summary_targets):
        worst = worst_source_targets(curves, summary_targets)
        worst_mean = mean_environment_curves(curves, worst, selected_methods)
    else:
        print("Skipping worst-10% summary: source_only_ols results are unavailable; combine with the baseline experiment to include it.")
    output_dir = (output_root or experiment_dir / "plots") / filename(response)
    saved = []
    for target in plot_targets:
        available = curves[target]
        target_methods = [method for method in selected_methods if method in available]
        if not target_methods:
            print(f'{target}: no completed fits for the requested methods; skipping.')
            continue
        counts = sorted({point["num_seeds"] for method in target_methods for point in available[method]})
        caps = sorted({(row['requested_invariant_features'],row['num_invariant_features'])
                       for row in result['results'] if row['target']==target and row['method'] in selected_methods
                       and row.get('requested_invariant_features') is not None
                       and row['num_invariant_features'] < row['requested_invariant_features']})
        cap_note = (' | Capped supports: '+', '.join(f'K={k}: |A|={actual}' for k,actual in caps)) if caps else ''
        saved.extend(draw_curves(
            available, target_methods, f"Response: {response} | Target: {target}" + (' | Partial' if partial else ''),
            f"Mean test MSE across available seeds | seed counts: {', '.join(map(str, counts))}{cap_note}", output_dir,
            filename(target[:-4] if target.endswith(".csv") else target), x_scale=x_scale, y_scale=y_scale, formats=formats))
    if all_mean is not None:
        saved.extend(draw_curves(
            all_mean, selected_methods, f"Response: {response} | All {len(summary_targets)} comparable environments" + (' | Partial' if partial else ''),
            "Equal environment weights; available seeds averaged first; shared budgets only", output_dir,
            "summary_all_environments", x_scale=x_scale, y_scale=y_scale, formats=formats))
    if worst_mean is not None:
        saved.extend(draw_curves(
            worst_mean, selected_methods, f"Response: {response} | Worst 10% by source-only MSE ({len(worst)}/{len(summary_targets)})" + (' | Partial cohort' if partial else ''),
            "Fixed environment subset; arithmetic mean of seed-averaged MSE", output_dir,
            "summary_worst_10pct_source", x_scale=x_scale, y_scale=y_scale, formats=formats))
        print(f"Worst-source environments ({len(worst)}): {', '.join(worst)}")
    saved.extend(draw_improvement_distributions(
        curves, plot_targets, selected_methods, response + (' (partial)' if partial else ''), output_dir,
        raw_result=result,
    ))
    write_task_first_score_reports(
        curves, selected_targets, selected_methods, output_dir,
        title=response, epsilon=epsilon, integration_min_budget=integration_min_budget,
        max_task_score=max_task_score,
    )
    pair = ["mixed_loss_estimator", "pseudo_lasso_centered_ranked_k5"]
    if all(method in selected_methods for method in pair):
        saved.extend(draw_improvement_distributions(
            curves, plot_targets, pair, response + (' (partial)' if partial else ''),
            output_dir, baselines=(pair[1],)))
    print(f"{experiment_dir.name}: plotted {len(plot_targets)} eligible targets plus summaries for response {response} in {output_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--experiment-dir", type=Path, nargs="+", help="Experiment folders containing per-method folders, or legacy results.json")
    inputs.add_argument("--results-root", type=Path, help="Plot every immediate experiment folder under this parent")
    parser.add_argument("--formats", nargs="+", choices=["png"], default=["png"])
    parser.add_argument("--x-scale", choices=["linear", "log"], default="log")
    parser.add_argument("--y-scale", choices=["linear", "log"], default="log")
    parser.add_argument("--target", nargs="+", help="Default: every saved target")
    parser.add_argument("--methods", nargs="+", help="Default: every saved estimator")
    parser.add_argument(
        "--pseudo-lasso-methods",
        nargs="+",
        help="For --pool-tasks, include only these pseudo-lasso variants while preserving baselines and other estimators.",
    )
    parser.add_argument("--combine", action="store_true", help="Compare methods from compatible experiment folders in the same plots")
    parser.add_argument("--output-dir", type=Path, help="Output parent for combined plots; default: first experiment's plots_combined/")
    parser.add_argument("--coverage-only", action="store_true", help="Print input files and saved environment/seed counts per budget without writing plots")
    parser.add_argument("--pool-tasks", action="store_true",
                        help="Only seven distribution comparisons, pooled over environment/task pairs; test MSE reductions")
    parser.add_argument("--headroom-epsilon", type=float, default=1e-6,
                        help="Minimum task-level log headroom required for normalized scores.")
    parser.add_argument("--integration-min-budget", type=int,
                        help="Exclude smaller budgets from task-first log-budget integration and scores.")
    parser.add_argument("--max-task-score", type=float,
                        help="Exclude a task from every score plot if any selected method has score I/H above this cap.")
    args = parser.parse_args()
    try:
        directories = args.experiment_dir
        if directories is None:
            directories = [path for path in sorted(args.results_root.iterdir()) if path.is_dir()
                           and ((path / "results.json").is_file() or any(path.glob("*/results.json")))]
        if not directories:
            raise ValueError("No experiment folders containing results.json were found")
        directories = list(dict.fromkeys(directories))
        if args.pool_tasks:
            if args.coverage_only:
                for directory in directories:
                    report_coverage(combine_experiments([directory]),args.methods)
                return
            plot_pooled_distributions(directories, args.output_dir or directories[0]/'plots_pooled_tasks',
                                      methods=args.methods, targets=args.target, epsilon=args.headroom_epsilon,
                                      integration_min_budget=args.integration_min_budget,
                                      pseudo_lasso_methods=args.pseudo_lasso_methods,
                                      max_task_score=args.max_task_score)
            return
        if args.combine:
            if args.experiment_dir is None or len(directories) < 2:
                raise ValueError("--combine requires at least two explicit --experiment-dir folders")
            combined = combine_experiments(directories)
            if args.coverage_only:
                report_coverage(combined,args.methods)
                return
            plot_experiment(directories[0], formats=args.formats, x_scale=args.x_scale, y_scale=args.y_scale,
                            targets=args.target, methods=args.methods, result=combined,
                            output_root=args.output_dir or directories[0] / "plots_combined", epsilon=args.headroom_epsilon,
                            integration_min_budget=args.integration_min_budget,
                            max_task_score=args.max_task_score)
            return
        if args.output_dir is not None:
            raise ValueError("--output-dir requires --combine")
        for directory in dict.fromkeys(directories):
            if args.coverage_only:
                report_coverage(combine_experiments([directory]),args.methods)
                continue
            plot_experiment(directory, formats=args.formats, x_scale=args.x_scale, y_scale=args.y_scale,
                            targets=args.target, methods=args.methods, epsilon=args.headroom_epsilon,
                            integration_min_budget=args.integration_min_budget,
                            max_task_score=args.max_task_score)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
