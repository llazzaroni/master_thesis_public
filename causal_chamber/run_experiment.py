"""Evaluate OLS baselines, pseudo-lasso, and mixed estimation across target budgets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import os
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

import numpy as np

from causal_chamber.estimators import fit_ols, fit_target_lasso, fit_pseudo_lasso, fit_centered_pseudo_lasso, fit_mixed_estimator, fit_mixed_loss_estimator, prepare_source_loss, average_source_fits, estimate_source_coefficients, screen_source_estimates, rank_source_features, ranked_source_screen, mse
from causal_chamber.methods import pseudo_method_info, expand_ranked_methods
from causal_chamber.splits import DEFAULT_DATA_DIR, verify_manifest
from causal_chamber.results import load_result_document


METADATA_COLUMNS = {"timestamp", "config", "counter", "flag", "intervention", "camera"}
# The three IR sensors are contemporaneous measurements of the same optical
# process.  Keeping a sibling IR reading as a covariate turns the intended
# cross-environment prediction tasks into sensor reconstruction.
IR_RESPONSE_COLUMNS = {"ir_1", "ir_2", "ir_3"}
EXCLUDED_PREDICTOR_COLUMNS = METADATA_COLUMNS | IR_RESPONSE_COLUMNS


DEFAULT_OUTPUT_ROOT = Path("results/causal_chamber")


def target_budget_pool(splits: dict[str, list[int]], target: str, seed: int) -> list[int]:
    file_seed = int.from_bytes(hashlib.sha256(f"{seed}:{target}".encode()).digest(), "big")
    rng = random.Random(file_seed)
    shuffled = {key: list(splits[key]) for key in ("train", "val")}
    for rows in shuffled.values():
        rng.shuffle(rows)
    return budget_pool(shuffled)


def budget_pool(splits: dict[str, list[int]]) -> list[int]:
    # Interleave the two saved lists proportionally, keeping their internal order.
    # Prefixes are nested and draw labels from both train and validation.
    entries = [( (i + 0.5) / len(splits[key]), k, row)
               for k, key in enumerate(("train", "val"))
               for i, row in enumerate(splits[key])]
    return [row for _, _, row in sorted(entries)]


def resolve_target(value: str, names: list[str]) -> str:
    if value in names:
        return value
    matches = [name for name in names if Path(name).stem == value or Path(name).name == value]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous target {value!r}; use a relative CSV path from inspect_data")
    return matches[0]


def load_environment(path: Path, features: list[str], response: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = features + [response]
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        values = []
        for i, row in enumerate(reader):
            try:
                values.append([float(row[key]) for key in required])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Nonnumeric/missing selected value in {path}, data row {i}") from error
    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Nonfinite selected values in {path}; no rows are silently dropped")
    return matrix[:, :-1], matrix[:, -1]


def result_key(row):
    return row['target'], row['method'], row['subset_seed'], row['target_budget']


def run(args, *, on_start=None, on_result=None) -> dict:
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    verify_manifest(manifest, args.data_dir)
    names = sorted(manifest["environments"])
    if len(names) < 2:
        raise ValueError("Source-only OLS requires at least two environments")
    features = args.features
    if features is None:
        columns = manifest["environments"][names[0]]["columns"]
        features = [
            column for column in columns
            if column != args.response and column not in EXCLUDED_PREDICTOR_COLUMNS
        ]
        if any(set(info["columns"]) != set(columns) for info in manifest["environments"].values()):
            raise ValueError("Automatic feature selection requires matching columns in every environment; specify --features")
    forbidden_features = sorted(set(features) & EXCLUDED_PREDICTOR_COLUMNS)
    if not features or args.response in features or len(set(features)) != len(features) or forbidden_features:
        detail = f"; forbidden predictors: {', '.join(forbidden_features)}" if forbidden_features else ""
        raise ValueError("Predictors must be unique, exclude the response, and exclude metadata and ir_1/ir_2/ir_3" + detail)
    if args.budgets is not None and any(b < 1 for b in args.budgets):
        raise ValueError("Target budgets must be positive integers")
    if any(not 0 < f <= 1 for f in args.budget_fractions):
        raise ValueError("Budget fractions must be in (0, 1]")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Methods must be distinct")
    if {"mixed_estimator", "mixed_loss_estimator"}.intersection(args.methods):
        if args.lasso_cv_folds < 2 or any(not np.isfinite(v) or not 0 <= v <= 1 for v in args.mixed_lambda_grid):
            raise ValueError("Mixed estimation requires at least two CV folds and weights in [0, 1]")
    pseudo_methods = [method for method in args.methods if pseudo_method_info(method)]
    if pseudo_methods or "target_only_lasso" in args.methods:
        if args.lasso_cv_folds < 2 or args.lasso_max_iter < 1 or not np.isfinite(args.lasso_tol) or args.lasso_tol <= 0:
            raise ValueError("Invalid lasso CV or solver settings")
        if any(not np.isfinite(v) or v < 0 for v in (args.screen_z_threshold, args.screen_stability_threshold)):
            raise ValueError("Screening thresholds must be finite and nonnegative")
        candidates = args.lasso_lambda_grid if args.lasso_lambda_grid is not None else args.lasso_lambda_ratios
        if any(not np.isfinite(v) or v <= 0 for v in candidates) or (args.lasso_lambda_grid is None and max(candidates) > 1):
            raise ValueError("Invalid lasso penalty grid")
    if len(set(args.subset_seeds)) != len(args.subset_seeds):
        raise ValueError("Subset seeds must be distinct")
    targets = [resolve_target(args.target, names)] if args.target else names
    excluded_sources = [name for name in names if Path(name).name == "uniform_reference.csv"]
    sources_by_target = {target: [name for name in names if name != target and name not in excluded_sources]
                         for target in targets}
    if any(not sources for sources in sources_by_target.values()):
        raise ValueError("No source environments remain after excluding the target and uniform_reference.csv")
    pools = {name: budget_pool(manifest["environments"][name]["splits"]) for name in names}
    budgets_by_target = {}
    for target in targets:
        n = len(pools[target])
        budgets = sorted(set(args.budgets)) if args.budgets is not None else sorted({max(1, math.ceil(f*n)) for f in args.budget_fractions})
        if max(budgets) > n:
            raise ValueError(f"Target {target} has {n} train+val rows but budget {max(budgets)} was requested")
        budgets_by_target[target] = budgets
    source_needed = pseudo_methods or {"source_only_ols", "mixed_estimator", "mixed_loss_estimator"}.intersection(args.methods)
    needed = set(targets).union(*(set(sources) for sources in sources_by_target.values())) if source_needed else set(targets)
    data = {name: load_environment(args.data_dir / name, features, args.response) for name in names if name in needed}
    print(f"Response: {args.response}; predictors ({len(features)}): {', '.join(features)}", flush=True)
    print(f"Excluded from sources: {', '.join(excluded_sources) or '(none present)'}", flush=True)
    results = []
    source_screens = {}
    source_screens_by_method = {method: {} for method in pseudo_methods}
    mixed_source_means = {}
    seeded_pools = {target: {str(seed): target_budget_pool(manifest["environments"][target]["splits"], target, seed)
                             for seed in args.subset_seeds} for target in targets}
    document = {
        "version": 9, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "target_lasso_definition": "Target-only lasso on all predictors; fold-specific standardization and target-subset CV; no source coefficients",
        "source_screens_by_method": source_screens_by_method,
        "ranking_definition": "abs(mean source coefficient) / sqrt(population coefficient variance + mean squared standard error); ties by predictor order; source-only ranking",
        "pseudo_lasso_definitions": {
            "pseudo_lasso": "Screened mean source coefficients fixed on A; target lasso correction on its complement",
            "pseudo_lasso_centered": "Screened mean source coefficients as center on A (zero elsewhere); target lasso correction on all features; standardized correction penalty",
        },
        "features": features, "response": args.response, "fit_intercept": not args.no_intercept,
        "excluded_source_environments": excluded_sources, "source_environments_by_target": sources_by_target,
        "methods": args.methods, "source_screens": source_screens, "mixed_source_means": mixed_source_means,
        "mixed_loss_estimator_definition": "lambda * pooled source MSE + (1-lambda) * target MSE; target-subset CV; lambda is source weight",
        "mixed_estimator_definition": "lambda * target OLS + (1-lambda) * equal-environment mean source OLS; target-subset CV",
        "source_estimator": "OLS on pooled source train+validation rows",
        "budget_definition": "Total labeled target rows from train+validation; seed-specific shuffling within splits, nested proportional interleaving",
        "subset_seeds": args.subset_seeds, "target_budget_pools": seeded_pools,
        "target_budgets": budgets_by_target,
        "environment_sizes": {name: {"total": info["num_rows"], **{key: len(rows) for key, rows in info["splits"].items()}}
                              for name, info in manifest["environments"].items()},
        "results": results,
    }
    if on_start:
        on_start(document)
    completed = {result_key(row) for row in results}
    # Cache source estimates once across targets and all ranked variants.
    source_ols_fits = {name: fit_ols(data[name][0][pools[name]], data[name][1][pools[name]], fit_intercept=not args.no_intercept)
                       for name in needed if name not in excluded_sources} if "mixed_estimator" in args.methods else {}
    source_estimates = {name: estimate_source_coefficients(data[name][0][pools[name]], data[name][1][pools[name]], fit_intercept=not args.no_intercept)
                        for name in needed if name not in excluded_sources} if pseudo_methods else {}
    for target in targets:
        if all((target,method,seed,budget) in completed for method in args.methods
               for seed in args.subset_seeds for budget in budgets_by_target[target]):
            print(f'{target}: all requested fits already saved; skipping.',flush=True)
            continue
        sources = sources_by_target[target]
        source_fit, source_n = None, 0
        if {"source_only_ols", "mixed_loss_estimator"}.intersection(args.methods):
            x_source = np.concatenate([data[name][0][pools[name]] for name in sources])
            y_source = np.concatenate([data[name][1][pools[name]] for name in sources])
            if "source_only_ols" in args.methods:
                source_fit = fit_ols(x_source, y_source, fit_intercept=not args.no_intercept)
            source_n = len(y_source)
        source_loss = (prepare_source_loss(x_source, y_source, fit_intercept=not args.no_intercept)
                       if "mixed_loss_estimator" in args.methods else None)
        source_mean = None
        if "mixed_estimator" in args.methods:
            source_mean = average_source_fits([source_ols_fits[name] for name in sources])
            mixed_source_means[target] = {"coefficients": source_mean.coefficients.tolist(),
                                          "intercept": source_mean.intercept}
        screen = None
        screens = {}
        if pseudo_methods:
            screen = screen_source_estimates([source_estimates[name] for name in sources],
                                    fit_intercept=not args.no_intercept,
                                    z_threshold=args.screen_z_threshold,
                                    stability_threshold=args.screen_stability_threshold,
                                    max_invariant_features=args.max_invariant_features)
            source_screens[target] = {
                "invariant_support": screen.invariant_support,
                "invariant_features": [features[j] for j in screen.invariant_support],
                "mean_coefficients": screen.mean_coefficients.tolist(),
                "mean_intercept": screen.mean_intercept,
                "details": [{**row, "feature_name": features[row["feature"]]} for row in screen.details],
                "settings": {"z_threshold": args.screen_z_threshold,
                             "stability_threshold": args.screen_stability_threshold,
                             "max_invariant_features": args.max_invariant_features},
                "passed_checks": {key: sum(row[key] for row in screen.details)
                                  for key in ("identifiable", "sign_consistent", "detectable", "selected")},
            }
            print(f"{target}: source screen selected {len(screen.invariant_support)}/{len(features)} features: "
                  f"{', '.join(features[j] for j in screen.invariant_support) or '(empty: target-only lasso)'}", flush=True)
            ranked_screens = {}
            ranking = rank_source_features(screen, include_unidentified=args.ranked_include_unidentified)
            for method in pseudo_methods:
                _, count = pseudo_method_info(method)
                if count is not None and count not in ranked_screens:
                    ranked_screens[count] = ranked_source_screen(screen, count, ranking=ranking)
                    actual = len(ranked_screens[count].invariant_support)
                    print(f"{target}: ranked K={count}, actual |A|={actual}: "
                          f"{', '.join(features[j] for j in ranked_screens[count].invariant_support)}", flush=True)
                    if actual < count:
                        print(f"{target}: K={count} capped at {actual} identifiable coefficients", flush=True)
                screens[method] = ranked_screens[count] if count is not None else screen
                chosen_screen = screens[method]
                saved = {**source_screens[target],
                         "invariant_support": chosen_screen.invariant_support,
                         "invariant_features": [features[j] for j in chosen_screen.invariant_support],
                         "details": [{**row, "feature_name": features[row["feature"]]} for row in chosen_screen.details]}
                if count is not None:
                    saved.update(settings={"mode": "ranked", "requested_features": count,
                                           "include_unidentified": args.ranked_include_unidentified},
                                 requested_features=count, actual_features=len(chosen_screen.invariant_support),
                                 ranking=[row["feature_name"] for row in sorted(saved["details"],key=lambda row: row["rank"] or math.inf)
                                          if row["rank"] is not None])
                    saved["passed_checks"] = {**saved["passed_checks"], "selected": len(chosen_screen.invariant_support)}
                source_screens_by_method[method][target] = saved
        x_target, y_target = data[target]
        test = manifest["environments"][target]["splits"]["test"]
        x_test, y_test = x_target[test], y_target[test]
        source_error = mse(source_fit, x_test, y_test) if source_fit is not None else None
        for seed in args.subset_seeds:
            for budget in budgets_by_target[target]:
                if all((target,method,seed,budget) in completed for method in args.methods):
                    continue
                selected = seeded_pools[target][str(seed)][:budget]
                target_fit = (fit_ols(x_target[selected], y_target[selected], fit_intercept=not args.no_intercept)
                              if {"target_only_ols", "mixed_estimator"}.intersection(args.methods) else None)
                fits = {"source_only_ols": (source_fit, source_n), "target_only_ols": (target_fit, budget)}
                errors = {}
                for method in args.methods:
                    if (target,method,seed,budget) in completed:
                        continue
                    info = pseudo_method_info(method)
                    if info:
                        fit_function = fit_centered_pseudo_lasso if info[0] == "pseudo_lasso_centered" else fit_pseudo_lasso
                        pseudo_fit = fit_function(
                            x_target[selected], y_target[selected], screens[method],
                            lambda_grid=args.lasso_lambda_grid, lambda_ratios=args.lasso_lambda_ratios,
                            cv_folds=args.lasso_cv_folds, seed=seed, fit_intercept=not args.no_intercept,
                            max_iter=args.lasso_max_iter, tol=args.lasso_tol)
                        fits[method] = (pseudo_fit, budget)
                        if not pseudo_fit.converged or any(not row["converged"] for row in pseudo_fit.cv_rows):
                            print(f"Warning: {method} solver did not converge for some fits: {target}, seed={seed}, budget={budget}", flush=True)
                    elif method == "target_only_lasso":
                        lasso_fit = fit_target_lasso(
                            x_target[selected], y_target[selected],
                            lambda_grid=args.lasso_lambda_grid, lambda_ratios=args.lasso_lambda_ratios,
                            cv_folds=args.lasso_cv_folds, seed=seed, fit_intercept=not args.no_intercept,
                            max_iter=args.lasso_max_iter, tol=args.lasso_tol)
                        fits[method] = (lasso_fit, budget)
                        if not lasso_fit.converged or any(not row["converged"] for row in lasso_fit.cv_rows):
                            print(f"Warning: {method} solver did not converge for some fits: {target}, seed={seed}, budget={budget}", flush=True)
                    elif method == "mixed_loss_estimator":
                        mixed_fit = fit_mixed_loss_estimator(
                            x_target[selected], y_target[selected], source_loss,
                            lambda_grid=args.mixed_lambda_grid, cv_folds=args.lasso_cv_folds,
                            seed=seed, fit_intercept=not args.no_intercept)
                        fits[method] = (mixed_fit, source_n + budget)
                    elif method == "mixed_estimator":
                        mixed_fit = fit_mixed_estimator(
                            x_target[selected], y_target[selected], source_mean,
                            lambda_grid=args.mixed_lambda_grid, cv_folds=args.lasso_cv_folds,
                            seed=seed, fit_intercept=not args.no_intercept, target_fit=target_fit)
                        fits[method] = (mixed_fit, budget)
                    fit, fit_n = fits[method]
                    error = source_error if method == "source_only_ols" else mse(fit, x_test, y_test)
                    errors[method] = error
                    is_pseudo = pseudo_method_info(method) is not None
                    is_lasso = is_pseudo or method == "target_only_lasso"
                    is_mixed = method in {"mixed_estimator", "mixed_loss_estimator"}
                    results.append({
                        "target": target, "method": method, "subset_seed": seed, "target_budget": budget,
                        "target_budget_fraction": budget / len(pools[target]),
                        "target_labels_used": 0 if method == "source_only_ols" else budget,
                        "fit_samples": fit_n, "test_samples": len(test), "target_test_mse": error,
                        "design_rank": None if is_lasso else fit.rank,
                        "num_parameters": len(features) + int(not args.no_intercept),
                        "coefficients": fit.coefficients.tolist(), "intercept": fit.intercept,
                        "lambda_lasso": fit.lambda_reg if is_lasso else None,
                        "lambda_ratio": fit.lambda_ratio if is_lasso else None,
                        "lasso_cv_folds": fit.cv_folds if is_lasso else None,
                        "lasso_cv_rows": fit.cv_rows if is_lasso else None,
                        "lasso_tuning": fit.tuning if is_lasso else None,
                        "lasso_iterations": fit.iterations if is_lasso else None,
                        "lasso_converged": fit.converged if is_lasso else None,
                        "num_invariant_features": len(fit.invariant_support) if is_pseudo else None,
                        "requested_invariant_features": pseudo_method_info(method)[1] if is_pseudo else None,
                        "mixed_lambda": fit.lambda_mix if is_mixed else None,
                        "mixed_cv_folds": fit.cv_folds if is_mixed else None,
                        "mixed_cv_rows": fit.cv_rows if is_mixed else None,
                        "mixed_tuning": fit.tuning if is_mixed else None,
                    })
                    if on_result:
                        on_result(document, results[-1])
                    completed.add(result_key(results[-1]))
                scores = " ".join(f"{method}_mse={error:.6g}" for method, error in errors.items())
                print(f"{target} seed={seed} budget={budget}/{len(pools[target])} {scores}", flush=True)
    return document


def write_tables(rows: list[dict], experiment_dir: Path) -> None:
    fields = [key for key in rows[0] if key not in {"coefficients", "intercept", "lasso_cv_rows", "mixed_cv_rows"}]
    with (experiment_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        key = (row["target"], row["method"], row["target_budget"])
        groups.setdefault(key, []).append(row["target_test_mse"])
    with (experiment_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["target", "method", "target_budget", "num_seeds", "mse_mean", "mse_std"])
        writer.writeheader()
        for (target, method, budget), values in groups.items():
            writer.writerow({"target": target, "method": method, "target_budget": budget,
                             "num_seeds": len(values), "mse_mean": float(np.mean(values)),
                             "mse_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0})


def validate_addition(experiment_dir: Path, result: dict):
    """Keep added methods comparable to the existing saved experiment."""
    paths = sorted(experiment_dir.glob("*/results.json"))
    if not paths:
        raise ValueError(f"No saved method results in existing experiment: {experiment_dir}")
    keys = ("manifest_sha256", "features", "response", "fit_intercept",
            "excluded_source_environments", "source_environments_by_target",
            "subset_seeds", "target_budget_pools")
    assignments = {(target, budget, seed) for target, budgets in result["target_budgets"].items()
                   for budget in budgets for seed in result["subset_seeds"]}
    for path in paths:
        saved = load_result_document(path)
        for key in keys:
            if key not in saved or saved[key] != result[key]:
                raise ValueError(f"Cannot add methods to {experiment_dir}: {key} differs from {path}")
        previous = ({(target, budget, seed) for target, budgets in saved["target_budgets"].items()
                     for budget in budgets for seed in saved["subset_seeds"]} if "target_budgets" in saved else
                    {(row["target"], row["target_budget"], row["subset_seed"]) for row in saved["results"]})
        if previous != assignments:
            raise ValueError(f"Cannot add methods to {experiment_dir}: targets, budgets or seeds differ from {path}")


def atomic_json(path, document):
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    atomic_text(path, text)


def atomic_text(path, text):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ProgressiveWriter:
    """Append each completed fit without repeatedly rewriting all saved rows."""

    def __init__(self, args, experiment_dir, adding):
        self.args, self.directory, self.adding = args, experiment_dir, adding
        self.started = False
        self.seen_targets = {method: set() for method in args.methods}
        self.document = None

    def method_document(self, document, method, *, complete=False, status="running"):
        saved = {**document, "methods": [method],
                 "source_screens": document["source_screens_by_method"].get(method, {}),
                 "mixed_source_means": document["mixed_source_means"] if method == "mixed_estimator" else {},
                 "results": [row for row in document["results"] if row["method"] == method] if complete else [],
                 "status": "complete" if complete else status,
                 "storage": "snapshot" if complete else "jsonl"}
        saved.pop("source_screens_by_method")
        return saved

    def start(self, document):
        if self.adding:
            validate_addition(self.directory, document)
        manifest_bytes = self.args.manifest.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != document["manifest_sha256"]:
            raise ValueError("Split manifest changed during initialization")
        resuming = getattr(self.args,'resume',False)
        saved_methods = {}
        if resuming:
            # Validate every requested method before modifying any saved file.
            common = ['response','no_intercept']
            for method in self.args.methods:
                folder = self.directory/method
                saved = load_result_document(folder/'results.json')
                config = json.loads((folder/'config.json').read_text())
                relevant = list(common)
                if pseudo_method_info(method) or method == 'target_only_lasso':
                    relevant += ['lasso_lambda_grid','lasso_lambda_ratios','lasso_cv_folds','lasso_max_iter','lasso_tol']
                if pseudo_method_info(method):
                    relevant += ['screen_z_threshold','screen_stability_threshold','max_invariant_features','ranked_include_unidentified']
                if method in {'mixed_estimator', 'mixed_loss_estimator'}:
                    relevant += ['mixed_lambda_grid','lasso_cv_folds']
                for key in relevant:
                    if key not in config or config[key] != getattr(self.args,key):
                        raise ValueError(f'Cannot resume {method}: {key} differs from the saved configuration')
                expected = {(target,method,seed,budget) for target,budgets in document['target_budgets'].items()
                            for seed in document['subset_seeds'] for budget in budgets}
                seen = set()
                for row in saved['results']:
                    key = result_key(row)
                    if key not in expected or key in seen:
                        raise ValueError(f'Unexpected or duplicate saved fit for {method}: {key}')
                    seen.add(key)
                saved_methods[method] = saved
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(self.args).items()}
        config.update(resolved_features=document["features"], command=sys.argv,
                      experiment_dir=str(self.directory), manifest_sha256=document["manifest_sha256"])
        self.args.output_root.mkdir(parents=True, exist_ok=True)
        if not self.adding:
            self.directory.mkdir()
            atomic_json(self.directory/'config.json', config)
            (self.directory/'splits.json').write_bytes(manifest_bytes)
        self.document = document
        for method in self.args.methods:
            folder = self.directory/method
            if resuming:
                saved = saved_methods[method]
                document['results'].extend(saved['results'])
                document['source_screens_by_method'].setdefault(method,{}).update(saved.get('source_screens',{}))
                if method == 'mixed_estimator':
                    document['mixed_source_means'].update(saved.get('mixed_source_means',{}))
                # Canonicalize the journal before appending, removing an incomplete
                # final line and ensuring snapshot-backed runs have a journal too.
                atomic_text(folder/'results.jsonl',''.join(json.dumps(row,allow_nan=False)+'\n' for row in saved['results']))
                if saved['results']:
                    write_tables(saved['results'],folder)
                elif (folder/'results.csv').exists():
                    (folder/'results.csv').unlink()
                print(f"Resuming {method}: keeping {len(saved['results'])} completed fits",flush=True)
            else:
                folder.mkdir()
                atomic_json(folder/'config.json', {**config, "methods": [method], "run_methods": self.args.methods})
            atomic_json(folder/'results.json', self.method_document(document, method))
        self.started = True
        print(f"Saving completed fits progressively to {self.directory}", flush=True)

    def append(self, document, row):
        method = row['method']
        folder = self.directory/method
        line = json.dumps(row, allow_nan=False) + '\n'
        if row['target'] not in self.seen_targets[method]:
            atomic_json(folder/'results.json', self.method_document(document, method))
            self.seen_targets[method].add(row['target'])
        with (folder/'results.jsonl').open('a') as stream:
            stream.write(line)
        fields = [key for key in row if key not in {"coefficients", "intercept", "lasso_cv_rows", "mixed_cv_rows"}]
        csv_path = folder/'results.csv'
        first_row = not csv_path.exists()
        with csv_path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            if first_row:
                writer.writeheader()
            writer.writerow(row)

    def finish(self, document):
        if hashlib.sha256(self.args.manifest.read_bytes()).hexdigest() != document['manifest_sha256']:
            raise ValueError('Split manifest changed during the run')
        for method in self.args.methods:
            folder = self.directory/method
            saved = self.method_document(document, method, complete=True)
            write_tables(saved['results'], folder)
            atomic_json(folder/'results.json', saved)
        print(f"Saved completed experiment to {self.directory}", flush=True)

    def mark_interrupted(self, status):
        if self.started:
            for method in self.args.methods:
                atomic_json(self.directory/method/'results.json', self.method_document(self.document, method, status=status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--manifest", type=Path, default=Path("causal_chamber/splits/lt_interventions_standard_v1.json"))
    parser.add_argument(
        "--features", nargs="+",
        help="Default: all columns except response, metadata, and ir_1/ir_2/ir_3. The IR sensors cannot be used as predictors.",
    )
    parser.add_argument("--response", default="ir_1")
    parser.add_argument("--target", help="Relative CSV path, filename, or unique filename stem; default: all")
    budget_args = parser.add_mutually_exclusive_group()
    budget_args.add_argument("--budgets", nargs="+", type=int)
    budget_args.add_argument("--budget-fractions", nargs="+", type=float, default=[.01, .02, .05, .1, .2, .5, 1.0])
    parser.add_argument("--methods", nargs="+", choices=["source_only_ols", "target_only_ols", "target_only_lasso", "pseudo_lasso", "pseudo_lasso_centered", "mixed_estimator", "mixed_loss_estimator"],
                        default=["source_only_ols", "target_only_ols", "pseudo_lasso", "mixed_estimator"])
    parser.add_argument("--screen-z-threshold", type=float, default=0.0,
                        help="Detectability gate; 0 disables the significance requirement (exploratory default)")
    parser.add_argument("--screen-stability-threshold", type=float, default=1.0,
                        help="Maximum relative source coefficient variation; larger is more permissive")
    parser.add_argument("--max-invariant-features", type=int)
    screening = parser.add_mutually_exclusive_group()
    screening.add_argument("--ranked-feature-counts", nargs="+", type=int,
                           help="Top-K grid for requested pseudo-lasso methods; default: 2 5 10")
    screening.add_argument("--threshold-screen", action="store_true",
                           help="Use the original threshold screening instead of the default ranked grid")
    parser.add_argument("--ranked-include-unidentified", action="store_true",
                        help="Allow pseudoinverse coefficients that are not individually identifiable; default caps K at identifiable count")
    penalty = parser.add_mutually_exclusive_group()
    penalty.add_argument("--lasso-lambda-grid", nargs="+", type=float, help="Absolute penalties, in standardized feature units")
    penalty.add_argument("--lasso-lambda-ratios", nargs="+", type=float, default=[1, .3, .1, .03, .01, .003, .001],
                         help="Default: penalties as fractions of training-fold lambda_max")
    parser.add_argument("--cv-folds", "--lasso-cv-folds", dest="lasso_cv_folds", type=int, default=5,
                        help="Shared target CV fold count for pseudo-lasso and mixed estimator")
    parser.add_argument("--mixed-lambda-grid", nargs="+", type=float, default=[0, .1, .2, .4, .6, .8, 1],
                        help="Weights in [0,1]: target weight for mixed_estimator; source-loss weight for mixed_loss_estimator")
    parser.add_argument("--lasso-max-iter", type=int, default=5000)
    parser.add_argument("--lasso-tol", type=float, default=1e-6)
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument("--subset-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-name", help="Folder name; default: linear_estimators_<UTC timestamp>_<unique suffix>")
    parser.add_argument("--resume", action="store_true", help="Continue an existing experiment using the same settings; skip completed fits")
    args = parser.parse_args()
    progressive = None
    try:
        if len(set(args.methods)) != len(args.methods):
            raise ValueError("Methods must be distinct")
        if (args.ranked_feature_counts is None and not args.threshold_screen
                and args.max_invariant_features is None
                and {"pseudo_lasso", "pseudo_lasso_centered"}.intersection(args.methods)):
            args.ranked_feature_counts = [2, 5, 10]
        if args.ranked_feature_counts is not None and args.max_invariant_features is not None:
            raise ValueError("Use ranked feature counts or a threshold-screen feature cap, not both")
        args.methods = expand_ranked_methods(args.methods, args.ranked_feature_counts)
        name = args.experiment_name or f"linear_estimators_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Experiment name must be a single folder name")
        experiment_dir = args.output_root / name
        adding = experiment_dir.exists()
        if args.resume and (not args.experiment_name or not adding):
            raise ValueError('--resume requires the name of an existing experiment')
        if adding:
            if (experiment_dir / "results.json").exists():
                raise ValueError("Existing experiment uses the legacy combined layout; choose a new experiment name")
            for method in args.methods:
                if args.resume and not (experiment_dir/method/'results.json').is_file():
                    raise ValueError(f'Cannot resume missing method: {experiment_dir/method}')
                if (experiment_dir / method).exists() and not args.resume:
                    raise ValueError(f"Method folder already exists: {experiment_dir / method}; choose another experiment name")
            if not (experiment_dir / "splits.json").is_file() or not list(experiment_dir.glob("*/results.json")):
                raise ValueError(f"Existing folder is not a completed method-folder experiment: {experiment_dir}")
            if (experiment_dir / "splits.json").read_bytes() != args.manifest.read_bytes():
                raise ValueError("Cannot add methods: split manifest differs from the saved experiment")
        progressive = ProgressiveWriter(args, experiment_dir, adding)
        result = run(args, on_start=progressive.start, on_result=progressive.append)
        progressive.finish(result)
    except KeyboardInterrupt:
        if progressive:
            progressive.mark_interrupted('interrupted')
        raise
    except (ValueError, OSError, KeyError, TypeError) as error:
        if progressive:
            try:
                progressive.mark_interrupted('failed')
            except OSError:
                pass
        parser.error(str(error))


if __name__ == "__main__":
    main()
