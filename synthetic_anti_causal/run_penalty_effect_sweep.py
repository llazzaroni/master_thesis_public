"""Synthetic experiment isolating target-moment penalties.

The target adaptation pool contains covariates only. Its labels are never
read by an estimator; every reported MSE is computed on an independent target
test set. Hyperparameters are selected once per budget through leave-one-
source-environment-out validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from synthetic_anti_causal.data import EnvironmentData, generate_synthetic_dataset
from synthetic_anti_causal.estimators import (
    shift_projector_inverse_fit,
    shift_projector_penalty,
    mse,
    ols_fit,
    source_cross_moment,
    stack_environments,
    subsample_unlabeled_env,
    unlabeled_second_moment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-source-envs", type=int, default=10)
    parser.add_argument("--num-target-envs", type=int, default=100)
    parser.add_argument("--source-samples-per-env", type=int, default=1000)
    parser.add_argument("--target-test-samples-per-env", type=int, default=400)
    parser.add_argument("--target-adaptation-samples-per-env", type=int, default=1500)
    parser.add_argument(
        "--target-budget-grid",
        type=int,
        nargs="+",
        default=[100, 200, 400, 800, 1200, 1500],
    )
    parser.add_argument("--pseudo-target-train-samples", type=int, default=800)
    parser.add_argument("--pseudo-target-val-samples", type=int, default=200)
    parser.add_argument("--num-features", type=int, default=20)
    parser.add_argument("--num-stable-features", type=int, default=5)
    parser.add_argument("--nonlinear-strength", type=float, default=0.25)
    parser.add_argument(
        "--confounder-dim",
        type=int,
        default=0,
        help="Dimension of the latent common cause U. Set with --confounding-strength to enable it.",
    )
    parser.add_argument(
        "--confounding-strength",
        type=float,
        default=0.0,
        help="Sets ||A||_F / ||s|| for the X <- A U contribution. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--outcome-confounding-strength",
        type=float,
        default=0.5,
        help="Sets ||rho|| in Y = rho^T U + sqrt(1-||rho||^2) eta when confounding is enabled.",
    )
    parser.add_argument("--stable-mean-scale", type=float, default=0.0)
    parser.add_argument("--stable-std-low", type=float, default=0.0)
    parser.add_argument("--stable-std-high", type=float, default=1.0)
    parser.add_argument("--unstable-mean-scale", type=float, default=0.8)
    parser.add_argument("--unstable-std-low", type=float, default=1.2)
    parser.add_argument("--unstable-std-high", type=float, default=1.6)
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument(
        "--num-structural-draws",
        type=int,
        default=5,
        help="Independent draws of the shared signal and confounding structure.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Hierarchical paired-bootstrap replicates for uncertainty intervals.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/synthetic_penalty_effect"),
    )
    return parser.parse_args()


def split_source_as_pseudo_target(
    env: EnvironmentData,
    *,
    train_samples: int,
    val_samples: int,
    seed: int,
) -> EnvironmentData:
    """Create an 800-X/200-label pseudo-target split from one source env."""
    required = train_samples + val_samples
    if required > env.x.shape[0]:
        raise ValueError(
            f"Pseudo-target split needs {required} samples, but env={env.env_id} has "
            f"only {env.x.shape[0]}."
        )
    rng = np.random.default_rng(np.random.SeedSequence([seed, env.env_id, 7103]))
    permutation = rng.permutation(env.x.shape[0])
    adaptation_indices = np.sort(permutation[:train_samples])
    validation_indices = np.sort(permutation[train_samples:required])
    return EnvironmentData(
        env_id=env.env_id,
        x=env.x[validation_indices],
        y=env.y[validation_indices],
        x_unlabeled=env.x[adaptation_indices],
        x_oracle=None,
        y_oracle=None,
        noise_mean=env.noise_mean,
        noise_std=env.noise_std,
    )


def pooled_inverse_fit(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    lambda_reg: float = 0.0,
    top_k: int,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Fit (M_pool + lambda Gamma_shift)^-1 m_source."""
    source_x, _ = stack_environments(source_envs)
    target_x = target_env.x_unlabeled
    if target_x is None:
        raise ValueError("Target adaptation covariates are required.")
    source_moment = (source_x.T @ source_x) / float(source_x.shape[0])
    target_moment = unlabeled_second_moment(target_env)
    pooled_moment = (
        float(source_x.shape[0]) * source_moment
        + float(target_x.shape[0]) * target_moment
    ) / float(source_x.shape[0] + target_x.shape[0])
    penalty = shift_projector_penalty(
        source_envs,
        target_env=target_env,
        top_k=top_k,
    )
    system = (
        pooled_moment
        + float(lambda_reg) * penalty
        + jitter * np.eye(pooled_moment.shape[0], dtype=np.float64)
    )
    return np.linalg.solve(system, source_cross_moment(source_envs))


def select_lambda(
    source_envs: list[EnvironmentData],
    *,
    budget: int,
    pseudo_target_train_samples: int,
    pseudo_target_val_samples: int,
    lambda_grid: list[float],
    top_k: int,
    seed: int,
    estimator: str,
) -> tuple[float, list[dict[str, float]]]:
    """Select one lambda for one budget with source-environment LOO CV."""
    rows: list[dict[str, float]] = []
    best_lambda = None
    best_loss = float("inf")
    for lambda_reg in sorted(set(float(value) for value in lambda_grid)):
        fold_losses = []
        for heldout_index, heldout_env in enumerate(source_envs):
            pseudo_target = split_source_as_pseudo_target(
                heldout_env,
                train_samples=pseudo_target_train_samples,
                val_samples=pseudo_target_val_samples,
                seed=seed,
            )
            pseudo_target = subsample_unlabeled_env(
                pseudo_target,
                budget=budget,
                seed=seed,
            )
            fold_source_envs = [
                env for index, env in enumerate(source_envs) if index != heldout_index
            ]
            if estimator == "target_penalty":
                beta = shift_projector_inverse_fit(
                    fold_source_envs,
                    target_env=pseudo_target,
                    lambda_reg=lambda_reg,
                    top_k=top_k,
                ).beta
            elif estimator == "pooled_penalty":
                beta = pooled_inverse_fit(
                    fold_source_envs,
                    target_env=pseudo_target,
                    lambda_reg=lambda_reg,
                    top_k=top_k,
                )
            else:
                raise ValueError(f"Unknown estimator family: {estimator}")
            validation_mse = mse(pseudo_target.x, pseudo_target.y, beta)
            fold_losses.append(validation_mse)
            rows.append(
                {
                    "lambda": lambda_reg,
                    "heldout_source_env": float(heldout_env.env_id),
                    "validation_mse": validation_mse,
                }
            )
        mean_loss = float(np.mean(fold_losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_lambda = lambda_reg
    if best_lambda is None:
        raise RuntimeError("Failed to select a penalty weight.")
    return best_lambda, rows


def save_average_plot(
    path: Path,
    *,
    budgets: list[int],
    averages: dict[str, list[float]],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for label, values in averages.items():
        axis.plot(budgets, values, marker="o", linewidth=2.0, label=label)
    # The sweep uses a small, discrete set of budgets.  Show every requested
    # value directly rather than letting a logarithmic axis hide tick labels.
    axis.set_xticks(budgets, [str(budget) for budget in budgets])
    axis.margins(x=0.04)
    axis.set_xlabel("Target adaptation-sample budget")
    axis.set_ylabel("Mean target-test MSE")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_paired_improvement_plot(
    path: Path,
    *,
    budgets: list[int],
    summaries: list[dict[str, float]],
    title: str,
    comparison: str,
) -> None:
    """Plot paired MSE reductions with hierarchical bootstrap intervals."""
    import matplotlib.pyplot as plt

    means = [summary["mean_mse_reduction"] for summary in summaries]
    lower = [summary["ci95_low_mse_reduction"] for summary in summaries]
    upper = [summary["ci95_high_mse_reduction"] for summary in summaries]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(budgets, means, marker="o", linewidth=2.0, label="anti-causal penalty")
    axis.fill_between(budgets, lower, upper, alpha=0.2, label="95% paired bootstrap CI")
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(budgets, [str(budget) for budget in budgets])
    axis.margins(x=0.04)
    axis.set_xlabel("Target adaptation-sample budget")
    axis.set_ylabel("MSE reduction")
    axis.set_title(f"{title}\nrelative to {comparison}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_dataset(args: argparse.Namespace, *, seed: int):
    """Construct sources and independent target test environments at their requested sizes."""
    common = {
        "seed": seed,
        "num_train_envs": args.num_source_envs,
        "unlabeled_samples_per_train_env": 0,
        "num_features": args.num_features,
        "num_stable_features": args.num_stable_features,
        "stable_mean_scale": args.stable_mean_scale,
        "stable_std_low": args.stable_std_low,
        "stable_std_high": args.stable_std_high,
        "unstable_mean_scale": args.unstable_mean_scale,
        "unstable_std_low": args.unstable_std_low,
        "unstable_std_high": args.unstable_std_high,
        "test_unstable_mean_scale": args.unstable_mean_scale,
        "test_unstable_std_low": args.unstable_std_low,
        "test_unstable_std_high": args.unstable_std_high,
        "nonlinear_strength": args.nonlinear_strength,
        "confounder_dim": args.confounder_dim,
        "confounding_strength": args.confounding_strength,
        "outcome_confounding_strength": args.outcome_confounding_strength,
    }
    source_dataset = generate_synthetic_dataset(
        **common,
        num_test_envs=0,
        samples_per_env=args.source_samples_per_env,
        unlabeled_samples_per_test_env=0,
    )
    target_dataset = generate_synthetic_dataset(
        **common,
        num_test_envs=args.num_target_envs,
        samples_per_env=args.target_test_samples_per_env,
        unlabeled_samples_per_test_env=args.target_adaptation_samples_per_env,
    )
    return type(target_dataset)(
        train_envs=source_dataset.train_envs,
        test_envs=target_dataset.test_envs,
        signal_direction=target_dataset.signal_direction,
        nonlinear_direction=target_dataset.nonlinear_direction,
        stable_feature_mask=target_dataset.stable_feature_mask,
        confounder_loading=target_dataset.confounder_loading,
        outcome_confounder_direction=target_dataset.outcome_confounder_direction,
    )


def hierarchical_paired_summary(
    rows: list[dict[str, float]],
    *,
    baseline_key: str,
    penalized_key: str,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, float]:
    """Bootstrap paired MSE reduction across structural draws and targets."""
    values_by_draw: list[np.ndarray] = []
    for draw in sorted({int(row["structural_draw"]) for row in rows}):
        draw_rows = [row for row in rows if int(row["structural_draw"]) == draw]
        values_by_draw.append(
            np.asarray(
                [row[baseline_key] - row[penalized_key] for row in draw_rows],
                dtype=np.float64,
            )
        )
    observed = np.concatenate(values_by_draw)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_replicates, dtype=np.float64)
    num_draws = len(values_by_draw)
    for replicate in range(bootstrap_replicates):
        sampled_draws = rng.integers(num_draws, size=num_draws)
        resampled = [
            values_by_draw[draw][rng.integers(values_by_draw[draw].shape[0], size=values_by_draw[draw].shape[0])]
            for draw in sampled_draws
        ]
        bootstrap_means[replicate] = np.mean(np.concatenate(resampled))
    return {
        "mean_mse_reduction": float(np.mean(observed)),
        "ci95_low_mse_reduction": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high_mse_reduction": float(np.quantile(bootstrap_means, 0.975)),
        "num_structural_draws": num_draws,
        "num_target_environments": int(observed.shape[0]),
    }


def main(args: argparse.Namespace) -> None:
    budgets = sorted(set(int(value) for value in args.target_budget_grid))
    if not budgets or budgets[0] <= 0:
        raise ValueError("Target budgets must be positive.")
    if budgets[-1] > args.target_adaptation_samples_per_env:
        raise ValueError("The largest target budget exceeds the target adaptation pool.")
    if args.pseudo_target_train_samples + args.pseudo_target_val_samples > args.source_samples_per_env:
        raise ValueError("Pseudo-target train/validation split exceeds source samples per environment.")
    if args.num_structural_draws <= 0:
        raise ValueError("num_structural_draws must be positive.")
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive.")

    top_k = args.num_features
    targets_by_budget: dict[int, list[dict[str, float]]] = {budget: [] for budget in budgets}
    cv_by_budget: dict[int, list[dict]] = {budget: [] for budget in budgets}
    for structural_draw in range(args.num_structural_draws):
        draw_seed = args.seed + structural_draw
        print(
            f"[penalty-effect] structural draw {structural_draw + 1}/{args.num_structural_draws} "
            f"seed={draw_seed}",
            flush=True,
        )
        dataset = build_dataset(args, seed=draw_seed)
        source_x, source_y = stack_environments(dataset.train_envs)
        source_beta = ols_fit(source_x, source_y)
        cv_cache: dict[int, tuple[float, list[dict[str, float]], float, list[dict[str, float]]]] = {}
        for budget in budgets:
            cv_budget = min(budget, args.pseudo_target_train_samples)
            if cv_budget not in cv_cache:
                print(f"[penalty-effect] cross-validating budget={cv_budget}", flush=True)
                target_lambda, target_cv_rows = select_lambda(
                    dataset.train_envs,
                    budget=cv_budget,
                    pseudo_target_train_samples=args.pseudo_target_train_samples,
                    pseudo_target_val_samples=args.pseudo_target_val_samples,
                    lambda_grid=args.lambda_grid,
                    top_k=top_k,
                    seed=draw_seed,
                    estimator="target_penalty",
                )
                pooled_lambda, pooled_cv_rows = select_lambda(
                    dataset.train_envs,
                    budget=cv_budget,
                    pseudo_target_train_samples=args.pseudo_target_train_samples,
                    pseudo_target_val_samples=args.pseudo_target_val_samples,
                    lambda_grid=args.lambda_grid,
                    top_k=top_k,
                    seed=draw_seed,
                    estimator="pooled_penalty",
                )
                cv_cache[cv_budget] = (target_lambda, target_cv_rows, pooled_lambda, pooled_cv_rows)
            else:
                print(
                    "[penalty-effect] reusing cross-validation result "
                    f"budget={cv_budget} for target budget={budget}",
                    flush=True,
                )
            target_lambda, target_cv_rows, pooled_lambda, pooled_cv_rows = cv_cache[cv_budget]
            cv_by_budget[budget].append(
                {
                    "structural_draw": structural_draw,
                    "seed": draw_seed,
                    "cross_validation_budget": cv_budget,
                    "selected_target_penalty_lambda": target_lambda,
                    "selected_pooled_penalty_lambda": pooled_lambda,
                    "target_penalty_cv": target_cv_rows,
                    "pooled_penalty_cv": pooled_cv_rows,
                }
            )
            for target_env in dataset.test_envs:
                target_subset = subsample_unlabeled_env(
                    target_env,
                    budget=budget,
                    seed=draw_seed,
                )
                target_beta = shift_projector_inverse_fit(
                    dataset.train_envs,
                    target_env=target_subset,
                    lambda_reg=0.0,
                    top_k=top_k,
                ).beta
                target_penalized_beta = shift_projector_inverse_fit(
                    dataset.train_envs,
                    target_env=target_subset,
                    lambda_reg=target_lambda,
                    top_k=top_k,
                ).beta
                pooled_beta = pooled_inverse_fit(
                    dataset.train_envs,
                    target_env=target_subset,
                    lambda_reg=0.0,
                    top_k=top_k,
                )
                pooled_penalized_beta = pooled_inverse_fit(
                    dataset.train_envs,
                    target_env=target_subset,
                    lambda_reg=pooled_lambda,
                    top_k=top_k,
                )
                targets_by_budget[budget].append(
                    {
                        "structural_draw": structural_draw,
                        "target_env": target_env.env_id,
                        "source_mse": mse(target_env.x, target_env.y, source_beta),
                        "target_inverse_mse": mse(target_env.x, target_env.y, target_beta),
                        "target_penalty_mse": mse(target_env.x, target_env.y, target_penalized_beta),
                        "pooled_inverse_mse": mse(target_env.x, target_env.y, pooled_beta),
                        "pooled_penalty_mse": mse(target_env.x, target_env.y, pooled_penalized_beta),
                    }
                )

    result_rows = []
    target_improvement_summaries = []
    pooled_improvement_summaries = []
    for budget in budgets:
        per_target = targets_by_budget[budget]
        target_summary = hierarchical_paired_summary(
            per_target,
            baseline_key="target_inverse_mse",
            penalized_key="target_penalty_mse",
            seed=args.seed + 100_000 + budget,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        pooled_summary = hierarchical_paired_summary(
            per_target,
            baseline_key="pooled_inverse_mse",
            penalized_key="pooled_penalty_mse",
            seed=args.seed + 200_000 + budget,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        target_improvement_summaries.append(target_summary)
        pooled_improvement_summaries.append(pooled_summary)
        result_rows.append(
            {
                "budget": budget,
                "mean_mse": {
                    key: float(np.mean([row[key] for row in per_target]))
                    for key in per_target[0]
                    if key not in {"structural_draw", "target_env"}
                },
                "target_penalty_vs_target_inverse": target_summary,
                "pooled_penalty_vs_pooled_inverse": pooled_summary,
                "cv_by_structural_draw": cv_by_budget[budget],
                "per_target_environment": per_target,
            }
        )

    output = {
        "setup": {
            "num_source_envs": args.num_source_envs,
            "num_target_envs": args.num_target_envs,
            "source_samples_per_env": args.source_samples_per_env,
            "target_adaptation_samples_per_env": args.target_adaptation_samples_per_env,
            "target_test_samples_per_env": args.target_test_samples_per_env,
            "pseudo_target_split": {
                "adaptation_samples": args.pseudo_target_train_samples,
                "validation_samples": args.pseudo_target_val_samples,
            },
            "top_k": top_k,
            "lambda_grid": args.lambda_grid,
            "num_structural_draws": args.num_structural_draws,
            "bootstrap_replicates": args.bootstrap_replicates,
            "confounding": {
                "confounder_dim": args.confounder_dim,
                "feature_confounding_strength": args.confounding_strength,
                "outcome_confounding_strength": args.outcome_confounding_strength,
            },
        },
        "rows": result_rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "results.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    target_averages = {
        "source OLS": [row["mean_mse"]["source_mse"] for row in result_rows],
        "target inverse": [row["mean_mse"]["target_inverse_mse"] for row in result_rows],
        "target inverse + anti-causal penalty": [row["mean_mse"]["target_penalty_mse"] for row in result_rows],
    }
    pooled_averages = {
        "source OLS": [row["mean_mse"]["source_mse"] for row in result_rows],
        "pooled inverse": [row["mean_mse"]["pooled_inverse_mse"] for row in result_rows],
        "pooled inverse + anti-causal penalty": [row["mean_mse"]["pooled_penalty_mse"] for row in result_rows],
    }
    save_average_plot(
        args.output_root / "target_penalty_average_mse.png",
        budgets=budgets,
        averages=target_averages,
        title="Target-moment anti-causal penalty",
    )
    save_average_plot(
        args.output_root / "pooled_penalty_average_mse.png",
        budgets=budgets,
        averages=pooled_averages,
        title="Pooled-moment anti-causal penalty",
    )
    save_paired_improvement_plot(
        args.output_root / "target_penalty_paired_improvement.png",
        budgets=budgets,
        summaries=target_improvement_summaries,
        title="Target-moment anti-causal penalty",
        comparison="target inverse",
    )
    save_paired_improvement_plot(
        args.output_root / "pooled_penalty_paired_improvement.png",
        budgets=budgets,
        summaries=pooled_improvement_summaries,
        title="Pooled-moment anti-causal penalty",
        comparison="pooled inverse",
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main(parse_args())
