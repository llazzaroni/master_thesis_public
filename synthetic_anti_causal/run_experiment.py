from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from synthetic_anti_causal.data import SyntheticDataset, generate_synthetic_dataset
from synthetic_anti_causal.estimators import (
    leave_one_environment_out_selection_inverse,
    shift_projector_inverse_fit,
    mir_gamma,
    plain_inverse_fit,
    pooled_inverse_fit,
    pooled_target_vir_fit,
    projector_inverse_fit,
    projector_inverse_vir_fit,
    projector_regularized_fit,
    select_projector_inverse_vir_by_leave_one_environment_out,
    select_projector_inverse_by_leave_one_environment_out,
    select_shift_projector_inverse_by_leave_one_environment_out,
    select_pooled_target_vir_by_leave_one_environment_out,
    select_projector_by_leave_one_environment_out,
    select_lambda_by_leave_one_environment_out,
    unlabeled_regularized_fit,
    leave_one_environment_out_selection,
    mse,
    ols_fit,
    SelectionResult,
    ScalarSelectionResult,
    stack_environments,
    target_aware_inverse_fit,
    target_aware_inverse_vir_fit,
    target_aware_penalized_fit,
    vir_gamma,
    select_target_aware_inverse_vir_by_leave_one_environment_out,
)


def save_test_error_plot(
    output_path: Path,
    *,
    per_test_environment: list[dict[str, float]],
    included_estimators: set[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if included_estimators is None:
        included_estimators = {
            "ols",
            "plain_inverse",
            "pooled_inverse",
            "pooled_target_vir",
            "target_aware",
            "target_aware_inverse",
            "target_aware_inverse_vir",
            "mir",
            "vir",
            "projector",
            "projector_inverse",
            "projector_inverse_vir",
            "shift_projector_inverse",
        }

    env_ids = [int(row["env_id"]) for row in per_test_environment]
    series = []
    if "ols" in included_estimators:
        series.append(("OLS", "ols_mse"))
    if "target_aware" in included_estimators:
        series.append(("Target-aware", "target_aware_penalized_mse"))
    if "plain_inverse" in included_estimators:
        series.append(("Plain inverse", "plain_inverse_mse"))
    if "pooled_inverse" in included_estimators:
        series.append(("Target inverse", "pooled_inverse_mse"))
    if "pooled_target_vir" in included_estimators:
        series.append(("Target inverse + target VIR", "pooled_target_vir_mse"))
    if "target_aware_inverse" in included_estimators:
        series.append(("Target-aware inverse", "target_aware_inverse_mse"))
    if "target_aware_inverse_vir" in included_estimators:
        series.append(("Target-aware inverse + VIR", "target_aware_inverse_vir_mse"))
    if "mir" in included_estimators:
        series.append(("MIR", "mir_mse"))
    if "vir" in included_estimators:
        series.append(("VIR", "vir_mse"))
    if "projector" in included_estimators:
        series.append(("Projector", "projector_mse"))
    if "projector_inverse" in included_estimators:
        series.append(("Projector inverse", "projector_inverse_mse"))
    if "projector_inverse_vir" in included_estimators:
        series.append(("Projector inverse + VIR", "projector_inverse_vir_mse"))
    if "shift_projector_inverse" in included_estimators:
        series.append(("Shift-projector inverse", "shift_projector_inverse_mse"))
    if not series:
        raise ValueError("No estimators left to plot.")

    x = np.arange(len(env_ids), dtype=float)
    width = 0.8 / max(1, len(series))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    offsets = np.linspace(-0.4 + width / 2.0, 0.4 - width / 2.0, len(series))
    for offset, (label, key) in zip(offsets, series):
        values = [float(row[key]) for row in per_test_environment]
        ax.bar(x + offset, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(env_id) for env_id in env_ids])
    ax.set_xlabel("Target Environment")
    ax.set_ylabel("MSE")
    ax.set_title("Test Error by Target Environment")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_3d_vector_plot(
    output_path: Path,
    *,
    signal_direction: np.ndarray,
    nonlinear_direction: np.ndarray,
    beta_ols: np.ndarray,
    beta_penalized: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    if signal_direction.shape[0] != 3:
        return

    vectors = [
        ("signal $s$", signal_direction, "tab:blue"),
        ("nonlinear $t$", nonlinear_direction, "tab:orange"),
        ("OLS", beta_ols, "tab:green"),
        ("penalized", beta_penalized, "tab:red"),
    ]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for label, vector, color in vectors:
        ax.quiver(
            0.0,
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            float(vector[2]),
            color=color,
            linewidth=2.0,
            label=label,
        )

    max_abs = max(float(np.max(np.abs(vector))) for _label, vector, _color in vectors)
    limit = max(1.0, 1.1 * max_abs)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title("True Directions and Estimated Coefficients")
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize_env_group(
    dataset: SyntheticDataset,
    *,
    beta_ols: np.ndarray,
    beta_plain_inverse: np.ndarray | None,
    target_aware_lambda_reg: float,
    target_aware_top_k: int,
    target_aware_inverse_lambda_reg: float,
    target_aware_inverse_top_k: int,
    target_aware_inverse_vir_lambda_reg: float,
    target_aware_inverse_vir_lambda_vir: float,
    target_aware_inverse_vir_top_k: int,
    beta_mir: np.ndarray | None,
    beta_vir: np.ndarray | None,
    beta_projector: np.ndarray | None,
    projector_inverse_lambda_u: float,
    projector_inverse_lambda_perp: float,
    projector_inverse_top_k: int,
    projector_inverse_vir_lambda_u: float,
    projector_inverse_vir_lambda_perp: float,
    projector_inverse_vir_lambda_vir: float,
    projector_inverse_vir_top_k: int,
    shift_projector_inverse_lambda_reg: float,
    shift_projector_inverse_top_k: int,
    pooled_target_vir_lambda_reg: float,
    include_full_target_oracle: bool = False,
    included_estimators: set[str] | None = None,
) -> dict[str, object]:
    if included_estimators is None:
        included_estimators = {
            "ols",
            "plain_inverse",
            "pooled_inverse",
            "pooled_target_vir",
            "target_aware",
            "target_aware_inverse",
            "target_aware_inverse_vir",
            "mir",
            "vir",
            "projector",
            "projector_inverse",
            "projector_inverse_vir",
            "shift_projector_inverse",
        }
    test_rows = []
    for env in dataset.test_envs:
        row = {
            "env_id": env.env_id,
            "ols_mse": mse(env.x, env.y, beta_ols),
        }
        target_inverse_baseline = plain_inverse_fit(
            dataset.train_envs,
            target_env=env,
        )
        target_inverse_mse = mse(env.x, env.y, target_inverse_baseline.beta)
        row["target_inverse_baseline_mse"] = target_inverse_mse
        row["oracle_best_source_target_inverse_mse"] = min(
            row["ols_mse"],
            target_inverse_mse,
        )
        if include_full_target_oracle:
            if env.x_oracle is None or env.y_oracle is None:
                raise ValueError("Oracle labels were requested but are absent from test environments.")
            beta_full_target_oracle = ols_fit(env.x_oracle, env.y_oracle)
            full_target_oracle_mse = mse(env.x, env.y, beta_full_target_oracle)
            row["full_target_oracle_mse"] = full_target_oracle_mse
        if "plain_inverse" in included_estimators:
            # The target inverse must use this test environment's unlabeled X,
            # not the first target environment selected for reporting metadata.
            plain_inverse_for_env = plain_inverse_fit(
                dataset.train_envs,
                target_env=env,
            )
            row["plain_inverse_mse"] = mse(
                env.x,
                env.y,
                plain_inverse_for_env.beta,
            )
        if "pooled_inverse" in included_estimators:
            pooled_fit = pooled_inverse_fit(dataset.train_envs, target_env=env)
            row["pooled_inverse_mse"] = mse(env.x, env.y, pooled_fit.beta)
        if "pooled_target_vir" in included_estimators:
            pooled_target_vir_fit_result = pooled_target_vir_fit(
                dataset.train_envs,
                target_env=env,
                lambda_reg=pooled_target_vir_lambda_reg,
            )
            row["pooled_target_vir_mse"] = mse(
                env.x,
                env.y,
                pooled_target_vir_fit_result.beta,
            )
        if "target_aware" in included_estimators:
            fit = target_aware_penalized_fit(
                dataset.train_envs,
                target_env=env,
                lambda_reg=target_aware_lambda_reg,
                top_k=target_aware_top_k,
            )
            row["target_aware_penalized_mse"] = mse(env.x, env.y, fit.beta)
            row["target_aware_train_mse"] = fit.train_mse
        if "target_aware_inverse" in included_estimators:
            inverse_fit = target_aware_inverse_fit(
                dataset.train_envs,
                target_env=env,
                lambda_reg=target_aware_inverse_lambda_reg,
                top_k=target_aware_inverse_top_k,
            )
            row["target_aware_inverse_mse"] = mse(env.x, env.y, inverse_fit.beta)
        if "target_aware_inverse_vir" in included_estimators:
            inverse_vir_fit = target_aware_inverse_vir_fit(
                dataset.train_envs,
                target_env=env,
                lambda_reg=target_aware_inverse_vir_lambda_reg,
                lambda_vir=target_aware_inverse_vir_lambda_vir,
                top_k=target_aware_inverse_vir_top_k,
            )
            row["target_aware_inverse_vir_mse"] = mse(env.x, env.y, inverse_vir_fit.beta)
        if "mir" in included_estimators:
            row["mir_mse"] = mse(env.x, env.y, beta_mir)
        if "vir" in included_estimators:
            row["vir_mse"] = mse(env.x, env.y, beta_vir)
        if "projector" in included_estimators:
            row["projector_mse"] = mse(env.x, env.y, beta_projector)
        if "projector_inverse" in included_estimators:
            projector_inverse_fit_result = projector_inverse_fit(
                dataset.train_envs,
                target_env=env,
                lambda_u=projector_inverse_lambda_u,
                lambda_perp=projector_inverse_lambda_perp,
                top_k=projector_inverse_top_k,
            )
            row["projector_inverse_mse"] = mse(
                env.x,
                env.y,
                projector_inverse_fit_result.beta,
            )
        if "projector_inverse_vir" in included_estimators:
            projector_inverse_vir_fit_result = projector_inverse_vir_fit(
                dataset.train_envs,
                target_env=env,
                lambda_u=projector_inverse_vir_lambda_u,
                lambda_perp=projector_inverse_vir_lambda_perp,
                lambda_vir=projector_inverse_vir_lambda_vir,
                top_k=projector_inverse_vir_top_k,
            )
            row["projector_inverse_vir_mse"] = mse(
                env.x,
                env.y,
                projector_inverse_vir_fit_result.beta,
            )
        if "shift_projector_inverse" in included_estimators:
            shift_projector_inverse_fit_result = shift_projector_inverse_fit(
                dataset.train_envs,
                target_env=env,
                lambda_reg=shift_projector_inverse_lambda_reg,
                top_k=shift_projector_inverse_top_k,
            )
            row["shift_projector_inverse_mse"] = mse(
                env.x,
                env.y,
                shift_projector_inverse_fit_result.beta,
            )
        test_rows.append(row)
    summary = {
        "per_test_environment": test_rows,
        "mean_test_ols_mse": float(np.mean([row["ols_mse"] for row in test_rows])),
    }
    summary["mean_test_target_inverse_baseline_mse"] = float(
        np.mean([row["target_inverse_baseline_mse"] for row in test_rows])
    )
    summary["mean_test_oracle_best_source_target_inverse_mse"] = float(
        np.mean([row["oracle_best_source_target_inverse_mse"] for row in test_rows])
    )
    if include_full_target_oracle:
        summary["mean_test_full_target_oracle_mse"] = float(
            np.mean([row["full_target_oracle_mse"] for row in test_rows])
        )
    if "plain_inverse" in included_estimators:
        summary["mean_test_plain_inverse_mse"] = float(
            np.mean([row["plain_inverse_mse"] for row in test_rows])
        )
    if "pooled_inverse" in included_estimators:
        summary["mean_test_pooled_inverse_mse"] = float(
            np.mean([row["pooled_inverse_mse"] for row in test_rows])
        )
    if "pooled_target_vir" in included_estimators:
        summary["mean_test_pooled_target_vir_mse"] = float(
            np.mean([row["pooled_target_vir_mse"] for row in test_rows])
        )
    if "target_aware" in included_estimators:
        summary["mean_test_target_aware_penalized_mse"] = float(
            np.mean([row["target_aware_penalized_mse"] for row in test_rows])
        )
    if "target_aware_inverse" in included_estimators:
        summary["mean_test_target_aware_inverse_mse"] = float(
            np.mean([row["target_aware_inverse_mse"] for row in test_rows])
        )
    if "target_aware_inverse_vir" in included_estimators:
        summary["mean_test_target_aware_inverse_vir_mse"] = float(
            np.mean([row["target_aware_inverse_vir_mse"] for row in test_rows])
        )
    if "mir" in included_estimators:
        summary["mean_test_mir_mse"] = float(np.mean([row["mir_mse"] for row in test_rows]))
    if "vir" in included_estimators:
        summary["mean_test_vir_mse"] = float(np.mean([row["vir_mse"] for row in test_rows]))
    if "projector" in included_estimators:
        summary["mean_test_projector_mse"] = float(
            np.mean([row["projector_mse"] for row in test_rows])
        )
    if "projector_inverse" in included_estimators:
        summary["mean_test_projector_inverse_mse"] = float(
            np.mean([row["projector_inverse_mse"] for row in test_rows])
        )
    if "projector_inverse_vir" in included_estimators:
        summary["mean_test_projector_inverse_vir_mse"] = float(
            np.mean([row["projector_inverse_vir_mse"] for row in test_rows])
        )
    if "shift_projector_inverse" in included_estimators:
        summary["mean_test_shift_projector_inverse_mse"] = float(
            np.mean([row["shift_projector_inverse_mse"] for row in test_rows])
        )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-train-envs", type=int, default=10)
    p.add_argument("--num-test-envs", type=int, default=3)
    p.add_argument("--samples-per-env", type=int, default=500)
    p.add_argument("--unlabeled-samples-per-train-env", type=int, default=500)
    p.add_argument("--unlabeled-samples-per-test-env", type=int, default=None)
    p.add_argument(
        "--oracle-labeled-samples-per-test-env",
        type=int,
        default=50_000,
        help=(
            "Independent target labeled samples reserved for oracle-only analysis. "
            "No estimator receives these labels."
        ),
    )
    p.add_argument("--num-features", type=int, default=50)
    p.add_argument("--num-stable-features", type=int, default=None)
    p.add_argument("--nonlinear-strength", type=float, default=0.0)
    p.add_argument("--stable-mean-scale", type=float, default=0.0)
    p.add_argument("--stable-std-low", type=float, default=0.8)
    p.add_argument("--stable-std-high", type=float, default=1.0)
    p.add_argument("--unstable-mean-scale", type=float, default=0.5)
    p.add_argument("--unstable-std-low", type=float, default=1.0)
    p.add_argument("--unstable-std-high", type=float, default=1.3)
    p.add_argument("--test-unstable-mean-scale", type=float, default=0.8)
    p.add_argument("--test-unstable-std-low", type=float, default=1.2)
    p.add_argument("--test-unstable-std-high", type=float, default=1.6)
    p.add_argument(
        "--lambda-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    )
    p.add_argument("--top-k-grid", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--mir-lambda-grid", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    p.add_argument("--vir-lambda-grid", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    p.add_argument("--projector-lambda-u-grid", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--projector-lambda-perp-grid", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--projector-top-k-grid", type=int, nargs="+", default=[0, 4, 8, 16, 32])
    p.add_argument("--lambda-expand-factor", type=float, default=5.0)
    p.add_argument("--top-k-expand-factor", type=int, default=2)
    p.add_argument(
        "--exclude-estimators",
        type=str,
        nargs="+",
        default=[],
        choices=[
            "ols",
            "plain_inverse",
            "pooled_inverse",
            "pooled_target_vir",
            "target_aware",
            "target_aware_inverse",
            "target_aware_inverse_vir",
            "mir",
            "vir",
            "projector",
            "projector_inverse",
            "projector_inverse_vir",
            "shift_projector_inverse",
        ],
    )
    p.add_argument("--output-json", type=str, default=None)
    default_plot_path = str(
        Path(__file__).resolve().parent / "synthetic_env_test_errors.png"
    )
    p.add_argument("--output-plot", type=str, default=default_plot_path)
    default_vector_plot_path = str(
        Path(__file__).resolve().parent / "synthetic_env_vectors_3d.png"
    )
    p.add_argument("--output-vector-plot", type=str, default=default_vector_plot_path)
    return p


def run_configuration(args, *, emit_progress: bool = True) -> dict[str, object]:
    included_estimators = {
        "ols",
        "plain_inverse",
        "pooled_inverse",
        "pooled_target_vir",
        "target_aware",
        "target_aware_inverse",
        "target_aware_inverse_vir",
        "mir",
        "vir",
        "projector",
        "projector_inverse",
        "projector_inverse_vir",
        "shift_projector_inverse",
    } - set(args.exclude_estimators)
    if emit_progress:
        print("[run] generating synthetic dataset")
    num_stable_features = args.num_stable_features
    if num_stable_features is None:
        num_stable_features = max(1, min(10, args.num_features // 4))
        if num_stable_features >= args.num_features:
            num_stable_features = max(1, args.num_features - 1)

    dataset = generate_synthetic_dataset(
        seed=args.seed,
        num_train_envs=args.num_train_envs,
        num_test_envs=args.num_test_envs,
        samples_per_env=args.samples_per_env,
        unlabeled_samples_per_train_env=args.unlabeled_samples_per_train_env,
        unlabeled_samples_per_test_env=args.unlabeled_samples_per_test_env,
        oracle_labeled_samples_per_test_env=args.oracle_labeled_samples_per_test_env,
        num_features=args.num_features,
        num_stable_features=num_stable_features,
        stable_mean_scale=args.stable_mean_scale,
        stable_std_low=args.stable_std_low,
        stable_std_high=args.stable_std_high,
        unstable_mean_scale=args.unstable_mean_scale,
        unstable_std_low=args.unstable_std_low,
        unstable_std_high=args.unstable_std_high,
        test_unstable_mean_scale=args.test_unstable_mean_scale,
        test_unstable_std_low=args.test_unstable_std_low,
        test_unstable_std_high=args.test_unstable_std_high,
        nonlinear_strength=args.nonlinear_strength,
    )
    if emit_progress:
        print(
            "[run] dataset ready "
            f"train_envs={len(dataset.train_envs)} test_envs={len(dataset.test_envs)} "
            f"samples_per_env={args.samples_per_env} num_features={args.num_features}"
        )

    if emit_progress:
        print("[run] fitting pooled OLS")
    x_train, y_train = stack_environments(dataset.train_envs)
    beta_ols = ols_fit(x_train, y_train)

    if emit_progress:
        print("[run] starting target-aware CV")
    selection = None
    if "target_aware" in included_estimators:
        selection = leave_one_environment_out_selection(
            dataset.train_envs,
            lambda_grid=[float(x) for x in args.lambda_grid],
            top_k_grid=[int(x) for x in args.top_k_grid],
            target_unlabeled_budget=args.unlabeled_samples_per_test_env,
            seed=args.seed,
            lambda_expand_factor=float(args.lambda_expand_factor),
            top_k_expand_factor=int(args.top_k_expand_factor),
        )
    if emit_progress and "target_aware_inverse" in included_estimators:
        print("[run] starting target-aware inverse CV")
    inverse_selection = None
    if "target_aware_inverse" in included_estimators:
        inverse_selection = leave_one_environment_out_selection_inverse(
            dataset.train_envs,
            lambda_grid=[float(x) for x in args.lambda_grid],
            top_k_grid=[int(x) for x in args.top_k_grid],
            target_unlabeled_budget=args.unlabeled_samples_per_test_env,
            seed=args.seed,
            lambda_expand_factor=float(args.lambda_expand_factor),
            top_k_expand_factor=int(args.top_k_expand_factor),
        )
    if emit_progress and "target_aware_inverse_vir" in included_estimators:
        print("[run] starting target-aware inverse + VIR CV")
    inverse_vir_selection = None
    if "target_aware_inverse_vir" in included_estimators:
        inverse_vir_selection = select_target_aware_inverse_vir_by_leave_one_environment_out(
            dataset.train_envs,
            lambda_grid=[float(x) for x in args.lambda_grid],
            lambda_vir_grid=[float(x) for x in args.vir_lambda_grid],
            top_k_grid=[int(x) for x in args.top_k_grid],
            target_unlabeled_budget=args.unlabeled_samples_per_test_env,
            seed=args.seed,
            lambda_expand_factor=float(args.lambda_expand_factor),
            top_k_expand_factor=int(args.top_k_expand_factor),
        )
    if emit_progress and "shift_projector_inverse" in included_estimators:
        print("[run] starting Shift-projector inverse CV")
    shift_projector_inverse_selection = None
    if "shift_projector_inverse" in included_estimators:
        shift_projector_inverse_selection = select_shift_projector_inverse_by_leave_one_environment_out(
            dataset.train_envs,
            lambda_grid=[float(x) for x in args.lambda_grid],
            top_k_grid=[int(x) for x in args.top_k_grid],
            target_unlabeled_budget=args.unlabeled_samples_per_test_env,
            seed=args.seed,
            lambda_expand_factor=float(args.lambda_expand_factor),
            top_k_expand_factor=int(args.top_k_expand_factor),
        )
    if emit_progress and "pooled_target_vir" in included_estimators:
        print("[run] starting pooled target-aware VIR CV")
    pooled_target_vir_selection = None
    if "pooled_target_vir" in included_estimators:
        pooled_target_vir_selection = select_pooled_target_vir_by_leave_one_environment_out(
            dataset.train_envs,
            lambda_grid=[float(x) for x in args.vir_lambda_grid],
            target_unlabeled_budget=args.unlabeled_samples_per_test_env,
            seed=args.seed,
            lambda_expand_factor=float(args.lambda_expand_factor),
        )
    best_lambda = selection.best_lambda_reg if selection is not None else None
    best_top_k = selection.best_top_k if selection is not None else None
    cv_rows = selection.rows if selection is not None else []
    inverse_best_lambda = (
        inverse_selection.best_lambda_reg if inverse_selection is not None else None
    )
    inverse_best_top_k = inverse_selection.best_top_k if inverse_selection is not None else None
    inverse_vir_best_lambda = (
        inverse_vir_selection.best_lambda_reg if inverse_vir_selection is not None else None
    )
    inverse_vir_best_lambda_vir = (
        inverse_vir_selection.best_lambda_vir if inverse_vir_selection is not None else None
    )
    inverse_vir_best_top_k = (
        inverse_vir_selection.best_top_k if inverse_vir_selection is not None else None
    )
    shift_projector_inverse_best_lambda = (
        shift_projector_inverse_selection.best_lambda_reg
        if shift_projector_inverse_selection is not None
        else None
    )
    shift_projector_inverse_best_top_k = (
        shift_projector_inverse_selection.best_top_k
        if shift_projector_inverse_selection is not None
        else None
    )
    pooled_target_vir_best_lambda = (
        pooled_target_vir_selection.best_lambda_reg
        if pooled_target_vir_selection is not None
        else None
    )
    reference_target_env = dataset.test_envs[0]
    if emit_progress and (
        "target_aware" in included_estimators
        or "target_aware_inverse" in included_estimators
        or "target_aware_inverse_vir" in included_estimators
        or "plain_inverse" in included_estimators
        or "pooled_inverse" in included_estimators
        or "pooled_target_vir" in included_estimators
    ):
        print(
            "[run] fitting final target-aware models "
            f"reference_target_env={reference_target_env.env_id}"
        )
    penalized_fit = None
    if "target_aware" in included_estimators:
        penalized_fit = target_aware_penalized_fit(
            dataset.train_envs,
            target_env=reference_target_env,
            lambda_reg=best_lambda,
            top_k=best_top_k,
        )
    target_aware_inverse = None
    if "target_aware_inverse" in included_estimators:
        target_aware_inverse = target_aware_inverse_fit(
            dataset.train_envs,
            target_env=reference_target_env,
            lambda_reg=inverse_best_lambda,
            top_k=inverse_best_top_k,
        )
    target_aware_inverse_vir = None
    if "target_aware_inverse_vir" in included_estimators:
        target_aware_inverse_vir = target_aware_inverse_vir_fit(
            dataset.train_envs,
            target_env=reference_target_env,
            lambda_reg=inverse_vir_best_lambda,
            lambda_vir=inverse_vir_best_lambda_vir,
            top_k=inverse_vir_best_top_k,
        )
    shift_projector_inverse = None
    if "shift_projector_inverse" in included_estimators:
        if emit_progress:
            print("[run] fitting final Shift-projector inverse model")
        shift_projector_inverse = shift_projector_inverse_fit(
            dataset.train_envs,
            target_env=reference_target_env,
            lambda_reg=shift_projector_inverse_best_lambda,
            top_k=shift_projector_inverse_best_top_k,
        )
    plain_inverse = None
    if "plain_inverse" in included_estimators:
        plain_inverse = plain_inverse_fit(
            dataset.train_envs,
            target_env=reference_target_env,
        )
    pooled_inverse = None
    if "pooled_inverse" in included_estimators:
        pooled_inverse = pooled_inverse_fit(
            dataset.train_envs,
            target_env=reference_target_env,
        )
    pooled_target_vir = None
    if "pooled_target_vir" in included_estimators:
        pooled_target_vir = pooled_target_vir_fit(
            dataset.train_envs,
            target_env=reference_target_env,
            lambda_reg=pooled_target_vir_best_lambda,
        )
    if emit_progress and "mir" in included_estimators:
        print("[run] starting MIR CV")
    mir_selection = None
    if "mir" in included_estimators:
        mir_selection = select_lambda_by_leave_one_environment_out(
            dataset.train_envs,
            lambda_grid=[float(x) for x in args.mir_lambda_grid],
            penalty_builder=mir_gamma,
            lambda_expand_factor=float(args.lambda_expand_factor),
        )
    if emit_progress and "vir" in included_estimators:
        print("[run] starting VIR CV")
    vir_selection = None
    if "vir" in included_estimators:
        vir_selection = select_lambda_by_leave_one_environment_out(
            dataset.train_envs,
            lambda_grid=[float(x) for x in args.vir_lambda_grid],
            penalty_builder=vir_gamma,
            lambda_expand_factor=float(args.lambda_expand_factor),
        )
    if emit_progress and ("mir" in included_estimators or "vir" in included_estimators):
        print("[run] fitting final MIR and VIR models")
    mir_fit = None
    if "mir" in included_estimators:
        mir_fit = unlabeled_regularized_fit(
            dataset.train_envs,
            lambda_reg=mir_selection.best_lambda_reg,
            penalty_matrix=mir_gamma(dataset.train_envs),
        )
    vir_fit = None
    if "vir" in included_estimators:
        vir_fit = unlabeled_regularized_fit(
            dataset.train_envs,
            lambda_reg=vir_selection.best_lambda_reg,
            penalty_matrix=vir_gamma(dataset.train_envs),
        )
    if emit_progress and "projector" in included_estimators:
        print("[run] starting projector CV")
    projector_selection = None
    if "projector" in included_estimators:
        projector_selection = select_projector_by_leave_one_environment_out(
            dataset.train_envs,
            lambda_u_grid=[float(x) for x in args.projector_lambda_u_grid],
            lambda_perp_grid=[float(x) for x in args.projector_lambda_perp_grid],
            top_k_grid=[int(x) for x in args.projector_top_k_grid],
            lambda_expand_factor=float(args.lambda_expand_factor),
            top_k_expand_factor=int(args.top_k_expand_factor),
        )
    if emit_progress and "projector" in included_estimators:
        print("[run] fitting final projector model")
    projector_fit = None
    if "projector" in included_estimators:
        projector_fit = projector_regularized_fit(
            dataset.train_envs,
            lambda_u=projector_selection.best_lambda_u,
            lambda_perp=projector_selection.best_lambda_perp,
            top_k=projector_selection.best_top_k,
        )
    if emit_progress and "projector_inverse" in included_estimators:
        print("[run] starting projector inverse CV")
    projector_inverse_selection = None
    if "projector_inverse" in included_estimators:
        projector_inverse_selection = select_projector_inverse_by_leave_one_environment_out(
            dataset.train_envs,
            lambda_u_grid=[float(x) for x in args.projector_lambda_u_grid],
            lambda_perp_grid=[float(x) for x in args.projector_lambda_perp_grid],
            top_k_grid=[int(x) for x in args.projector_top_k_grid],
            target_unlabeled_budget=args.unlabeled_samples_per_test_env,
            seed=args.seed,
            lambda_expand_factor=float(args.lambda_expand_factor),
            top_k_expand_factor=int(args.top_k_expand_factor),
        )
    if emit_progress and "projector_inverse_vir" in included_estimators:
        print("[run] starting projector inverse + VIR CV")
    projector_inverse_vir_selection = None
    if "projector_inverse_vir" in included_estimators:
        projector_inverse_vir_selection = select_projector_inverse_vir_by_leave_one_environment_out(
            dataset.train_envs,
            lambda_u_grid=[float(x) for x in args.projector_lambda_u_grid],
            lambda_perp_grid=[float(x) for x in args.projector_lambda_perp_grid],
            lambda_vir_grid=[float(x) for x in args.vir_lambda_grid],
            top_k_grid=[int(x) for x in args.projector_top_k_grid],
            target_unlabeled_budget=args.unlabeled_samples_per_test_env,
            seed=args.seed,
            lambda_expand_factor=float(args.lambda_expand_factor),
            top_k_expand_factor=int(args.top_k_expand_factor),
        )
    if emit_progress and "projector_inverse" in included_estimators:
        print("[run] fitting final projector inverse model")
    projector_inverse = None
    if "projector_inverse" in included_estimators:
        projector_inverse = projector_inverse_fit(
            dataset.train_envs,
            target_env=reference_target_env,
            lambda_u=projector_inverse_selection.best_lambda_u,
            lambda_perp=projector_inverse_selection.best_lambda_perp,
            top_k=projector_inverse_selection.best_top_k,
        )
    projector_inverse_vir = None
    if "projector_inverse_vir" in included_estimators:
        projector_inverse_vir = projector_inverse_vir_fit(
            dataset.train_envs,
            target_env=reference_target_env,
            lambda_u=projector_inverse_vir_selection.best_lambda_u,
            lambda_perp=projector_inverse_vir_selection.best_lambda_perp,
            lambda_vir=projector_inverse_vir_selection.best_lambda_vir,
            top_k=projector_inverse_vir_selection.best_top_k,
        )
    if emit_progress:
        print("[run] evaluating on test environments")

    results = {
        "seed": args.seed,
        "included_estimators": sorted(included_estimators),
        "excluded_estimators": sorted(set(args.exclude_estimators)),
        "num_train_envs": args.num_train_envs,
        "num_test_envs": args.num_test_envs,
        "samples_per_env": args.samples_per_env,
        "unlabeled_samples_per_train_env": args.unlabeled_samples_per_train_env,
        "unlabeled_samples_per_test_env": (
            args.unlabeled_samples_per_test_env
            if args.unlabeled_samples_per_test_env is not None
            else args.samples_per_env
        ),
        "num_features": args.num_features,
        "num_stable_features": num_stable_features,
        "nonlinear_strength": args.nonlinear_strength,
        "stable_mean_scale": args.stable_mean_scale,
        "stable_std_low": args.stable_std_low,
        "stable_std_high": args.stable_std_high,
        "unstable_mean_scale": args.unstable_mean_scale,
        "unstable_std_low": args.unstable_std_low,
        "unstable_std_high": args.unstable_std_high,
        "test_unstable_mean_scale": args.test_unstable_mean_scale,
        "test_unstable_std_low": args.test_unstable_std_low,
        "test_unstable_std_high": args.test_unstable_std_high,
        "initial_lambda_grid": [float(x) for x in args.lambda_grid],
        "initial_top_k_grid": [int(x) for x in args.top_k_grid],
        "final_lambda_grid": selection.final_lambda_grid if selection is not None else None,
        "final_top_k_grid": selection.final_top_k_grid if selection is not None else None,
        "num_hyperparameter_expansions": selection.num_expansions if selection is not None else None,
        "selected_lambda_reg": best_lambda,
        "selected_top_k": best_top_k,
        "inverse_final_lambda_grid": (
            inverse_selection.final_lambda_grid if inverse_selection is not None else None
        ),
        "inverse_final_top_k_grid": (
            inverse_selection.final_top_k_grid if inverse_selection is not None else None
        ),
        "inverse_num_hyperparameter_expansions": (
            inverse_selection.num_expansions if inverse_selection is not None else None
        ),
        "selected_inverse_lambda_reg": inverse_best_lambda,
        "selected_inverse_top_k": inverse_best_top_k,
        "inverse_vir_final_lambda_grid": (
            inverse_vir_selection.final_lambda_grid if inverse_vir_selection is not None else None
        ),
        "inverse_vir_final_lambda_vir_grid": (
            inverse_vir_selection.final_lambda_vir_grid if inverse_vir_selection is not None else None
        ),
        "inverse_vir_final_top_k_grid": (
            inverse_vir_selection.final_top_k_grid if inverse_vir_selection is not None else None
        ),
        "inverse_vir_num_hyperparameter_expansions": (
            inverse_vir_selection.num_expansions if inverse_vir_selection is not None else None
        ),
        "selected_inverse_vir_lambda_reg": inverse_vir_best_lambda,
        "selected_inverse_vir_lambda_vir": inverse_vir_best_lambda_vir,
        "selected_inverse_vir_top_k": inverse_vir_best_top_k,
        "shift_projector_inverse_final_lambda_grid": (
            shift_projector_inverse_selection.final_lambda_grid
            if shift_projector_inverse_selection is not None
            else None
        ),
        "shift_projector_inverse_final_top_k_grid": (
            shift_projector_inverse_selection.final_top_k_grid
            if shift_projector_inverse_selection is not None
            else None
        ),
        "shift_projector_inverse_num_hyperparameter_expansions": (
            shift_projector_inverse_selection.num_expansions
            if shift_projector_inverse_selection is not None
            else None
        ),
        "selected_shift_projector_inverse_lambda_reg": shift_projector_inverse_best_lambda,
        "selected_shift_projector_inverse_top_k": shift_projector_inverse_best_top_k,
        "pooled_target_vir_final_lambda_grid": (
            pooled_target_vir_selection.final_lambda_grid
            if pooled_target_vir_selection is not None
            else None
        ),
        "pooled_target_vir_num_hyperparameter_expansions": (
            pooled_target_vir_selection.num_expansions
            if pooled_target_vir_selection is not None
            else None
        ),
        "selected_pooled_target_vir_lambda_reg": pooled_target_vir_best_lambda,
        "selected_mir_lambda_reg": mir_selection.best_lambda_reg if mir_selection is not None else None,
        "selected_vir_lambda_reg": vir_selection.best_lambda_reg if vir_selection is not None else None,
        "selected_projector_lambda_u": projector_selection.best_lambda_u if projector_selection is not None else None,
        "selected_projector_lambda_perp": projector_selection.best_lambda_perp if projector_selection is not None else None,
        "selected_projector_top_k": projector_selection.best_top_k if projector_selection is not None else None,
        "selected_projector_inverse_lambda_u": (
            projector_inverse_selection.best_lambda_u if projector_inverse_selection is not None else None
        ),
        "selected_projector_inverse_lambda_perp": (
            projector_inverse_selection.best_lambda_perp if projector_inverse_selection is not None else None
        ),
        "selected_projector_inverse_top_k": (
            projector_inverse_selection.best_top_k if projector_inverse_selection is not None else None
        ),
        "selected_projector_inverse_vir_lambda_u": (
            projector_inverse_vir_selection.best_lambda_u if projector_inverse_vir_selection is not None else None
        ),
        "selected_projector_inverse_vir_lambda_perp": (
            projector_inverse_vir_selection.best_lambda_perp if projector_inverse_vir_selection is not None else None
        ),
        "selected_projector_inverse_vir_lambda_vir": (
            projector_inverse_vir_selection.best_lambda_vir if projector_inverse_vir_selection is not None else None
        ),
        "selected_projector_inverse_vir_top_k": (
            projector_inverse_vir_selection.best_top_k if projector_inverse_vir_selection is not None else None
        ),
        "train_ols_mse": mse(x_train, y_train, beta_ols),
        "reference_target_env_id_for_reported_train_mse": reference_target_env.env_id,
        "train_plain_inverse_mse": plain_inverse.train_mse if plain_inverse is not None else None,
        "train_pooled_inverse_mse": pooled_inverse.train_mse if pooled_inverse is not None else None,
        "train_pooled_target_vir_mse": (
            pooled_target_vir.train_mse if pooled_target_vir is not None else None
        ),
        "train_target_aware_penalized_mse": penalized_fit.train_mse if penalized_fit is not None else None,
        "train_target_aware_inverse_mse": target_aware_inverse.train_mse if target_aware_inverse is not None else None,
        "train_target_aware_inverse_vir_mse": (
            target_aware_inverse_vir.train_mse if target_aware_inverse_vir is not None else None
        ),
        "train_shift_projector_inverse_mse": (
            shift_projector_inverse.train_mse if shift_projector_inverse is not None else None
        ),
        "train_mir_mse": mir_fit.train_mse if mir_fit is not None else None,
        "train_vir_mse": vir_fit.train_mse if vir_fit is not None else None,
        "train_projector_mse": projector_fit.train_mse if projector_fit is not None else None,
        "train_projector_inverse_mse": projector_inverse.train_mse if projector_inverse is not None else None,
        "train_projector_inverse_vir_mse": (
            projector_inverse_vir.train_mse if projector_inverse_vir is not None else None
        ),
        "cv_rows": cv_rows,
        "inverse_cv_rows": inverse_selection.rows if inverse_selection is not None else [],
        "inverse_vir_cv_rows": inverse_vir_selection.rows if inverse_vir_selection is not None else [],
        "shift_projector_inverse_cv_rows": (
            shift_projector_inverse_selection.rows
            if shift_projector_inverse_selection is not None
            else []
        ),
        "pooled_target_vir_cv_rows": (
            pooled_target_vir_selection.rows if pooled_target_vir_selection is not None else []
        ),
        "mir_cv_rows": mir_selection.rows if mir_selection is not None else [],
        "vir_cv_rows": vir_selection.rows if vir_selection is not None else [],
        "projector_cv_rows": projector_selection.rows if projector_selection is not None else [],
        "projector_inverse_cv_rows": (
            projector_inverse_selection.rows if projector_inverse_selection is not None else []
        ),
        "projector_inverse_vir_cv_rows": (
            projector_inverse_vir_selection.rows if projector_inverse_vir_selection is not None else []
        ),
        "final_mir_lambda_grid": mir_selection.final_lambda_grid if mir_selection is not None else None,
        "final_vir_lambda_grid": vir_selection.final_lambda_grid if vir_selection is not None else None,
        "final_projector_lambda_u_grid": (
            projector_selection.final_lambda_u_grid if projector_selection is not None else None
        ),
        "final_projector_lambda_perp_grid": (
            projector_selection.final_lambda_perp_grid if projector_selection is not None else None
        ),
        "final_projector_top_k_grid": (
            projector_selection.final_top_k_grid if projector_selection is not None else None
        ),
        "final_projector_inverse_lambda_u_grid": (
            projector_inverse_selection.final_lambda_u_grid if projector_inverse_selection is not None else None
        ),
        "final_projector_inverse_lambda_perp_grid": (
            projector_inverse_selection.final_lambda_perp_grid if projector_inverse_selection is not None else None
        ),
        "final_projector_inverse_top_k_grid": (
            projector_inverse_selection.final_top_k_grid if projector_inverse_selection is not None else None
        ),
        "final_projector_inverse_vir_lambda_u_grid": (
            projector_inverse_vir_selection.final_lambda_u_grid if projector_inverse_vir_selection is not None else None
        ),
        "final_projector_inverse_vir_lambda_perp_grid": (
            projector_inverse_vir_selection.final_lambda_perp_grid if projector_inverse_vir_selection is not None else None
        ),
        "final_projector_inverse_vir_lambda_vir_grid": (
            projector_inverse_vir_selection.final_lambda_vir_grid if projector_inverse_vir_selection is not None else None
        ),
        "final_projector_inverse_vir_top_k_grid": (
            projector_inverse_vir_selection.final_top_k_grid if projector_inverse_vir_selection is not None else None
        ),
        "mir_num_hyperparameter_expansions": mir_selection.num_expansions if mir_selection is not None else None,
        "vir_num_hyperparameter_expansions": vir_selection.num_expansions if vir_selection is not None else None,
        "projector_num_hyperparameter_expansions": (
            projector_selection.num_expansions if projector_selection is not None else None
        ),
        "projector_inverse_num_hyperparameter_expansions": (
            projector_inverse_selection.num_expansions if projector_inverse_selection is not None else None
        ),
        "projector_inverse_vir_num_hyperparameter_expansions": (
            projector_inverse_vir_selection.num_expansions if projector_inverse_vir_selection is not None else None
        ),
        "test_summary": summarize_env_group(
            dataset,
            beta_ols=beta_ols,
            beta_plain_inverse=plain_inverse.beta if plain_inverse is not None else None,
            target_aware_lambda_reg=best_lambda,
            target_aware_top_k=best_top_k,
            target_aware_inverse_lambda_reg=inverse_best_lambda,
            target_aware_inverse_top_k=inverse_best_top_k,
            target_aware_inverse_vir_lambda_reg=inverse_vir_best_lambda,
            target_aware_inverse_vir_lambda_vir=inverse_vir_best_lambda_vir,
            target_aware_inverse_vir_top_k=inverse_vir_best_top_k,
            beta_mir=mir_fit.beta if mir_fit is not None else None,
            beta_vir=vir_fit.beta if vir_fit is not None else None,
            beta_projector=projector_fit.beta if projector_fit is not None else None,
            projector_inverse_lambda_u=(
                projector_inverse_selection.best_lambda_u
                if projector_inverse_selection is not None
                else 0.0
            ),
            projector_inverse_lambda_perp=(
                projector_inverse_selection.best_lambda_perp
                if projector_inverse_selection is not None
                else 0.0
            ),
            projector_inverse_top_k=(
                projector_inverse_selection.best_top_k
                if projector_inverse_selection is not None
                else 0
            ),
            projector_inverse_vir_lambda_u=(
                projector_inverse_vir_selection.best_lambda_u
                if projector_inverse_vir_selection is not None
                else 0.0
            ),
            projector_inverse_vir_lambda_perp=(
                projector_inverse_vir_selection.best_lambda_perp
                if projector_inverse_vir_selection is not None
                else 0.0
            ),
            projector_inverse_vir_lambda_vir=(
                projector_inverse_vir_selection.best_lambda_vir
                if projector_inverse_vir_selection is not None
                else 0.0
            ),
            projector_inverse_vir_top_k=(
                projector_inverse_vir_selection.best_top_k
                if projector_inverse_vir_selection is not None
                else 0
            ),
            shift_projector_inverse_lambda_reg=(
                shift_projector_inverse_selection.best_lambda_reg
                if shift_projector_inverse_selection is not None
                else 0.0
            ),
            shift_projector_inverse_top_k=(
                shift_projector_inverse_selection.best_top_k
                if shift_projector_inverse_selection is not None
                else 0
            ),
            pooled_target_vir_lambda_reg=(
                pooled_target_vir_selection.best_lambda_reg
                if pooled_target_vir_selection is not None
                else 0.0
            ),
            include_full_target_oracle=args.oracle_labeled_samples_per_test_env > 0,
            included_estimators=included_estimators,
        ),
        "stable_feature_indices": np.flatnonzero(dataset.stable_feature_mask).tolist(),
        "plain_inverse_projection_on_signal_direction": (
            float(dataset.signal_direction @ plain_inverse.beta) if plain_inverse is not None else None
        ),
        "pooled_inverse_projection_on_signal_direction": (
            float(dataset.signal_direction @ pooled_inverse.beta)
            if pooled_inverse is not None
            else None
        ),
        "pooled_target_vir_projection_on_signal_direction": (
            float(dataset.signal_direction @ pooled_target_vir.beta)
            if pooled_target_vir is not None
            else None
        ),
        "target_aware_inverse_projection_on_signal_direction": (
            float(dataset.signal_direction @ target_aware_inverse.beta)
            if target_aware_inverse is not None
            else None
        ),
        "target_aware_inverse_vir_projection_on_signal_direction": (
            float(dataset.signal_direction @ target_aware_inverse_vir.beta)
            if target_aware_inverse_vir is not None
            else None
        ),
        "mir_projection_on_signal_direction": (
            float(dataset.signal_direction @ mir_fit.beta) if mir_fit is not None else None
        ),
        "vir_projection_on_signal_direction": (
            float(dataset.signal_direction @ vir_fit.beta) if vir_fit is not None else None
        ),
        "projector_projection_on_signal_direction": (
            float(dataset.signal_direction @ projector_fit.beta) if projector_fit is not None else None
        ),
        "projector_inverse_projection_on_signal_direction": (
            float(dataset.signal_direction @ projector_inverse.beta)
            if projector_inverse is not None
            else None
        ),
        "projector_inverse_vir_projection_on_signal_direction": (
            float(dataset.signal_direction @ projector_inverse_vir.beta)
            if projector_inverse_vir is not None
            else None
        ),
        "shift_projector_inverse_projection_on_signal_direction": (
            float(dataset.signal_direction @ shift_projector_inverse.beta)
            if shift_projector_inverse is not None
            else None
        ),
        "ols_projection_on_signal_direction": float(dataset.signal_direction @ beta_ols),
    }

    return results


def main(args) -> None:
    results = run_configuration(args, emit_progress=True)

    print(json.dumps(results, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.output_plot is not None:
        plot_path = Path(args.output_plot)
        save_test_error_plot(
            plot_path,
            per_test_environment=results["test_summary"]["per_test_environment"],
            included_estimators=set(results["included_estimators"]),
        )
        print(f"Saved plot to {plot_path}")

    if args.output_vector_plot is not None and args.num_features == 3:
        vector_plot_path = Path(args.output_vector_plot)
        save_3d_vector_plot(
            vector_plot_path,
            signal_direction=dataset.signal_direction,
            nonlinear_direction=dataset.nonlinear_direction,
            beta_ols=beta_ols,
            beta_penalized=penalized_fit.beta,
        )
        print(f"Saved 3D vector plot to {vector_plot_path}")


if __name__ == "__main__":
    main(build_argument_parser().parse_args())
