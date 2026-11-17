from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from synthetic_anti_causal.data import EnvironmentData


@dataclass(frozen=True)
class FitResult:
    beta: np.ndarray
    lambda_reg: float
    top_k: int
    train_mse: float


@dataclass(frozen=True)
class SelectionResult:
    best_lambda_reg: float
    best_top_k: int
    rows: list[dict[str, float]]
    final_lambda_grid: list[float]
    final_top_k_grid: list[int]
    num_expansions: int


@dataclass(frozen=True)
class ScalarSelectionResult:
    best_lambda_reg: float
    rows: list[dict[str, float]]
    final_lambda_grid: list[float]
    num_expansions: int


@dataclass(frozen=True)
class ProjectorSelectionResult:
    best_lambda_u: float
    best_lambda_perp: float
    best_top_k: int
    rows: list[dict[str, float]]
    final_lambda_u_grid: list[float]
    final_lambda_perp_grid: list[float]
    final_top_k_grid: list[int]
    num_expansions: int


@dataclass(frozen=True)
class InverseVirSelectionResult:
    best_lambda_reg: float
    best_lambda_vir: float
    best_top_k: int
    rows: list[dict[str, float]]
    final_lambda_grid: list[float]
    final_lambda_vir_grid: list[float]
    final_top_k_grid: list[int]
    num_expansions: int


@dataclass(frozen=True)
class ProjectorInverseVirSelectionResult:
    best_lambda_u: float
    best_lambda_perp: float
    best_lambda_vir: float
    best_top_k: int
    rows: list[dict[str, float]]
    final_lambda_u_grid: list[float]
    final_lambda_perp_grid: list[float]
    final_lambda_vir_grid: list[float]
    final_top_k_grid: list[int]
    num_expansions: int


def source_cross_moment(envs: list[EnvironmentData]) -> np.ndarray:
    x, y = stack_environments(envs)
    n = float(x.shape[0])
    return (x.T @ y) / n


def stack_environments(envs: list[EnvironmentData]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([env.x for env in envs], axis=0)
    y = np.concatenate([env.y for env in envs], axis=0)
    return x, y


def unlabeled_x(env: EnvironmentData) -> np.ndarray:
    if env.x_unlabeled is not None:
        return env.x_unlabeled
    return env.x


def observed_x(env: EnvironmentData) -> np.ndarray:
    if env.x_unlabeled is None:
        return env.x
    return np.concatenate([env.x, env.x_unlabeled], axis=0)


def unlabeled_second_moment(env: EnvironmentData) -> np.ndarray:
    x_unlab = unlabeled_x(env)
    return (x_unlab.T @ x_unlab) / float(x_unlab.shape[0])


def subsample_unlabeled_env(
    env: EnvironmentData,
    *,
    budget: int | None,
    seed: int,
) -> EnvironmentData:
    if budget is None or env.x_unlabeled is None or budget >= env.x_unlabeled.shape[0]:
        return env
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(env.env_id), int(budget)]))
    indices = rng.choice(env.x_unlabeled.shape[0], size=int(budget), replace=False)
    indices = np.sort(indices)
    return EnvironmentData(
        env_id=env.env_id,
        x=env.x,
        y=env.y,
        x_unlabeled=env.x_unlabeled[indices],
        x_oracle=env.x_oracle,
        y_oracle=env.y_oracle,
        noise_mean=env.noise_mean,
        noise_std=env.noise_std,
    )


def ols_fit(x: np.ndarray, y: np.ndarray, *, jitter: float = 1e-8) -> np.ndarray:
    xtx = x.T @ x
    xty = x.T @ y
    return np.linalg.solve(xtx + jitter * np.eye(xtx.shape[0]), xty)


def predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return x @ beta


def mse(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    residuals = predict(x, beta) - y
    return float(np.mean(residuals**2))


def environment_second_moment(env: EnvironmentData) -> np.ndarray:
    # Use all available covariates for geometry estimation: labeled and unlabeled.
    x_obs = observed_x(env)
    return (x_obs.T @ x_obs) / float(x_obs.shape[0])


def environment_mean(env: EnvironmentData) -> np.ndarray:
    x_obs = observed_x(env)
    return np.mean(x_obs, axis=0)


def environment_covariance(env: EnvironmentData) -> np.ndarray:
    x_obs = observed_x(env)
    centered = x_obs - np.mean(x_obs, axis=0, keepdims=True)
    return (centered.T @ centered) / float(x_obs.shape[0])


def source_statistics(envs: list[EnvironmentData]) -> tuple[np.ndarray, np.ndarray]:
    moments = np.stack([environment_second_moment(env) for env in envs], axis=0)
    mean_moment = np.mean(moments, axis=0)
    variation = np.zeros_like(mean_moment)
    for moment in moments:
        delta = moment - mean_moment
        variation += delta @ delta
    variation /= float(len(envs))
    return mean_moment, variation


def target_weighted_gamma(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    top_k: int,
    eps: float = 1e-12,
) -> np.ndarray:
    mean_moment, variation = source_statistics(source_envs)
    target_moment = unlabeled_second_moment(target_env)

    eigvals, eigvecs = np.linalg.eigh(variation)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    top_k = max(1, min(int(top_k), eigvals.shape[0]))
    top_eigvals = eigvals[:top_k]
    top_eigvecs = eigvecs[:, :top_k]

    s1 = max(float(top_eigvals[0]), eps)
    delta_target = target_moment - mean_moment
    raw_weights = np.zeros(top_k, dtype=np.float64)
    for idx in range(top_k):
        u = top_eigvecs[:, idx]
        d_i = abs(float(u.T @ delta_target @ u))
        raw_weights[idx] = (max(float(top_eigvals[idx]), 0.0) / s1) * d_i

    weight_sum = float(raw_weights.sum())
    if weight_sum <= eps:
        normalized_weights = np.full(top_k, 1.0 / top_k, dtype=np.float64)
    else:
        normalized_weights = raw_weights / weight_sum

    gamma = np.zeros_like(variation)
    for idx in range(top_k):
        u = top_eigvecs[:, idx]
        gamma += normalized_weights[idx] * np.outer(u, u)
    return gamma


def mir_gamma(source_envs: list[EnvironmentData]) -> np.ndarray:
    means = np.stack([environment_mean(env) for env in source_envs], axis=1)
    grand_mean = np.mean(means, axis=1, keepdims=True)
    centered = means - grand_mean
    return (centered @ centered.T) / float(means.shape[1])


def vir_gamma(source_envs: list[EnvironmentData]) -> np.ndarray:
    covariances = np.stack([environment_covariance(env) for env in source_envs], axis=0)
    mean_covariance = np.mean(covariances, axis=0)
    variation = np.zeros_like(mean_covariance)
    for covariance in covariances:
        delta = covariance - mean_covariance
        variation += delta @ delta
    return variation / float(covariances.shape[0])


def target_aware_vir_gamma(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
) -> np.ndarray:
    """VIR covariance variation estimated from source environments and target X."""
    target_x = unlabeled_x(target_env)
    target_centered = target_x - np.mean(target_x, axis=0, keepdims=True)
    target_covariance = (target_centered.T @ target_centered) / float(target_x.shape[0])
    covariances = np.concatenate(
        [
            np.stack([environment_covariance(env) for env in source_envs], axis=0),
            target_covariance[None, :, :],
        ],
        axis=0,
    )
    mean_covariance = np.mean(covariances, axis=0)
    variation = np.zeros_like(mean_covariance)
    for covariance in covariances:
        delta = covariance - mean_covariance
        variation += delta @ delta
    return variation / float(covariances.shape[0])


def source_variation_matrix(source_envs: list[EnvironmentData]) -> np.ndarray:
    mean_moment, variation = source_statistics(source_envs)
    _ = mean_moment
    return variation


def projector_penalty_matrices(
    source_envs: list[EnvironmentData],
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    variation = source_variation_matrix(source_envs)
    eigvals, eigvecs = np.linalg.eigh(variation)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    dim = variation.shape[0]
    top_k = max(0, min(int(top_k), dim))
    if top_k == 0:
        p_u = np.zeros((dim, dim), dtype=np.float64)
    else:
        u = eigvecs[:, :top_k]
        p_u = u @ u.T
    p_perp = np.eye(dim, dtype=np.float64) - p_u
    return p_u, p_perp


def target_aware_penalized_fit(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    lambda_reg: float,
    top_k: int,
    jitter: float = 1e-8,
) -> FitResult:
    x, y = stack_environments(source_envs)
    gamma = target_weighted_gamma(source_envs, target_env=target_env, top_k=top_k)
    n = float(x.shape[0])
    lhs = (x.T @ x) / n + float(lambda_reg) * gamma + jitter * np.eye(x.shape[1])
    rhs = (x.T @ y) / n
    beta = np.linalg.solve(lhs, rhs)
    return FitResult(
        beta=beta,
        lambda_reg=float(lambda_reg),
        top_k=int(top_k),
        train_mse=mse(x, y, beta),
    )


def target_aware_inverse_fit(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    lambda_reg: float,
    top_k: int,
    jitter: float = 1e-8,
) -> FitResult:
    x, y = stack_environments(source_envs)
    target_moment = unlabeled_second_moment(target_env)
    gamma = float(lambda_reg) * target_weighted_gamma(
        source_envs,
        target_env=target_env,
        top_k=top_k,
    )
    beta = np.linalg.solve(
        target_moment + gamma + jitter * np.eye(target_moment.shape[0], dtype=np.float64),
        source_cross_moment(source_envs),
    )
    return FitResult(
        beta=beta,
        lambda_reg=float(lambda_reg),
        top_k=int(top_k),
        train_mse=mse(x, y, beta),
    )


def plain_inverse_fit(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    jitter: float = 1e-8,
) -> FitResult:
    x, y = stack_environments(source_envs)
    target_moment = unlabeled_second_moment(target_env)
    beta = np.linalg.solve(
        target_moment + jitter * np.eye(target_moment.shape[0], dtype=np.float64),
        source_cross_moment(source_envs),
    )
    return FitResult(
        beta=beta,
        lambda_reg=0.0,
        top_k=0,
        train_mse=mse(x, y, beta),
    )


def target_aware_inverse_vir_fit(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    lambda_reg: float,
    lambda_vir: float,
    top_k: int,
    jitter: float = 1e-8,
) -> FitResult:
    x, y = stack_environments(source_envs)
    target_moment = unlabeled_second_moment(target_env)
    gamma = float(lambda_reg) * target_weighted_gamma(
        source_envs,
        target_env=target_env,
        top_k=top_k,
    ) + float(lambda_vir) * vir_gamma(source_envs)
    beta = np.linalg.solve(
        target_moment + gamma + jitter * np.eye(target_moment.shape[0], dtype=np.float64),
        source_cross_moment(source_envs),
    )
    return FitResult(
        beta=beta,
        lambda_reg=float(lambda_reg),
        top_k=int(top_k),
        train_mse=mse(x, y, beta),
    )


def unlabeled_regularized_fit(
    source_envs: list[EnvironmentData],
    *,
    lambda_reg: float,
    penalty_matrix: np.ndarray,
    jitter: float = 1e-8,
) -> FitResult:
    x, y = stack_environments(source_envs)
    n = float(x.shape[0])
    lhs = (x.T @ x) / n + float(lambda_reg) * penalty_matrix + jitter * np.eye(x.shape[1])
    rhs = (x.T @ y) / n
    beta = np.linalg.solve(lhs, rhs)
    return FitResult(
        beta=beta,
        lambda_reg=float(lambda_reg),
        top_k=0,
        train_mse=mse(x, y, beta),
    )


def projector_regularized_fit(
    source_envs: list[EnvironmentData],
    *,
    lambda_u: float,
    lambda_perp: float,
    top_k: int,
    jitter: float = 1e-8,
) -> FitResult:
    x, y = stack_environments(source_envs)
    p_u, p_perp = projector_penalty_matrices(source_envs, top_k=top_k)
    penalty = float(lambda_u) * p_u + float(lambda_perp) * p_perp
    n = float(x.shape[0])
    lhs = (x.T @ x) / n + penalty + jitter * np.eye(x.shape[1], dtype=np.float64)
    rhs = (x.T @ y) / n
    beta = np.linalg.solve(lhs, rhs)
    return FitResult(
        beta=beta,
        lambda_reg=0.0,
        top_k=int(top_k),
        train_mse=mse(x, y, beta),
    )


def projector_inverse_fit(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    lambda_u: float,
    lambda_perp: float,
    top_k: int,
    jitter: float = 1e-8,
) -> FitResult:
    x, y = stack_environments(source_envs)
    target_moment = unlabeled_second_moment(target_env)
    p_u, p_perp = projector_penalty_matrices(source_envs, top_k=top_k)
    gamma = float(lambda_u) * p_u + float(lambda_perp) * p_perp
    beta = np.linalg.solve(
        target_moment + gamma + jitter * np.eye(target_moment.shape[0], dtype=np.float64),
        source_cross_moment(source_envs),
    )
    return FitResult(
        beta=beta,
        lambda_reg=0.0,
        top_k=int(top_k),
        train_mse=mse(x, y, beta),
    )


def shift_projector_penalty(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    top_k: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build the shift-projector penalty from unlabeled covariates.

    This matches the pooled target-projector construction: source environments define directions of
    second-moment variation around the pooled source mean.  A direction is
    penalized only when its target variance has positive standardized excess
    over the source mean variance.
    """
    source_x = [observed_x(env) for env in source_envs]
    pooled_source_x = np.concatenate(source_x, axis=0)
    source_mean = np.mean(pooled_source_x, axis=0)

    source_matrices = np.stack(
        [
            ((x - source_mean).T @ (x - source_mean)) / float(x.shape[0])
            for x in source_x
        ],
        axis=0,
    )
    source_mean_matrix = np.mean(source_matrices, axis=0)
    variation = np.zeros_like(source_mean_matrix)
    for matrix in source_matrices:
        delta = matrix - source_mean_matrix
        variation += delta @ delta
    variation /= float(len(source_matrices))

    eigvals, eigvecs = np.linalg.eigh(variation)
    order = np.argsort(eigvals)[::-1]
    dim = variation.shape[0]
    top_k = max(0, min(int(top_k), dim))
    if top_k == 0:
        return np.zeros((dim, dim), dtype=np.float64)
    basis = eigvecs[:, order[:top_k]]

    target_x = unlabeled_x(target_env)
    target_centered = target_x - source_mean
    target_matrix = (target_centered.T @ target_centered) / float(target_centered.shape[0])

    weights = np.zeros(top_k, dtype=np.float64)
    for index in range(top_k):
        direction = basis[:, index]
        source_direction_values = np.einsum(
            "i,eij,j->e", direction, source_matrices, direction
        )
        source_mean_value = float(direction @ source_mean_matrix @ direction)
        target_value = float(direction @ target_matrix @ direction)
        source_variance = (
            float(np.var(source_direction_values, ddof=1))
            if len(source_direction_values) > 1
            else 0.0
        )
        target_squared_projection = (target_centered @ direction) ** 2
        target_mean_variance = (
            float(np.var(target_squared_projection, ddof=1))
            / float(target_squared_projection.shape[0])
            if target_squared_projection.shape[0] > 1
            else 0.0
        )
        standardized_excess = (target_value - source_mean_value) / np.sqrt(
            max(source_variance + target_mean_variance, eps)
        )
        weights[index] = max(float(standardized_excess), 0.0)

    return (basis * weights) @ basis.T


def shift_projector_inverse_fit(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    lambda_reg: float,
    top_k: int,
    jitter: float = 1e-8,
) -> FitResult:
    """Use target geometry in the inverse and the shift-projector penalty."""
    x, y = stack_environments(source_envs)
    target_moment = unlabeled_second_moment(target_env)
    penalty = shift_projector_penalty(
        source_envs,
        target_env=target_env,
        top_k=top_k,
    )
    beta = np.linalg.solve(
        target_moment
        + float(lambda_reg) * penalty
        + jitter * np.eye(target_moment.shape[0], dtype=np.float64),
        source_cross_moment(source_envs),
    )
    return FitResult(
        beta=beta,
        lambda_reg=float(lambda_reg),
        top_k=int(top_k),
        train_mse=mse(x, y, beta),
    )


def pooled_inverse_fit(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    jitter: float = 1e-8,
) -> FitResult:
    """No-penalty control using the target second-moment matrix."""
    x, y = stack_environments(source_envs)
    target_moment = unlabeled_second_moment(target_env)
    beta = np.linalg.solve(
        target_moment + jitter * np.eye(target_moment.shape[0], dtype=np.float64),
        source_cross_moment(source_envs),
    )
    return FitResult(
        beta=beta,
        lambda_reg=0.0,
        top_k=0,
        train_mse=mse(x, y, beta),
    )


def pooled_target_vir_fit(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    lambda_reg: float,
    jitter: float = 1e-8,
) -> FitResult:
    """Target inverse with a VIR penalty that includes target unlabeled X."""
    x, y = stack_environments(source_envs)
    target_moment = unlabeled_second_moment(target_env)
    penalty = target_aware_vir_gamma(source_envs, target_env=target_env)
    beta = np.linalg.solve(
        target_moment
        + float(lambda_reg) * penalty
        + jitter * np.eye(target_moment.shape[0], dtype=np.float64),
        source_cross_moment(source_envs),
    )
    return FitResult(
        beta=beta,
        lambda_reg=float(lambda_reg),
        top_k=0,
        train_mse=mse(x, y, beta),
    )


def projector_inverse_vir_fit(
    source_envs: list[EnvironmentData],
    *,
    target_env: EnvironmentData,
    lambda_u: float,
    lambda_perp: float,
    lambda_vir: float,
    top_k: int,
    jitter: float = 1e-8,
) -> FitResult:
    x, y = stack_environments(source_envs)
    target_moment = unlabeled_second_moment(target_env)
    p_u, p_perp = projector_penalty_matrices(source_envs, top_k=top_k)
    gamma = (
        float(lambda_u) * p_u
        + float(lambda_perp) * p_perp
        + float(lambda_vir) * vir_gamma(source_envs)
    )
    beta = np.linalg.solve(
        target_moment + gamma + jitter * np.eye(target_moment.shape[0], dtype=np.float64),
        source_cross_moment(source_envs),
    )
    return FitResult(
        beta=beta,
        lambda_reg=0.0,
        top_k=int(top_k),
        train_mse=mse(x, y, beta),
    )


def _expand_lambda_grid(
    grid: list[float],
    *,
    selected: float,
    factor: float,
) -> tuple[list[float], str | None]:
    values = sorted({float(x) for x in grid})
    if not values:
        return values, None

    low = values[0]
    high = values[-1]
    tol = 1e-15

    if abs(selected - low) < tol:
        if abs(selected) < tol:
            if len(values) >= 2:
                candidate = max(values[1] / factor, 1e-8)
            else:
                candidate = 1e-8
        else:
            candidate = max(selected / factor, 1e-12)
        if all(abs(candidate - existing) >= tol for existing in values):
            values.append(candidate)
        return sorted(values), "low"

    if abs(selected - high) < tol:
        candidate = max(selected * factor, 1e-8)
        if all(abs(candidate - existing) >= tol for existing in values):
            values.append(candidate)
        return sorted(values), "high"

    return values, None


def _expand_top_k_grid(
    grid: list[int],
    *,
    selected: int,
    max_top_k: int,
    factor: int,
) -> tuple[list[int], str | None]:
    values = sorted({max(0, int(x)) for x in grid})
    if not values:
        return values, None

    low = values[0]
    high = values[-1]

    if selected == low:
        if low <= 0:
            return values, None
        candidate = max(0, low // max(2, factor))
        if candidate == low:
            candidate = low - 1
        if candidate not in values:
            values.append(candidate)
        return sorted(values), "low"

    if selected == high:
        if high >= max_top_k:
            return values, None
        candidate = min(max_top_k, high * max(2, factor))
        if candidate == high:
            return values, None
        if candidate not in values:
            values.append(candidate)
        return sorted(values), "high"

    return values, None


def leave_one_environment_out_selection(
    envs: list[EnvironmentData],
    *,
    lambda_grid: list[float],
    top_k_grid: list[int],
    target_unlabeled_budget: int | None = None,
    seed: int = 0,
    lambda_expand_factor: float = 5.0,
    top_k_expand_factor: int = 2,
) -> SelectionResult:
    if len(envs) < 2:
        raise ValueError("Need at least two environments for leave-one-environment-out selection.")

    rows: list[dict[str, float]] = []
    cached_results: dict[tuple[float, int, int], float] = {}
    current_lambda_grid = sorted({float(x) for x in lambda_grid})
    current_top_k_grid = sorted({max(1, int(x)) for x in top_k_grid})
    max_top_k = int(envs[0].x.shape[1])
    num_expansions = 0

    while True:
        print(
            "[target-aware-cv] round="
            f"{num_expansions} lambda_grid={current_lambda_grid} top_k_grid={current_top_k_grid}"
        )
        best_params = None
        best_score = float("inf")

        for lambda_reg in current_lambda_grid:
            for top_k in current_top_k_grid:
                fold_mses = []
                for heldout_idx in range(len(envs)):
                    target_env = subsample_unlabeled_env(
                        envs[heldout_idx],
                        budget=target_unlabeled_budget,
                        seed=seed,
                    )
                    cache_key = (float(lambda_reg), int(top_k), int(target_env.env_id))
                    if cache_key in cached_results:
                        val_mse = cached_results[cache_key]
                    else:
                        source_envs = [env for idx, env in enumerate(envs) if idx != heldout_idx]
                        fit = target_aware_penalized_fit(
                            source_envs,
                            target_env=target_env,
                            lambda_reg=lambda_reg,
                            top_k=top_k,
                        )
                        val_mse = mse(target_env.x, target_env.y, fit.beta)
                        cached_results[cache_key] = float(val_mse)
                        rows.append(
                            {
                                "lambda_reg": float(lambda_reg),
                                "top_k": float(top_k),
                                "heldout_env": float(target_env.env_id),
                                "val_mse": float(val_mse),
                            }
                        )

                    fold_mses.append(float(val_mse))

                mean_val_mse = float(np.mean(fold_mses))
                if mean_val_mse < best_score:
                    best_score = mean_val_mse
                    best_params = (float(lambda_reg), int(top_k))

        if best_params is None:
            raise RuntimeError("Failed to select hyperparameters.")

        expanded_lambda_grid, lambda_boundary = _expand_lambda_grid(
            current_lambda_grid,
            selected=best_params[0],
            factor=lambda_expand_factor,
        )
        expanded_top_k_grid, top_k_boundary = _expand_top_k_grid(
            current_top_k_grid,
            selected=best_params[1],
            max_top_k=max_top_k,
            factor=top_k_expand_factor,
        )

        changed = False
        if lambda_boundary is not None and expanded_lambda_grid != current_lambda_grid:
            current_lambda_grid = expanded_lambda_grid
            changed = True
        if top_k_boundary is not None and expanded_top_k_grid != current_top_k_grid:
            current_top_k_grid = expanded_top_k_grid
            changed = True

        if not changed:
            print(
                "[target-aware-cv] selected "
                f"lambda={best_params[0]} top_k={best_params[1]} mean_val_mse={best_score}"
            )
            return SelectionResult(
                best_lambda_reg=best_params[0],
                best_top_k=best_params[1],
                rows=rows,
                final_lambda_grid=current_lambda_grid,
                final_top_k_grid=current_top_k_grid,
                num_expansions=num_expansions,
            )

        num_expansions += 1
        print(
            "[target-aware-cv] boundary expansion triggered "
            f"new_lambda_grid={current_lambda_grid} new_top_k_grid={current_top_k_grid}"
        )


def select_lambda_by_leave_one_environment_out(
    envs: list[EnvironmentData],
    *,
    lambda_grid: list[float],
    penalty_builder,
    lambda_expand_factor: float = 5.0,
) -> ScalarSelectionResult:
    if len(envs) < 2:
        raise ValueError("Need at least two environments for leave-one-environment-out selection.")

    rows: list[dict[str, float]] = []
    cached_results: dict[tuple[float, int], float] = {}
    cached_penalties: dict[tuple[int, ...], np.ndarray] = {}
    current_lambda_grid = sorted({float(x) for x in lambda_grid})
    num_expansions = 0

    while True:
        penalty_name = getattr(penalty_builder, "__name__", "penalty")
        print(
            f"[{penalty_name}-cv] round={num_expansions} lambda_grid={current_lambda_grid}"
        )
        best_lambda = None
        best_score = float("inf")

        for lambda_reg in current_lambda_grid:
            fold_mses = []
            for heldout_idx in range(len(envs)):
                target_env = envs[heldout_idx]
                cache_key = (float(lambda_reg), int(target_env.env_id))
                if cache_key in cached_results:
                    val_mse = cached_results[cache_key]
                else:
                    source_envs = [env for idx, env in enumerate(envs) if idx != heldout_idx]
                    env_key = tuple(env.env_id for env in source_envs)
                    penalty = cached_penalties.get(env_key)
                    if penalty is None:
                        penalty = penalty_builder(source_envs)
                        cached_penalties[env_key] = penalty
                    fit = unlabeled_regularized_fit(
                        source_envs,
                        lambda_reg=lambda_reg,
                        penalty_matrix=penalty,
                    )
                    val_mse = mse(target_env.x, target_env.y, fit.beta)
                    cached_results[cache_key] = float(val_mse)
                    rows.append(
                        {
                            "lambda_reg": float(lambda_reg),
                            "heldout_env": float(target_env.env_id),
                            "val_mse": float(val_mse),
                        }
                    )
                fold_mses.append(float(val_mse))

            mean_val_mse = float(np.mean(fold_mses))
            if mean_val_mse < best_score:
                best_score = mean_val_mse
                best_lambda = float(lambda_reg)

        if best_lambda is None:
            raise RuntimeError("Failed to select regularization strength.")

        expanded_lambda_grid, lambda_boundary = _expand_lambda_grid(
            current_lambda_grid,
            selected=best_lambda,
            factor=lambda_expand_factor,
        )
        if lambda_boundary is None or expanded_lambda_grid == current_lambda_grid:
            print(
                f"[{penalty_name}-cv] selected lambda={best_lambda} mean_val_mse={best_score}"
            )
            return ScalarSelectionResult(
                best_lambda_reg=best_lambda,
                rows=rows,
                final_lambda_grid=current_lambda_grid,
                num_expansions=num_expansions,
            )

        current_lambda_grid = expanded_lambda_grid
        num_expansions += 1
        print(
            f"[{penalty_name}-cv] boundary expansion triggered new_lambda_grid={current_lambda_grid}"
        )


def select_pooled_target_vir_by_leave_one_environment_out(
    envs: list[EnvironmentData],
    *,
    lambda_grid: list[float],
    target_unlabeled_budget: int | None = None,
    seed: int = 0,
    lambda_expand_factor: float = 5.0,
) -> ScalarSelectionResult:
    """Tune pooled target-aware VIR by leave-one-environment-out CV."""
    if len(envs) < 2:
        raise ValueError("Need at least two environments for leave-one-environment-out selection.")

    rows: list[dict[str, float]] = []
    cached_results: dict[tuple[float, int], float] = {}
    current_lambda_grid = sorted({max(0.0, float(x)) for x in lambda_grid})
    num_expansions = 0
    while True:
        print(
            "[pooled-target-vir-cv] round="
            f"{num_expansions} lambda_grid={current_lambda_grid}"
        )
        best_lambda = None
        best_score = float("inf")
        for lambda_reg in current_lambda_grid:
            fold_mses = []
            for heldout_idx in range(len(envs)):
                target_env = subsample_unlabeled_env(
                    envs[heldout_idx],
                    budget=target_unlabeled_budget,
                    seed=seed,
                )
                cache_key = (float(lambda_reg), int(target_env.env_id))
                if cache_key not in cached_results:
                    source_envs = [env for idx, env in enumerate(envs) if idx != heldout_idx]
                    fit = pooled_target_vir_fit(
                        source_envs,
                        target_env=target_env,
                        lambda_reg=lambda_reg,
                    )
                    cached_results[cache_key] = mse(target_env.x, target_env.y, fit.beta)
                    rows.append(
                        {
                            "lambda_reg": float(lambda_reg),
                            "heldout_env": float(target_env.env_id),
                            "val_mse": float(cached_results[cache_key]),
                        }
                    )
                fold_mses.append(float(cached_results[cache_key]))
            mean_val_mse = float(np.mean(fold_mses))
            if mean_val_mse < best_score:
                best_score = mean_val_mse
                best_lambda = float(lambda_reg)

        if best_lambda is None:
            raise RuntimeError("Failed to select regularization strength.")
        expanded_lambda_grid, lambda_boundary = _expand_lambda_grid(
            current_lambda_grid,
            selected=best_lambda,
            factor=lambda_expand_factor,
        )
        if lambda_boundary is None or expanded_lambda_grid == current_lambda_grid:
            print(
                "[pooled-target-vir-cv] selected "
                f"lambda={best_lambda} mean_val_mse={best_score}"
            )
            return ScalarSelectionResult(
                best_lambda_reg=best_lambda,
                rows=rows,
                final_lambda_grid=current_lambda_grid,
                num_expansions=num_expansions,
            )
        current_lambda_grid = expanded_lambda_grid
        num_expansions += 1
        print(
            "[pooled-target-vir-cv] boundary expansion triggered "
            f"new_lambda_grid={current_lambda_grid}"
        )


def select_projector_by_leave_one_environment_out(
    envs: list[EnvironmentData],
    *,
    lambda_u_grid: list[float],
    lambda_perp_grid: list[float],
    top_k_grid: list[int],
    lambda_expand_factor: float = 5.0,
    top_k_expand_factor: int = 2,
) -> ProjectorSelectionResult:
    if len(envs) < 2:
        raise ValueError("Need at least two environments for leave-one-environment-out selection.")

    rows: list[dict[str, float]] = []
    cache: dict[tuple[float, float, int, int], float] = {}
    projector_cache: dict[tuple[tuple[int, ...], int], tuple[np.ndarray, np.ndarray]] = {}
    current_lambda_u_grid = sorted({max(0.0, float(x)) for x in lambda_u_grid})
    current_lambda_perp_grid = sorted({max(0.0, float(x)) for x in lambda_perp_grid})
    current_top_k_grid = sorted({max(0, int(x)) for x in top_k_grid})
    max_top_k = int(envs[0].x.shape[1])
    num_expansions = 0

    while True:
        print(
            "[projector-cv] round="
            f"{num_expansions} lambda_u_grid={current_lambda_u_grid} "
            f"lambda_perp_grid={current_lambda_perp_grid} top_k_grid={current_top_k_grid}"
        )
        best_params = None
        best_score = float("inf")
        combinations = [
            (lambda_u, lambda_perp, top_k)
            for lambda_u in current_lambda_u_grid
            for lambda_perp in current_lambda_perp_grid
            for top_k in current_top_k_grid
        ]
        for lambda_u, lambda_perp, top_k in tqdm(
            combinations,
            desc=f"projector-cv round {num_expansions}",
            leave=False,
        ):
                    fold_mses = []
                    for heldout_idx in range(len(envs)):
                        target_env = envs[heldout_idx]
                        cache_key = (float(lambda_u), float(lambda_perp), int(top_k), int(target_env.env_id))
                        if cache_key in cache:
                            val_mse = cache[cache_key]
                        else:
                            source_envs = [env for idx, env in enumerate(envs) if idx != heldout_idx]
                            proj_key = (tuple(env.env_id for env in source_envs), int(top_k))
                            if proj_key not in projector_cache:
                                projector_cache[proj_key] = projector_penalty_matrices(
                                    source_envs,
                                    top_k=top_k,
                                )
                            p_u, p_perp = projector_cache[proj_key]
                            x, y = stack_environments(source_envs)
                            n = float(x.shape[0])
                            penalty = float(lambda_u) * p_u + float(lambda_perp) * p_perp
                            beta = np.linalg.solve(
                                (x.T @ x) / n + penalty + 1e-8 * np.eye(x.shape[1], dtype=np.float64),
                                (x.T @ y) / n,
                            )
                            val_mse = mse(target_env.x, target_env.y, beta)
                            cache[cache_key] = float(val_mse)
                            rows.append(
                                {
                                    "lambda_u": float(lambda_u),
                                    "lambda_perp": float(lambda_perp),
                                    "top_k": float(top_k),
                                    "heldout_env": float(target_env.env_id),
                                    "val_mse": float(val_mse),
                                }
                            )
                        fold_mses.append(float(val_mse))

                    mean_val_mse = float(np.mean(fold_mses))
                    if mean_val_mse < best_score:
                        best_score = mean_val_mse
                        best_params = (float(lambda_u), float(lambda_perp), int(top_k))

        if best_params is None:
            raise RuntimeError("Failed to select projector-penalty hyperparameters.")

        expanded_lambda_u_grid, lambda_u_boundary = _expand_lambda_grid(
            current_lambda_u_grid,
            selected=best_params[0],
            factor=lambda_expand_factor,
        )
        expanded_lambda_perp_grid, lambda_perp_boundary = _expand_lambda_grid(
            current_lambda_perp_grid,
            selected=best_params[1],
            factor=lambda_expand_factor,
        )
        expanded_top_k_grid, top_k_boundary = _expand_top_k_grid(
            current_top_k_grid,
            selected=best_params[2],
            max_top_k=max_top_k,
            factor=top_k_expand_factor,
        )

        changed = False
        if lambda_u_boundary is not None and expanded_lambda_u_grid != current_lambda_u_grid:
            current_lambda_u_grid = expanded_lambda_u_grid
            changed = True
        if lambda_perp_boundary is not None and expanded_lambda_perp_grid != current_lambda_perp_grid:
            current_lambda_perp_grid = expanded_lambda_perp_grid
            changed = True
        if top_k_boundary is not None and expanded_top_k_grid != current_top_k_grid:
            current_top_k_grid = expanded_top_k_grid
            changed = True

        if not changed:
            print(
                "[projector-cv] selected "
                f"lambda_u={best_params[0]} lambda_perp={best_params[1]} "
                f"top_k={best_params[2]} mean_val_mse={best_score}"
            )
            return ProjectorSelectionResult(
                best_lambda_u=best_params[0],
                best_lambda_perp=best_params[1],
                best_top_k=best_params[2],
                rows=rows,
                final_lambda_u_grid=current_lambda_u_grid,
                final_lambda_perp_grid=current_lambda_perp_grid,
                final_top_k_grid=current_top_k_grid,
                num_expansions=num_expansions,
            )

        num_expansions += 1
        print(
            "[projector-cv] boundary expansion triggered "
            f"new_lambda_u_grid={current_lambda_u_grid} "
            f"new_lambda_perp_grid={current_lambda_perp_grid} "
            f"new_top_k_grid={current_top_k_grid}"
        )


def leave_one_environment_out_selection_inverse(
    envs: list[EnvironmentData],
    *,
    lambda_grid: list[float],
    top_k_grid: list[int],
    target_unlabeled_budget: int | None = None,
    seed: int = 0,
    lambda_expand_factor: float = 5.0,
    top_k_expand_factor: int = 2,
) -> SelectionResult:
    if len(envs) < 2:
        raise ValueError("Need at least two environments for leave-one-environment-out selection.")

    rows: list[dict[str, float]] = []
    cached_results: dict[tuple[float, int, int], float] = {}
    current_lambda_grid = sorted({float(x) for x in lambda_grid})
    current_top_k_grid = sorted({max(1, int(x)) for x in top_k_grid})
    max_top_k = int(envs[0].x.shape[1])
    num_expansions = 0

    while True:
        print(
            "[target-aware-inverse-cv] round="
            f"{num_expansions} lambda_grid={current_lambda_grid} top_k_grid={current_top_k_grid}"
        )
        best_params = None
        best_score = float("inf")

        for lambda_reg in current_lambda_grid:
            for top_k in current_top_k_grid:
                fold_mses = []
                for heldout_idx in range(len(envs)):
                    target_env = subsample_unlabeled_env(
                        envs[heldout_idx],
                        budget=target_unlabeled_budget,
                        seed=seed,
                    )
                    cache_key = (float(lambda_reg), int(top_k), int(target_env.env_id))
                    if cache_key in cached_results:
                        val_mse = cached_results[cache_key]
                    else:
                        source_envs = [env for idx, env in enumerate(envs) if idx != heldout_idx]
                        fit = target_aware_inverse_fit(
                            source_envs,
                            target_env=target_env,
                            lambda_reg=lambda_reg,
                            top_k=top_k,
                        )
                        val_mse = mse(target_env.x, target_env.y, fit.beta)
                        cached_results[cache_key] = float(val_mse)
                        rows.append(
                            {
                                "lambda_reg": float(lambda_reg),
                                "top_k": float(top_k),
                                "heldout_env": float(target_env.env_id),
                                "val_mse": float(val_mse),
                            }
                        )
                    fold_mses.append(float(val_mse))

                mean_val_mse = float(np.mean(fold_mses))
                if mean_val_mse < best_score:
                    best_score = mean_val_mse
                    best_params = (float(lambda_reg), int(top_k))

        if best_params is None:
            raise RuntimeError("Failed to select hyperparameters.")

        expanded_lambda_grid, lambda_boundary = _expand_lambda_grid(
            current_lambda_grid,
            selected=best_params[0],
            factor=lambda_expand_factor,
        )
        expanded_top_k_grid, top_k_boundary = _expand_top_k_grid(
            current_top_k_grid,
            selected=best_params[1],
            max_top_k=max_top_k,
            factor=top_k_expand_factor,
        )

        changed = False
        if lambda_boundary is not None and expanded_lambda_grid != current_lambda_grid:
            current_lambda_grid = expanded_lambda_grid
            changed = True
        if top_k_boundary is not None and expanded_top_k_grid != current_top_k_grid:
            current_top_k_grid = expanded_top_k_grid
            changed = True

        if not changed:
            print(
                "[target-aware-inverse-cv] selected "
                f"lambda={best_params[0]} top_k={best_params[1]} mean_val_mse={best_score}"
            )
            return SelectionResult(
                best_lambda_reg=best_params[0],
                best_top_k=best_params[1],
                rows=rows,
                final_lambda_grid=current_lambda_grid,
                final_top_k_grid=current_top_k_grid,
                num_expansions=num_expansions,
            )

        num_expansions += 1
        print(
            "[target-aware-inverse-cv] boundary expansion triggered "
            f"new_lambda_grid={current_lambda_grid} new_top_k_grid={current_top_k_grid}"
        )


def select_shift_projector_inverse_by_leave_one_environment_out(
    envs: list[EnvironmentData],
    *,
    lambda_grid: list[float],
    top_k_grid: list[int],
    target_unlabeled_budget: int | None = None,
    seed: int = 0,
    lambda_expand_factor: float = 5.0,
    top_k_expand_factor: int = 2,
) -> SelectionResult:
    """Tune the Shift-projector inverse by pseudo-target CV."""
    if len(envs) < 2:
        raise ValueError("Need at least two environments for leave-one-environment-out selection.")

    rows: list[dict[str, float]] = []
    cached_results: dict[tuple[float, int, int], float] = {}
    current_lambda_grid = sorted({max(0.0, float(x)) for x in lambda_grid})
    current_top_k_grid = sorted({max(0, int(x)) for x in top_k_grid})
    max_top_k = int(envs[0].x.shape[1])
    num_expansions = 0

    while True:
        print(
            "[shift-projector-inverse-cv] round="
            f"{num_expansions} lambda_grid={current_lambda_grid} "
            f"top_k_grid={current_top_k_grid}"
        )
        best_params = None
        best_score = float("inf")
        for lambda_reg in current_lambda_grid:
            for top_k in current_top_k_grid:
                fold_mses = []
                for heldout_idx in range(len(envs)):
                    target_env = subsample_unlabeled_env(
                        envs[heldout_idx],
                        budget=target_unlabeled_budget,
                        seed=seed,
                    )
                    cache_key = (float(lambda_reg), int(top_k), int(target_env.env_id))
                    if cache_key not in cached_results:
                        source_envs = [env for idx, env in enumerate(envs) if idx != heldout_idx]
                        fit = shift_projector_inverse_fit(
                            source_envs,
                            target_env=target_env,
                            lambda_reg=lambda_reg,
                            top_k=top_k,
                        )
                        cached_results[cache_key] = mse(target_env.x, target_env.y, fit.beta)
                        rows.append(
                            {
                                "lambda_reg": float(lambda_reg),
                                "top_k": float(top_k),
                                "heldout_env": float(target_env.env_id),
                                "val_mse": float(cached_results[cache_key]),
                            }
                        )
                    fold_mses.append(float(cached_results[cache_key]))

                mean_val_mse = float(np.mean(fold_mses))
                if mean_val_mse < best_score:
                    best_score = mean_val_mse
                    best_params = (float(lambda_reg), int(top_k))

        if best_params is None:
            raise RuntimeError("Failed to select hyperparameters.")
        expanded_lambda_grid, lambda_boundary = _expand_lambda_grid(
            current_lambda_grid,
            selected=best_params[0],
            factor=lambda_expand_factor,
        )
        expanded_top_k_grid, top_k_boundary = _expand_top_k_grid(
            current_top_k_grid,
            selected=best_params[1],
            max_top_k=max_top_k,
            factor=top_k_expand_factor,
        )
        changed = False
        if lambda_boundary is not None and expanded_lambda_grid != current_lambda_grid:
            current_lambda_grid = expanded_lambda_grid
            changed = True
        if top_k_boundary is not None and expanded_top_k_grid != current_top_k_grid:
            current_top_k_grid = expanded_top_k_grid
            changed = True
        if not changed:
            print(
                "[shift-projector-inverse-cv] selected "
                f"lambda={best_params[0]} top_k={best_params[1]} mean_val_mse={best_score}"
            )
            return SelectionResult(
                best_lambda_reg=best_params[0],
                best_top_k=best_params[1],
                rows=rows,
                final_lambda_grid=current_lambda_grid,
                final_top_k_grid=current_top_k_grid,
                num_expansions=num_expansions,
            )
        num_expansions += 1
        print(
            "[shift-projector-inverse-cv] boundary expansion triggered "
            f"new_lambda_grid={current_lambda_grid} new_top_k_grid={current_top_k_grid}"
        )


def select_projector_inverse_by_leave_one_environment_out(
    envs: list[EnvironmentData],
    *,
    lambda_u_grid: list[float],
    lambda_perp_grid: list[float],
    top_k_grid: list[int],
    target_unlabeled_budget: int | None = None,
    seed: int = 0,
    lambda_expand_factor: float = 5.0,
    top_k_expand_factor: int = 2,
) -> ProjectorSelectionResult:
    if len(envs) < 2:
        raise ValueError("Need at least two environments for leave-one-environment-out selection.")

    rows: list[dict[str, float]] = []
    cache: dict[tuple[float, float, int, int], float] = {}
    current_lambda_u_grid = sorted({max(0.0, float(x)) for x in lambda_u_grid})
    current_lambda_perp_grid = sorted({max(0.0, float(x)) for x in lambda_perp_grid})
    current_top_k_grid = sorted({max(0, int(x)) for x in top_k_grid})
    max_top_k = int(envs[0].x.shape[1])
    num_expansions = 0

    while True:
        print(
            "[projector-inverse-cv] round="
            f"{num_expansions} lambda_u_grid={current_lambda_u_grid} "
            f"lambda_perp_grid={current_lambda_perp_grid} top_k_grid={current_top_k_grid}"
        )
        best_params = None
        best_score = float("inf")
        combinations = [
            (lambda_u, lambda_perp, top_k)
            for lambda_u in current_lambda_u_grid
            for lambda_perp in current_lambda_perp_grid
            for top_k in current_top_k_grid
        ]
        for lambda_u, lambda_perp, top_k in tqdm(
            combinations,
            desc=f"projector-inverse-cv round {num_expansions}",
            leave=False,
        ):
                    fold_mses = []
                    for heldout_idx in range(len(envs)):
                        target_env = subsample_unlabeled_env(
                            envs[heldout_idx],
                            budget=target_unlabeled_budget,
                            seed=seed,
                        )
                        cache_key = (float(lambda_u), float(lambda_perp), int(top_k), int(target_env.env_id))
                        if cache_key in cache:
                            val_mse = cache[cache_key]
                        else:
                            source_envs = [env for idx, env in enumerate(envs) if idx != heldout_idx]
                            fit = projector_inverse_fit(
                                source_envs,
                                target_env=target_env,
                                lambda_u=lambda_u,
                                lambda_perp=lambda_perp,
                                top_k=top_k,
                            )
                            val_mse = mse(target_env.x, target_env.y, fit.beta)
                            cache[cache_key] = float(val_mse)
                            rows.append(
                                {
                                    "lambda_u": float(lambda_u),
                                    "lambda_perp": float(lambda_perp),
                                    "top_k": float(top_k),
                                    "heldout_env": float(target_env.env_id),
                                    "val_mse": float(val_mse),
                                }
                            )
                        fold_mses.append(float(val_mse))

                    mean_val_mse = float(np.mean(fold_mses))
                    if mean_val_mse < best_score:
                        best_score = mean_val_mse
                        best_params = (float(lambda_u), float(lambda_perp), int(top_k))

        if best_params is None:
            raise RuntimeError("Failed to select hyperparameters.")

        expanded_lambda_u_grid, lambda_u_boundary = _expand_lambda_grid(
            current_lambda_u_grid,
            selected=best_params[0],
            factor=lambda_expand_factor,
        )
        expanded_lambda_perp_grid, lambda_perp_boundary = _expand_lambda_grid(
            current_lambda_perp_grid,
            selected=best_params[1],
            factor=lambda_expand_factor,
        )
        expanded_top_k_grid, top_k_boundary = _expand_top_k_grid(
            current_top_k_grid,
            selected=best_params[2],
            max_top_k=max_top_k,
            factor=top_k_expand_factor,
        )

        changed = False
        if lambda_u_boundary is not None and expanded_lambda_u_grid != current_lambda_u_grid:
            current_lambda_u_grid = expanded_lambda_u_grid
            changed = True
        if lambda_perp_boundary is not None and expanded_lambda_perp_grid != current_lambda_perp_grid:
            current_lambda_perp_grid = expanded_lambda_perp_grid
            changed = True
        if top_k_boundary is not None and expanded_top_k_grid != current_top_k_grid:
            current_top_k_grid = expanded_top_k_grid
            changed = True

        if not changed:
            print(
                "[projector-inverse-cv] selected "
                f"lambda_u={best_params[0]} lambda_perp={best_params[1]} "
                f"top_k={best_params[2]} mean_val_mse={best_score}"
            )
            return ProjectorSelectionResult(
                best_lambda_u=best_params[0],
                best_lambda_perp=best_params[1],
                best_top_k=best_params[2],
                rows=rows,
                final_lambda_u_grid=current_lambda_u_grid,
                final_lambda_perp_grid=current_lambda_perp_grid,
                final_top_k_grid=current_top_k_grid,
                num_expansions=num_expansions,
            )

        num_expansions += 1
        print(
            "[projector-inverse-cv] boundary expansion triggered "
            f"new_lambda_u_grid={current_lambda_u_grid} "
            f"new_lambda_perp_grid={current_lambda_perp_grid} "
            f"new_top_k_grid={current_top_k_grid}"
        )


def select_target_aware_inverse_vir_by_leave_one_environment_out(
    envs: list[EnvironmentData],
    *,
    lambda_grid: list[float],
    lambda_vir_grid: list[float],
    top_k_grid: list[int],
    target_unlabeled_budget: int | None = None,
    seed: int = 0,
    lambda_expand_factor: float = 5.0,
    top_k_expand_factor: int = 2,
) -> InverseVirSelectionResult:
    if len(envs) < 2:
        raise ValueError("Need at least two environments for leave-one-environment-out selection.")

    rows: list[dict[str, float]] = []
    cache: dict[tuple[float, float, int, int], float] = {}
    current_lambda_grid = sorted({float(x) for x in lambda_grid})
    current_lambda_vir_grid = sorted({max(0.0, float(x)) for x in lambda_vir_grid})
    current_top_k_grid = sorted({max(1, int(x)) for x in top_k_grid})
    max_top_k = int(envs[0].x.shape[1])
    num_expansions = 0

    while True:
        print(
            "[target-aware-inverse-vir-cv] round="
            f"{num_expansions} lambda_grid={current_lambda_grid} "
            f"lambda_vir_grid={current_lambda_vir_grid} top_k_grid={current_top_k_grid}"
        )
        best_params = None
        best_score = float("inf")
        combinations = [
            (lambda_reg, lambda_vir, top_k)
            for lambda_reg in current_lambda_grid
            for lambda_vir in current_lambda_vir_grid
            for top_k in current_top_k_grid
        ]
        for lambda_reg, lambda_vir, top_k in tqdm(
            combinations,
            desc=f"target-aware-inverse-vir-cv round {num_expansions}",
            leave=False,
        ):
                    fold_mses = []
                    for heldout_idx in range(len(envs)):
                        target_env = subsample_unlabeled_env(
                            envs[heldout_idx],
                            budget=target_unlabeled_budget,
                            seed=seed,
                        )
                        cache_key = (
                            float(lambda_reg),
                            float(lambda_vir),
                            int(top_k),
                            int(target_env.env_id),
                        )
                        if cache_key in cache:
                            val_mse = cache[cache_key]
                        else:
                            source_envs = [env for idx, env in enumerate(envs) if idx != heldout_idx]
                            fit = target_aware_inverse_vir_fit(
                                source_envs,
                                target_env=target_env,
                                lambda_reg=lambda_reg,
                                lambda_vir=lambda_vir,
                                top_k=top_k,
                            )
                            val_mse = mse(target_env.x, target_env.y, fit.beta)
                            cache[cache_key] = float(val_mse)
                            rows.append(
                                {
                                    "lambda_reg": float(lambda_reg),
                                    "lambda_vir": float(lambda_vir),
                                    "top_k": float(top_k),
                                    "heldout_env": float(target_env.env_id),
                                    "val_mse": float(val_mse),
                                }
                            )
                        fold_mses.append(float(val_mse))
                    mean_val_mse = float(np.mean(fold_mses))
                    if mean_val_mse < best_score:
                        best_score = mean_val_mse
                        best_params = (float(lambda_reg), float(lambda_vir), int(top_k))

        if best_params is None:
            raise RuntimeError("Failed to select hyperparameters.")

        expanded_lambda_grid, lambda_boundary = _expand_lambda_grid(
            current_lambda_grid, selected=best_params[0], factor=lambda_expand_factor
        )
        expanded_lambda_vir_grid, lambda_vir_boundary = _expand_lambda_grid(
            current_lambda_vir_grid, selected=best_params[1], factor=lambda_expand_factor
        )
        expanded_top_k_grid, top_k_boundary = _expand_top_k_grid(
            current_top_k_grid,
            selected=best_params[2],
            max_top_k=max_top_k,
            factor=top_k_expand_factor,
        )
        changed = False
        if lambda_boundary is not None and expanded_lambda_grid != current_lambda_grid:
            current_lambda_grid = expanded_lambda_grid
            changed = True
        if lambda_vir_boundary is not None and expanded_lambda_vir_grid != current_lambda_vir_grid:
            current_lambda_vir_grid = expanded_lambda_vir_grid
            changed = True
        if top_k_boundary is not None and expanded_top_k_grid != current_top_k_grid:
            current_top_k_grid = expanded_top_k_grid
            changed = True
        if not changed:
            print(
                "[target-aware-inverse-vir-cv] selected "
                f"lambda={best_params[0]} lambda_vir={best_params[1]} "
                f"top_k={best_params[2]} mean_val_mse={best_score}"
            )
            return InverseVirSelectionResult(
                best_lambda_reg=best_params[0],
                best_lambda_vir=best_params[1],
                best_top_k=best_params[2],
                rows=rows,
                final_lambda_grid=current_lambda_grid,
                final_lambda_vir_grid=current_lambda_vir_grid,
                final_top_k_grid=current_top_k_grid,
                num_expansions=num_expansions,
            )
        num_expansions += 1


def select_projector_inverse_vir_by_leave_one_environment_out(
    envs: list[EnvironmentData],
    *,
    lambda_u_grid: list[float],
    lambda_perp_grid: list[float],
    lambda_vir_grid: list[float],
    top_k_grid: list[int],
    target_unlabeled_budget: int | None = None,
    seed: int = 0,
    lambda_expand_factor: float = 5.0,
    top_k_expand_factor: int = 2,
) -> ProjectorInverseVirSelectionResult:
    if len(envs) < 2:
        raise ValueError("Need at least two environments for leave-one-environment-out selection.")

    rows: list[dict[str, float]] = []
    cache: dict[tuple[float, float, float, int, int], float] = {}
    current_lambda_u_grid = sorted({max(0.0, float(x)) for x in lambda_u_grid})
    current_lambda_perp_grid = sorted({max(0.0, float(x)) for x in lambda_perp_grid})
    current_lambda_vir_grid = sorted({max(0.0, float(x)) for x in lambda_vir_grid})
    current_top_k_grid = sorted({max(0, int(x)) for x in top_k_grid})
    max_top_k = int(envs[0].x.shape[1])
    num_expansions = 0

    while True:
        print(
            "[projector-inverse-vir-cv] round="
            f"{num_expansions} lambda_u_grid={current_lambda_u_grid} "
            f"lambda_perp_grid={current_lambda_perp_grid} "
            f"lambda_vir_grid={current_lambda_vir_grid} top_k_grid={current_top_k_grid}"
        )
        best_params = None
        best_score = float("inf")
        combinations = [
            (lambda_u, lambda_perp, lambda_vir, top_k)
            for lambda_u in current_lambda_u_grid
            for lambda_perp in current_lambda_perp_grid
            for lambda_vir in current_lambda_vir_grid
            for top_k in current_top_k_grid
        ]
        for lambda_u, lambda_perp, lambda_vir, top_k in tqdm(
            combinations,
            desc=f"projector-inverse-vir-cv round {num_expansions}",
            leave=False,
        ):
                        fold_mses = []
                        for heldout_idx in range(len(envs)):
                            target_env = subsample_unlabeled_env(
                                envs[heldout_idx],
                                budget=target_unlabeled_budget,
                                seed=seed,
                            )
                            cache_key = (
                                float(lambda_u),
                                float(lambda_perp),
                                float(lambda_vir),
                                int(top_k),
                                int(target_env.env_id),
                            )
                            if cache_key in cache:
                                val_mse = cache[cache_key]
                            else:
                                source_envs = [env for idx, env in enumerate(envs) if idx != heldout_idx]
                                fit = projector_inverse_vir_fit(
                                    source_envs,
                                    target_env=target_env,
                                    lambda_u=lambda_u,
                                    lambda_perp=lambda_perp,
                                    lambda_vir=lambda_vir,
                                    top_k=top_k,
                                )
                                val_mse = mse(target_env.x, target_env.y, fit.beta)
                                cache[cache_key] = float(val_mse)
                                rows.append(
                                    {
                                        "lambda_u": float(lambda_u),
                                        "lambda_perp": float(lambda_perp),
                                        "lambda_vir": float(lambda_vir),
                                        "top_k": float(top_k),
                                        "heldout_env": float(target_env.env_id),
                                        "val_mse": float(val_mse),
                                    }
                                )
                            fold_mses.append(float(val_mse))
                        mean_val_mse = float(np.mean(fold_mses))
                        if mean_val_mse < best_score:
                            best_score = mean_val_mse
                            best_params = (
                                float(lambda_u),
                                float(lambda_perp),
                                float(lambda_vir),
                                int(top_k),
                            )

        if best_params is None:
            raise RuntimeError("Failed to select hyperparameters.")

        expanded_lambda_u_grid, lambda_u_boundary = _expand_lambda_grid(
            current_lambda_u_grid, selected=best_params[0], factor=lambda_expand_factor
        )
        expanded_lambda_perp_grid, lambda_perp_boundary = _expand_lambda_grid(
            current_lambda_perp_grid, selected=best_params[1], factor=lambda_expand_factor
        )
        expanded_lambda_vir_grid, lambda_vir_boundary = _expand_lambda_grid(
            current_lambda_vir_grid, selected=best_params[2], factor=lambda_expand_factor
        )
        expanded_top_k_grid, top_k_boundary = _expand_top_k_grid(
            current_top_k_grid,
            selected=best_params[3],
            max_top_k=max_top_k,
            factor=top_k_expand_factor,
        )
        changed = False
        if lambda_u_boundary is not None and expanded_lambda_u_grid != current_lambda_u_grid:
            current_lambda_u_grid = expanded_lambda_u_grid
            changed = True
        if lambda_perp_boundary is not None and expanded_lambda_perp_grid != current_lambda_perp_grid:
            current_lambda_perp_grid = expanded_lambda_perp_grid
            changed = True
        if lambda_vir_boundary is not None and expanded_lambda_vir_grid != current_lambda_vir_grid:
            current_lambda_vir_grid = expanded_lambda_vir_grid
            changed = True
        if top_k_boundary is not None and expanded_top_k_grid != current_top_k_grid:
            current_top_k_grid = expanded_top_k_grid
            changed = True
        if not changed:
            print(
                "[projector-inverse-vir-cv] selected "
                f"lambda_u={best_params[0]} lambda_perp={best_params[1]} "
                f"lambda_vir={best_params[2]} top_k={best_params[3]} mean_val_mse={best_score}"
            )
            return ProjectorInverseVirSelectionResult(
                best_lambda_u=best_params[0],
                best_lambda_perp=best_params[1],
                best_lambda_vir=best_params[2],
                best_top_k=best_params[3],
                rows=rows,
                final_lambda_u_grid=current_lambda_u_grid,
                final_lambda_perp_grid=current_lambda_perp_grid,
                final_lambda_vir_grid=current_lambda_vir_grid,
                final_top_k_grid=current_top_k_grid,
                num_expansions=num_expansions,
            )
        num_expansions += 1
