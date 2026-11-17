from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from synthetic_shared_support.data import EnvironmentData


@dataclass(frozen=True)
class OLSFit:
    beta: np.ndarray
    sigma2_hat: float
    stderr: np.ndarray


@dataclass(frozen=True)
class MixedFit:
    beta: np.ndarray
    alpha: float
    a_hat: float
    b_hat: float
    rho2_hat: float
    sigma2_target_hat: float
    sigma2_source_hat: float


@dataclass(frozen=True)
class MixedCVResult:
    best_alpha: float
    rows: list[dict[str, float]]
    effective_train_size: int


@dataclass(frozen=True)
class PseudoLassoFit:
    beta: np.ndarray
    invariant_support: list[int]
    lambda_lasso: float
    screening_details: list[dict[str, float]]


@dataclass(frozen=True)
class TargetOnlyLassoFit:
    beta: np.ndarray
    lambda_lasso: float


def _expand_lambda_grid(lambda_grid: list[float], best_lambda: float) -> tuple[list[float], bool]:
    sorted_grid = sorted({float(x) for x in lambda_grid})
    if len(sorted_grid) < 2:
        return sorted_grid, False

    expanded_grid = list(sorted_grid)
    changed = False

    if np.isclose(best_lambda, sorted_grid[0]):
        lower_candidate = max(sorted_grid[0] / 5.0, 1e-8)
        if lower_candidate < sorted_grid[0] and not any(np.isclose(lower_candidate, x) for x in expanded_grid):
            expanded_grid.insert(0, lower_candidate)
            changed = True

    if np.isclose(best_lambda, sorted_grid[-1]):
        upper_candidate = sorted_grid[-1] * 5.0
        if upper_candidate > sorted_grid[-1] and not any(np.isclose(upper_candidate, x) for x in expanded_grid):
            expanded_grid.append(upper_candidate)
            changed = True

    return expanded_grid, changed


def _expand_alpha_grid(alpha_grid: list[float], best_alpha: float) -> tuple[list[float], bool]:
    sorted_grid = sorted({float(np.clip(x, 0.0, 1.0)) for x in alpha_grid})
    if len(sorted_grid) < 2:
        return sorted_grid, False

    expanded_grid = list(sorted_grid)
    changed = False

    if np.isclose(best_alpha, sorted_grid[0]) and sorted_grid[0] > 0.0:
        lower_candidate = max(sorted_grid[0] / 5.0, 0.0)
        if lower_candidate < sorted_grid[0] and not any(np.isclose(lower_candidate, x) for x in expanded_grid):
            expanded_grid.insert(0, lower_candidate)
            changed = True

    if np.isclose(best_alpha, sorted_grid[-1]) and sorted_grid[-1] < 1.0:
        gap_to_one = 1.0 - sorted_grid[-1]
        upper_candidate = min(sorted_grid[-1] + gap_to_one / 5.0, 1.0)
        if upper_candidate > sorted_grid[-1] and not any(np.isclose(upper_candidate, x) for x in expanded_grid):
            expanded_grid.append(upper_candidate)
            changed = True

    return expanded_grid, changed



def stack_environments(envs: list[EnvironmentData]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([env.x for env in envs], axis=0)
    y = np.concatenate([env.y for env in envs], axis=0)
    return x, y


def predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return x @ beta


def mse(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    residuals = predict(x, beta) - y
    return float(np.mean(residuals**2))



def fit_ols(x: np.ndarray, y: np.ndarray, *, jitter: float = 1e-8) -> OLSFit:
    xtx = x.T @ x
    rank = int(np.linalg.matrix_rank(x))
    if x.shape[0] < x.shape[1] or rank < x.shape[1]:
        # The Moore-Penrose solution is the unique minimum-norm OLS solution.
        beta = np.linalg.pinv(x) @ y
        inv_xtx = np.linalg.pinv(xtx)
    else:
        inv_xtx = np.linalg.inv(xtx + jitter * np.eye(xtx.shape[0], dtype=np.float64))
        beta = inv_xtx @ (x.T @ y)
    residuals = y - x @ beta
    dof = max(x.shape[0] - x.shape[1], 1)
    sigma2_hat = float((residuals @ residuals) / dof)
    stderr = np.sqrt(np.clip(np.diag(inv_xtx) * sigma2_hat, 0.0, None))
    return OLSFit(beta=beta, sigma2_hat=sigma2_hat, stderr=stderr)



def average_source_ols(source_envs: list[EnvironmentData]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fits = [fit_ols(env.x, env.y) for env in source_envs]
    betas = np.stack([fit.beta for fit in fits], axis=0)
    sigmas = np.array([fit.sigma2_hat for fit in fits], dtype=np.float64)
    stderrs = np.stack([fit.stderr for fit in fits], axis=0)
    return betas.mean(axis=0), sigmas, stderrs


def fit_mixed_estimator(
    source_envs: list[EnvironmentData],
    target_train: EnvironmentData,
) -> MixedFit:
    target_fit = fit_ols(target_train.x, target_train.y)
    source_mean_beta, source_sigmas, _source_stderrs = average_source_ols(source_envs)

    mean_source_n = float(np.mean([env.num_samples for env in source_envs]))
    num_source_envs = float(len(source_envs))
    target_n = float(target_train.num_samples)
    dim = float(target_train.x.shape[1])

    sigma2_source_hat = float(np.mean(source_sigmas))
    sigma2_target_hat = float(target_fit.sigma2_hat)

    a_hat = sigma2_target_hat * dim / max(target_n, 1.0)
    source_estimation_term = sigma2_source_hat * dim / max(mean_source_n * num_source_envs, 1.0)
    diff2 = float(np.sum((source_mean_beta - target_fit.beta) ** 2))
    rho2_hat = max(0.0, diff2 - a_hat - source_estimation_term)
    b_hat = rho2_hat + source_estimation_term

    denom = a_hat + b_hat
    alpha = b_hat / denom if denom > 0.0 else 0.5
    alpha = float(np.clip(alpha, 0.0, 1.0))
    beta = alpha * target_fit.beta + (1.0 - alpha) * source_mean_beta
    return MixedFit(
        beta=beta,
        alpha=alpha,
        a_hat=a_hat,
        b_hat=b_hat,
        rho2_hat=rho2_hat,
        sigma2_target_hat=sigma2_target_hat,
        sigma2_source_hat=sigma2_source_hat,
    )


def fit_mixed_estimator_with_alpha(
    source_envs: list[EnvironmentData],
    target_train: EnvironmentData,
    *,
    alpha: float,
) -> MixedFit:
    target_fit = fit_ols(target_train.x, target_train.y)
    source_mean_beta, source_sigmas, _source_stderrs = average_source_ols(source_envs)

    mean_source_n = float(np.mean([env.num_samples for env in source_envs]))
    num_source_envs = float(len(source_envs))
    target_n = float(target_train.num_samples)
    dim = float(target_train.x.shape[1])

    sigma2_source_hat = float(np.mean(source_sigmas))
    sigma2_target_hat = float(target_fit.sigma2_hat)

    a_hat = sigma2_target_hat * dim / max(target_n, 1.0)
    source_estimation_term = sigma2_source_hat * dim / max(mean_source_n * num_source_envs, 1.0)
    diff2 = float(np.sum((source_mean_beta - target_fit.beta) ** 2))
    rho2_hat = max(0.0, diff2 - a_hat - source_estimation_term)
    b_hat = rho2_hat + source_estimation_term

    alpha = float(np.clip(alpha, 0.0, 1.0))
    beta = alpha * target_fit.beta + (1.0 - alpha) * source_mean_beta
    return MixedFit(
        beta=beta,
        alpha=alpha,
        a_hat=a_hat,
        b_hat=b_hat,
        rho2_hat=rho2_hat,
        sigma2_target_hat=sigma2_target_hat,
        sigma2_source_hat=sigma2_source_hat,
    )


def _split_environment_for_target_cv(
    env: EnvironmentData,
    *,
    target_train_size: int,
    seed: int,
) -> tuple[EnvironmentData, np.ndarray, np.ndarray]:
    n = env.num_samples
    if n < 2:
        raise ValueError("Need at least two samples in a held-out environment for mixed-estimator CV.")

    effective_train_size = min(max(1, int(target_train_size)), n - 1)
    rng = np.random.default_rng(seed + env.env_id)
    permutation = rng.permutation(n)
    train_idx = permutation[:effective_train_size]
    val_idx = permutation[effective_train_size:]
    target_train = EnvironmentData(
        env_id=env.env_id,
        x=env.x[train_idx],
        y=env.y[train_idx],
        beta_star=env.beta_star,
        support_specific=env.support_specific,
    )
    return target_train, env.x[val_idx], env.y[val_idx]


def select_mixed_alpha_by_target_cv(
    source_envs: list[EnvironmentData],
    *,
    target_train: EnvironmentData,
    alpha_grid: list[float],
    cv_folds: int = 5,
    seed: int = 0,
) -> MixedCVResult:
    """Choose the target weight using only folds of the actual target subset."""
    if not source_envs or target_train.num_samples < 1 or cv_folds < 2:
        raise ValueError("Mixed target-subset CV requires source data, target data, and at least two folds.")
    alpha_grid = sorted({float(np.clip(alpha, 0.0, 1.0)) for alpha in alpha_grid})
    if not alpha_grid:
        raise ValueError("Mixed target-subset CV requires at least one alpha candidate.")
    if target_train.num_samples < 2:
        return MixedCVResult(best_alpha=0.0, rows=[], effective_train_size=1)

    count = min(cv_folds, target_train.num_samples)
    folds = np.array_split(np.random.default_rng(seed).permutation(target_train.num_samples), count)
    rows: list[dict[str, float]] = []
    candidate_losses = []
    for alpha in tqdm(alpha_grid, desc="mixed target-subset CV", unit="alpha", leave=False):
        fold_losses = []
        for fold_index, validation_idx in enumerate(folds):
            train_idx = np.setdiff1d(np.arange(target_train.num_samples), validation_idx)
            fold_target = EnvironmentData(
                env_id=target_train.env_id,
                x=target_train.x[train_idx],
                y=target_train.y[train_idx],
                beta_star=target_train.beta_star,
                support_specific=target_train.support_specific,
            )
            fit = fit_mixed_estimator_with_alpha(source_envs, fold_target, alpha=alpha)
            val_loss = mse(target_train.x[validation_idx], target_train.y[validation_idx], fit.beta)
            fold_losses.append(val_loss)
            rows.append({
                "alpha": float(alpha),
                "fold_index": float(fold_index),
                "train_size": float(len(train_idx)),
                "val_size": float(len(validation_idx)),
                "val_mse": float(val_loss),
            })
        candidate_losses.append((float(np.mean(fold_losses)), float(alpha)))

    # On an exact validation tie, prefer the more source-regularized fit.
    best_alpha = min(candidate_losses)[1]

    return MixedCVResult(
        best_alpha=best_alpha,
        rows=rows,
        effective_train_size=int(target_train.num_samples - len(folds[0])),
    )




def recover_invariant_support(
    source_envs: list[EnvironmentData],
    *,
    z_threshold: float = 2.0,
    stability_threshold: float = 0.35,
    max_invariant_features: int | None = None,
) -> tuple[list[int], list[dict[str, float]]]:
    fits = [fit_ols(env.x, env.y) for env in source_envs]
    beta_mat = np.stack([fit.beta for fit in fits], axis=0)
    stderr_mat = np.stack([fit.stderr for fit in fits], axis=0)
    num_features = beta_mat.shape[1]

    details: list[dict[str, float]] = []
    support: list[int] = []
    for feature_idx in range(num_features):
        coefficients = beta_mat[:, feature_idx]
        stderr = stderr_mat[:, feature_idx]
        mean_abs = float(np.mean(np.abs(coefficients)))
        max_stderr = float(np.max(stderr))
        stability = float(np.std(coefficients) / (mean_abs + 1e-8))
        all_positive = bool(np.all(coefficients > 0.0))
        all_negative = bool(np.all(coefficients < 0.0))
        sign_consistent = all_positive or all_negative
        detectable = (
            True if z_threshold <= 0
            else float(np.min(np.abs(coefficients))) > z_threshold * max_stderr
        )
        selected = bool(sign_consistent and detectable and stability <= stability_threshold)
        details.append(
            {
                "feature": float(feature_idx),
                "mean_abs_coef": mean_abs,
                "max_stderr": max_stderr,
                "stability_ratio": stability,
                "sign_consistent": float(sign_consistent),
                "detectable": float(detectable),
                "selected": float(selected),
            }
        )
        if selected:
            support.append(feature_idx)

    support = sorted(
        support,
        key=lambda idx: (
            details[idx]["stability_ratio"],
            -details[idx]["mean_abs_coef"],
        ),
    )
    if max_invariant_features is not None:
        support = support[: max_invariant_features]
    support = sorted(support)
    return support, details


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def fit_lasso_coordinate_descent(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lambda_reg: float,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    if x.shape[1] == 0:
        return np.zeros(0, dtype=np.float64), float(y.mean())

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    xs = (x - x_mean) / x_std
    y_mean = float(y.mean())
    yc = y - y_mean

    n, p = xs.shape
    beta_scaled = np.zeros(p, dtype=np.float64)
    residual = yc.copy()
    column_norm = (xs * xs).sum(axis=0) / float(n)

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            residual += xs[:, j] * beta_scaled[j]
            rho = float(xs[:, j] @ residual) / float(n)
            updated = _soft_threshold(rho, lambda_reg) / float(column_norm[j] + 1e-12)
            residual -= xs[:, j] * updated
            max_change = max(max_change, abs(updated - beta_scaled[j]))
            beta_scaled[j] = updated
        if max_change < tol:
            break

    beta = beta_scaled / x_std
    intercept = y_mean - float(x_mean @ beta)
    return beta, intercept


def select_lasso_lambda(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lambda_grid: list[float],
    num_folds: int = 5,
    seed: int = 0,
    max_grid_expansions: int = 10,
) -> float:
    if x.shape[1] == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    n = x.shape[0]
    fold_count = max(2, min(num_folds, n))
    indices = rng.permutation(n)
    fold_splits = np.array_split(indices, fold_count)

    lambda_grid = sorted({float(x) for x in lambda_grid})
    best_lambda = float(lambda_grid[0])
    expansion_round = 0

    while True:
        best_score = float("inf")
        round_best_lambda = float(lambda_grid[0])
        lambda_iterator = tqdm(
            lambda_grid,
            desc=f"lasso CV round {expansion_round}",
            unit="lambda",
            leave=False,
        )
        for lambda_reg in lambda_iterator:
            fold_scores = []
            for val_idx in fold_splits:
                train_mask = np.ones(n, dtype=bool)
                train_mask[val_idx] = False
                train_idx = np.flatnonzero(train_mask)
                beta, intercept = fit_lasso_coordinate_descent(
                    x[train_idx],
                    y[train_idx],
                    lambda_reg=float(lambda_reg),
                )
                predictions = x[val_idx] @ beta + intercept
                fold_scores.append(float(np.mean((y[val_idx] - predictions) ** 2)))
            mean_score = float(np.mean(fold_scores))
            if mean_score < best_score:
                best_score = mean_score
                round_best_lambda = float(lambda_reg)
            lambda_iterator.set_postfix(best_lambda=round_best_lambda, best_mse=best_score)

        best_lambda = round_best_lambda
        expanded_grid, changed = _expand_lambda_grid(lambda_grid, best_lambda)
        if not changed or expansion_round >= max_grid_expansions:
            break
        lambda_grid = expanded_grid
        expansion_round += 1

    return best_lambda


def fit_target_only_lasso(
    target_train: EnvironmentData,
    *,
    lambda_grid: list[float] | None = None,
    cv_folds: int = 5,
    seed: int = 0,
) -> TargetOnlyLassoFit:
    if lambda_grid is None:
        lambda_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    lambda_lasso = select_lasso_lambda(
        target_train.x,
        target_train.y,
        lambda_grid=[float(x) for x in lambda_grid],
        num_folds=cv_folds,
        seed=seed,
    )
    beta, _intercept = fit_lasso_coordinate_descent(
        target_train.x,
        target_train.y,
        lambda_reg=lambda_lasso,
    )
    return TargetOnlyLassoFit(beta=beta, lambda_lasso=lambda_lasso)


def fit_pseudo_lasso(
    source_envs: list[EnvironmentData],
    target_train: EnvironmentData,
    *,
    z_threshold: float = 2.0,
    stability_threshold: float = 0.35,
    max_invariant_features: int | None = None,
    lambda_grid: list[float] | None = None,
    cv_folds: int = 5,
    seed: int = 0,
) -> PseudoLassoFit:
    if lambda_grid is None:
        lambda_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

    source_mean_beta, _source_sigmas, _source_stderrs = average_source_ols(source_envs)
    invariant_support, details = recover_invariant_support(
        source_envs,
        z_threshold=z_threshold,
        stability_threshold=stability_threshold,
        max_invariant_features=max_invariant_features,
    )

    dim = target_train.x.shape[1]
    beta_a = np.zeros(dim, dtype=np.float64)
    if invariant_support:
        beta_a[np.array(invariant_support, dtype=int)] = source_mean_beta[np.array(invariant_support, dtype=int)]

    complement = [idx for idx in range(dim) if idx not in set(invariant_support)]
    if not complement:
        return PseudoLassoFit(
            beta=beta_a,
            invariant_support=invariant_support,
            lambda_lasso=0.0,
            screening_details=details,
        )

    residual_target = target_train.y - target_train.x[:, invariant_support] @ beta_a[invariant_support]
    x_b = target_train.x[:, complement]
    lambda_lasso = select_lasso_lambda(
        x_b,
        residual_target,
        lambda_grid=[float(x) for x in lambda_grid],
        num_folds=cv_folds,
        seed=seed,
    )
    zeta_b, intercept = fit_lasso_coordinate_descent(
        x_b,
        residual_target,
        lambda_reg=lambda_lasso,
    )
    beta = beta_a.copy()
    beta[np.array(complement, dtype=int)] += zeta_b
    if abs(intercept) > 1e-6:
        details = details + [{"feature": -1.0, "mean_abs_coef": abs(intercept), "max_stderr": 0.0, "stability_ratio": 0.0, "sign_consistent": 0.0, "detectable": 0.0, "selected": 0.0}]
    return PseudoLassoFit(
        beta=beta,
        invariant_support=invariant_support,
        lambda_lasso=lambda_lasso,
        screening_details=details,
    )
