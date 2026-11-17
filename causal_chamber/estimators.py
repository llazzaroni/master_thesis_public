"""OLS baselines, pseudo-lasso, and a source/target mixed estimator."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OLSFit:
    coefficients: np.ndarray
    intercept: float
    rank: int
    num_parameters: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coefficients + self.intercept


def minimum_norm_least_squares(design: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, int]:
    """Solve least squares with an explicit Moore-Penrose branch when n < p.

    ``np.linalg.lstsq`` already returns the Moore-Penrose minimum-norm solution
    for rank-deficient designs.  The explicit underdetermined branch makes this
    intended behavior unambiguous for small target budgets, and applies the
    same SVD tolerance used by ``lstsq(..., rcond=None)``.
    """
    if design.ndim != 2 or response.ndim != 1 or design.shape[0] != len(response):
        raise ValueError("Least squares requires a 2D design and matching 1D response")
    rows, columns = design.shape
    if rows < columns:
        # A = U diag(s) V^T, so A^+ y is the unique least-squares solution
        # with minimum Euclidean norm.
        u, singular_values, vh = np.linalg.svd(design, full_matrices=False)
        if not len(singular_values) or singular_values[0] == 0:
            return np.zeros(columns, dtype=np.result_type(design, response)), 0
        tolerance = np.finfo(singular_values.dtype).eps * max(rows, columns) * singular_values[0]
        keep = singular_values > tolerance
        beta = (vh[keep].T / singular_values[keep]) @ (u[:, keep].T @ response)
        return beta, int(np.count_nonzero(keep))
    beta, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    return beta, int(rank)


def fit_ols(x: np.ndarray, y: np.ndarray, *, fit_intercept: bool = True) -> OLSFit:
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError("OLS requires a nonempty 2D design and matching 1D response")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OLS input contains nonfinite values")
    design = np.column_stack((np.ones(len(x)), x)) if fit_intercept else x
    beta, rank = minimum_norm_least_squares(design, y)
    return OLSFit(
        coefficients=beta[1:] if fit_intercept else beta,
        intercept=float(beta[0]) if fit_intercept else 0.0,
        rank=int(rank),
        num_parameters=design.shape[1],
    )


def mse(fit, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((fit.predict(x) - y) ** 2))


@dataclass(frozen=True)
class SourceMeanFit:
    coefficients: np.ndarray
    intercept: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coefficients + self.intercept


def average_source_fits(fits: list[OLSFit]) -> SourceMeanFit:
    if not fits:
        raise ValueError("Mixed estimation requires at least one source fit")
    return SourceMeanFit(np.mean([fit.coefficients for fit in fits], axis=0),
                         float(np.mean([fit.intercept for fit in fits])))


@dataclass(frozen=True)
class MixedFit:
    coefficients: np.ndarray
    intercept: float
    lambda_mix: float
    cv_rows: list[dict]
    cv_folds: int
    tuning: str
    rank: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coefficients + self.intercept


def fit_mixed_estimator(x, y, source_mean: SourceMeanFit, *,
                        lambda_grid=(0, .1, .2, .4, .6, .8, 1), cv_folds=5,
                        seed=0, fit_intercept=True, target_fit=None) -> MixedFit:
    """Tune lambda*target OLS + (1-lambda)*mean source OLS on target folds."""
    if not len(y) or cv_folds < 2:
        raise ValueError("Mixed estimation requires samples and at least two requested CV folds")
    candidates = sorted(set(float(v) for v in lambda_grid))
    if not candidates or any(not np.isfinite(v) or not 0 <= v <= 1 for v in candidates):
        raise ValueError("Mixed lambda candidates must be finite and in [0, 1]")
    if not fit_intercept and source_mean.intercept != 0:
        raise ValueError("A no-intercept mixed fit requires no-intercept source fits")
    count = min(cv_folds, len(y))
    cv_rows = []
    if len(candidates) == 1:
        chosen, count, tuning = candidates[0], 0, "fixed_weight"
    elif len(y) < 2:
        chosen, count, tuning = candidates[0], 0, "insufficient_samples_smallest_target_weight"
    else:
        folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), count)
        errors = np.zeros(len(candidates))
        for validation in folds:
            train = np.setdiff1d(np.arange(len(y)), validation)
            fold_fit = fit_ols(x[train], y[train], fit_intercept=fit_intercept)
            target_prediction = fold_fit.predict(x[validation])
            source_prediction = source_mean.predict(x[validation])
            for i, candidate in enumerate(candidates):
                prediction = candidate * target_prediction + (1-candidate) * source_prediction
                errors[i] += float(np.sum((prediction - y[validation]) ** 2))
        cv_rows = [{"candidate": candidate, "mse": float(error / len(y))}
                   for candidate, error in zip(candidates, errors)]
        chosen = min(cv_rows, key=lambda row: (row["mse"], row["candidate"]))["candidate"]
        tuning = "target_subset_cv"
    # CV target fits use only their training folds. Refit on the full budget
    # after choosing the weight; the cached full-budget fit can be reused here.
    if target_fit is None:
        target_fit = fit_ols(x, y, fit_intercept=fit_intercept)
    return MixedFit(chosen * target_fit.coefficients + (1-chosen) * source_mean.coefficients,
                    chosen * target_fit.intercept + (1-chosen) * source_mean.intercept,
                    chosen, cv_rows, count, tuning, target_fit.rank)


def prepare_source_loss(x, y, *, fit_intercept=True):
    """Compress pooled source least squares once; retain its sample count."""
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError("Source loss requires a nonempty design and matching response")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Source loss input contains nonfinite values")
    design = np.column_stack((np.ones(len(x)), x)) if fit_intercept else x
    q, r = np.linalg.qr(design, mode="reduced")
    return r, q.T @ y, len(y)


def fit_mixed_loss_estimator(x, y, source_loss, *,
                             lambda_grid=(0, .1, .2, .4, .6, .8, 1),
                             cv_folds=5, seed=0, fit_intercept=True):
    """Minimize lambda*source MSE + (1-lambda)*target MSE; tune on target CV."""
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y) or cv_folds < 2:
        raise ValueError("Mixed loss requires samples and at least two requested CV folds")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Mixed loss input contains nonfinite values")
    candidates = sorted(set(float(v) for v in lambda_grid))
    if not candidates or any(not np.isfinite(v) or not 0 <= v <= 1 for v in candidates):
        raise ValueError("Mixed lambda candidates must be finite and in [0, 1]")
    source_design, source_response, source_n = source_loss
    design = np.column_stack((np.ones(len(x)), x)) if fit_intercept else x
    if source_n < 1 or source_design.shape[1] != design.shape[1]:
        raise ValueError("Source and target loss designs must have matching columns")

    def solve(rows, weight):
        if weight == 0:
            a, b = design[rows], y[rows]
        elif weight == 1:
            a, b = source_design, source_response
        else:
            source_scale = np.sqrt(weight / source_n)
            target_scale = np.sqrt((1-weight) / len(rows))
            a = np.vstack((source_scale*source_design, target_scale*design[rows]))
            b = np.concatenate((source_scale*source_response, target_scale*y[rows]))
        beta, rank = minimum_norm_least_squares(a, b)
        return beta, rank

    count = min(cv_folds, len(y))
    cv_rows = []
    if len(candidates) == 1:
        chosen, count, tuning = candidates[0], 0, "fixed_weight"
    elif len(y) < 2:
        chosen, count, tuning = candidates[-1], 0, "insufficient_samples_largest_source_weight"
    else:
        folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), count)
        errors = np.zeros(len(candidates))
        for validation in folds:
            train = np.setdiff1d(np.arange(len(y)), validation)
            for i, candidate in enumerate(candidates):
                beta, _ = solve(train, candidate)
                errors[i] += float(np.sum((design[validation] @ beta-y[validation])**2))
        cv_rows = [{"candidate": candidate, "mse": float(error/len(y))}
                   for candidate, error in zip(candidates, errors)]
        chosen = min(cv_rows, key=lambda row: (row["mse"], -row["candidate"]))["candidate"]
        tuning = "target_subset_cv"
    beta, rank = solve(np.arange(len(y)), chosen)
    return MixedFit(beta[1:] if fit_intercept else beta,
                    float(beta[0]) if fit_intercept else 0.0,
                    chosen, cv_rows, count, tuning, rank)


@dataclass(frozen=True)
class SourceScreen:
    mean_coefficients: np.ndarray
    mean_intercept: float
    invariant_support: list[int]
    details: list[dict]


@dataclass(frozen=True)
class LassoFit:
    coefficients: np.ndarray
    intercept: float
    lambda_reg: float
    iterations: int
    converged: bool

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coefficients + self.intercept


@dataclass(frozen=True)
class PseudoLassoFit:
    coefficients: np.ndarray
    intercept: float
    invariant_support: list[int]
    lambda_reg: float
    lambda_ratio: float | None
    cv_rows: list[dict]
    cv_folds: int
    iterations: int
    converged: bool
    tuning: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coefficients + self.intercept


def _standardize(x, y, fit_intercept):
    x_mean = x.mean(axis=0) if fit_intercept else np.zeros(x.shape[1])
    y_mean = float(y.mean()) if fit_intercept else 0.0
    centered = x - x_mean
    scale = np.sqrt(np.mean(centered ** 2, axis=0))
    scale[scale < 1e-12] = 1.0
    return centered / scale, y - y_mean, x_mean, scale, y_mean


def estimate_source_coefficients(x, y, *, fit_intercept=True):
    """Fit one source once; return coefficients, errors, identification, intercept."""
    xs, yc, mean, scale, y_mean = _standardize(x, y, fit_intercept)
    beta, rank = minimum_norm_least_squares(xs, yc)
    inverse = np.linalg.pinv(xs)
    residual = yc - xs @ beta
    dof = max(len(y) - int(rank) - int(fit_intercept), 1)
    sigma2 = float(residual @ residual) / dof
    return (beta / scale, np.sqrt(sigma2 * np.sum(inverse ** 2, axis=1)) / scale,
            np.isclose(np.diag(inverse @ xs), 1.0, atol=1e-7, rtol=0),
            y_mean - float(mean @ (beta / scale)))


def screen_sources(source_data: list[tuple[np.ndarray, np.ndarray]], **kwargs) -> SourceScreen:
    estimates = [estimate_source_coefficients(x, y, fit_intercept=kwargs.get("fit_intercept", True))
                 for x, y in source_data]
    return screen_source_estimates(estimates, **kwargs)


def screen_source_estimates(estimates, *,
                   fit_intercept=True, z_threshold=0.0, stability_threshold=1.0,
                   max_invariant_features=None) -> SourceScreen:
    """Synthetic coefficient-stability screen, with identification checks."""
    if not estimates or z_threshold < 0 or stability_threshold < 0:
        raise ValueError("Source data and nonnegative screening thresholds are required")
    if max_invariant_features is not None and max_invariant_features < 0:
        raise ValueError("Maximum invariant features must be nonnegative")
    coefficients, errors, identified, intercepts = zip(*estimates)
    coefficients, errors, identified = map(np.asarray, (coefficients, errors, identified))
    details, support = [], []
    for j in range(coefficients.shape[1]):
        coef, stderr = coefficients[:, j], errors[:, j]
        mean_abs = float(np.mean(np.abs(coef)))
        stability = float(np.std(coef) / (mean_abs + 1e-8))
        sign_consistent = bool(np.all(coef > 0) or np.all(coef < 0))
        min_abs = float(np.min(np.abs(coef)))
        max_stderr = float(np.max(stderr))
        detectable = bool(min_abs > z_threshold * max_stderr)
        identifiable = bool(np.all(identified[:, j]))
        selected = identifiable and sign_consistent and detectable and stability <= stability_threshold
        mean_coef = float(np.mean(coef))
        coef_variance = float(np.var(coef))
        mean_se_squared = float(np.mean(stderr ** 2))
        denominator = max(float(np.sqrt(coef_variance + mean_se_squared)), abs(mean_coef)*1e-12, 1e-300)
        ranking_score = abs(mean_coef) / denominator
        details.append({"feature": j, "mean_abs_coef": mean_abs, "min_abs_coef": min_abs, "max_stderr": max_stderr,
                        "stability_ratio": stability, "sign_consistent": sign_consistent,
                        "detectable": detectable, "identifiable": identifiable, "selected": selected,
                        "mean_coef": mean_coef, "coef_variance": coef_variance,
                        "mean_se_squared": mean_se_squared, "ranking_score": ranking_score,
                        "rejection_reasons": [name for name, passed in (
                            ("identifiability", identifiable), ("sign_consistency", sign_consistent),
                            ("detectability", detectable), ("stability", stability <= stability_threshold)) if not passed]})
        if selected:
            support.append(j)
    support.sort(key=lambda j: (details[j]["stability_ratio"], -details[j]["mean_abs_coef"]))
    if max_invariant_features is not None:
        support = support[:max_invariant_features]
    for row in details:
        if row["selected"] and row["feature"] not in support:
            row["rejection_reasons"].append("feature_cap")
        row["selected"] = row["feature"] in support
    return SourceScreen(coefficients.mean(axis=0), float(np.mean(intercepts)), sorted(support), details)


def rank_source_features(screen: SourceScreen, *, include_unidentified=False):
    eligible = [row for row in screen.details if include_unidentified or row["identifiable"]]
    return [row["feature"] for row in sorted(eligible, key=lambda row: (-row["ranking_score"], row["feature"]))]


def ranked_source_screen(screen: SourceScreen, count: int, *, include_unidentified=False, ranking=None) -> SourceScreen:
    """Top K by source effect / variability and uncertainty; deterministic ties."""
    if count < 1:
        raise ValueError("Ranked support size must be positive")
    order = rank_source_features(screen, include_unidentified=include_unidentified) if ranking is None else ranking
    support = sorted(order[:count])
    ranks = {feature: rank for rank, feature in enumerate(order, 1)}
    details = [{**row, "rank": ranks.get(row["feature"]), "selected": row["feature"] in support,
                "rejection_reasons": ([] if row["feature"] in support else
                                      ["below_top_k"] if row["feature"] in ranks else ["identifiability"])}
               for row in screen.details]
    return SourceScreen(screen.mean_coefficients, screen.mean_intercept, support, details)


def fit_lasso_coordinate_descent(x, y, *, lambda_reg, fit_intercept=True,
                                 max_iter=5000, tol=1e-6) -> LassoFit:
    """Minimize ||yc-Xs beta||²/(2n) + lambda*||beta||_1."""
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Lasso requires nonempty finite data")
    if not np.isfinite(lambda_reg) or lambda_reg < 0 or max_iter < 1 or tol <= 0:
        raise ValueError("Invalid lasso penalty or solver settings")
    xs, yc, mean, scale, y_mean = _standardize(x, y, fit_intercept)
    gram = xs.T @ xs / len(y)
    cross = xs.T @ yc / len(y)
    beta = np.zeros(x.shape[1])
    converged = False
    # Relative KKT tolerance remains meaningful for large light-intensity units.
    tolerance = tol * max(float(np.max(np.abs(cross), initial=0)), lambda_reg, 1e-12)
    for iteration in range(1, max_iter + 1):
        for j in range(len(beta)):
            if gram[j, j] <= 1e-15:
                continue
            rho = cross[j] - gram[j] @ beta + gram[j, j] * beta[j]
            beta[j] = np.sign(rho) * max(abs(rho) - lambda_reg, 0.0) / gram[j, j]
        gradient = cross - gram @ beta
        violation = np.where(beta != 0, np.abs(gradient - lambda_reg * np.sign(beta)),
                             np.maximum(np.abs(gradient) - lambda_reg, 0))
        if float(np.max(violation, initial=0)) <= tolerance:
            converged = True
            break
    coefficients = beta / scale
    return LassoFit(coefficients, y_mean - float(mean @ coefficients), float(lambda_reg), iteration, converged)


def _lambda_max(x, y, fit_intercept):
    xs, yc, _, _, _ = _standardize(x, y, fit_intercept)
    return float(np.max(np.abs(xs.T @ yc / len(y)), initial=0))


def fit_target_lasso(x, y, **kwargs) -> PseudoLassoFit:
    """Target-only lasso with the same subset CV and scaling as pseudo-lasso."""
    empty = SourceScreen(np.zeros(x.shape[1]), 0.0, [], [])
    return _fit_pseudo_lasso(x, y, empty, correction_on_all=True, **kwargs)


def fit_pseudo_lasso(x, y, screen: SourceScreen, **kwargs) -> PseudoLassoFit:
    """Keep the screened source coefficients fixed; correct only outside A."""
    return _fit_pseudo_lasso(x, y, screen, correction_on_all=False, **kwargs)


def fit_centered_pseudo_lasso(x, y, screen: SourceScreen, **kwargs) -> PseudoLassoFit:
    """Center on screened source coefficients and allow corrections everywhere."""
    return _fit_pseudo_lasso(x, y, screen, correction_on_all=True, **kwargs)


def _fit_pseudo_lasso(x, y, screen: SourceScreen, *, correction_on_all=False, lambda_grid=None,
                     lambda_ratios=(1.0, .3, .1, .03, .01, .003, .001), cv_folds=5,
                     seed=0, fit_intercept=True, max_iter=5000, tol=1e-6) -> PseudoLassoFit:
    if not len(y) or cv_folds < 2:
        raise ValueError("Pseudo-lasso requires samples and at least two requested CV folds")
    candidates = sorted(set(float(v) for v in (lambda_grid if lambda_grid is not None else lambda_ratios)))
    if not candidates or any(not np.isfinite(v) or v <= 0 for v in candidates):
        raise ValueError("Lasso penalty candidates must be finite and positive")
    if lambda_grid is None and max(candidates) > 1:
        raise ValueError("Relative penalties must be at most 1")
    beta = np.zeros(x.shape[1])
    beta[screen.invariant_support] = screen.mean_coefficients[screen.invariant_support]
    source_intercept = screen.mean_intercept if fit_intercept else 0.0
    residual = y - x @ beta - source_intercept
    correction_features = [j for j in range(x.shape[1]) if correction_on_all or j not in screen.invariant_support]
    xb = x[:, correction_features]
    if not correction_features:
        intercept = source_intercept + float(residual.mean()) if fit_intercept else 0.0
        return PseudoLassoFit(beta, intercept, screen.invariant_support, 0.0, None, [], 0, 0, True, "no_complement")
    count = min(cv_folds, len(y))
    cv_rows = []
    if len(y) < 2 or len(candidates) == 1:
        chosen = candidates[-1]
        count = 0
        tuning = "fixed_penalty" if len(candidates) == 1 else "insufficient_samples_strongest_penalty"
    else:
        # Seeded CV sees only the chosen target labels. Scaling is fitted
        # independently inside each training fold; no test/held-out scaling.
        folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), count)
        for candidate in candidates:
            squared_error, observations = 0.0, 0
            all_converged = True
            for validation in folds:
                train = np.setdiff1d(np.arange(len(y)), validation)
                penalty = candidate if lambda_grid is not None else candidate * _lambda_max(xb[train], residual[train], fit_intercept)
                fit = fit_lasso_coordinate_descent(xb[train], residual[train], lambda_reg=penalty,
                                                  fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)
                squared_error += float(np.sum((fit.predict(xb[validation]) - residual[validation]) ** 2))
                observations += len(validation)
                all_converged = all_converged and fit.converged
            cv_rows.append({"candidate": candidate, "mse": squared_error / observations, "converged": all_converged})
        # Prefer the stronger penalty on exact ties.
        chosen = min(cv_rows, key=lambda row: (row["mse"], -row["candidate"]))["candidate"]
        tuning = "target_subset_cv"
    penalty = chosen if lambda_grid is not None else chosen * _lambda_max(xb, residual, fit_intercept)
    correction = fit_lasso_coordinate_descent(xb, residual, lambda_reg=penalty,
                                             fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)
    beta[correction_features] += correction.coefficients
    return PseudoLassoFit(beta, source_intercept + correction.intercept, screen.invariant_support,
                         penalty, chosen if lambda_grid is None else None, cv_rows, count,
                         correction.iterations, correction.converged, tuning)
