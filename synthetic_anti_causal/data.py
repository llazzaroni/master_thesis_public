from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvironmentData:
    env_id: int
    x: np.ndarray
    y: np.ndarray
    x_unlabeled: np.ndarray | None
    # Test environments optionally carry labels reserved for oracle analysis.
    # Estimators never receive these arrays.
    x_oracle: np.ndarray | None
    y_oracle: np.ndarray | None
    noise_mean: np.ndarray
    noise_std: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True)
class SyntheticDataset:
    train_envs: list[EnvironmentData]
    test_envs: list[EnvironmentData]
    signal_direction: np.ndarray
    nonlinear_direction: np.ndarray
    stable_feature_mask: np.ndarray
    confounder_loading: np.ndarray | None = None
    outcome_confounder_direction: np.ndarray | None = None


def _sample_noise_parameters(
    rng: np.random.Generator,
    *,
    num_features: int,
    stable_feature_mask: np.ndarray,
    stable_mean_scale: float,
    stable_std_low: float,
    stable_std_high: float,
    unstable_mean_scale: float,
    unstable_std_low: float,
    unstable_std_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty(num_features, dtype=np.float64)
    stds = np.empty(num_features, dtype=np.float64)

    stable_idx = stable_feature_mask
    unstable_idx = ~stable_feature_mask

    means[stable_idx] = rng.normal(
        loc=0.0,
        scale=stable_mean_scale,
        size=int(stable_idx.sum()),
    )
    means[unstable_idx] = rng.normal(
        loc=0.0,
        scale=unstable_mean_scale,
        size=int(unstable_idx.sum()),
    )

    stds[stable_idx] = rng.uniform(
        stable_std_low,
        stable_std_high,
        size=int(stable_idx.sum()),
    )
    stds[unstable_idx] = rng.uniform(
        unstable_std_low,
        unstable_std_high,
        size=int(unstable_idx.sum()),
    )

    return means, stds


def generate_environment(
    rng: np.random.Generator,
    *,
    env_id: int,
    num_samples: int,
    num_unlabeled_samples: int,
    num_oracle_samples: int,
    signal_direction: np.ndarray,
    nonlinear_direction: np.ndarray,
    stable_feature_mask: np.ndarray,
    stable_mean_scale: float,
    stable_std_low: float,
    stable_std_high: float,
    unstable_mean_scale: float,
    unstable_std_low: float,
    unstable_std_high: float,
    nonlinear_strength: float,
    confounder_loading: np.ndarray | None = None,
    outcome_confounder_direction: np.ndarray | None = None,
) -> EnvironmentData:
    num_features = int(signal_direction.shape[0])
    noise_mean, noise_std = _sample_noise_parameters(
        rng,
        num_features=num_features,
        stable_feature_mask=stable_feature_mask,
        stable_mean_scale=stable_mean_scale,
        stable_std_low=stable_std_low,
        stable_std_high=stable_std_high,
        unstable_mean_scale=unstable_mean_scale,
        unstable_std_low=unstable_std_low,
        unstable_std_high=unstable_std_high,
    )
    # Split the stream before variable-size draws so changing an unlabeled
    # budget cannot change either the evaluation or oracle samples.
    sample_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
    unlabeled_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
    oracle_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))

    def sample_xy(
        local_rng: np.random.Generator,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if confounder_loading is None or outcome_confounder_direction is None:
            # Preserve the original, unconfounded sampling path exactly.
            y_local = local_rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64)
            confounders = None
        else:
            confounders = local_rng.normal(
                loc=0.0,
                scale=1.0,
                size=(n, outcome_confounder_direction.shape[0]),
            ).astype(np.float64)
            outcome_strength_sq = float(outcome_confounder_direction @ outcome_confounder_direction)
            y_local = (
                confounders @ outcome_confounder_direction
                + np.sqrt(1.0 - outcome_strength_sq)
                * local_rng.normal(loc=0.0, scale=1.0, size=n)
            ).astype(np.float64)
        eps_local = local_rng.normal(
            loc=noise_mean,
            scale=noise_std,
            size=(n, num_features),
        ).astype(np.float64)
        nonlinear_feature = (y_local**2 - 1.0).astype(np.float64)
        x_local = (
            np.outer(y_local, signal_direction).astype(np.float64)
            + float(nonlinear_strength)
            * np.outer(nonlinear_feature, nonlinear_direction).astype(np.float64)
            + eps_local
        )
        if confounders is not None:
            x_local += confounders @ confounder_loading.T
        return x_local, y_local

    x, y = sample_xy(sample_rng, num_samples)
    x_unlabeled = None
    if num_unlabeled_samples > 0:
        x_unlabeled, _ = sample_xy(unlabeled_rng, num_unlabeled_samples)
    x_oracle = None
    y_oracle = None
    if num_oracle_samples > 0:
        x_oracle, y_oracle = sample_xy(oracle_rng, num_oracle_samples)
    return EnvironmentData(
        env_id=env_id,
        x=x,
        y=y,
        x_unlabeled=x_unlabeled,
        x_oracle=x_oracle,
        y_oracle=y_oracle,
        noise_mean=noise_mean,
        noise_std=noise_std,
    )


def generate_synthetic_dataset(
    *,
    seed: int = 0,
    num_train_envs: int = 10,
    num_test_envs: int = 3,
    samples_per_env: int = 500,
    unlabeled_samples_per_train_env: int = 500,
    unlabeled_samples_per_test_env: int | None = None,
    oracle_labeled_samples_per_test_env: int = 0,
    num_features: int = 50,
    num_stable_features: int = 10,
    stable_mean_scale: float = 0.0,
    stable_std_low: float = 0.8,
    stable_std_high: float = 1.0,
    unstable_mean_scale: float = 0.5,
    unstable_std_low: float = 1.0,
    unstable_std_high: float = 1.3,
    test_unstable_mean_scale: float = 0.8,
    test_unstable_std_low: float = 1.2,
    test_unstable_std_high: float = 1.6,
    nonlinear_strength: float = 0.0,
    confounder_dim: int = 0,
    confounding_strength: float = 0.0,
    outcome_confounding_strength: float = 0.5,
) -> SyntheticDataset:
    if not 0 < num_stable_features < num_features:
        raise ValueError("num_stable_features must be in (0, num_features).")
    if confounder_dim < 0:
        raise ValueError("confounder_dim must be non-negative.")
    if confounding_strength < 0:
        raise ValueError("confounding_strength must be non-negative.")
    if not 0.0 <= outcome_confounding_strength < 1.0:
        raise ValueError("outcome_confounding_strength must be in [0, 1).")
    if unlabeled_samples_per_test_env is None:
        unlabeled_samples_per_test_env = samples_per_env

    seed_sequence = np.random.SeedSequence(seed)
    base_rng = np.random.default_rng(seed_sequence)
    signal_direction = base_rng.normal(size=num_features).astype(np.float64)
    signal_direction /= np.linalg.norm(signal_direction)
    nonlinear_direction = base_rng.normal(size=num_features).astype(np.float64)
    nonlinear_direction -= float(signal_direction @ nonlinear_direction) * signal_direction
    nonlinear_norm = np.linalg.norm(nonlinear_direction)
    if nonlinear_norm <= 1e-12:
        nonlinear_direction = np.roll(signal_direction, 1)
        nonlinear_norm = np.linalg.norm(nonlinear_direction)
    nonlinear_direction /= nonlinear_norm

    confounder_loading = None
    outcome_confounder_direction = None
    if confounder_dim > 0 and confounding_strength > 0.0:
        confounder_loading = base_rng.normal(
            size=(num_features, confounder_dim)
        ).astype(np.float64)
        confounder_loading *= (
            float(confounding_strength) * np.linalg.norm(signal_direction)
            / np.linalg.norm(confounder_loading)
        )
        outcome_confounder_direction = base_rng.normal(
            size=confounder_dim
        ).astype(np.float64)
        outcome_confounder_direction *= (
            float(outcome_confounding_strength)
            / np.linalg.norm(outcome_confounder_direction)
        )

    stable_feature_mask = np.zeros(num_features, dtype=bool)
    stable_feature_mask[:num_stable_features] = True

    env_seed_sequences = seed_sequence.spawn(num_train_envs + num_test_envs)

    train_envs = [
        generate_environment(
            np.random.default_rng(env_seed_sequences[env_id]),
            env_id=env_id,
            num_samples=samples_per_env,
            num_unlabeled_samples=unlabeled_samples_per_train_env,
            num_oracle_samples=0,
            signal_direction=signal_direction,
            nonlinear_direction=nonlinear_direction,
            stable_feature_mask=stable_feature_mask,
            stable_mean_scale=stable_mean_scale,
            stable_std_low=stable_std_low,
            stable_std_high=stable_std_high,
            unstable_mean_scale=unstable_mean_scale,
            unstable_std_low=unstable_std_low,
            unstable_std_high=unstable_std_high,
            nonlinear_strength=nonlinear_strength,
            confounder_loading=confounder_loading,
            outcome_confounder_direction=outcome_confounder_direction,
        )
        for env_id in range(num_train_envs)
    ]

    test_envs = [
        generate_environment(
            np.random.default_rng(env_seed_sequences[num_train_envs + env_id]),
            env_id=num_train_envs + env_id,
            num_samples=samples_per_env,
            num_unlabeled_samples=int(unlabeled_samples_per_test_env),
            num_oracle_samples=int(oracle_labeled_samples_per_test_env),
            signal_direction=signal_direction,
            nonlinear_direction=nonlinear_direction,
            stable_feature_mask=stable_feature_mask,
            stable_mean_scale=stable_mean_scale,
            stable_std_low=stable_std_low,
            stable_std_high=stable_std_high,
            unstable_mean_scale=test_unstable_mean_scale,
            unstable_std_low=test_unstable_std_low,
            unstable_std_high=test_unstable_std_high,
            nonlinear_strength=nonlinear_strength,
            confounder_loading=confounder_loading,
            outcome_confounder_direction=outcome_confounder_direction,
        )
        for env_id in range(num_test_envs)
    ]

    return SyntheticDataset(
        train_envs=train_envs,
        test_envs=test_envs,
        signal_direction=signal_direction,
        nonlinear_direction=nonlinear_direction,
        stable_feature_mask=stable_feature_mask,
        confounder_loading=confounder_loading,
        outcome_confounder_direction=outcome_confounder_direction,
    )
