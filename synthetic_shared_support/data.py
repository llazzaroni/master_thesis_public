from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvironmentData:
    env_id: int
    x: np.ndarray
    y: np.ndarray
    beta_star: np.ndarray
    support_specific: list[int]

    @property
    def num_samples(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True)
class SyntheticDataset:
    source_envs: list[EnvironmentData]
    target_train: EnvironmentData
    target_test: EnvironmentData
    shared_support: list[int]
    shared_beta: np.ndarray


def _sample_sparse_coefficients(
    rng: np.random.Generator,
    support: list[int],
    num_features: int,
    *,
    magnitude_low: float,
    magnitude_high: float,
) -> np.ndarray:
    beta = np.zeros(num_features, dtype=np.float64)
    if not support:
        return beta
    magnitudes = rng.uniform(magnitude_low, magnitude_high, size=len(support))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=len(support))
    beta[np.array(support, dtype=int)] = magnitudes * signs
    return beta


def _sample_specific_support(
    rng: np.random.Generator,
    *,
    num_features: int,
    forbidden: set[int],
    support_size: int,
) -> list[int]:
    candidates = [idx for idx in range(num_features) if idx not in forbidden]
    if support_size <= 0:
        return []
    if support_size > len(candidates):
        raise ValueError("Requested environment-specific support larger than remaining features.")
    return sorted(rng.choice(np.array(candidates, dtype=int), size=support_size, replace=False).tolist())


def _sample_design(
    rng: np.random.Generator,
    *,
    num_samples: int,
    num_features: int,
) -> np.ndarray:
    return rng.normal(size=(num_samples, num_features)).astype(np.float64)


def generate_dataset(
    *,
    seed: int = 0,
    num_source_envs: int = 8,
    source_samples_per_env: int = 500,
    target_train_samples: int = 80,
    target_test_samples: int = 2000,
    num_features: int = 40,
    shared_support_size: int = 5,
    source_specific_support_size: int = 4,
    target_specific_support_size: int = 4,
    coefficient_low: float = 0.8,
    coefficient_high: float = 1.2,
    noise_std: float = 1.0,
) -> SyntheticDataset:
    if not 0 < shared_support_size < num_features:
        raise ValueError("shared_support_size must be in (0, num_features).")

    rng = np.random.default_rng(seed)
    shared_support = sorted(
        rng.choice(np.arange(num_features, dtype=int), size=shared_support_size, replace=False).tolist()
    )
    shared_beta = _sample_sparse_coefficients(
        rng,
        shared_support,
        num_features,
        magnitude_low=coefficient_low,
        magnitude_high=coefficient_high,
    )

    source_envs: list[EnvironmentData] = []
    for env_id in range(num_source_envs):
        specific_support = _sample_specific_support(
            rng,
            num_features=num_features,
            forbidden=set(shared_support),
            support_size=source_specific_support_size,
        )
        gamma = _sample_sparse_coefficients(
            rng,
            specific_support,
            num_features,
            magnitude_low=coefficient_low,
            magnitude_high=coefficient_high,
        )
        beta_star = shared_beta + gamma
        x = _sample_design(
            rng,
            num_samples=source_samples_per_env,
            num_features=num_features,
        )
        y = x @ beta_star + rng.normal(scale=noise_std, size=source_samples_per_env)
        source_envs.append(
            EnvironmentData(
                env_id=env_id,
                x=x,
                y=y.astype(np.float64),
                beta_star=beta_star,
                support_specific=specific_support,
            )
        )

    target_specific_support = _sample_specific_support(
        rng,
        num_features=num_features,
        forbidden=set(shared_support),
        support_size=target_specific_support_size,
    )
    target_gamma = _sample_sparse_coefficients(
        rng,
        target_specific_support,
        num_features,
        magnitude_low=coefficient_low,
        magnitude_high=coefficient_high,
    )
    target_beta_star = shared_beta + target_gamma

    x_target_train = _sample_design(
        rng,
        num_samples=target_train_samples,
        num_features=num_features,
    )
    y_target_train = x_target_train @ target_beta_star + rng.normal(
        scale=noise_std,
        size=target_train_samples,
    )
    target_train = EnvironmentData(
        env_id=num_source_envs,
        x=x_target_train,
        y=y_target_train.astype(np.float64),
        beta_star=target_beta_star,
        support_specific=target_specific_support,
    )

    x_target_test = _sample_design(
        rng,
        num_samples=target_test_samples,
        num_features=num_features,
    )
    y_target_test = x_target_test @ target_beta_star + rng.normal(
        scale=noise_std,
        size=target_test_samples,
    )
    target_test = EnvironmentData(
        env_id=num_source_envs + 1,
        x=x_target_test,
        y=y_target_test.astype(np.float64),
        beta_star=target_beta_star,
        support_specific=target_specific_support,
    )

    return SyntheticDataset(
        source_envs=source_envs,
        target_train=target_train,
        target_test=target_test,
        shared_support=shared_support,
        shared_beta=shared_beta,
    )
