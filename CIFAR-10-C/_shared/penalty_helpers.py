#!/usr/bin/env python3
"""Target-aware inverse linear penalty for CIFAR-10-C PCA features.

The PCA projector is reused from the source-only linear baseline. Hyperparameters
are selected with leave-one-source-environment-out validation over every source
environment. The selected estimator is then fitted with unlabeled real-target
data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CORRUPTIONS = (
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
)

DEFAULT_CLUSTER_DATA_ROOT = Path("data")
DEFAULT_LAMBDA_GRID = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
DEFAULT_TOP_K_GRID = (10,)
DEFAULT_BUDGET_GRID = (50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 35000)
INPUT_SCALE = 255.0


@dataclass(frozen=True)
class EnvironmentStats:
    count: int
    feature_sum: np.ndarray
    second_moment_sum: np.ndarray
    feature_label_sum: np.ndarray
    label_sum: np.ndarray


@dataclass(frozen=True)
class SourceGeometry:
    environment_ids: tuple[str, ...]
    global_mean: np.ndarray
    environment_matrices: np.ndarray
    mean_matrix: np.ndarray
    variation: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    cross_moment: np.ndarray
    label_mean: np.ndarray
    sample_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=CORRUPTIONS)
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        choices=CLASS_NAMES,
        help="One-vs-rest task. Repeat as needed; defaults to all ten classes.",
    )
    parser.add_argument(
        "--budget",
        action="append",
        dest="budgets",
        type=int,
        help="Number of unlabeled target rows; -1 means all available rows.",
    )
    parser.add_argument(
        "--lambda-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDA_GRID),
    )
    parser.add_argument(
        "--top-k-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOP_K_GRID),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter", type=float, default=1e-8)
    parser.add_argument(
        "--source-ridge",
        type=float,
        default=0.0,
        help="Slope-only eta * I added to source-covariance estimators.",
    )
    parser.add_argument(
        "--cv-selection-metric",
        choices=("mse", "auroc"),
        default="mse",
        help="Metric averaged across pseudo-target folds for hyperparameter selection.",
    )
    parser.add_argument(
        "--pseudo-target-count",
        type=int,
        default=-1,
        help=(
            "Use only this many source environments as pseudo targets, selected by "
            "RBF-MMD to the real target's permitted unlabeled budget. -1 uses every "
            "source environment, preserving the original validation procedure."
        ),
    )
    parser.add_argument(
        "--pseudo-target-mmd-samples",
        type=int,
        default=1024,
        help="Maximum rows per environment used for MMD ranking (default: 1024).",
    )
    parser.add_argument(
        "--gamma-weighting",
        choices=(
            "directional_discrepancy",
            "outside_source_range",
            "positive_standardized_excess",
            "bayesian_directional_shrinkage",
            "bayesian_source_posterior_update",
            "bayesian_positive_excess",
        ),
        default="directional_discrepancy",
        help="Rule used to weight the selected variation eigenvectors.",
    )
    parser.add_argument(
        "--project-psd",
        action="store_true",
        help=(
            "Project the corrected target second-moment matrix onto the PSD cone "
            "before solving. Required by the directional Bayesian shrinkage estimator."
        ),
    )
    parser.add_argument(
        "--estimator-base",
        choices=("target_second_moment", "source_covariance", "pooled_covariance"),
        default="target_second_moment",
        help="Unpenalized matrix used in the estimator linear system.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_CLUSTER_DATA_ROOT / "CIFAR-10-C",
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        help="Use row-aligned frozen embeddings from this cache instead of PCA features.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_CLUSTER_DATA_ROOT / "cifar10c_disjoint_environment_splits_seed_42.npz",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_CLUSTER_DATA_ROOT / "disjoint_source_baseline",
    )
    parser.add_argument(
        "--pca-path",
        type=Path,
        help="Explicit baseline pca.npz; otherwise resolve it from --baseline-root.",
    )
    parser.add_argument("--pca-components", type=int, default=200)
    parser.add_argument("--pca-iterated-power", type=int, default=3)
    parser.add_argument("--pca-seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CLUSTER_DATA_ROOT / "disjoint_anti_causal_penalty",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_data_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, resolved / "CIFAR-10-C"):
        if (candidate / "labels.npy").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find labels.npy under {resolved}")


def resolve_pca_path(args: argparse.Namespace) -> Path:
    if args.pca_path is not None:
        path = args.pca_path.expanduser().resolve()
    else:
        filename = (
            f"pca_{args.pca_components}_randomized_"
            f"power_{args.pca_iterated_power}_seed_{args.pca_seed}.npz"
        )
        path = (
            args.baseline_root.expanduser().resolve()
            / "_pca_by_target"
            / args.target
            / filename
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Baseline PCA artifact not found: {path}. Run the linear baseline "
            f"for target={args.target} first, or pass --pca-path."
        )
    return path


def resolve_embedding_cache(path: Path, row_count: int) -> tuple[Path, int, str]:
    root = path.expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Embedding-cache metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    dimension = int(metadata.get("embedding_dimension", 0))
    if dimension <= 0:
        raise ValueError(f"Invalid embedding_dimension in {metadata_path}")
    for environment in CORRUPTIONS:
        embedding_path = root / f"{environment}.npy"
        if not embedding_path.is_file():
            raise FileNotFoundError(f"Embedding cache is missing {embedding_path}")
        values = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (row_count, dimension) or values.dtype != np.float32:
            raise ValueError(
                f"Invalid embedding cache {embedding_path}: expected ({row_count}, {dimension}) "
                f"float32, got {values.shape} {values.dtype}"
            )
    return root, dimension, sha256(metadata_path)


def load_splits(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        splits = {
            name: np.asarray(artifact[name], dtype=np.int64)
            for name in ("train", "val", "test")
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(splits[left], splits[right]).size:
            raise ValueError(f"Split arrays {left} and {right} overlap")
    return splits


def load_pca(
    path: Path,
    target: str,
    expected_sources: Sequence[str],
    split_hash: str,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        components = np.asarray(artifact["components"], dtype=np.float32)
        mean = np.asarray(artifact["mean"], dtype=np.float32)
        artifact_target = str(artifact["target"])
        artifact_sources = tuple(artifact["source_environments"].tolist())
        artifact_split_hash = str(artifact["split_sha256"])
        input_scale = float(artifact["input_scale"][0])
    if artifact_target != target:
        raise ValueError(f"PCA target mismatch: expected {target}, got {artifact_target}")
    if artifact_sources != tuple(expected_sources):
        raise ValueError("PCA source-environment list does not match this experiment")
    if artifact_split_hash != split_hash:
        raise ValueError("PCA and experiment split-file checksums differ")
    if input_scale != INPUT_SCALE:
        raise ValueError(f"Unsupported PCA input scale: {input_scale}")
    if components.ndim != 2 or mean.shape != (components.shape[1],):
        raise ValueError("Invalid PCA component/mean shapes")
    return components, mean


def project_all_environments(
    data_dir: Path,
    components: np.ndarray,
    pca_mean: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    projected: dict[str, np.ndarray] = {}
    for environment_index, environment in enumerate(CORRUPTIONS, start=1):
        started = time.perf_counter()
        images = np.load(
            data_dir / f"{environment}.npy", mmap_mode="r", allow_pickle=False
        )
        result = np.empty((images.shape[0], components.shape[0]), dtype=np.float32)
        for start in range(0, images.shape[0], batch_size):
            stop = min(start + batch_size, images.shape[0])
            flattened = np.asarray(images[start:stop], dtype=np.float32).reshape(
                stop - start, -1
            )
            flattened *= 1.0 / INPUT_SCALE
            result[start:stop] = (flattened - pca_mean) @ components.T
        projected[environment] = result
        print(
            f"Projected [{environment_index}/{len(CORRUPTIONS)}] "
            f"environment={environment} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return projected


def load_all_embedding_environments(embedding_root: Path) -> dict[str, np.ndarray]:
    """Open row-aligned frozen feature arrays without materializing all of them."""
    result: dict[str, np.ndarray] = {}
    for environment in CORRUPTIONS:
        result[environment] = np.load(
            embedding_root / f"{environment}.npy", mmap_mode="r", allow_pickle=False
        )
    print(f"Opened frozen embedding cache from {embedding_root}", flush=True)
    return result


def binary_targets(labels: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    return np.where(labels[:, None] == class_ids[None, :], 1.0, -1.0)


def compute_environment_stats(
    x: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
) -> EnvironmentStats:
    selected = np.asarray(x[indices], dtype=np.float64)
    targets = binary_targets(labels[indices], class_ids)
    return EnvironmentStats(
        count=int(len(indices)),
        feature_sum=selected.sum(axis=0),
        second_moment_sum=selected.T @ selected,
        feature_label_sum=selected.T @ targets,
        label_sum=targets.sum(axis=0),
    )


def centered_environment_matrix(
    stats: EnvironmentStats, global_mean: np.ndarray
) -> np.ndarray:
    environment_mean = stats.feature_sum / float(stats.count)
    second_moment = stats.second_moment_sum / float(stats.count)
    matrix = (
        second_moment
        - np.outer(environment_mean, global_mean)
        - np.outer(global_mean, environment_mean)
        + np.outer(global_mean, global_mean)
    )
    return 0.5 * (matrix + matrix.T)


def build_source_geometry(
    environment_ids: Sequence[str],
    stats_by_environment: dict[str, EnvironmentStats],
) -> SourceGeometry:
    stats = [stats_by_environment[environment] for environment in environment_ids]
    total_count = sum(item.count for item in stats)
    feature_sum = sum((item.feature_sum for item in stats), start=np.zeros_like(stats[0].feature_sum))
    label_sum = sum((item.label_sum for item in stats), start=np.zeros_like(stats[0].label_sum))
    feature_label_sum = sum(
        (item.feature_label_sum for item in stats),
        start=np.zeros_like(stats[0].feature_label_sum),
    )
    global_mean = feature_sum / float(total_count)
    environment_matrices = np.stack(
        [centered_environment_matrix(item, global_mean) for item in stats], axis=0
    )
    mean_matrix = environment_matrices.mean(axis=0)
    variation = np.zeros_like(mean_matrix)
    for matrix in environment_matrices:
        difference = matrix - mean_matrix
        variation += difference @ difference
    variation /= float(len(environment_ids))
    variation = 0.5 * (variation + variation.T)
    eigenvalues, eigenvectors = np.linalg.eigh(variation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    cross_moment = (
        feature_label_sum - np.outer(global_mean, label_sum)
    ) / float(total_count)
    label_mean = label_sum / float(total_count)
    return SourceGeometry(
        environment_ids=tuple(environment_ids),
        global_mean=global_mean,
        environment_matrices=environment_matrices,
        mean_matrix=mean_matrix,
        variation=variation,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        cross_moment=cross_moment,
        label_mean=label_mean,
        sample_count=total_count,
    )


def target_matrix(x: np.ndarray, indices: np.ndarray, source_mean: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError("Cannot compute a target matrix from zero samples")
    centered = np.asarray(x[indices], dtype=np.float64) - source_mean
    matrix = (centered.T @ centered) / float(len(indices))
    return 0.5 * (matrix + matrix.T)


def build_gamma(
    geometry: SourceGeometry,
    target_second_moment: np.ndarray,
    top_k: int,
    weighting: str = "directional_discrepancy",
    target_centered_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    effective_k = max(1, min(int(top_k), geometry.eigenvectors.shape[1]))
    eigenvalues = np.maximum(geometry.eigenvalues[:effective_k], 0.0)
    basis = geometry.eigenvectors[:, :effective_k]
    denominator = max(float(eigenvalues[0]), 1e-12)
    target_difference = target_second_moment - geometry.mean_matrix
    directional_discrepancy = np.abs(
        np.einsum("ik,ij,jk->k", basis, target_difference, basis)
    )
    eigenvalue_scale = eigenvalues / denominator
    q_target = np.einsum("ik,ij,jk->k", basis, target_second_moment, basis)
    q_source = np.einsum("ik,ij,jk->k", basis, geometry.mean_matrix, basis)
    q_environments = np.einsum(
        "ik,eij,jk->ek", basis, geometry.environment_matrices, basis
    )
    q_source_min = q_environments.min(axis=0)
    q_source_max = q_environments.max(axis=0)
    outside_discrepancy = np.maximum.reduce(
        (
            np.zeros_like(q_target),
            q_target - q_source_max,
            q_source_min - q_target,
        )
    )
    source_direction_variance = np.var(q_environments, axis=0, ddof=1)
    target_observation_variance = np.full(effective_k, np.nan, dtype=np.float64)
    target_mean_variance = np.full(effective_k, np.nan, dtype=np.float64)
    standardized_excess = np.full(effective_k, np.nan, dtype=np.float64)

    if weighting == "directional_discrepancy":
        raw_weights = eigenvalue_scale * directional_discrepancy
        raw_sum = float(raw_weights.sum())
        if raw_sum > 1e-12:
            weights = raw_weights / raw_sum
        else:
            weights = np.full(effective_k, 1.0 / effective_k, dtype=np.float64)
        normalization = "divide_by_sum_with_uniform_fallback"
    elif weighting == "outside_source_range":
        raw_weights = eigenvalue_scale * outside_discrepancy
        raw_mean = float(raw_weights.mean()) if raw_weights.size else 0.0
        if raw_mean > 1e-12:
            weights = raw_weights / raw_mean
        else:
            weights = np.zeros_like(raw_weights)
        normalization = "divide_by_selected_mean_with_zero_fallback"
    elif weighting == "positive_standardized_excess":
        if target_centered_samples is None:
            raise ValueError(
                "positive_standardized_excess requires target-centered samples"
            )
        centered = np.asarray(target_centered_samples, dtype=np.float64)
        if centered.ndim != 2 or centered.shape[1] != basis.shape[0]:
            raise ValueError("Target-centered samples have an incompatible shape")
        if centered.shape[0] < 2:
            raise ValueError(
                "positive_standardized_excess requires at least two target samples"
            )
        squared_projections = np.square(centered @ basis)
        target_observation_variance = np.var(squared_projections, axis=0, ddof=1)
        target_mean_variance = target_observation_variance / float(centered.shape[0])
        variance_sum = source_direction_variance + target_mean_variance
        variance_floor = np.finfo(np.float64).eps * np.maximum(
            1.0, np.square(np.abs(q_source) + np.abs(q_target))
        )
        standardized_excess = (q_target - q_source) / np.sqrt(
            np.maximum(variance_sum, variance_floor)
        )
        raw_weights = np.maximum(standardized_excess, 0.0)
        weights = raw_weights
        normalization = "none_positive_part_no_threshold"
    elif weighting == "bayesian_directional_shrinkage":
        if target_centered_samples is None:
            raise ValueError(
                "bayesian_directional_shrinkage requires target-centered samples"
            )
        centered = np.asarray(target_centered_samples, dtype=np.float64)
        if centered.ndim != 2 or centered.shape[1] != basis.shape[0]:
            raise ValueError("Target-centered samples have an incompatible shape")
        if centered.shape[0] < 2:
            raise ValueError(
                "bayesian_directional_shrinkage requires at least two target samples"
            )
        squared_projections = np.square(centered @ basis)
        target_observation_variance = np.var(squared_projections, axis=0, ddof=1)
        target_mean_variance = target_observation_variance / float(centered.shape[0])
        total_variance = source_direction_variance + target_mean_variance
        variance_floor = np.finfo(np.float64).eps * np.maximum(
            1.0, np.square(np.abs(q_source) + np.abs(q_target))
        )
        # 1 - kappa: uncertainty-driven shrinkage from the noisy target
        # directional moment toward its source-environment prior mean.
        shrinkage_strength = np.divide(
            target_mean_variance,
            total_variance,
            out=np.zeros_like(total_variance),
            where=total_variance > variance_floor,
        )
        raw_weights = shrinkage_strength * (q_source - q_target)
        weights = raw_weights
        normalization = "none_signed_posterior_moment_correction"
    elif weighting == "bayesian_source_posterior_update":
        if target_centered_samples is None:
            raise ValueError(
                "bayesian_source_posterior_update requires target-centered samples"
            )
        centered = np.asarray(target_centered_samples, dtype=np.float64)
        if centered.ndim != 2 or centered.shape[1] != basis.shape[0]:
            raise ValueError("Target-centered samples have an incompatible shape")
        if centered.shape[0] < 2:
            raise ValueError(
                "bayesian_source_posterior_update requires at least two target samples"
            )
        squared_projections = np.square(centered @ basis)
        target_observation_variance = np.var(squared_projections, axis=0, ddof=1)
        target_mean_variance = target_observation_variance / float(centered.shape[0])
        total_variance = source_direction_variance + target_mean_variance
        variance_floor = np.finfo(np.float64).eps * np.maximum(
            1.0, np.square(np.abs(q_source) + np.abs(q_target))
        )
        # kappa: trust in the observed target directional moment. Adding this
        # correction to the source covariance gives the posterior moment.
        target_trust = np.divide(
            source_direction_variance,
            total_variance,
            out=np.zeros_like(total_variance),
            where=total_variance > variance_floor,
        )
        raw_weights = target_trust * (q_target - q_source)
        weights = raw_weights
        normalization = "none_signed_source_to_posterior_update"
    elif weighting == "bayesian_positive_excess":
        if target_centered_samples is None:
            raise ValueError(
                "bayesian_positive_excess requires target-centered samples"
            )
        centered = np.asarray(target_centered_samples, dtype=np.float64)
        if centered.ndim != 2 or centered.shape[1] != basis.shape[0]:
            raise ValueError("Target-centered samples have an incompatible shape")
        if centered.shape[0] < 2:
            raise ValueError(
                "bayesian_positive_excess requires at least two target samples"
            )
        squared_projections = np.square(centered @ basis)
        target_observation_variance = np.var(squared_projections, axis=0, ddof=1)
        target_mean_variance = target_observation_variance / float(centered.shape[0])
        total_variance = source_direction_variance + target_mean_variance
        variance_floor = np.finfo(np.float64).eps * np.maximum(
            1.0, np.square(np.abs(q_source) + np.abs(q_target))
        )
        target_trust = np.divide(
            source_direction_variance,
            total_variance,
            out=np.zeros_like(total_variance),
            where=total_variance > variance_floor,
        )
        raw_weights = target_trust * np.maximum(q_target - q_source, 0.0)
        weights = raw_weights
        normalization = "none_positive_source_to_target_excess_update"
    else:
        raise ValueError(f"Unknown gamma weighting: {weighting}")
    gamma = (basis * weights[None, :]) @ basis.T
    gamma = 0.5 * (gamma + gamma.T)
    return gamma, {
        "effective_top_k": effective_k,
        "basis": basis,
        "selected_eigenvalues": eigenvalues,
        "weighting": weighting,
        "normalization": normalization,
        "directional_discrepancy": directional_discrepancy,
        "outside_discrepancy": outside_discrepancy,
        "source_direction_variance": source_direction_variance,
        "target_observation_variance": target_observation_variance,
        "target_mean_variance": target_mean_variance,
        "standardized_excess": standardized_excess,
        "q_target": q_target,
        "q_source": q_source,
        "q_environments": q_environments,
        "q_source_min": q_source_min,
        "q_source_max": q_source_max,
        "raw_weights": raw_weights,
        "weights": weights,
    }


def solve_estimator(
    geometry: SourceGeometry,
    target_second_moment: np.ndarray,
    gamma: np.ndarray,
    lambda_value: float,
    jitter: float,
    estimator_base: str = "target_second_moment",
    target_sample_count: int | None = None,
    project_psd: bool = False,
    source_ridge: float = 0.0,
    compute_condition_number: bool = True,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    dimension = target_second_moment.shape[0]
    if estimator_base in {"target_second_moment", "pooled_covariance"}:
        if estimator_base == "pooled_covariance":
            if target_sample_count is None or target_sample_count <= 0:
                raise ValueError("pooled_covariance requires a positive target sample count")
            base_matrix = (
                float(geometry.sample_count) * geometry.mean_matrix
                + float(target_sample_count) * target_second_moment
            ) / float(geometry.sample_count + target_sample_count)
        else:
            base_matrix = target_second_moment
        system = base_matrix + float(lambda_value) * gamma
        if project_psd:
            eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (system + system.T))
            system = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            system = 0.5 * (system + system.T)
        system = system + float(jitter) * np.eye(dimension, dtype=np.float64)
        try:
            beta = np.linalg.solve(system, geometry.cross_moment)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(system, geometry.cross_moment, rcond=None)[0]
        intercept = geometry.label_mean - geometry.global_mean @ beta
    elif estimator_base == "source_covariance":
        # Use the same augmented normal equations as the baseline, normalized
        # by sample count. The only modification is lambda * Gamma in the slope
        # block; the intercept is unpenalized and no isotropic jitter is added.
        centered_system = (
            geometry.mean_matrix
            + float(lambda_value) * gamma
            + float(source_ridge) * np.eye(dimension, dtype=np.float64)
        )
        if project_psd and float(lambda_value) != 0.0:
            eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (centered_system + centered_system.T)
            )
            centered_system = (
                eigenvectors * np.maximum(eigenvalues, 0.0)
            ) @ eigenvectors.T
            centered_system = 0.5 * (centered_system + centered_system.T)
        raw_second_moment = centered_system + np.outer(
            geometry.global_mean, geometry.global_mean
        )
        raw_cross_moment = geometry.cross_moment + np.outer(
            geometry.global_mean, geometry.label_mean
        )
        system = np.empty((dimension + 1, dimension + 1), dtype=np.float64)
        system[:dimension, :dimension] = raw_second_moment
        system[:dimension, dimension] = geometry.global_mean
        system[dimension, :dimension] = geometry.global_mean
        system[dimension, dimension] = 1.0
        right_hand_side = np.vstack((raw_cross_moment, geometry.label_mean))
        weights = np.linalg.lstsq(system, right_hand_side, rcond=None)[0]
        beta = weights[:dimension]
        intercept = weights[dimension]
    else:
        raise ValueError(f"Unknown estimator base: {estimator_base}")
    condition_number = (
        float(np.linalg.cond(system)) if compute_condition_number else None
    )
    return beta, intercept, condition_number


def select_budget_indices(
    available_indices: np.ndarray,
    budget: int,
    seed: int,
    target_name: str,
    environment_name: str,
) -> np.ndarray:
    if budget == -1 or budget >= len(available_indices):
        return np.asarray(available_indices, dtype=np.int64)
    if budget <= 0:
        raise ValueError(f"Budget must be -1 or positive, got {budget}")
    seed_sequence = np.random.SeedSequence(
        [seed, CORRUPTIONS.index(target_name), CORRUPTIONS.index(environment_name)]
    )
    rng = np.random.default_rng(seed_sequence)
    permutation = rng.permutation(available_indices)
    return np.sort(np.asarray(permutation[:budget], dtype=np.int64))


def _mmd_subsample(indices: np.ndarray, maximum: int, seed_parts: Sequence[int]) -> np.ndarray:
    if len(indices) <= maximum:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def _median_rbf_bandwidth_squared(reference: np.ndarray) -> float:
    squared_norms = np.einsum("ij,ij->i", reference, reference)
    distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (reference @ reference.T)
    positive = distances[np.triu_indices_from(distances, k=1)]
    positive = positive[positive > 0.0]
    if not len(positive):
        return 1.0
    return max(float(np.median(positive)), np.finfo(np.float64).eps)


def _rbf_mmd_squared(left: np.ndarray, right: np.ndarray, bandwidth_squared: float) -> float:
    def kernel_mean(first: np.ndarray, second: np.ndarray) -> float:
        first_norms = np.einsum("ij,ij->i", first, first)
        second_norms = np.einsum("ij,ij->i", second, second)
        distances = first_norms[:, None] + second_norms[None, :] - 2.0 * (first @ second.T)
        return float(np.exp(-np.maximum(distances, 0.0) / (2.0 * bandwidth_squared)).mean())

    return max(kernel_mean(left, left) + kernel_mean(right, right) - 2.0 * kernel_mean(left, right), 0.0)


def rank_pseudo_targets_by_mmd(
    *, target: str, budget: int, seed: int, target_indices: np.ndarray,
    dev_indices: np.ndarray, source_environments: Sequence[str],
    embeddings: dict[str, np.ndarray], max_samples: int,
) -> list[tuple[str, float]]:
    """Rank source environments using only permitted real-target unlabeled rows."""
    if max_samples <= 0:
        raise ValueError("--pseudo-target-mmd-samples must be positive")
    target_rows = _mmd_subsample(
        target_indices, max_samples, (seed, CORRUPTIONS.index(target), 0, budget)
    )
    target_x = np.asarray(embeddings[target][target_rows], dtype=np.float64)
    # A fixed, balanced source reference gives one common median-heuristic
    # bandwidth for all candidate-environment MMD values.
    reference_parts = [target_x]
    source_samples: dict[str, np.ndarray] = {}
    for environment in source_environments:
        rows = _mmd_subsample(
            dev_indices, max_samples,
            (seed, CORRUPTIONS.index(target), CORRUPTIONS.index(environment), budget),
        )
        source_samples[environment] = np.asarray(embeddings[environment][rows], dtype=np.float64)
        reference_parts.append(source_samples[environment])
    reference = np.concatenate(reference_parts, axis=0)
    if len(reference) > 2048:
        reference_rows = _mmd_subsample(
            np.arange(len(reference), dtype=np.int64), 2048,
            (seed, CORRUPTIONS.index(target), 1, budget),
        )
        reference = reference[reference_rows]
    bandwidth_squared = _median_rbf_bandwidth_squared(reference)
    ranking = [
        (environment, _rbf_mmd_squared(target_x, source_samples[environment], bandwidth_squared))
        for environment in source_environments
    ]
    return sorted(ranking, key=lambda item: (item[1], item[0]))


def run_cross_validation(
    *,
    target: str,
    budget: int,
    source_environments: Sequence[str],
    pseudo_targets: Sequence[str],
    dev_stats: dict[str, EnvironmentStats],
    embeddings: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    labels: np.ndarray,
    class_ids: np.ndarray,
    lambda_grid: Sequence[float],
    top_k_grid: Sequence[int],
    seed: int,
    jitter: float,
    gamma_weighting: str,
    estimator_base: str,
    project_psd: bool,
    selection_metric: str,
    source_ridge: float,
) -> tuple[dict[int, tuple[float, int]], dict[str, object]]:
    combinations = [
        (float(lambda_value), int(top_k))
        for lambda_value in lambda_grid
        for top_k in top_k_grid
    ]
    losses = {
        combination: np.zeros((len(pseudo_targets), len(class_ids)), dtype=np.float64)
        for combination in combinations
    }
    aurocs = {
        combination: np.zeros((len(pseudo_targets), len(class_ids)), dtype=np.float64)
        for combination in combinations
    }
    fold_payloads: list[dict[str, object]] = []

    for fold_index, pseudo_target in enumerate(pseudo_targets):
        fold_sources = tuple(
            environment
            for environment in source_environments
            if environment != pseudo_target
        )
        geometry = build_source_geometry(fold_sources, dev_stats)
        selected_target_indices = select_budget_indices(
            splits["train"], budget, seed, target, pseudo_target
        )
        pseudo_target_matrix = target_matrix(
            embeddings[pseudo_target], selected_target_indices, geometry.global_mean
        )
        pseudo_target_centered = (
            np.asarray(
                embeddings[pseudo_target][selected_target_indices], dtype=np.float64
            )
            - geometry.global_mean
        )
        validation_x = np.asarray(
            embeddings[pseudo_target][splits["val"]], dtype=np.float64
        )
        validation_y = binary_targets(labels[splits["val"]], class_ids)
        gamma_by_k = {
            int(top_k): build_gamma(
                geometry,
                pseudo_target_matrix,
                int(top_k),
                weighting=gamma_weighting,
                target_centered_samples=pseudo_target_centered,
            )[0]
            for top_k in top_k_grid
        }
        fold_rows: list[dict[str, object]] = []
        for lambda_value, top_k in combinations:
            beta, intercept, condition_number = solve_estimator(
                geometry,
                pseudo_target_matrix,
                gamma_by_k[top_k],
                lambda_value,
                jitter,
                estimator_base=estimator_base,
                target_sample_count=len(selected_target_indices),
                project_psd=project_psd,
                source_ridge=source_ridge,
                compute_condition_number=False,
            )
            scores = validation_x @ beta + intercept
            mse = np.mean(np.square(scores - validation_y), axis=0)
            auroc = np.asarray(
                [
                    roc_auc_score(validation_y[:, index] == 1.0, scores[:, index])
                    for index in range(len(class_ids))
                ],
                dtype=np.float64,
            )
            losses[(lambda_value, top_k)][fold_index] = mse
            aurocs[(lambda_value, top_k)][fold_index] = auroc
            fold_rows.append(
                {
                    "lambda": lambda_value,
                    "top_k": top_k,
                    "mse_by_task": {
                        CLASS_NAMES[int(class_id)]: float(mse[index])
                        for index, class_id in enumerate(class_ids)
                    },
                    "auroc_by_task": {
                        CLASS_NAMES[int(class_id)]: float(auroc[index])
                        for index, class_id in enumerate(class_ids)
                    },
                }
            )
        fold_payloads.append(
            {
                "fold": fold_index,
                "pseudo_target": pseudo_target,
                "source_environments": list(fold_sources),
                "unlabeled_train_count": int(len(selected_target_indices)),
                "validation_count": int(len(splits["val"])),
                "rows": fold_rows,
            }
        )

    selected: dict[int, tuple[float, int]] = {}
    average_rows: list[dict[str, object]] = []
    for lambda_value, top_k in combinations:
        mean_mse = losses[(lambda_value, top_k)].mean(axis=0)
        mean_auroc = aurocs[(lambda_value, top_k)].mean(axis=0)
        average_rows.append(
            {
                "lambda": lambda_value,
                "top_k": top_k,
                "mean_mse_by_task": {
                    CLASS_NAMES[int(class_id)]: float(mean_mse[index])
                    for index, class_id in enumerate(class_ids)
                },
                "mean_auroc_by_task": {
                    CLASS_NAMES[int(class_id)]: float(mean_auroc[index])
                    for index, class_id in enumerate(class_ids)
                },
            }
        )
    for task_index, class_id in enumerate(class_ids):
        if selection_metric == "mse":
            selected[int(class_id)] = min(
                combinations,
                key=lambda combination: (
                    float(losses[combination][:, task_index].mean()),
                    combination[0],
                    combination[1],
                ),
            )
        else:
            selected[int(class_id)] = min(
                combinations,
                key=lambda combination: (
                    -float(aurocs[combination][:, task_index].mean()),
                    combination[0],
                    combination[1],
                ),
            )
    return selected, {
        "selection_metric": selection_metric,
        "pseudo_targets": list(pseudo_targets),
        "folds": fold_payloads,
        "averages": average_rows,
    }


def is_complete(run_dir: Path, expected: dict[str, object]) -> bool:
    required = (
        run_dir / "config.json",
        run_dir / "metrics.json",
        run_dir / "model.npz",
        run_dir / "cv_results.json",
        run_dir / "penalty_matrices.npz",
        run_dir / "target_test_predictions.npz",
    )
    representation_artifact = (
        run_dir / "representation_metadata.json"
        if expected.get("representation_mode") == "frozen_embedding_cache"
        else run_dir / "pca.npz"
    )
    if not all(path.is_file() for path in (*required, representation_artifact)):
        return False
    try:
        with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return config.get("status") == "complete" and all(
            config.get(key) == value for key, value in expected.items()
        )
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    tasks = list(dict.fromkeys(args.tasks or CLASS_NAMES))
    budgets = list(dict.fromkeys(args.budgets or DEFAULT_BUDGET_GRID))
    if any(budget == 0 or budget < -1 for budget in budgets):
        raise ValueError("Budgets must be -1 or positive integers")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.jitter < 0.0:
        raise ValueError("--jitter must be non-negative")
    if args.source_ridge < 0.0:
        raise ValueError("--source-ridge must be non-negative")
    lambda_grid = sorted(set(float(value) for value in args.lambda_grid))
    top_k_grid = sorted(set(int(value) for value in args.top_k_grid))
    if any(value < 0.0 for value in lambda_grid):
        raise ValueError("Lambda values must be non-negative")
    if any(value <= 0 for value in top_k_grid):
        raise ValueError("Top-k values must be positive")
    if args.pseudo_target_count == 0 or args.pseudo_target_count < -1:
        raise ValueError("--pseudo-target-count must be -1 or a positive integer")
    if args.pseudo_target_mmd_samples <= 0:
        raise ValueError("--pseudo-target-mmd-samples must be positive")

    data_dir = resolve_data_dir(args.data_dir)
    split_path = args.split_file.expanduser().resolve()
    splits = load_splits(split_path)
    split_hash = sha256(split_path)
    source_environments = tuple(
        environment for environment in CORRUPTIONS if environment != args.target
    )
    labels = np.load(data_dir / "labels.npy", mmap_mode="r", allow_pickle=False)
    if labels.ndim != 1:
        raise ValueError(f"Expected one-dimensional labels, got shape {labels.shape}")
    embedding_root: Path | None = None
    embedding_dimension: int | None = None
    embedding_metadata_hash: str | None = None
    pca_path: Path | None = None
    pca_hash: str | None = None
    if args.embedding_root is None:
        pca_path = resolve_pca_path(args)
        pca_hash = sha256(pca_path)
    else:
        embedding_root, embedding_dimension, embedding_metadata_hash = resolve_embedding_cache(
            args.embedding_root, len(labels)
        )
    output_root = args.output_root.expanduser().resolve()

    run_dirs = {
        (task, budget): output_root / f"{task}_vs_rest" / args.target / str(budget)
        for task in tasks
        for budget in budgets
    }
    expected = {
        "target_environment": args.target,
        "split_sha256": split_hash,
        "lambda_grid": lambda_grid,
        "top_k_grid": top_k_grid,
        "seed": args.seed,
        "gamma_weighting": args.gamma_weighting,
        "estimator_base": args.estimator_base,
        "pseudo_target_selection": "leave_one_source_environment_out",
    }
    if args.pseudo_target_count != -1:
        expected.update({
            "pseudo_target_count": args.pseudo_target_count,
            "pseudo_target_ranking": "rbf_mmd_real_target_unlabeled_budget",
            "pseudo_target_mmd_samples": args.pseudo_target_mmd_samples,
        })
    if embedding_root is None:
        expected.update({"representation_mode": "pca", "pca_sha256": pca_hash})
    else:
        expected.update({
            "representation_mode": "frozen_embedding_cache",
            "embedding_root": str(embedding_root),
            "embedding_dimension": embedding_dimension,
            "embedding_metadata_sha256": embedding_metadata_hash,
        })
    if args.cv_selection_metric != "mse":
        # Older penalty artifacts implicitly used MSE. Only a non-default
        # selection metric needs to invalidate those completed runs.
        expected["cv_selection_metric"] = args.cv_selection_metric
    if args.source_ridge != 0.0:
        expected["source_ridge"] = args.source_ridge
    if args.project_psd:
        # Preserve completion compatibility with existing non-PSD penalty runs.
        expected["project_psd"] = True
    if args.estimator_base == "source_covariance":
        expected["estimator_solver"] = (
            "source_covariance_gamma_ridge_augmented_v1"
            if args.source_ridge != 0.0
            else (
                "source_covariance_psd_projected_gamma_augmented_v1"
                if args.project_psd
                else "baseline_plus_gamma_augmented_v4_exact_zero"
            )
        )
    else:
        expected["jitter"] = args.jitter
    pending = {
        (task, budget)
        for task in tasks
        for budget in budgets
        if args.force
        or not is_complete(
            run_dirs[(task, budget)],
            dict(expected, task=f"{task}_vs_rest", budget=budget),
        )
    }
    if not pending:
        print(f"All requested runs are complete for target={args.target}")
        return

    if embedding_root is None:
        components, pca_mean = load_pca(
            pca_path, args.target, source_environments, split_hash
        )
        embeddings = project_all_environments(
            data_dir, components, pca_mean, args.batch_size
        )
    else:
        embeddings = load_all_embedding_environments(embedding_root)
    class_ids = np.asarray([CLASS_NAMES.index(task) for task in tasks], dtype=np.int64)
    dev_indices = np.sort(np.concatenate((splits["train"], splits["val"])))
    dev_stats = {
        environment: compute_environment_stats(
            embeddings[environment], dev_indices, labels, class_ids
        )
        for environment in source_environments
    }

    for budget in budgets:
        pending_tasks = [task for task in tasks if (task, budget) in pending]
        if not pending_tasks:
            continue
        real_target_indices = select_budget_indices(
            dev_indices, budget, args.seed, args.target, args.target
        )
        if args.pseudo_target_count == -1 or args.pseudo_target_count >= len(source_environments):
            pseudo_targets = source_environments
            pseudo_target_ranking: list[tuple[str, float]] | None = None
        else:
            pseudo_target_ranking = rank_pseudo_targets_by_mmd(
                target=args.target,
                budget=budget,
                seed=args.seed,
                target_indices=real_target_indices,
                dev_indices=dev_indices,
                source_environments=source_environments,
                embeddings=embeddings,
                max_samples=args.pseudo_target_mmd_samples,
            )
            pseudo_targets = tuple(
                environment for environment, _distance in pseudo_target_ranking[:args.pseudo_target_count]
            )
            print(
                "Selected MMD-nearest pseudo targets: "
                + ", ".join(f"{name} ({distance:.6g})" for name, distance in pseudo_target_ranking[:args.pseudo_target_count]),
                flush=True,
            )
        print(
            f"Cross-validating target={args.target} budget={budget} "
            f"pseudo_target_folds={len(pseudo_targets)}",
            flush=True,
        )
        selected, cv_payload = run_cross_validation(
            target=args.target,
            budget=budget,
            source_environments=source_environments,
            pseudo_targets=pseudo_targets,
            dev_stats=dev_stats,
            embeddings=embeddings,
            splits=splits,
            labels=labels,
            class_ids=class_ids,
            lambda_grid=lambda_grid,
            top_k_grid=top_k_grid,
            seed=args.seed,
            jitter=args.jitter,
            gamma_weighting=args.gamma_weighting,
            estimator_base=args.estimator_base,
            project_psd=args.project_psd,
            selection_metric=args.cv_selection_metric,
            source_ridge=args.source_ridge,
        )
        if pseudo_target_ranking is not None:
            cv_payload["pseudo_target_mmd_ranking"] = [
                {"environment": environment, "mmd_squared": distance}
                for environment, distance in pseudo_target_ranking
            ]

        final_geometry = build_source_geometry(source_environments, dev_stats)
        real_target_matrix = target_matrix(
            embeddings[args.target], real_target_indices, final_geometry.global_mean
        )
        real_target_centered = (
            np.asarray(embeddings[args.target][real_target_indices], dtype=np.float64)
            - final_geometry.global_mean
        )
        gamma_cache = {
            top_k: build_gamma(
                final_geometry,
                real_target_matrix,
                top_k,
                weighting=args.gamma_weighting,
                target_centered_samples=real_target_centered,
            )
            for top_k in {combination[1] for combination in selected.values()}
        }
        test_x = np.asarray(embeddings[args.target][splits["test"]], dtype=np.float64)
        test_targets = binary_targets(labels[splits["test"]], class_ids)

        for task_index, task in enumerate(tasks):
            if task not in pending_tasks:
                continue
            class_id = int(class_ids[task_index])
            selected_lambda, selected_top_k = selected[class_id]
            gamma, gamma_stats = gamma_cache[selected_top_k]
            task_geometry = SourceGeometry(
                environment_ids=final_geometry.environment_ids,
                global_mean=final_geometry.global_mean,
                environment_matrices=final_geometry.environment_matrices,
                mean_matrix=final_geometry.mean_matrix,
                variation=final_geometry.variation,
                eigenvalues=final_geometry.eigenvalues,
                eigenvectors=final_geometry.eigenvectors,
                cross_moment=final_geometry.cross_moment[:, [task_index]],
                label_mean=final_geometry.label_mean[[task_index]],
                sample_count=final_geometry.sample_count,
            )
            targets = test_targets[:, task_index]
            if (
                args.estimator_base == "source_covariance"
                and selected_lambda == 0.0
                and args.source_ridge == 0.0
            ):
                baseline_dir = (
                    args.baseline_root.expanduser().resolve()
                    / f"{task}_vs_rest"
                    / args.target
                )
                with np.load(
                    baseline_dir / "linear_model.npz", allow_pickle=False
                ) as artifact:
                    beta = np.asarray(artifact["coefficients"], dtype=np.float64)
                    intercept = float(np.asarray(artifact["intercept"])[0])
                with np.load(
                    baseline_dir / "target_test_predictions.npz", allow_pickle=False
                ) as artifact:
                    baseline_indices = np.asarray(artifact["indices"], dtype=np.int64)
                    baseline_targets = np.asarray(artifact["targets"], dtype=np.float64)
                    scores = np.asarray(artifact["scores"], dtype=np.float64)
                if not np.array_equal(baseline_indices, splits["test"]):
                    raise ValueError("Baseline and penalty test indices differ")
                if not np.array_equal(baseline_targets, targets):
                    raise ValueError("Baseline and penalty test targets differ")
                with (baseline_dir / "metrics.json").open(
                    "r", encoding="utf-8"
                ) as handle:
                    baseline_metrics = json.load(handle)
                mse = float(baseline_metrics["target_test_mse"])
                auroc = float(baseline_metrics["target_test_auroc"])
                condition_number = baseline_metrics.get(
                    "normal_equation_condition_number"
                )
            else:
                beta_matrix, intercept_vector, condition_number = solve_estimator(
                    task_geometry,
                    real_target_matrix,
                    gamma,
                    selected_lambda,
                    args.jitter,
                    estimator_base=args.estimator_base,
                    target_sample_count=len(real_target_indices),
                    project_psd=args.project_psd,
                    source_ridge=args.source_ridge,
                )
                beta = beta_matrix[:, 0]
                intercept = float(intercept_vector[0])
                scores = test_x @ beta + intercept
                mse = float(np.mean(np.square(scores - targets)))
                auroc = float(roc_auc_score(targets == 1.0, scores))
            run_dir = run_dirs[(task, budget)]
            run_dir.mkdir(parents=True, exist_ok=True)

            task_cv_payload = dict(cv_payload)
            task_cv_payload["task"] = f"{task}_vs_rest"
            task_cv_payload["selected_lambda"] = selected_lambda
            task_cv_payload["selected_top_k"] = selected_top_k
            atomic_write_json(run_dir / "cv_results.json", task_cv_payload)
            atomic_savez(
                run_dir / "model.npz",
                beta=beta,
                intercept=np.asarray([intercept]),
                source_mean=final_geometry.global_mean,
                positive_class_id=np.asarray([class_id], dtype=np.int64),
                selected_lambda=np.asarray([selected_lambda]),
                selected_top_k=np.asarray([selected_top_k], dtype=np.int64),
            )
            atomic_savez(
                run_dir / "target_test_predictions.npz",
                indices=splits["test"],
                targets=targets,
                scores=scores,
            )
            atomic_savez(
                run_dir / "penalty_matrices.npz",
                source_mean=final_geometry.global_mean,
                source_environment_matrices=final_geometry.environment_matrices,
                source_mean_matrix=final_geometry.mean_matrix,
                variation=final_geometry.variation,
                eigenvalues=final_geometry.eigenvalues,
                eigenvectors=final_geometry.eigenvectors,
                target_matrix=real_target_matrix,
                estimator_base_matrix=(
                    real_target_matrix
                    if args.estimator_base == "target_second_moment"
                    else (
                        (
                            float(final_geometry.sample_count) * final_geometry.mean_matrix
                            + float(len(real_target_indices)) * real_target_matrix
                        )
                        / float(final_geometry.sample_count + len(real_target_indices))
                        if args.estimator_base == "pooled_covariance"
                        else final_geometry.mean_matrix
                    )
                ),
                gamma=gamma,
                source_cross_moment=task_geometry.cross_moment[:, 0],
                source_label_mean=task_geometry.label_mean,
                selected_basis=np.asarray(gamma_stats["basis"]),
                directional_discrepancy=np.asarray(
                    gamma_stats["directional_discrepancy"]
                ),
                outside_discrepancy=np.asarray(gamma_stats["outside_discrepancy"]),
                source_direction_variance=np.asarray(
                    gamma_stats["source_direction_variance"]
                ),
                tau_squared=np.asarray(gamma_stats["source_direction_variance"]),
                target_observation_variance=np.asarray(
                    gamma_stats["target_observation_variance"]
                ),
                r=np.asarray(gamma_stats["target_observation_variance"]),
                target_mean_variance=np.asarray(gamma_stats["target_mean_variance"]),
                var_q_target=np.asarray(gamma_stats["target_mean_variance"]),
                standardized_excess=np.asarray(gamma_stats["standardized_excess"]),
                q_target=np.asarray(gamma_stats["q_target"]),
                q_source=np.asarray(gamma_stats["q_source"]),
                q_environments=np.asarray(gamma_stats["q_environments"]),
                q_source_min=np.asarray(gamma_stats["q_source_min"]),
                q_source_max=np.asarray(gamma_stats["q_source_max"]),
                raw_weights=np.asarray(gamma_stats["raw_weights"]),
                weights=np.asarray(gamma_stats["weights"]),
                gamma_weighting=np.asarray(args.gamma_weighting),
                environment_ids=np.asarray(final_geometry.environment_ids),
                target_unlabeled_indices=real_target_indices,
            )
            metrics: dict[str, object] = {
                "task": f"{task}_vs_rest",
                "positive_class_id": class_id,
                "target_environment": args.target,
                "budget": budget,
                "effective_target_unlabeled_count": int(len(real_target_indices)),
                "source_train_val_count": final_geometry.sample_count,
                "target_test_count": int(len(splits["test"])),
                "target_test_positive_count": int(np.count_nonzero(targets == 1.0)),
                "target_test_mse": mse,
                "target_test_auroc": auroc,
                "selected_lambda": selected_lambda,
                "selected_top_k": selected_top_k,
                "system_condition_number": condition_number,
                "pseudo_targets": list(pseudo_targets),
                "gamma_weighting": args.gamma_weighting,
                "estimator_base": args.estimator_base,
                "project_psd": bool(args.project_psd),
                "cv_selection_metric": args.cv_selection_metric,
                "source_ridge": args.source_ridge,
            }
            atomic_write_json(run_dir / "metrics.json", metrics)
            config = dict(
                expected,
                status="complete",
                task=f"{task}_vs_rest",
                positive_class_id=class_id,
                budget=budget,
                effective_target_unlabeled_count=int(len(real_target_indices)),
                source_environments=list(source_environments),
                pseudo_targets=list(pseudo_targets),
                pseudo_target_count=len(pseudo_targets),
                pseudo_target_ranking=(
                    "all_source_environments"
                    if pseudo_target_ranking is None
                    else "rbf_mmd_real_target_unlabeled_budget"
                ),
                pseudo_target_mmd_samples=(
                    None if pseudo_target_ranking is None else args.pseudo_target_mmd_samples
                ),
                source_splits=["train", "val"],
                pseudo_target_unlabeled_split="train",
                pseudo_target_evaluation_split="val",
                real_target_unlabeled_splits=["train", "val"],
                real_target_evaluation_split="test",
                data_dir=str(data_dir),
                split_file=str(split_path),
                representation_mode=(
                    "pca" if embedding_root is None else "frozen_embedding_cache"
                ),
                pca_path=str(pca_path) if pca_path is not None else None,
                embedding_root=str(embedding_root) if embedding_root is not None else None,
                embedding_dimension=embedding_dimension,
                embedding_metadata_sha256=embedding_metadata_hash,
                centering="global source mean",
                cross_moment="mean(y * (x - source_mean))",
                gamma_weighting_rule=(
                    "directional discrepancy normalized by sum"
                    if args.gamma_weighting == "directional_discrepancy"
                    else (
                        "outside source min-max interval normalized by mean"
                        if args.gamma_weighting == "outside_source_range"
                        else (
                            "signed target-to-source posterior directional correction: "
                            "(r / n) / (tau_squared + r / n) * "
                            "(q_source_mean - q_target); no normalization"
                            if args.gamma_weighting
                            == "bayesian_directional_shrinkage"
                            else (
                                "signed source-to-posterior directional update: "
                                "tau_squared / (tau_squared + r / n) * "
                                "(q_target - q_source_mean); no normalization"
                                if args.gamma_weighting
                                == "bayesian_source_posterior_update"
                                else (
                                    (
                                        "one-sided source-to-target update: "
                                        "tau_squared / (tau_squared + r / n) * "
                                        "positive_part(q_target - q_source_mean); "
                                        "no normalization"
                                        if args.gamma_weighting
                                        == "bayesian_positive_excess"
                                        else (
                                            "positive part of (q_target - q_source_mean) / "
                                            "sqrt(source_environment_variance + "
                                            "target_observation_variance / target_sample_count); "
                                            "no threshold and no weight normalization"
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                estimator_formula=(
                    (
                        "PSD_projection(target_second_moment + lambda * gamma) "
                        "+ jitter * I"
                        if args.project_psd
                        else "target_second_moment + lambda * gamma + jitter * I"
                    )
                    if args.estimator_base == "target_second_moment"
                    else (
                        (
                            "PSD_projection(pooled source-target centered covariance "
                            "+ lambda * gamma) + jitter * I"
                            if args.project_psd
                            else "pooled source-target centered covariance + lambda * "
                            "gamma + jitter * I"
                        )
                        if args.estimator_base == "pooled_covariance"
                        else (
                            "PSD_projection(source centered covariance + lambda * gamma) "
                            "embedded in source augmented normal equations; no isotropic jitter"
                            if args.project_psd
                            else (
                                "baseline augmented normal equations plus slope-only "
                                "lambda * gamma + source_ridge * I"
                                if args.source_ridge != 0.0
                                else "baseline augmented normal equations plus slope-only "
                                "lambda * gamma; no isotropic jitter"
                            )
                        )
                    )
                ),
                artifacts={
                    "model": "model.npz",
                    "metrics": "metrics.json",
                    "cv_results": "cv_results.json",
                    "penalty_matrices": "penalty_matrices.npz",
                    "target_test_predictions": "target_test_predictions.npz",
                },
            )
            if pca_path is not None:
                link_or_copy(pca_path, run_dir / "pca.npz")
                config["artifacts"]["representation"] = "pca.npz"
            else:
                link_or_copy(embedding_root / "metadata.json", run_dir / "representation_metadata.json")
                config["artifacts"]["representation"] = "representation_metadata.json"
            atomic_write_json(run_dir / "config.json", config)
            print(
                f"Completed task={task}_vs_rest target={args.target} budget={budget}: "
                f"lambda={selected_lambda:g} top_k={selected_top_k} "
                f"MSE={mse:.6f} AUROC={auroc:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
