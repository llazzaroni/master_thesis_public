#!/usr/bin/env python3
"""Fit severity-restricted source OLS baselines on disjoint CIFAR-10-C splits.

PCA is fitted once per severity on the train+validation images from all
corruptions. Frozen row-aligned embeddings can bypass PCA. For each held-out
target corruption, the supervised source head uses only the other corruptions'
train+validation labels; target test images are never used for the linear head.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_PATH = SCRIPT_DIR.parent / "_shared" / "baseline_helpers.py"
spec = importlib.util.spec_from_file_location("cifar10c_baseline_helpers", SHARED_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load shared baseline helpers from {SHARED_PATH}")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

CLASS_NAMES = shared.CLASS_NAMES
CORRUPTIONS = shared.CORRUPTIONS
DEFAULT_DATA_ROOT = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=CORRUPTIONS)
    parser.add_argument("--task", action="append", dest="tasks", choices=CLASS_NAMES)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_ROOT / "CIFAR-10-C")
    parser.add_argument(
        "--split-file", type=Path,
        default=DEFAULT_DATA_ROOT / "cifar10c_disjoint_environment_splits_seed_42.npz",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT / "disjoint_source_baseline")
    parser.add_argument("--severity", type=int, default=5, help="CIFAR-10-C severity block, from 1 to 5.")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument(
        "--embedding-root", type=Path,
        help="Frozen row-aligned embedding cache; bypasses pixel loading and PCA.",
    )
    parser.add_argument(
        "--embedding-pca-components", type=int,
        help="Fit this many global PCA components on frozen embedding train+validation rows.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pca-iterated-power", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refit-pca", action="store_true")
    return parser.parse_args()


def load_disjoint_split_metadata(path: Path) -> tuple[int, int, str]:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as artifact:
        required = {"schema_version", "base_size", "num_severities", "environments"}
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"Disjoint split file is missing {sorted(missing)}")
        base_size = int(artifact["base_size"][0])
        num_severities = int(artifact["num_severities"][0])
        environments = tuple(str(value) for value in artifact["environments"].tolist())
    if environments != tuple(CORRUPTIONS):
        raise ValueError("Disjoint split environments do not match CIFAR-10-C corruptions")
    return base_size, num_severities, shared.sha256(path)


def environment_indices(path: Path, environment: str, split: str, severity: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        key = f"{environment}_{split}_base"
        if key not in artifact.files:
            raise ValueError(f"Disjoint split file is missing {key}")
        base_size = int(artifact["base_size"][0])
        base_indices = np.asarray(artifact[key], dtype=np.int64)
    return base_indices + (severity - 1) * base_size


def load_pixel_rows(data_dir: Path, environment: str, indices: np.ndarray) -> np.ndarray:
    images = np.load(data_dir / f"{environment}.npy", mmap_mode="r", allow_pickle=False)
    rows = np.asarray(images[indices], dtype=np.float32).reshape(len(indices), -1)
    rows *= 1.0 / shared.INPUT_SCALE
    return rows


def load_disjoint_source_embeddings(
    embedding_root: Path,
    source_indices: dict[str, np.ndarray],
    embedding_dimension: int,
) -> np.ndarray:
    total_rows = sum(len(indices) for indices in source_indices.values())
    result = np.empty((total_rows, embedding_dimension), dtype=np.float32)
    start = 0
    for environment in CORRUPTIONS:
        if environment not in source_indices:
            continue
        indices = source_indices[environment]
        stop = start + len(indices)
        embeddings = np.load(embedding_root / f"{environment}.npy", mmap_mode="r", allow_pickle=False)
        result[start:stop] = embeddings[indices]
        start = stop
    return result


def global_embedding_pca_cache_path(output_root: Path, args: argparse.Namespace) -> Path:
    return output_root / "_global_embedding_pca" / (
        f"severity_{args.severity}_pca_{args.embedding_pca_components}_randomized_"
        f"power_{args.pca_iterated_power}_seed_{args.seed}.npz"
    )


def load_or_fit_global_embedding_pca(
    *, args: argparse.Namespace, embedding_root: Path, split_file: Path,
    output_root: Path, split_hash: str, embedding_metadata_hash: str,
) -> tuple[Path, np.ndarray, np.ndarray]:
    cache_path = global_embedding_pca_cache_path(output_root, args)
    expected = {
        "split_sha256": split_hash,
        "embedding_metadata_sha256": embedding_metadata_hash,
        "severity": args.severity,
        "n_components": args.embedding_pca_components,
        "pca_iterated_power": args.pca_iterated_power,
        "seed": args.seed,
        "pca_fit_environments": list(CORRUPTIONS),
        "pca_fit_splits": ["train", "val"],
    }
    if cache_path.is_file() and not args.refit_pca:
        with np.load(cache_path, allow_pickle=False) as artifact:
            components = np.asarray(artifact["components"], dtype=np.float32)
            mean = np.asarray(artifact["mean"], dtype=np.float32)
            metadata = json.loads(str(artifact["metadata_json"]))
        if metadata != expected:
            raise ValueError(f"Embedding PCA metadata mismatch at {cache_path}; use --refit-pca")
        return cache_path, components, mean

    matrices = []
    for environment in CORRUPTIONS:
        indices = np.sort(np.concatenate((
            environment_indices(split_file, environment, "train", args.severity),
            environment_indices(split_file, environment, "val", args.severity),
        )))
        embeddings = np.load(embedding_root / f"{environment}.npy", mmap_mode="r", allow_pickle=False)
        matrices.append(np.asarray(embeddings[indices], dtype=np.float32))
    all_dev_embeddings = np.concatenate(matrices, axis=0)
    print(
        f"Fitting global {args.embedding_pca_components}-component embedding PCA on "
        f"{len(all_dev_embeddings)} train+validation rows at severity={args.severity}", flush=True,
    )
    fitted_pca, _unused = shared.fit_pca(
        all_dev_embeddings, args.embedding_pca_components, args.pca_iterated_power, args.seed
    )
    shared.atomic_savez(
        cache_path,
        components=np.asarray(fitted_pca.components_, dtype=np.float32),
        mean=np.asarray(fitted_pca.mean_, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(expected, sort_keys=True)),
    )
    return cache_path, np.asarray(fitted_pca.components_, dtype=np.float32), np.asarray(fitted_pca.mean_, dtype=np.float32)


def evaluate_projected_embeddings(
    embedding_path: Path, test_indices: np.ndarray, labels: np.ndarray,
    class_ids: np.ndarray, weights: np.ndarray, components: np.ndarray, mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    projected = (np.asarray(embeddings[test_indices], dtype=np.float32) - mean) @ components.T
    scores = projected @ weights[:-1] + weights[-1]
    targets = shared.binary_targets(labels[test_indices], class_ids)
    mse = np.mean(np.square(scores - targets), axis=0)
    auroc = np.asarray([
        roc_auc_score(targets[:, index] == 1.0, scores[:, index])
        for index in range(len(class_ids))
    ], dtype=np.float64)
    return scores, targets, mse, auroc


def global_pca_cache_path(output_root: Path, args: argparse.Namespace) -> Path:
    return output_root / "_global_pca" / (
        f"severity_{args.severity}_pca_{args.n_components}_randomized_"
        f"power_{args.pca_iterated_power}_seed_{args.seed}.npz"
    )


def load_or_fit_global_pca(
    *, args: argparse.Namespace, data_dir: Path, split_file: Path,
    output_root: Path, split_hash: str,
) -> tuple[Path, np.ndarray, np.ndarray]:
    cache_path = global_pca_cache_path(output_root, args)
    expected = {
        "split_sha256": split_hash,
        "severity": args.severity,
        "n_components": args.n_components,
        "pca_iterated_power": args.pca_iterated_power,
        "seed": args.seed,
        "pca_fit_environments": list(CORRUPTIONS),
        "pca_fit_splits": ["train", "val"],
    }
    if cache_path.is_file() and not args.refit_pca:
        with np.load(cache_path, allow_pickle=False) as artifact:
            components = np.asarray(artifact["components"], dtype=np.float32)
            mean = np.asarray(artifact["mean"], dtype=np.float32)
            metadata = json.loads(str(artifact["metadata_json"]))
        if metadata != expected:
            raise ValueError(f"Global PCA metadata mismatch at {cache_path}; use --refit-pca")
        return cache_path, components, mean

    matrices = []
    for environment in CORRUPTIONS:
        indices = np.sort(np.concatenate((
            environment_indices(split_file, environment, "train", args.severity),
            environment_indices(split_file, environment, "val", args.severity),
        )))
        matrices.append(load_pixel_rows(data_dir, environment, indices))
    all_dev_images = np.concatenate(matrices, axis=0)
    print(
        f"Fitting global PCA on {len(all_dev_images)} train+validation images "
        f"from all {len(CORRUPTIONS)} corruptions at severity={args.severity}", flush=True,
    )
    fitted_pca, _unused_embeddings = shared.fit_pca(
        all_dev_images, args.n_components, args.pca_iterated_power, args.seed
    )
    shared.atomic_savez(
        cache_path,
        components=np.asarray(fitted_pca.components_, dtype=np.float32),
        mean=np.asarray(fitted_pca.mean_, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(expected, sort_keys=True)),
    )
    return cache_path, np.asarray(fitted_pca.components_, dtype=np.float32), np.asarray(fitted_pca.mean_, dtype=np.float32)


def is_complete(run_dir: Path, expected: dict[str, object]) -> bool:
    required = (
        run_dir / "config.json", run_dir / "metrics.json", run_dir / "linear_model.npz",
        run_dir / "target_test_predictions.npz",
    )
    representation = (
        run_dir / "representation_metadata.json"
        if expected.get("representation_mode") == "frozen_embedding_cache"
        else run_dir / "pca.npz"
    )
    if not all(path.is_file() for path in (*required, representation)):
        return False
    try:
        with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return config.get("status") == "complete" and all(config.get(key) == value for key, value in expected.items())
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    if not 1 <= args.severity <= 5:
        raise ValueError("--severity must be in [1, 5]")
    if args.n_components <= 0 or args.batch_size <= 0:
        raise ValueError("--n-components and --batch-size must be positive")
    data_dir = shared.resolve_data_dir(args.data_dir)
    split_file = args.split_file.expanduser().resolve()
    base_size, num_severities, split_hash = load_disjoint_split_metadata(split_file)
    if args.severity > num_severities:
        raise ValueError(f"Requested severity {args.severity}, but split file has {num_severities}")
    labels = np.load(data_dir / "labels.npy", mmap_mode="r", allow_pickle=False)
    if len(labels) != base_size * num_severities:
        raise ValueError("Split-file base size does not match labels.npy")

    embedding_root = None
    embedding_dimension = None
    embedding_metadata_hash = None
    if args.embedding_root is not None:
        embedding_root, embedding_dimension, embedding_metadata_hash = shared.resolve_embedding_cache(
            args.embedding_root.expanduser().resolve(), len(labels)
        )
        if args.embedding_pca_components is not None:
            if not 0 < args.embedding_pca_components <= embedding_dimension:
                raise ValueError("--embedding-pca-components must be in [1, embedding_dimension]")

    tasks = list(dict.fromkeys(args.tasks or CLASS_NAMES))
    output_root = args.output_root.expanduser().resolve()
    expected = {
        "target_environment": args.target,
        "split_file": str(split_file),
        "split_sha256": split_hash,
        "split_schema": "environment_disjoint_v1",
        "severity": args.severity,
        "seed": args.seed,
    }
    if embedding_root is None:
        expected.update({
            "n_components": args.n_components,
            "pca_method": "sklearn.decomposition.PCA(randomized)",
            "pca_iterated_power": args.pca_iterated_power,
        })
    else:
        expected.update({
            "representation_mode": (
                "frozen_embedding_pca" if args.embedding_pca_components is not None
                else "frozen_embedding_cache"
            ),
            "embedding_root": str(embedding_root),
            "embedding_dimension": embedding_dimension,
            "embedding_metadata_sha256": embedding_metadata_hash,
        })
        if args.embedding_pca_components is not None:
            expected.update({
                "embedding_pca_components": args.embedding_pca_components,
                "pca_iterated_power": args.pca_iterated_power,
            })
    run_dirs = {task: output_root / f"{task}_vs_rest" / args.target for task in tasks}
    pending = [task for task in tasks if args.force or not is_complete(run_dirs[task], expected)]
    if not pending:
        print(f"All requested tasks are already complete for target={args.target}")
        return

    source_environments = tuple(environment for environment in CORRUPTIONS if environment != args.target)
    source_indices = {
        environment: np.sort(np.concatenate((
            environment_indices(split_file, environment, "train", args.severity),
            environment_indices(split_file, environment, "val", args.severity),
        )))
        for environment in source_environments
    }
    if embedding_root is None:
        pca_path, components, mean = load_or_fit_global_pca(
            args=args, data_dir=data_dir, split_file=split_file, output_root=output_root, split_hash=split_hash,
        )
        source_pixels = np.concatenate([
            load_pixel_rows(data_dir, environment, source_indices[environment])
            for environment in source_environments
        ], axis=0)
        source_embeddings = shared.transform_source_matrix(source_pixels, components, mean, args.batch_size)
    else:
        source_embeddings = load_disjoint_source_embeddings(
            embedding_root, source_indices, int(embedding_dimension)
        )
        if args.embedding_pca_components is None:
            pca_path = components = mean = None
        else:
            pca_path, components, mean = load_or_fit_global_embedding_pca(
                args=args, embedding_root=embedding_root, split_file=split_file,
                output_root=output_root, split_hash=split_hash,
                embedding_metadata_hash=str(embedding_metadata_hash),
            )
            source_embeddings = shared.transform_source_matrix(
                source_embeddings, components, mean, args.batch_size
            )
    source_labels = np.concatenate([
        np.asarray(labels[source_indices[environment]], dtype=np.int64)
        for environment in source_environments
    ])
    class_ids = np.asarray([CLASS_NAMES.index(task) for task in pending], dtype=np.int64)
    weights, source_mse, source_rows, rank, condition_number = shared.fit_linear_models(
        source_embeddings, source_labels, class_ids, args.batch_size
    )
    target_test_indices = environment_indices(split_file, args.target, "test", args.severity)
    if embedding_root is None:
        scores, targets, test_mse, test_auroc = shared.evaluate_target(
            data_dir / f"{args.target}.npy", target_test_indices, labels, class_ids,
            weights, components, mean, args.batch_size,
        )
    else:
        if args.embedding_pca_components is None:
            scores, targets, test_mse, test_auroc = shared.evaluate_target_embeddings(
                embedding_root / f"{args.target}.npy", target_test_indices, labels,
                class_ids, weights, args.batch_size,
            )
        else:
            scores, targets, test_mse, test_auroc = evaluate_projected_embeddings(
                embedding_root / f"{args.target}.npy", target_test_indices, labels,
                class_ids, weights, components, mean,
            )

    for task_index, (task, class_id) in enumerate(zip(pending, class_ids, strict=True)):
        run_dir = run_dirs[task]
        run_dir.mkdir(parents=True, exist_ok=True)
        if pca_path is not None:
            shared.link_or_copy(pca_path, run_dir / "pca.npz")
        else:
            shared.link_or_copy(embedding_root / "metadata.json", run_dir / "representation_metadata.json")
        shared.atomic_savez(
            run_dir / "linear_model.npz",
            coefficients=weights[:-1, task_index], intercept=np.asarray([weights[-1, task_index]]),
            positive_class_id=np.asarray([class_id], dtype=np.int64), positive_class_name=np.asarray(task),
        )
        shared.atomic_savez(
            run_dir / "target_test_predictions.npz", indices=target_test_indices,
            targets=targets[:, task_index], scores=scores[:, task_index],
        )
        shared.atomic_write_json(run_dir / "metrics.json", {
            "task": f"{task}_vs_rest", "positive_class_id": int(class_id),
            "target_environment": args.target, "severity": args.severity,
            "source_train_val_count": source_rows, "target_test_count": int(len(target_test_indices)),
            "target_test_positive_count": int(np.count_nonzero(targets[:, task_index] == 1.0)),
            "source_train_val_mse": float(source_mse[task_index]),
            "target_test_mse": float(test_mse[task_index]), "target_test_auroc": float(test_auroc[task_index]),
            "normal_equation_rank": rank, "normal_equation_condition_number": condition_number,
        })
        shared.atomic_write_json(run_dir / "config.json", dict(
            expected, status="complete", task=f"{task}_vs_rest", positive_class_id=int(class_id),
            data_dir=str(data_dir), source_environments=list(source_environments),
            source_splits=["train", "val"], evaluation_split="test",
            pca_fit_environments=list(CORRUPTIONS) if pca_path is not None else None,
            pca_fit_splits=["train", "val"] if pca_path is not None else None,
            pca_path=str(pca_path) if pca_path is not None else None,
            regression="unregularized ordinary least squares with intercept",
            artifacts={"representation": "pca.npz" if pca_path is not None else "representation_metadata.json", "linear_model": "linear_model.npz",
                       "target_test_predictions": "target_test_predictions.npz", "metrics": "metrics.json"},
        ))
        print(
            f"Completed task={task}_vs_rest target={args.target} severity={args.severity}: "
            f"MSE={test_mse[task_index]:.6f} AUROC={test_auroc[task_index]:.6f}", flush=True,
        )


if __name__ == "__main__":
    main()
