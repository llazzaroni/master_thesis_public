#!/usr/bin/env python3
"""Create class-stratified, environment-disjoint CIFAR-10-C split indices.

Every one of the 10,000 underlying CIFAR-10 image IDs is assigned to exactly
one corruption environment. Within that environment, its base IDs are split
into train, validation, and test sets. Each base-ID split is then expanded over
all five CIFAR-10-C severity blocks. Thus images are shared across severity
levels within one environment, but never across corruption environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


CORRUPTIONS = (
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
    "snow", "spatter", "speckle_noise", "zoom_blur",
)
SPLIT_NAMES = ("train", "val", "test")
SPLIT_PROPORTIONS = (0.70, 0.15, 0.15)
DEFAULT_DATA_ROOT = Path("/cluster/scratch/llazzaroni/CIFAR_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_ROOT / "CIFAR-10-C",
        help="Extracted CIFAR-10-C directory, or its parent.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_DATA_ROOT / "cifar10c_disjoint_environment_splits_seed_42.npz",
        help="Output NPZ file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def resolve_data_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    for candidate in (path, path / "CIFAR-10-C"):
        if (candidate / "labels.npy").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find extracted labels.npy under {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def split_counts(count: int) -> tuple[int, int, int]:
    exact = np.asarray(SPLIT_PROPORTIONS, dtype=np.float64) * count
    allocated = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - allocated))[:count - int(allocated.sum())]:
        allocated[index] += 1
    return tuple(int(value) for value in allocated)


def expand_severities(base_indices: np.ndarray, base_size: int, num_severities: int) -> np.ndarray:
    return np.concatenate([
        base_indices + severity * base_size for severity in range(num_severities)
    ]).astype(np.int64, copy=False)


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def main(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output}; pass --force to replace it")
    data_dir = resolve_data_dir(args.data_dir)
    labels = np.asarray(np.load(data_dir / "labels.npy", allow_pickle=False), dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0 or len(labels) % 5:
        raise ValueError(f"Expected labels repeated over five severity blocks, got {labels.shape}")
    base_size = len(labels) // 5
    base_labels = labels[:base_size]
    if any(not np.array_equal(labels[severity * base_size:(severity + 1) * base_size], base_labels) for severity in range(5)):
        raise ValueError("CIFAR-10-C label blocks do not match across severity levels")

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "seed": np.asarray([args.seed], dtype=np.int64),
        "base_size": np.asarray([base_size], dtype=np.int64),
        "num_severities": np.asarray([5], dtype=np.int64),
        "split_percentages": np.asarray([70, 15, 15], dtype=np.int64),
        "environments": np.asarray(CORRUPTIONS),
    }
    assigned_base_ids: list[np.ndarray] = []
    environment_split_base: dict[str, dict[str, list[np.ndarray]]] = {
        environment: {split: [] for split in SPLIT_NAMES} for environment in CORRUPTIONS
    }

    # Partition each class independently so every corruption receives nearly
    # identical class frequencies while base image IDs remain disjoint globally.
    for class_id in sorted(np.unique(base_labels)):
        class_ids = np.flatnonzero(base_labels == class_id)
        partition_rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(class_id), 0]))
        shuffled = partition_rng.permutation(class_ids)
        for environment_index, base_ids in enumerate(np.array_split(shuffled, len(CORRUPTIONS))):
            environment = CORRUPTIONS[environment_index]
            assigned_base_ids.append(base_ids)
            split_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, int(class_id), environment_index + 1])
            )
            shuffled_environment_ids = split_rng.permutation(base_ids)
            train_count, val_count, _test_count = split_counts(len(base_ids))
            environment_split_base[environment]["train"].append(shuffled_environment_ids[:train_count])
            environment_split_base[environment]["val"].append(
                shuffled_environment_ids[train_count:train_count + val_count]
            )
            environment_split_base[environment]["test"].append(
                shuffled_environment_ids[train_count + val_count:]
            )

    assigned = np.concatenate(assigned_base_ids)
    if len(assigned) != base_size or len(np.unique(assigned)) != base_size:
        raise RuntimeError("Base image IDs were not partitioned exactly once across environments")

    report: dict[str, object] = {
        "schema": "cifar10c_environment_disjoint_v1",
        "seed": args.seed,
        "labels_sha256": sha256(data_dir / "labels.npy"),
        "base_images": base_size,
        "severity_levels": 5,
        "environments": {},
    }
    for environment in CORRUPTIONS:
        report["environments"][environment] = {}
        assigned_for_environment: list[np.ndarray] = []
        for split in SPLIT_NAMES:
            base_indices = np.sort(np.concatenate(environment_split_base[environment][split])).astype(np.int64)
            expanded = expand_severities(base_indices, base_size, 5)
            arrays[f"{environment}_{split}_base"] = base_indices
            arrays[f"{environment}_{split}"] = expanded
            assigned_for_environment.append(base_indices)
            report["environments"][environment][split] = {
                "base_images": int(len(base_indices)), "rows_over_all_severities": int(len(expanded)),
                "class_counts": {str(class_id): int(np.sum(base_labels[base_indices] == class_id)) for class_id in sorted(np.unique(base_labels))},
            }
        environment_union = np.concatenate(assigned_for_environment)
        if len(environment_union) != len(np.unique(environment_union)):
            raise RuntimeError(f"Within-environment split overlap for {environment}")

    atomic_savez(output, arrays)
    atomic_write_json(output.with_suffix(".json"), report)
    print(f"Wrote environment-disjoint split file: {output}")
    print(f"Wrote split report: {output.with_suffix('.json')}")
    for environment in CORRUPTIONS:
        counts = report["environments"][environment]
        print(
            f"{environment}: base train/val/test="
            f"{counts['train']['base_images']}/{counts['val']['base_images']}/{counts['test']['base_images']}",
            flush=True,
        )


if __name__ == "__main__":
    main(parse_args())
