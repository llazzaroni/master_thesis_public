#!/usr/bin/env python3
"""Extract one frozen DINOv2 ViT-S/14 embedding cache for CIFAR-10-C.

The output preserves the row order of every CIFAR-10-C corruption array, so
the existing ``cifar10c_split_indices.npz`` can index these embeddings without
any conversion.  Each corruption is written independently, which makes an
interrupted extraction safely resumable at corruption granularity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np


CORRUPTIONS = (
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
    "snow", "spatter", "speckle_noise", "zoom_blur",
)
DEFAULT_DATA_ROOT = Path("/cluster/scratch/llazzaroni/CIFAR_data")
MODEL_REPOSITORY = "facebookresearch/dinov2"
MODEL_NAME = "dinov2_vits14"
EMBEDDING_DIMENSION = 384
IMAGE_SIZE = 224
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_ROOT / "CIFAR-10-C",
        help="Extracted CIFAR-10-C directory, or its parent (default: cluster cache).",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=DEFAULT_DATA_ROOT / "dinov2_vits14_embeddings",
        help="Directory for the row-aligned embedding cache.",
    )
    parser.add_argument(
        "--corruption", action="append", choices=CORRUPTIONS,
        help="Extract only this corruption. Repeatable; defaults to all corruptions.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda).")
    parser.add_argument(
        "--model-repo", type=Path,
        help=(
            "Local clone of facebookresearch/dinov2. Required on offline compute nodes "
            "unless the repository is already in the torch.hub cache."
        ),
    )
    parser.add_argument(
        "--weights-path", type=Path,
        help=(
            "Local dinov2_vits14_pretrain.pth checkpoint. Required with --model-repo; "
            "avoids any checkpoint download."
        ),
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable CUDA autocast. Embeddings are always saved as float32.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute completed corruption caches."
    )
    return parser.parse_args()


def resolve_data_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    for candidate in (path, path / "CIFAR-10-C"):
        if (candidate / "labels.npy").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find labels.npy under {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def validate_cached_embedding(path: Path, row_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        return cached.shape == (row_count, EMBEDDING_DIMENSION) and cached.dtype == np.float32
    except (OSError, ValueError):
        return False


def load_model(device: str, model_repo: Path | None, weights_path: Path | None):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if (model_repo is None) != (weights_path is None):
        raise ValueError("--model-repo and --weights-path must be provided together")
    if model_repo is None:
        # This mode downloads the open-source repository and checkpoint on the
        # first invocation, then reuses the torch.hub cache.
        model = torch.hub.load(MODEL_REPOSITORY, MODEL_NAME, trust_repo=True)
    else:
        model_repo = model_repo.expanduser().resolve()
        weights_path = weights_path.expanduser().resolve()
        if not (model_repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"DINOv2 hubconf.py not found under {model_repo}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINOv2 checkpoint not found: {weights_path}")
        model = torch.hub.load(
            str(model_repo), MODEL_NAME, source="local", pretrained=False
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


def extract_corruption(
    *, image_path: Path, output_path: Path, model, device: str, batch_size: int,
    use_amp: bool,
) -> int:
    import torch
    import torch.nn.functional as functional

    images = np.load(image_path, mmap_mode="r", allow_pickle=False)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB images in {image_path}, got {images.shape}")
    row_count = int(images.shape[0])
    temporary_path = output_path.with_name(f".{output_path.stem}.partial.npy")
    if temporary_path.exists():
        temporary_path.unlink()
    embeddings = np.lib.format.open_memmap(
        temporary_path, mode="w+", dtype=np.float32, shape=(row_count, EMBEDDING_DIMENSION)
    )
    mean = torch.tensor(NORMALIZATION_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(NORMALIZATION_STD, device=device).view(1, 3, 1, 1)
    started = time.perf_counter()
    try:
        for start in range(0, row_count, batch_size):
            stop = min(start + batch_size, row_count)
            batch = torch.from_numpy(np.asarray(images[start:stop])).to(device=device, dtype=torch.float32)
            batch = batch.permute(0, 3, 1, 2).div_(255.0)
            batch = functional.interpolate(batch, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False)
            batch = (batch - mean) / std
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp and device.startswith("cuda")
            ):
                output = model(batch)
            if output.shape != (stop - start, EMBEDDING_DIMENSION):
                raise ValueError(
                    f"Unexpected {MODEL_NAME} output shape {tuple(output.shape)}; "
                    f"expected ({stop - start}, {EMBEDDING_DIMENSION})"
                )
            embeddings[start:stop] = output.float().cpu().numpy()
            elapsed = time.perf_counter() - started
            rate = stop / elapsed if elapsed else 0.0
            print(f"{image_path.stem}: {stop}/{row_count} rows ({rate:.1f} images/s)", flush=True)
    except BaseException:
        del embeddings
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    del embeddings
    os.replace(temporary_path, output_path)
    return row_count


def main(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    data_dir = resolve_data_dir(args.data_dir)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    labels_path = data_dir / "labels.npy"
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if labels.ndim != 1:
        raise ValueError(f"Expected one-dimensional labels, got {labels.shape}")
    corruptions = tuple(args.corruption or CORRUPTIONS)
    for corruption in corruptions:
        if not (data_dir / f"{corruption}.npy").is_file():
            raise FileNotFoundError(data_dir / f"{corruption}.npy")

    model = load_model(args.device, args.model_repo, args.weights_path)
    extracted: dict[str, str] = {}
    for corruption in corruptions:
        image_path = data_dir / f"{corruption}.npy"
        output_path = output_root / f"{corruption}.npy"
        if not args.force and validate_cached_embedding(output_path, len(labels)):
            print(f"Skipping valid cache: {output_path}", flush=True)
        else:
            print(f"Extracting {corruption} -> {output_path}", flush=True)
            extract_corruption(
                image_path=image_path, output_path=output_path, model=model,
                device=args.device, batch_size=args.batch_size, use_amp=not args.no_amp,
            )
        extracted[corruption] = sha256(output_path)

    labels_output = output_root / "labels.npy"
    if not labels_output.is_file() or args.force:
        np.save(labels_output, np.asarray(labels))
    write_json(output_root / "metadata.json", {
        "model_repository": MODEL_REPOSITORY,
        "model_name": MODEL_NAME,
        "model_repo": str(args.model_repo.expanduser().resolve()) if args.model_repo else None,
        "weights_path": str(args.weights_path.expanduser().resolve()) if args.weights_path else None,
        "weights_sha256": sha256(args.weights_path.expanduser().resolve()) if args.weights_path else None,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_dtype": "float32",
        "image_size": IMAGE_SIZE,
        "normalization_mean": NORMALIZATION_MEAN,
        "normalization_std": NORMALIZATION_STD,
        "row_order": "identical to each extracted CIFAR-10-C corruption .npy input",
        "labels_sha256": sha256(labels_path),
        "corruption_embedding_sha256": extracted,
    })
    print(f"Wrote DINOv2 cache metadata: {output_root / 'metadata.json'}", flush=True)


if __name__ == "__main__":
    main(parse_args())
