"""Create and verify persistent, per-CSV train/validation/test row assignments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


DEFAULT_DATA_DIR = "data/causal_chamber"


def describe_csv(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        columns = next(reader, None)
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"Missing or duplicate column names in {path}")
        count = 0
        for row in reader:
            if len(row) != len(columns):
                raise ValueError(f"Invalid data row {count} in {path}")
            count += 1
    return {"sha256": digest.hexdigest(), "columns": columns, "num_rows": count}


def discover(data_dir: Path) -> dict[str, Path]:
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    paths = {p.relative_to(data_dir).as_posix(): p for p in sorted(data_dir.rglob("*.csv"))}
    if not paths:
        raise ValueError(f"No CSV files found in {data_dir}")
    return paths


def create_manifest(data_dir: Path, *, seed: int, strategy: str, fractions: list[float]) -> dict:
    if strategy not in {"random", "contiguous"}:
        raise ValueError("Strategy must be random or contiguous")
    if len(fractions) != 3 or any(not 0 < f < 1 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("Provide three positive fractions summing to 1")
    environments = {}
    for name, path in discover(data_dir).items():
        info = describe_csv(path)
        n = info["num_rows"]
        n_train, n_val = int(n * fractions[0]), int(n * fractions[1])
        if min(n_train, n_val, n - n_train - n_val) < 1:
            raise ValueError(f"{name} has too few rows ({n}) for nonempty splits")
        rows = list(range(n))
        if strategy == "random":
            # A per-file seed keeps existing assignments stable if new CSVs are added.
            file_seed = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest(), "big")
            random.Random(file_seed).shuffle(rows)
        info["splits"] = {
            "train": rows[:n_train],
            "val": rows[n_train:n_train + n_val],
            "test": rows[n_train + n_val:],
        }
        environments[name] = info
    return {
        "version": 1,
        "seed": seed,
        "strategy": strategy,
        "fractions": dict(zip(("train", "val", "test"), fractions)),
        "row_index_convention": "Zero-based CSV data rows, excluding the header; preserve saved order.",
        "environments": environments,
    }


def verify_manifest(manifest: dict, data_dir: Path) -> None:
    if manifest.get("version") != 1:
        raise ValueError("Unsupported split manifest version")
    paths = discover(data_dir)
    if set(paths) != set(manifest["environments"]):
        raise ValueError("CSV file list differs from the saved manifest")
    for name, path in paths.items():
        saved = manifest["environments"][name]
        current = describe_csv(path)
        if any(saved[key] != current[key] for key in current):
            raise ValueError(f"Data changed since splits were created: {name}")
        splits = saved["splits"]
        if set(splits) != {"train", "val", "test"} or any(not rows for rows in splits.values()):
            raise ValueError(f"Missing or empty split in {name}")
        rows = [row for values in splits.values() for row in values]
        if any(type(row) is not int for row in rows) or sorted(rows) != list(range(current["num_rows"])):
            raise ValueError(f"Splits overlap or do not partition the rows of {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Check an existing manifest against the CSVs")
    parser.add_argument("--strategy", choices=["random", "contiguous"], help="Required when creating splits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fractions", type=float, nargs=3, default=[0.6, 0.2, 0.2], metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()
    try:
        if args.verify:
            manifest = json.loads(args.manifest.read_text())
            verify_manifest(manifest, args.data_dir)
            print(f"Verified splits for {len(manifest['environments'])} environments.")
        else:
            if args.manifest.exists():
                raise ValueError("Manifest already exists; use --verify to check it. Existing splits are never overwritten.")
            if args.strategy is None:
                raise ValueError("Choose --strategy random or contiguous")
            manifest = create_manifest(args.data_dir, seed=args.seed, strategy=args.strategy, fractions=args.fractions)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            with args.manifest.open("x") as stream:
                json.dump(manifest, stream)
                stream.write("\n")
            print(f"Saved splits for {len(manifest['environments'])} environments to {args.manifest}")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
