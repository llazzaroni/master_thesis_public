"""Summarize saved pseudo-lasso source supports without fitting models."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from causal_chamber.methods import pseudo_method_info
from causal_chamber.results import load_result_document


def load_results(path: Path, method: str | None = None) -> dict:
    if path.is_dir():
        available = sorted(folder.name for folder in path.iterdir()
                           if folder.is_dir() and pseudo_method_info(folder.name) and (folder/'results.json').is_file())
        selected = method or ("pseudo_lasso" if "pseudo_lasso" in available else
                              "pseudo_lasso_centered" if "pseudo_lasso_centered" in available else
                              available[0] if available else "pseudo_lasso")
        method_file = path / selected / "results.json"
        path = method_file if method_file.is_file() else path / "results.json"
    return load_result_document(path)


def summarize_support(result: dict) -> dict:
    screens = result.get("source_screens", {})
    if not screens:
        raise ValueError("No saved pseudo-lasso source screens found in these results")
    targets = {row["target"] for row in result.get("results", []) if pseudo_method_info(row["method"])}
    missing = targets - screens.keys()
    if missing:
        raise ValueError(f"Missing source screens for: {', '.join(sorted(missing))}")
    supports = {}
    for target, screen in sorted(screens.items()):
        support = screen["invariant_support"]
        if len(support) != len(set(support)):
            raise ValueError(f"Duplicate support indices for {target}")
        names = screen.get("invariant_features")
        if names is None:
            names = [result["features"][index] for index in support]
        if len(names) != len(support):
            raise ValueError(f"Feature names and support size differ for {target}")
        supports[target] = list(names)
    sizes = [len(names) for names in supports.values()]
    nonempty = [size for size in sizes if size]
    return {
        "supports": supports,
        "num_targets": len(sizes),
        "empty_targets": [target for target, names in supports.items() if not names],
        "mean_size": statistics.mean(sizes),
        "median_size": statistics.median(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "std_size": statistics.pstdev(sizes),
        "nonempty_mean_size": statistics.mean(nonempty) if nonempty else None,
        "size_counts": Counter(sizes),
        "feature_counts": Counter(name for names in supports.values() for name in names),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Experiment folder, pseudo_lasso folder, or results.json")
    parser.add_argument("--details", action="store_true", help="List A for every target, including empty sets")
    parser.add_argument("--method",
                        help="Method folder to inspect within an experiment; defaults to original if available")
    args = parser.parse_args()
    try:
        if args.method and not pseudo_method_info(args.method):
            raise ValueError("Method must be a pseudo-lasso variant, including ranked_k variants")
        result = load_results(args.results, args.method)
        summary = summarize_support(result)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.error(str(error))
    total = summary["num_targets"]
    empty = len(summary["empty_targets"])
    if result.get('methods'):
        print(f"Saved methods: {', '.join(result['methods'])}")
    print("One source screen per target environment; shared across target budgets and subset seeds.")
    print(f"Target environments: {total}")
    print(f"Empty A (target-only lasso): {empty}/{total} ({100 * empty / total:.1f}%)")
    print(f"Nonempty A: {total - empty}/{total} ({100 * (total - empty) / total:.1f}%)")
    print(f"|A| across all targets: mean={summary['mean_size']:.2f}, median={summary['median_size']:g}, "
          f"min={summary['min_size']}, max={summary['max_size']}, std={summary['std_size']:.2f}")
    if summary["nonempty_mean_size"] is not None:
        print(f"Mean |A| among nonempty sets: {summary['nonempty_mean_size']:.2f}")
    print("\nSize distribution:")
    for size, count in sorted(summary["size_counts"].items()):
        print(f"  |A|={size}: {count} targets ({100 * count / total:.1f}%)")
    if summary["feature_counts"]:
        print("\nFeature selection frequency:")
        for name, count in sorted(summary["feature_counts"].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {name}: {count}/{total} targets ({100 * count / total:.1f}%)")
    if args.details:
        print("\nSets by target:")
        for target, names in summary["supports"].items():
            print(f"  {target}: |A|={len(names)}, A={{{', '.join(names)}}}")
    elif summary["empty_targets"]:
        print("\nTargets with empty A:")
        for target in summary["empty_targets"]:
            print(f"  {target}")


if __name__ == "__main__":
    main()
