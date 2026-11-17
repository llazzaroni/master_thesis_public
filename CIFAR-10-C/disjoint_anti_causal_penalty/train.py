#!/usr/bin/env python3
"""Anti-causal penalty on environment-disjoint, single-severity CIFAR-10-C."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
FIRST_PATH = SCRIPT_DIR.parent / "_shared" / "penalty_helpers.py"
spec = importlib.util.spec_from_file_location("cifar10c_penalty_helpers", FIRST_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {FIRST_PATH}")
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)

CLASS_NAMES, CORRUPTIONS = shared.CLASS_NAMES, shared.CORRUPTIONS
DEFAULT_ROOT = Path("data")
DEFAULT_BUDGETS = (50, 100, 200, 300, -1)
DEFAULT_LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 0.7, 1.0, 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=CORRUPTIONS)
    parser.add_argument("--task", action="append", dest="tasks", choices=CLASS_NAMES)
    parser.add_argument("--budget", action="append", dest="budgets", type=int)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT / "CIFAR-10-C")
    parser.add_argument(
        "--embedding-root", type=Path,
        help="Frozen row-aligned embeddings; bypasses pixel loading and PCA.",
    )
    parser.add_argument(
        "--embedding-pca-components", type=int,
        help="Use this global PCA projection fitted by the embedding source baseline.",
    )
    parser.add_argument(
        "--embedding-pca-path", type=Path,
        help="Explicit global embedding-PCA artifact; otherwise resolve it from --baseline-root.",
    )
    parser.add_argument("--split-file", type=Path, default=DEFAULT_ROOT / "cifar10c_disjoint_environment_splits_seed_42.npz")
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_ROOT / "disjoint_source_baseline_severity5")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT / "disjoint_anti_causal_penalty_severity5")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--pca-path", type=Path)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--pca-iterated-power", type=int, default=3)
    parser.add_argument("--pca-seed", type=int, default=42)
    parser.add_argument("--lambda-grid", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--top-k-grid", nargs="+", type=int, default=[200])
    parser.add_argument(
        "--estimator-base",
        choices=("source_covariance", "pooled_covariance"),
        default="source_covariance",
    )
    parser.add_argument(
        "--gamma-weighting",
        choices=("positive_standardized_excess", "bayesian_positive_excess"),
        default="positive_standardized_excess",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter", type=float, default=1e-8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--cv-selection-metric", choices=("mse", "auroc"), default="mse")
    parser.add_argument(
        "--pseudo-target-count", type=int, default=-1,
        help="Number of MMD-nearest source environments used as pseudo targets; -1 uses all.",
    )
    parser.add_argument("--pseudo-target-mmd-samples", type=int, default=1024)
    parser.add_argument(
        "--require-pseudo-auroc-improvement", action="store_true",
        help="Accept the mean-AUROC winner only if it improves over lambda=0 on MMD-nearest pseudo targets.",
    )
    parser.add_argument(
        "--pseudo-auroc-improvement-mmd-count", type=int, default=2,
        help="Number of MMD-nearest pseudo targets used by --require-pseudo-auroc-improvement.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_metadata(path: Path) -> tuple[int, int, str]:
    with np.load(path, allow_pickle=False) as artifact:
        base_size, severities = int(artifact["base_size"][0]), int(artifact["num_severities"][0])
        environments = tuple(str(value) for value in artifact["environments"].tolist())
    if environments != tuple(CORRUPTIONS):
        raise ValueError("Split-file environments do not match CIFAR-10-C")
    return base_size, severities, shared.sha256(path)


def split_base_indices(path: Path, environment: str, split: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        return np.asarray(artifact[f"{environment}_{split}_base"], dtype=np.int64)


def resolve_pca(args: argparse.Namespace, split_hash: str) -> tuple[Path, np.ndarray, np.ndarray]:
    path = args.pca_path.expanduser().resolve() if args.pca_path else (
        args.baseline_root.expanduser().resolve() / "_global_pca" /
        f"severity_{args.severity}_pca_{args.n_components}_randomized_power_{args.pca_iterated_power}_seed_{args.pca_seed}.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Global PCA artifact not found: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        components = np.asarray(artifact["components"], dtype=np.float32)
        mean = np.asarray(artifact["mean"], dtype=np.float32)
        metadata = json.loads(str(artifact["metadata_json"]))
    expected = {"split_sha256": split_hash, "severity": args.severity,
                "n_components": args.n_components, "pca_iterated_power": args.pca_iterated_power,
                "seed": args.pca_seed, "pca_fit_environments": list(CORRUPTIONS),
                "pca_fit_splits": ["train", "val"]}
    if metadata != expected:
        raise ValueError(f"PCA metadata mismatch at {path}")
    return path, components, mean


def project_severity(data_dir: Path, environment: str, severity: int, base_size: int,
                     components: np.ndarray, mean: np.ndarray, batch_size: int) -> np.ndarray:
    images = np.load(data_dir / f"{environment}.npy", mmap_mode="r", allow_pickle=False)
    start = (severity - 1) * base_size
    projected = np.empty((base_size, components.shape[0]), dtype=np.float32)
    for batch_start in range(0, base_size, batch_size):
        batch_stop = min(batch_start + batch_size, base_size)
        pixels = np.asarray(
            images[start + batch_start:start + batch_stop], dtype=np.float32
        ).reshape(batch_stop - batch_start, -1)
        pixels *= 1.0 / shared.INPUT_SCALE
        projected[batch_start:batch_stop] = (pixels - mean) @ components.T
    return projected


def load_embedding_severity(
    embedding_root: Path, severity: int, base_size: int, embedding_dimension: int,
) -> dict[str, np.ndarray]:
    start = (severity - 1) * base_size
    result: dict[str, np.ndarray] = {}
    for environment in CORRUPTIONS:
        embeddings = np.load(
            embedding_root / f"{environment}.npy", mmap_mode="r", allow_pickle=False
        )
        result[environment] = embeddings[start:start + base_size]
        if result[environment].shape != (base_size, embedding_dimension):
            raise ValueError(f"Invalid severity slice for {environment}")
    return result


def resolve_embedding_pca(
    args: argparse.Namespace, split_hash: str, embedding_metadata_hash: str,
) -> tuple[Path, np.ndarray, np.ndarray]:
    if args.embedding_pca_components is None:
        raise ValueError("--embedding-pca-components is required for embedding PCA")
    path = args.embedding_pca_path.expanduser().resolve() if args.embedding_pca_path else (
        args.baseline_root.expanduser().resolve() / "_global_embedding_pca" /
        f"severity_{args.severity}_pca_{args.embedding_pca_components}_randomized_"
        f"power_{args.pca_iterated_power}_seed_{args.pca_seed}.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Global embedding PCA artifact not found: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        components = np.asarray(artifact["components"], dtype=np.float32)
        mean = np.asarray(artifact["mean"], dtype=np.float32)
        metadata = json.loads(str(artifact["metadata_json"]))
    expected = {
        "split_sha256": split_hash,
        "embedding_metadata_sha256": embedding_metadata_hash,
        "severity": args.severity,
        "n_components": args.embedding_pca_components,
        "pca_iterated_power": args.pca_iterated_power,
        "seed": args.pca_seed,
        "pca_fit_environments": list(CORRUPTIONS),
        "pca_fit_splits": ["train", "val"],
    }
    if metadata != expected:
        raise ValueError(f"Embedding PCA metadata mismatch at {path}")
    return path, components, mean


def project_embedding_severity(
    embeddings: dict[str, np.ndarray], components: np.ndarray, mean: np.ndarray, batch_size: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for environment, values in embeddings.items():
        projected = np.empty((len(values), components.shape[0]), dtype=np.float32)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            projected[start:stop] = (np.asarray(values[start:stop], dtype=np.float32) - mean) @ components.T
        result[environment] = projected
    return result


def rank_disjoint_pseudo_targets_by_mmd(
    *, target: str, budget: int, seed: int, target_indices: np.ndarray,
    source_environments: tuple[str, ...], source_dev_indices: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray], max_samples: int,
) -> list[tuple[str, float]]:
    if max_samples <= 0:
        raise ValueError("--pseudo-target-mmd-samples must be positive")
    # SeedSequence accepts unsigned entropy only; preserve a distinct,
    # deterministic stream for the unlimited-budget sentinel.
    budget_seed = budget if budget >= 0 else np.iinfo(np.uint32).max
    target_rows = shared._mmd_subsample(
        target_indices, max_samples, (seed, CORRUPTIONS.index(target), 0, budget_seed)
    )
    target_values = np.asarray(embeddings[target][target_rows], dtype=np.float64)
    samples: dict[str, np.ndarray] = {}
    reference_parts = [target_values]
    for environment in source_environments:
        rows = shared._mmd_subsample(
            source_dev_indices[environment], max_samples,
            (seed, CORRUPTIONS.index(target), CORRUPTIONS.index(environment), budget_seed),
        )
        samples[environment] = np.asarray(embeddings[environment][rows], dtype=np.float64)
        reference_parts.append(samples[environment])
    reference = np.concatenate(reference_parts, axis=0)
    if len(reference) > 2048:
        rows = shared._mmd_subsample(
            np.arange(len(reference), dtype=np.int64), 2048,
            (seed, CORRUPTIONS.index(target), 1, budget_seed),
        )
        reference = reference[rows]
    bandwidth_squared = shared._median_rbf_bandwidth_squared(reference)
    return sorted(
        ((environment, shared._rbf_mmd_squared(target_values, samples[environment], bandwidth_squared))
         for environment in source_environments),
        key=lambda item: (item[1], item[0]),
    )


def select_and_validate(*, target: str, budget: int, source_envs: tuple[str, ...], dev_stats,
                        embeddings, splits, labels, class_ids, lambdas, top_ks, seed, jitter, metric,
                        estimator_base: str, gamma_weighting: str, pseudo_targets: tuple[str, ...],
                        auroc_guard_pseudo_targets: tuple[str, ...]):
    combinations = [(float(value), int(top_k)) for value in lambdas for top_k in top_ks]
    losses = {combo: np.zeros((len(pseudo_targets), len(class_ids))) for combo in combinations}
    aurocs = {combo: np.zeros((len(pseudo_targets), len(class_ids))) for combo in combinations}
    fold_rows = []
    for fold_index, pseudo in enumerate(pseudo_targets):
        fold_sources = tuple(env for env in source_envs if env != pseudo)
        geometry = shared.build_source_geometry(fold_sources, dev_stats)
        selected = shared.select_budget_indices(splits[pseudo]["train"], budget, seed, target, pseudo)
        target_matrix = shared.target_matrix(embeddings[pseudo], selected, geometry.global_mean)
        centered = np.asarray(embeddings[pseudo][selected], dtype=np.float64) - geometry.global_mean
        gamma_by_k = {top_k: shared.build_gamma(geometry, target_matrix, top_k,
                      weighting=gamma_weighting, target_centered_samples=centered)[0]
                      for top_k in top_ks}
        validation_x = np.asarray(embeddings[pseudo][splits[pseudo]["val"]], dtype=np.float64)
        validation_y = shared.binary_targets(labels[splits[pseudo]["val"]], class_ids)
        candidates = []
        for lambda_value, top_k in combinations:
            beta, intercept, _condition = shared.solve_estimator(
                geometry, target_matrix, gamma_by_k[top_k], lambda_value, jitter,
                estimator_base=estimator_base, target_sample_count=len(selected),
                source_ridge=0.0, compute_condition_number=False,
            )
            scores = validation_x @ beta + intercept
            mse = np.mean(np.square(scores - validation_y), axis=0)
            auc = np.asarray([roc_auc_score(validation_y[:, i] == 1.0, scores[:, i]) for i in range(len(class_ids))])
            losses[(lambda_value, top_k)][fold_index] = mse
            aurocs[(lambda_value, top_k)][fold_index] = auc
            candidates.append({"lambda": lambda_value, "top_k": top_k,
                               "mse_by_task": {CLASS_NAMES[int(cid)]: float(mse[i]) for i, cid in enumerate(class_ids)},
                               "auroc_by_task": {CLASS_NAMES[int(cid)]: float(auc[i]) for i, cid in enumerate(class_ids)}})
        fold_rows.append({
            "fold": fold_index,
            "pseudo_target": pseudo,
            "source_environments": list(fold_sources),
            "unlabeled_train_count": int(len(selected)),
            "validation_count": int(len(splits[pseudo]["val"])),
            "rows": candidates,
        })
    selected_by_class = {}
    averages = []
    for combo in combinations:
        mean_loss, mean_auc = losses[combo].mean(axis=0), aurocs[combo].mean(axis=0)
        averages.append({"lambda": combo[0], "top_k": combo[1],
                         "mean_mse_by_task": {CLASS_NAMES[int(cid)]: float(mean_loss[i]) for i, cid in enumerate(class_ids)},
                         "mean_auroc_by_task": {CLASS_NAMES[int(cid)]: float(mean_auc[i]) for i, cid in enumerate(class_ids)}})
    baseline_candidates = [combo for combo in combinations if combo[0] == 0.0]
    if auroc_guard_pseudo_targets and not baseline_candidates:
        raise ValueError("AUROC-improvement selection requires lambda=0 in --lambda-grid")
    baseline_combo = min(baseline_candidates, key=lambda combo: combo[1]) if baseline_candidates else None
    guard_indices = [pseudo_targets.index(target) for target in auroc_guard_pseudo_targets]
    pass_guard_by_combo: dict[tuple[float, int], np.ndarray] = {}
    if baseline_combo is not None and guard_indices:
        for combo in combinations:
            pass_guard_by_combo[combo] = np.all(
                aurocs[combo][guard_indices] > aurocs[baseline_combo][guard_indices], axis=0
            )
    for index, class_id in enumerate(class_ids):
        if metric == "mse":
            key = lambda combo: (float(losses[combo][:, index].mean()), combo[0], combo[1])
        else:
            key = lambda combo: (-float(aurocs[combo][:, index].mean()), combo[0], combo[1])
        winner = min(combinations, key=key)
        if auroc_guard_pseudo_targets and (
            winner[0] == 0.0 or not bool(pass_guard_by_combo[winner][index])
        ):
            winner = baseline_combo
        selected_by_class[int(class_id)] = winner
    for row, combo in zip(averages, combinations, strict=True):
        if baseline_combo is not None and guard_indices:
            row["passes_mmd_guard_auroc_improvement_by_task"] = {
                CLASS_NAMES[int(class_id)]: bool(pass_guard_by_combo[combo][index])
                for index, class_id in enumerate(class_ids)
            }
    return selected_by_class, {
        "selection_metric": metric,
        "auroc_guard_pseudo_targets": list(auroc_guard_pseudo_targets),
        "pseudo_targets": list(pseudo_targets), "folds": fold_rows, "averages": averages,
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.severity <= 5 or args.jitter < 0 or args.batch_size <= 0:
        raise ValueError("Invalid severity, jitter, or batch size")
    budgets = list(dict.fromkeys(args.budgets or DEFAULT_BUDGETS))
    if any(value == 0 or value < -1 for value in budgets):
        raise ValueError("Budgets must be positive or -1")
    lambdas, top_ks = sorted(set(args.lambda_grid)), sorted(set(args.top_k_grid))
    if any(value < 0 for value in lambdas) or any(value <= 0 for value in top_ks):
        raise ValueError("Invalid lambda or top-k grid")
    if args.pseudo_target_count == 0 or args.pseudo_target_count < -1:
        raise ValueError("--pseudo-target-count must be -1 or positive")
    if args.pseudo_target_mmd_samples <= 0:
        raise ValueError("--pseudo-target-mmd-samples must be positive")
    if args.require_pseudo_auroc_improvement and args.cv_selection_metric != "auroc":
        raise ValueError("--require-pseudo-auroc-improvement requires --cv-selection-metric auroc")
    if args.pseudo_auroc_improvement_mmd_count <= 0:
        raise ValueError("--pseudo-auroc-improvement-mmd-count must be positive")
    data_dir, split_path = shared.resolve_data_dir(args.data_dir), args.split_file.expanduser().resolve()
    base_size, severities, split_hash = split_metadata(split_path)
    if args.severity > severities:
        raise ValueError("Requested severity unavailable in split file")
    labels = np.asarray(np.load(data_dir / "labels.npy", mmap_mode="r", allow_pickle=False)[:base_size], dtype=np.int64)
    embedding_root = None
    embedding_dimension = None
    embedding_metadata_hash = None
    if args.embedding_root is not None:
        embedding_root, embedding_dimension, embedding_metadata_hash = shared.resolve_embedding_cache(
            args.embedding_root.expanduser().resolve(), base_size * severities
        )
        if args.embedding_pca_components is not None:
            if not 0 < args.embedding_pca_components <= embedding_dimension:
                raise ValueError("--embedding-pca-components must be in [1, embedding_dimension]")
            pca_path, components, mean = resolve_embedding_pca(
                args, split_hash, str(embedding_metadata_hash)
            )
        else:
            pca_path = components = mean = None
    else:
        pca_path, components, mean = resolve_pca(args, split_hash)
    tasks = list(dict.fromkeys(args.tasks or CLASS_NAMES))
    output_root = args.output_root.expanduser().resolve()
    source_envs = tuple(env for env in CORRUPTIONS if env != args.target)
    if (
        args.require_pseudo_auroc_improvement
        and args.pseudo_target_count != -1
        and args.pseudo_target_count < args.pseudo_auroc_improvement_mmd_count
    ):
        raise ValueError(
            "The AUROC MMD guard must be included among the validation pseudo targets"
        )
    expected = {"target_environment": args.target, "split_sha256": split_hash,
                "split_schema": "environment_disjoint_v1", "severity": args.severity,
                "lambda_grid": lambdas, "top_k_grid": top_ks,
                "gamma_weighting": args.gamma_weighting, "estimator_base": args.estimator_base,
                "pseudo_target_selection": (
                    "leave_one_source_environment_out_all_sources"
                    if args.pseudo_target_count == -1 else "rbf_mmd_real_target_unlabeled_budget"
                ), "seed": args.seed}
    if args.pseudo_target_count != -1:
        expected.update({
            "pseudo_target_count": args.pseudo_target_count,
            "pseudo_target_ranking": "rbf_mmd_real_target_unlabeled_budget",
            "pseudo_target_mmd_samples": args.pseudo_target_mmd_samples,
        })
    if args.cv_selection_metric != "mse":
        expected["cv_selection_metric"] = args.cv_selection_metric
    if args.require_pseudo_auroc_improvement:
        expected.update({
            "require_pseudo_auroc_improvement": True,
            "pseudo_auroc_improvement_mmd_count": args.pseudo_auroc_improvement_mmd_count,
        })
    if embedding_root is None:
        expected["pca_sha256"] = shared.sha256(pca_path)
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
                "embedding_pca_sha256": shared.sha256(pca_path),
                "pca_iterated_power": args.pca_iterated_power,
                "pca_seed": args.pca_seed,
            })
    run_dirs = {(task, budget): output_root / f"{task}_vs_rest" / args.target / str(budget) for task in tasks for budget in budgets}
    pending = {(task, budget) for task in tasks for budget in budgets if args.force or not shared.is_complete(
        run_dirs[(task, budget)], dict(expected, task=f"{task}_vs_rest", budget=budget))}
    if not pending:
        print(f"All requested runs are complete for target={args.target}")
        return
    splits = {env: {split: split_base_indices(split_path, env, split) for split in ("train", "val", "test")} for env in CORRUPTIONS}
    source_dev_indices = {
        env: np.sort(np.concatenate((splits[env]["train"], splits[env]["val"])))
        for env in source_envs
    }
    if embedding_root is None:
        embeddings = {
            env: project_severity(data_dir, env, args.severity, base_size, components, mean, args.batch_size)
            for env in CORRUPTIONS
        }
    else:
        embeddings = load_embedding_severity(
            embedding_root, args.severity, base_size, int(embedding_dimension)
        )
        if args.embedding_pca_components is not None:
            embeddings = project_embedding_severity(
                embeddings, components, mean, args.batch_size
            )
    dev_stats = {env: shared.compute_environment_stats(embeddings[env], source_dev_indices[env], labels,
                 np.asarray([CLASS_NAMES.index(task) for task in tasks], dtype=np.int64)) for env in source_envs}
    class_ids = np.asarray([CLASS_NAMES.index(task) for task in tasks], dtype=np.int64)
    target_dev = np.sort(np.concatenate((splits[args.target]["train"], splits[args.target]["val"])))
    target_test = splits[args.target]["test"]
    for budget in budgets:
        pending_tasks = [task for task in tasks if (task, budget) in pending]
        if not pending_tasks:
            continue
        real_target = shared.select_budget_indices(target_dev, budget, args.seed, args.target, args.target)
        needs_mmd_ranking = (
            args.pseudo_target_count != -1
            or args.require_pseudo_auroc_improvement
        )
        if needs_mmd_ranking:
            pseudo_target_ranking = rank_disjoint_pseudo_targets_by_mmd(
                target=args.target, budget=budget, seed=args.seed, target_indices=real_target,
                source_environments=source_envs, source_dev_indices=source_dev_indices,
                embeddings=embeddings, max_samples=args.pseudo_target_mmd_samples,
            )
        else:
            pseudo_target_ranking = None
        if args.pseudo_target_count == -1 or args.pseudo_target_count >= len(source_envs):
            pseudo_targets = source_envs
        else:
            pseudo_targets = tuple(
                environment for environment, _distance in pseudo_target_ranking[:args.pseudo_target_count]
            )
            print(
                "Selected MMD-nearest pseudo targets: " + ", ".join(
                    f"{environment} ({distance:.6g})"
                    for environment, distance in pseudo_target_ranking[:args.pseudo_target_count]
                ), flush=True,
            )
        auroc_guard_pseudo_targets = (
            tuple(
                environment for environment, _distance in pseudo_target_ranking[
                    :args.pseudo_auroc_improvement_mmd_count
                ]
            ) if args.require_pseudo_auroc_improvement else ()
        )
        if auroc_guard_pseudo_targets:
            print(
                "AUROC guard pseudo targets: " + ", ".join(auroc_guard_pseudo_targets),
                flush=True,
            )
        selected, cv = select_and_validate(target=args.target, budget=budget, source_envs=source_envs, dev_stats=dev_stats,
            embeddings=embeddings, splits=splits, labels=labels, class_ids=class_ids, lambdas=lambdas, top_ks=top_ks,
            seed=args.seed, jitter=args.jitter, metric=args.cv_selection_metric, estimator_base=args.estimator_base,
            gamma_weighting=args.gamma_weighting, pseudo_targets=pseudo_targets,
            auroc_guard_pseudo_targets=auroc_guard_pseudo_targets)
        if pseudo_target_ranking is not None:
            cv["pseudo_target_mmd_ranking"] = [
                {"environment": environment, "mmd_squared": distance}
                for environment, distance in pseudo_target_ranking
            ]
        geometry = shared.build_source_geometry(source_envs, dev_stats)
        matrix = shared.target_matrix(embeddings[args.target], real_target, geometry.global_mean)
        centered = np.asarray(embeddings[args.target][real_target], dtype=np.float64) - geometry.global_mean
        gamma_cache = {top_k: shared.build_gamma(geometry, matrix, top_k, weighting=args.gamma_weighting, target_centered_samples=centered)
                       for top_k in set(value[1] for value in selected.values())}
        test_x, test_y = np.asarray(embeddings[args.target][target_test], dtype=np.float64), shared.binary_targets(labels[target_test], class_ids)
        for task_index, task in enumerate(tasks):
            if task not in pending_tasks:
                continue
            class_id, (lambda_value, top_k) = int(class_ids[task_index]), selected[int(class_ids[task_index])]
            gamma, gamma_stats = gamma_cache[top_k]
            task_geometry = shared.SourceGeometry(geometry.environment_ids, geometry.global_mean, geometry.environment_matrices,
                geometry.mean_matrix, geometry.variation, geometry.eigenvalues, geometry.eigenvectors,
                geometry.cross_moment[:, [task_index]], geometry.label_mean[[task_index]], geometry.sample_count)
            targets = test_y[:, task_index]
            if args.estimator_base == "source_covariance" and lambda_value == 0.0:
                baseline_dir = args.baseline_root.expanduser().resolve() / f"{task}_vs_rest" / args.target
                with np.load(baseline_dir / "linear_model.npz", allow_pickle=False) as artifact:
                    beta = np.asarray(artifact["coefficients"], dtype=np.float64)
                    intercept = float(np.asarray(artifact["intercept"])[0])
                with np.load(baseline_dir / "target_test_predictions.npz", allow_pickle=False) as artifact:
                    baseline_indices = np.asarray(artifact["indices"], dtype=np.int64)
                    baseline_targets = np.asarray(artifact["targets"], dtype=np.float64)
                    scores = np.asarray(artifact["scores"], dtype=np.float64)
                expected_indices = target_test + (args.severity - 1) * base_size
                if not np.array_equal(baseline_indices, expected_indices) or not np.array_equal(baseline_targets, targets):
                    raise ValueError("Disjoint baseline and penalty test sets differ")
                with (baseline_dir / "metrics.json").open("r", encoding="utf-8") as handle:
                    baseline_metrics = json.load(handle)
                mse, auroc = float(baseline_metrics["target_test_mse"]), float(baseline_metrics["target_test_auroc"])
                condition = baseline_metrics.get("normal_equation_condition_number")
            else:
                beta, intercept_array, condition = shared.solve_estimator(
                    task_geometry, matrix, gamma, lambda_value, args.jitter,
                    estimator_base=args.estimator_base, target_sample_count=len(real_target), source_ridge=0.0,
                )
                beta, intercept = beta[:, 0], float(intercept_array[0])
                scores = test_x @ beta + intercept
                mse, auroc = float(np.mean(np.square(scores - targets))), float(roc_auc_score(targets == 1.0, scores))
            run_dir = run_dirs[(task, budget)]; run_dir.mkdir(parents=True, exist_ok=True)
            if pca_path is not None:
                shared.link_or_copy(pca_path, run_dir / "pca.npz")
            else:
                shared.link_or_copy(embedding_root / "metadata.json", run_dir / "representation_metadata.json")
            task_cv = dict(cv, task=f"{task}_vs_rest", selected_lambda=lambda_value, selected_top_k=top_k)
            shared.atomic_write_json(run_dir / "cv_results.json", task_cv)
            shared.atomic_savez(run_dir / "model.npz", beta=beta, intercept=np.asarray([intercept]), source_mean=geometry.global_mean,
                positive_class_id=np.asarray([class_id]), selected_lambda=np.asarray([lambda_value]), selected_top_k=np.asarray([top_k]))
            shared.atomic_savez(run_dir / "target_test_predictions.npz", indices=target_test + (args.severity - 1) * base_size,
                targets=targets, scores=scores)
            shared.atomic_savez(run_dir / "penalty_matrices.npz", source_mean=geometry.global_mean, source_environment_matrices=geometry.environment_matrices,
                source_mean_matrix=geometry.mean_matrix, variation=geometry.variation, eigenvalues=geometry.eigenvalues,
                eigenvectors=geometry.eigenvectors, target_matrix=matrix,
                estimator_base_matrix=(
                    (float(geometry.sample_count) * geometry.mean_matrix + float(len(real_target)) * matrix)
                    / float(geometry.sample_count + len(real_target))
                    if args.estimator_base == "pooled_covariance" else geometry.mean_matrix
                ),
                gamma=gamma, weights=np.asarray(gamma_stats["weights"]),
                target_unlabeled_indices=real_target + (args.severity - 1) * base_size)
            shared.atomic_write_json(run_dir / "metrics.json", {"task": f"{task}_vs_rest", "positive_class_id": class_id,
                "target_environment": args.target, "budget": budget, "severity": args.severity,
                "effective_target_unlabeled_count": int(len(real_target)), "source_train_val_count": geometry.sample_count,
                "target_test_count": int(len(target_test)), "target_test_positive_count": int(np.count_nonzero(targets == 1.0)),
                "target_test_mse": mse, "target_test_auroc": auroc, "selected_lambda": lambda_value,
                "selected_top_k": top_k, "system_condition_number": condition, "pseudo_targets": list(pseudo_targets),
                "gamma_weighting": args.gamma_weighting, "estimator_base": args.estimator_base})
            shared.atomic_write_json(run_dir / "config.json", dict(expected, status="complete", task=f"{task}_vs_rest", budget=budget,
                positive_class_id=class_id, pca_path=str(pca_path) if pca_path is not None else None,
                source_environments=list(source_envs), pseudo_targets=list(pseudo_targets),
                source_splits=["train", "val"], pseudo_target_unlabeled_split="train", pseudo_target_evaluation_split="val",
                real_target_unlabeled_splits=["train", "val"], real_target_evaluation_split="test",
                artifacts={"representation": "pca.npz" if pca_path is not None else "representation_metadata.json", "model": "model.npz", "metrics": "metrics.json", "cv_results": "cv_results.json",
                           "penalty_matrices": "penalty_matrices.npz", "target_test_predictions": "target_test_predictions.npz"}))
            print(f"Completed task={task}_vs_rest target={args.target} budget={budget}: lambda={lambda_value:g} top_k={top_k} AUROC={auroc:.6f}", flush=True)


if __name__ == "__main__":
    main()
