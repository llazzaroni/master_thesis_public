#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="/cluster/scratch/llazzaroni/CIFAR_data"
DATA_DIR="${DATA_ROOT}/CIFAR-10-C"
SPLIT_FILE="${DATA_ROOT}/cifar10c_disjoint_environment_splits_seed_42.npz"
BASELINE_ROOT="${DATA_ROOT}/disjoint_source_baseline_severity5"
OUTPUT_ROOT="${DATA_ROOT}/disjoint_anti_causal_penalty_severity5"
EMBEDDING_ROOT=""
EMBEDDING_PCA_COMPONENTS=""
PSEUDO_TARGET_COUNT=""
PSEUDO_TARGET_MMD_SAMPLES=""
CV_SELECTION_METRIC="mse"
REQUIRE_PSEUDO_AUROC_IMPROVEMENT=0
PSEUDO_AUROC_IMPROVEMENT_MMD_COUNT=""
SEVERITY=5
N_COMPONENTS=20
TASKS=(); TARGETS=(); BUDGETS=()
ALL_TASKS=(airplane automobile bird cat deer dog frog horse ship truck)
ALL_TARGETS=(brightness contrast defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur)
DEFAULT_BUDGETS=(50 100 200 300 -1)

append_values() { local variable="$1" raw="$2" value; local values=(); IFS=',' read -r -a values <<< "$raw"; for value in "${values[@]}"; do [[ -n "$value" ]] && eval "${variable}+=(\"\$value\")"; done; }
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --task) append_values TASKS "$2"; shift 2 ;;
    --target) append_values TARGETS "$2"; shift 2 ;;
    --budget) append_values BUDGETS "$2"; shift 2 ;;
    --severity) SEVERITY="$2"; shift 2 ;;
    --n-components) N_COMPONENTS="$2"; shift 2 ;;
    --split-file) SPLIT_FILE="$2"; shift 2 ;;
    --embedding-root) EMBEDDING_ROOT="$2"; shift 2 ;;
    --embedding-pca-components) EMBEDDING_PCA_COMPONENTS="$2"; shift 2 ;;
    --pseudo-target-count) PSEUDO_TARGET_COUNT="$2"; shift 2 ;;
    --pseudo-target-mmd-samples) PSEUDO_TARGET_MMD_SAMPLES="$2"; shift 2 ;;
    --cv-selection-metric) CV_SELECTION_METRIC="$2"; shift 2 ;;
    --require-pseudo-auroc-improvement) REQUIRE_PSEUDO_AUROC_IMPROVEMENT=1; shift ;;
    --pseudo-auroc-improvement-mmd-count) PSEUDO_AUROC_IMPROVEMENT_MMD_COUNT="$2"; shift 2 ;;
    --baseline-root) BASELINE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) echo "Usage: run_disjoint_anti_causal_penalty.sh [--data-dir DIR] [--task X] [--target X] [--budget N] [--severity 1..5] [--n-components N] [--embedding-root DIR]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ ${#TASKS[@]} -gt 0 ]] || TASKS=("${ALL_TASKS[@]}")
[[ ${#TARGETS[@]} -gt 0 ]] || TARGETS=("${ALL_TARGETS[@]}")
[[ ${#BUDGETS[@]} -gt 0 ]] || BUDGETS=("${DEFAULT_BUDGETS[@]}")
task_args=(); budget_args=()
for task in "${TASKS[@]}"; do task_args+=(--task "$task"); done
for budget in "${BUDGETS[@]}"; do budget_args+=(--budget "$budget"); done
for target in "${TARGETS[@]}"; do
  command=(python3 "${SCRIPT_DIR}/train.py" --target "$target" "${task_args[@]}" "${budget_args[@]}"
    --data-dir "$DATA_DIR" --split-file "$SPLIT_FILE" --baseline-root "$BASELINE_ROOT" --output-root "$OUTPUT_ROOT" --severity "$SEVERITY" --n-components "$N_COMPONENTS")
  [[ -z "$EMBEDDING_ROOT" ]] || command+=(--embedding-root "$EMBEDDING_ROOT")
  [[ -z "$EMBEDDING_PCA_COMPONENTS" ]] || command+=(--embedding-pca-components "$EMBEDDING_PCA_COMPONENTS")
  [[ -z "$PSEUDO_TARGET_COUNT" ]] || command+=(--pseudo-target-count "$PSEUDO_TARGET_COUNT")
  [[ -z "$PSEUDO_TARGET_MMD_SAMPLES" ]] || command+=(--pseudo-target-mmd-samples "$PSEUDO_TARGET_MMD_SAMPLES")
  command+=(--cv-selection-metric "$CV_SELECTION_METRIC")
  [[ "$REQUIRE_PSEUDO_AUROC_IMPROVEMENT" -eq 0 ]] || command+=(--require-pseudo-auroc-improvement)
  [[ -z "$PSEUDO_AUROC_IMPROVEMENT_MMD_COUNT" ]] || command+=(--pseudo-auroc-improvement-mmd-count "$PSEUDO_AUROC_IMPROVEMENT_MMD_COUNT")
  [[ "$FORCE" -eq 0 ]] || command+=(--force)
  "${command[@]}"
done
