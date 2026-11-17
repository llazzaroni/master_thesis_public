#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="/cluster/scratch/llazzaroni/CIFAR_data"
DATA_DIR="${DATA_ROOT}/CIFAR-10-C"
SPLIT_FILE="${DATA_ROOT}/cifar10c_disjoint_environment_splits_seed_42.npz"
OUTPUT_ROOT="${DATA_ROOT}/disjoint_source_baseline"
EMBEDDING_ROOT=""
EMBEDDING_PCA_COMPONENTS=""
SEVERITY=5
N_COMPONENTS=20
TASKS=()
TARGETS=()

ALL_TASKS=(airplane automobile bird cat deer dog frog horse ship truck)
ALL_TARGETS=(brightness contrast defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur)

usage() {
  cat <<'EOF'
Usage: run_disjoint_source_baseline.sh [options]
  --data-dir DIR           Extracted CIFAR-10-C directory.
  --task NAME[,NAME]       CIFAR class; defaults to all.
  --target NAME[,NAME]     Held-out corruption; defaults to all.
  --severity N             One CIFAR-10-C severity level, 1..5 (default: 5).
  --n-components N         Global pixel-PCA dimension (default: 20).
  --split-file FILE        Environment-disjoint split NPZ.
  --embedding-root DIR     Frozen row-aligned embeddings; bypasses PCA.
  --embedding-pca-components N
                         Fit a global PCA on frozen embedding train+validation rows.
  --output-root DIR        Result and global-PCA cache root.
  --force                  Recompute completed tasks.
EOF
}

append_values() {
  local variable="$1" raw="$2" value
  local values=()
  IFS=',' read -r -a values <<< "$raw"
  for value in "${values[@]}"; do [[ -n "$value" ]] && eval "${variable}+=(\"\$value\")"; done
}

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --task) append_values TASKS "$2"; shift 2 ;;
    --target) append_values TARGETS "$2"; shift 2 ;;
    --severity) SEVERITY="$2"; shift 2 ;;
    --n-components) N_COMPONENTS="$2"; shift 2 ;;
    --split-file) SPLIT_FILE="$2"; shift 2 ;;
    --embedding-root) EMBEDDING_ROOT="$2"; shift 2 ;;
    --embedding-pca-components) EMBEDDING_PCA_COMPONENTS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ ${#TASKS[@]} -gt 0 ]] || TASKS=("${ALL_TASKS[@]}")
[[ ${#TARGETS[@]} -gt 0 ]] || TARGETS=("${ALL_TARGETS[@]}")

task_args=()
for task in "${TASKS[@]}"; do task_args+=(--task "$task"); done
for target in "${TARGETS[@]}"; do
  command=(python3 "${SCRIPT_DIR}/train.py" --target "$target" "${task_args[@]}"
    --data-dir "$DATA_DIR" --split-file "$SPLIT_FILE" --output-root "$OUTPUT_ROOT" --severity "$SEVERITY" --n-components "$N_COMPONENTS")
  [[ -z "$EMBEDDING_ROOT" ]] || command+=(--embedding-root "$EMBEDDING_ROOT")
  [[ -z "$EMBEDDING_PCA_COMPONENTS" ]] || command+=(--embedding-pca-components "$EMBEDDING_PCA_COMPONENTS")
  [[ "$FORCE" -eq 0 ]] || command+=(--force)
  "${command[@]}"
done
