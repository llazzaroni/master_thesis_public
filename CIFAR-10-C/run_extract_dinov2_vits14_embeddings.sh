#!/usr/bin/env bash
# One-time extraction of frozen 384-dimensional DINOv2 ViT-S/14 features.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="/cluster/scratch/llazzaroni/CIFAR_data/CIFAR-10-C"
OUTPUT_ROOT="/cluster/scratch/llazzaroni/CIFAR_data/dinov2_vits14_embeddings"
BATCH_SIZE=512
DEVICE=cuda
MODEL_REPO=""
WEIGHTS_PATH=""
CORRUPTIONS=()
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_extract_dinov2_vits14_embeddings.sh [options]

Options:
  --data-dir DIR       Extracted CIFAR-10-C directory.
  --output-root DIR    Destination embedding cache directory.
  --batch-size N       GPU inference batch size (default: 512).
  --device DEVICE      PyTorch device (default: cuda).
  --model-repo DIR     Local facebookresearch/dinov2 clone for offline inference.
  --weights-path FILE  Local dinov2_vits14_pretrain.pth checkpoint for offline inference.
  --corruption NAME    Extract one corruption; repeatable. Defaults to all.
  --force              Recompute existing corruption cache files.
  --no-amp             Disable CUDA mixed-precision inference.
  -h, --help           Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --model-repo) MODEL_REPO="$2"; shift 2 ;;
    --weights-path) WEIGHTS_PATH="$2"; shift 2 ;;
    --corruption) CORRUPTIONS+=("$2"); shift 2 ;;
    --force|--no-amp) EXTRA_ARGS+=("$1"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command=(python3 "$ROOT_DIR/CIFAR-10-C/extract_dinov2_embeddings.py"
  --data-dir "$DATA_DIR"
  --output-root "$OUTPUT_ROOT"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
)
for corruption in "${CORRUPTIONS[@]}"; do
  command+=(--corruption "$corruption")
done
if [[ -n "$MODEL_REPO" || -n "$WEIGHTS_PATH" ]]; then
  [[ -n "$MODEL_REPO" && -n "$WEIGHTS_PATH" ]] || {
    echo "--model-repo and --weights-path must be supplied together" >&2
    exit 2
  }
  command+=(--model-repo "$MODEL_REPO" --weights-path "$WEIGHTS_PATH")
fi
command+=("${EXTRA_ARGS[@]}")
"${command[@]}"
