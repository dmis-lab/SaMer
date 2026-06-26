#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-}"
MODEL_NAME="${MODEL_NAME:-vidore/colpali-v1.3-hf}"
ADAPTER_PATH="${ADAPTER_PATH:-checkpoints/samer_k64_colpali}"
CACHE_DIR="${CACHE_DIR:-outputs/flickr_samer_k64}"
SPLITS="${SPLITS:-val test}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CACHE_DTYPE="${CACHE_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"

NUM_REGIONS="${NUM_REGIONS:-64}"
ASSIGNMENT_TEMPERATURE="${ASSIGNMENT_TEMPERATURE:-0.07}"
SPATIAL_WEIGHT="${SPATIAL_WEIGHT:-0.1}"
CLUSTER_ITERS="${CLUSTER_ITERS:-3}"

if [[ -z "$DATA_ROOT" ]]; then
  echo "[error] DATA_ROOT must point to Flickr30k-Entities." >&2
  echo "Example: DATA_ROOT=/data/flickr30k_entities bash bash/cache.sh" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"

args=(
  scripts/cache.py
  --data-root "$DATA_ROOT"
  --model-name "$MODEL_NAME"
  --output-dir "$CACHE_DIR"
  --splits
)

read -r -a split_array <<< "$SPLITS"
args+=("${split_array[@]}")

args+=(
  --batch-size "$BATCH_SIZE"
  --cache-dtype "$CACHE_DTYPE"
  --num-regions "$NUM_REGIONS"
  --cluster-iters "$CLUSTER_ITERS"
  --spatial-weight "$SPATIAL_WEIGHT"
  --assignment-temperature "$ASSIGNMENT_TEMPERATURE"
  --device "$DEVICE"
)

if [[ -n "$ADAPTER_PATH" ]]; then
  args+=(--adapter-path "$ADAPTER_PATH")
fi

if [[ "$LOCAL_FILES_ONLY" == "true" || "$LOCAL_FILES_ONLY" == "1" ]]; then
  args+=(--local-files-only)
else
  args+=(--no-local-files-only)
fi

echo "[cache] output_dir=${CACHE_DIR}"
echo "[cache] splits=${SPLITS}"
echo "[cache] adapter_path=${ADAPTER_PATH:-<none>}"

python "${args[@]}"
