#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

CACHE_DIR="${CACHE_DIR:-outputs/flickr_samer_k64}"
SPLIT="${SPLIT:-test}"
OUTPUT_JSON="${OUTPUT_JSON:-${CACHE_DIR}/${SPLIT}_metrics.json}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "${CACHE_DIR}/${SPLIT}.pt" ]]; then
  echo "[error] cache file not found: ${CACHE_DIR}/${SPLIT}.pt" >&2
  echo "Run cache first: DATA_ROOT=/data/flickr30k_entities bash bash/cache.sh" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_JSON")"

echo "[inference] cache_dir=${CACHE_DIR}"
echo "[inference] split=${SPLIT}"
echo "[inference] output_json=${OUTPUT_JSON}"

python scripts/eval.py \
  --cache-dir "$CACHE_DIR" \
  --split "$SPLIT" \
  --batch-size "$BATCH_SIZE" \
  --score-chunk-size "$SCORE_CHUNK_SIZE" \
  --output-json "$OUTPUT_JSON" \
  --device "$DEVICE"
