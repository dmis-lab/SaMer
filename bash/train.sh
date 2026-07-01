#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/samer_k64_colpali.yaml}"
RUN_NAME="${RUN_NAME:-samer_k64_colpali}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"
CONFIG_OUT="${CONFIG_OUT:-${OUTPUT_DIR}/config.yaml}"

DATA_ROOT="${DATA_ROOT:-}"
MODEL_NAME="${MODEL_NAME:-vidore/colpali-v1.3-hf}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"
REPORT_TO="${REPORT_TO:-none}"

NUM_REGIONS="${NUM_REGIONS:-64}"
ASSIGNMENT_TEMPERATURE="${ASSIGNMENT_TEMPERATURE:-0.07}"
SPATIAL_WEIGHT="${SPATIAL_WEIGHT:-0.1}"
CLUSTER_ITERS="${CLUSTER_ITERS:-3}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
BF16="${BF16:-true}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ -z "$DATA_ROOT" ]]; then
  echo "[error] DATA_ROOT must point to Flickr30k-Entities." >&2
  echo "Example: DATA_ROOT=/data/flickr30k_entities bash bash/train.sh" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export DATA_ROOT MODEL_NAME LOCAL_FILES_ONLY RUN_NAME OUTPUT_DIR
export NUM_REGIONS ASSIGNMENT_TEMPERATURE SPATIAL_WEIGHT CLUSTER_ITERS
export TRAIN_BATCH_SIZE EVAL_BATCH_SIZE GRAD_ACCUM_STEPS LEARNING_RATE NUM_TRAIN_EPOCHS
export DATALOADER_NUM_WORKERS BF16 REPORT_TO

python - "$CONFIG_TEMPLATE" "$CONFIG_OUT" <<'PY'
import os
import sys
from pathlib import Path

import yaml


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_int_or_none(name: str):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))

cfg["model"]["model_name_or_path"] = os.environ["MODEL_NAME"]
cfg["model"]["local_files_only"] = env_bool("LOCAL_FILES_ONLY", False)
cfg["data"]["root"] = os.environ["DATA_ROOT"]
cfg["data"]["limit_train_images"] = env_int_or_none("LIMIT_TRAIN_IMAGES")
cfg["data"]["limit_eval_images"] = env_int_or_none("LIMIT_EVAL_IMAGES")

cfg["merge"]["num_regions"] = int(os.environ["NUM_REGIONS"])
cfg["merge"]["cluster_iters"] = int(os.environ["CLUSTER_ITERS"])
cfg["merge"]["spatial_weight"] = float(os.environ["SPATIAL_WEIGHT"])
cfg["merge"]["assignment_temperature"] = float(os.environ["ASSIGNMENT_TEMPERATURE"])

cfg["training"]["output_dir"] = os.environ["OUTPUT_DIR"]
cfg["training"]["run_name"] = os.environ["RUN_NAME"]
cfg["training"]["per_device_train_batch_size"] = int(os.environ["TRAIN_BATCH_SIZE"])
cfg["training"]["per_device_eval_batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
cfg["training"]["gradient_accumulation_steps"] = int(os.environ["GRAD_ACCUM_STEPS"])
cfg["training"]["learning_rate"] = float(os.environ["LEARNING_RATE"])
cfg["training"]["num_train_epochs"] = float(os.environ["NUM_TRAIN_EPOCHS"])
cfg["training"]["dataloader_num_workers"] = int(os.environ["DATALOADER_NUM_WORKERS"])
cfg["training"]["bf16"] = env_bool("BF16", True)
cfg["training"]["report_to"] = os.environ["REPORT_TO"]

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

echo "[train] config=${CONFIG_OUT}"
echo "[train] output_dir=${OUTPUT_DIR}"
echo "[train] run_name=${RUN_NAME}"
echo "[train] nproc_per_node=${NPROC_PER_NODE}"

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  torchrun --nproc_per_node "$NPROC_PER_NODE" scripts/train.py --config "$CONFIG_OUT"
else
  python scripts/train.py --config "$CONFIG_OUT"
fi
