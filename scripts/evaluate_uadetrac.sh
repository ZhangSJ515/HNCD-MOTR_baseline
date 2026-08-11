#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_PATH="${CONFIG_PATH:-./configs/train_uadetrac_hncd.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
EVAL_DIR="${EVAL_DIR:-./outputs/hncd_uadetrac}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the checkpoint filename or path under EVAL_DIR}"
SPLIT="${SPLIT:-test}"
GPU_ID="${GPU_ID:-0}"
EVAL_THREADS="${EVAL_THREADS:-1}"
DET_DB="${DET_DB:-/home/zsj/data/datasets/uadetrac_yolox_det_db.json}"

EVAL_MODEL="${EVAL_MODEL:-${MODEL_PATH}}"
if [[ "${EVAL_MODEL}" == "${EVAL_DIR}/"* ]]; then
  EVAL_MODEL="${EVAL_MODEL#${EVAL_DIR}/}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python main.py \
  --config-path "${CONFIG_PATH}" \
  --mode eval \
  --data-root "${DATA_ROOT}" \
  --available-gpus "${GPU_ID}" \
  --eval-dir "${EVAL_DIR}" \
  --eval-mode specific \
  --eval-model "${EVAL_MODEL}" \
  --eval-data-split "${SPLIT}" \
  --eval-threads "${EVAL_THREADS}" \
  --det-db "${DET_DB}"
