#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_PATH="${CONFIG_PATH:-./configs/train_uadetrac_hncd.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
SUBMIT_DIR="${SUBMIT_DIR:-./outputs/hncd_uadetrac}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a checkpoint filename or path under SUBMIT_DIR}"
SPLIT="${SPLIT:-test}"
GPU_ID="${GPU_ID:-0}"
DET_DB="${DET_DB:-/home/zsj/data/datasets/uadetrac_yolox_det_db.json}"

# main.py expects SUBMIT_MODEL relative to SUBMIT_DIR. An absolute MODEL_PATH is
# accepted by copying only when it already resides inside SUBMIT_DIR; otherwise
# pass SUBMIT_MODEL explicitly.
SUBMIT_MODEL="${SUBMIT_MODEL:-${MODEL_PATH}}"
if [[ "${SUBMIT_MODEL}" == "${SUBMIT_DIR}/"* ]]; then
  SUBMIT_MODEL="${SUBMIT_MODEL#${SUBMIT_DIR}/}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python main.py \
  --config-path "${CONFIG_PATH}" \
  --mode submit \
  --data-root "${DATA_ROOT}" \
  --available-gpus "${GPU_ID}" \
  --submit-dir "${SUBMIT_DIR}" \
  --submit-model "${SUBMIT_MODEL}" \
  --submit-data-split "${SPLIT}" \
  --det-db "${DET_DB}"
