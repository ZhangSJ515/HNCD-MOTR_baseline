#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_PATH="${CONFIG_PATH:-./configs/train_uadetrac_hncd.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
OUTPUTS_DIR="${OUTPUTS_DIR:-./outputs/hncd_uadetrac}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-}"
DET_DB="${DET_DB:-/home/zsj/data/datasets/uadetrac_yolox_det_db.json}"

COMMON_ARGS=(
  --config-path "${CONFIG_PATH}"
  --mode train
  --data-root "${DATA_ROOT}"
  --outputs-dir "${OUTPUTS_DIR}"
  --available-gpus "${GPU_IDS}"
  --det-db "${DET_DB}"
)

if [[ -n "${PRETRAINED_MODEL}" ]]; then
  COMMON_ARGS+=(--pretrained-model "${PRETRAINED_MODEL}")
fi

if (( NUM_GPUS > 1 )); then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  python -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    main.py \
    "${COMMON_ARGS[@]}" \
    --use-distributed
else
  CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" \
  python main.py "${COMMON_ARGS[@]}"
fi
