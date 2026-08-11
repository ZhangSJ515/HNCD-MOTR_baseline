#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
DETECTIONS_ROOT="${DETECTIONS_ROOT:?Set DETECTIONS_ROOT to per-sequence detector txt outputs}"
OUTPUT="${OUTPUT:-/home/zsj/data/datasets/uadetrac_yolox_det_db.json}"
SPLITS="${SPLITS:-train test}"
MIN_SCORE="${MIN_SCORE:-0.0}"
STRICT="${STRICT:-1}"

ARGS=(
  --data-root "${DATA_ROOT}"
  --detections-root "${DETECTIONS_ROOT}"
  --output "${OUTPUT}"
  --min-score "${MIN_SCORE}"
  --splits
)
read -r -a SPLIT_ARRAY <<< "${SPLITS}"
ARGS+=("${SPLIT_ARRAY[@]}")

if [[ "${STRICT}" == "1" ]]; then
  ARGS+=(--strict)
fi

python tools/build_uadetrac_det_db.py "${ARGS[@]}"
