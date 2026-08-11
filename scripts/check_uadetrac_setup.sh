#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
SPLITS="${SPLITS:-train test}"
GT_FORMAT="${GT_FORMAT:-auto}"
MIN_VISIBILITY="${MIN_VISIBILITY:-0.0}"
FILTER_GT_BY_MARK="${FILTER_GT_BY_MARK:-1}"
DET_DB="${DET_DB:-/home/zsj/data/datasets/uadetrac_yolox_det_db.json}"
REQUIRE_DET_DB="${REQUIRE_DET_DB:-1}"

read -r -a SPLIT_ARRAY <<< "${SPLITS}"
ARGS=(
  --data-root "${DATA_ROOT}"
  --splits "${SPLIT_ARRAY[@]}"
  --gt-format "${GT_FORMAT}"
  --min-visibility "${MIN_VISIBILITY}"
)

if [[ "${FILTER_GT_BY_MARK}" == "1" ]]; then
  ARGS+=(--filter-gt-by-mark)
fi
if [[ -n "${DET_DB}" ]]; then
  ARGS+=(--det-db "${DET_DB}")
fi
if [[ "${REQUIRE_DET_DB}" == "1" ]]; then
  ARGS+=(--require-det-db)
fi

python tools/check_uadetrac_setup.py "${ARGS[@]}"
