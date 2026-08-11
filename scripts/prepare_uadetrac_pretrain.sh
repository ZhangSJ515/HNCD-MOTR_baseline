#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT to the official DAB-Deformable-DETR COCO checkpoint}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-./pretrains/dab_deformable_detr_coco_uadetrac.pth}"

python tools/convert_uadetrac_dab_pretrain.py \
  --source "${SOURCE_CHECKPOINT}" \
  --output "${OUTPUT_CHECKPOINT}"
