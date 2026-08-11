#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

IMAGES_ROOT="${IMAGES_ROOT:?Set IMAGES_ROOT to official UA-DETRAC images}"
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:?Set ANNOTATIONS_ROOT to official UA-DETRAC XML annotations}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zsj/data/datasets/UA-DETRAC}"
SPLIT="${SPLIT:-train}"
LINK_MODE="${LINK_MODE:-symlink}"
FPS="${FPS:-25}"

python tools/convert_uadetrac_xml_to_mot.py \
  --images-root "${IMAGES_ROOT}" \
  --annotations-root "${ANNOTATIONS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --split "${SPLIT}" \
  --link-mode "${LINK_MODE}" \
  --fps "${FPS}"
