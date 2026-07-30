#!/usr/bin/env bash
set -euo pipefail

python tools/check_airmot_setup.py \
  --data-root /home/zsj/data/datasets \
  --det-db /home/zsj/data/datasets/airmot_yolox_det_db.json
