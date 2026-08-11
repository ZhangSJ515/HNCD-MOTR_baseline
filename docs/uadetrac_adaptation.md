# HNCD-MOTR: Complete UA-DETRAC Adaptation

This branch adds a complete single-class UA-DETRAC pipeline to HNCD-MOTR while preserving the proposal-driven MOTRv2 + HNCD architecture.

## 1. Design choice

UA-DETRAC is treated as a **single `vehicle` class**. Internally the config keeps:

```yaml
DATASET: DanceTrack
DATASET_ADAPTER: UADETRAC
DATASET_NAME: UA-DETRAC
```

This is intentional. The original HNCD model and criterion already implement a one-class head for `DanceTrack`; `DATASET_ADAPTER` selects the UA-DETRAC loader and UA-specific submission/evaluation path without rewriting HNCD losses or query propagation.

The default HNCD UA-DETRAC variant is proposal-driven:

```yaml
ARCH: motrv2
USE_PROPOSALS: True
NUM_DET_QUERIES: 10
MERGE_DET_TRACK_LAYER: 0
```

A real detector proposal database is therefore required for a faithful HNCD-MOTR/MOTRv2-style experiment.

## 2. Dataset layout

The adapter accepts `UA-DETRAC`, `UADETRAC`, or `UA_DETRAC`. Canonical layout:

```text
/home/zsj/data/datasets/
└── UA-DETRAC/
    ├── train/
    │   └── MVI_XXXX/
    │       ├── img1/
    │       ├── gt/gt.txt
    │       └── seqinfo.ini
    └── test/
        └── MVI_XXXX/
            ├── img1/
            ├── gt/gt.txt          # required for local evaluation
            └── seqinfo.ini
```

Supported GT layouts:

```text
# MOTChallenge
frame,id,x,y,width,height,mark,class,visibility[,unused]

# Class-first native8
frame id class x y width height flag

# Compact single-class
frame id x y width height [mark]
```

Set `UADETRAC_GT_FORMAT` to `auto`, `mot`, `native8`, or `simple` when needed.

## 3. Convert official XML data

```bash
cd /data/zsj/workspace/multi-object_tracking/HNCD-MOTR_baseline

IMAGES_ROOT=/path/to/official/train/images \
ANNOTATIONS_ROOT=/path/to/official/train/xml \
OUTPUT_ROOT=/home/zsj/data/datasets/UA-DETRAC \
SPLIT=train \
LINK_MODE=symlink \
bash scripts/convert_uadetrac.sh
```

Repeat for the evaluation split:

```bash
IMAGES_ROOT=/path/to/official/test/images \
ANNOTATIONS_ROOT=/path/to/official/test/xml \
OUTPUT_ROOT=/home/zsj/data/datasets/UA-DETRAC \
SPLIT=test \
LINK_MODE=symlink \
bash scripts/convert_uadetrac.sh
```

The converter writes:

```text
frame,id,x,y,w,h,1,1,visibility
```

where `visibility = 1 - truncation_ratio` when available in the XML.

## 4. Build the MOTRv2 detector proposal DB

HNCD-MOTR must use detector proposals rather than GT boxes. Prepare one detector result file per sequence. Supported rows:

```text
frame,id,x,y,w,h,score[, ...]
frame,x,y,w,h,score
```

Accepted file locations include:

```text
<DETECTIONS_ROOT>/<sequence>.txt
<DETECTIONS_ROOT>/<split>/<sequence>.txt
<DETECTIONS_ROOT>/<sequence>/det/det.txt
<DETECTIONS_ROOT>/<split>/<sequence>/det/det.txt
```

Build the JSON DB:

```bash
DATA_ROOT=/home/zsj/data/datasets \
DETECTIONS_ROOT=/path/to/uadetrac_detector_outputs \
OUTPUT=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
SPLITS="train test" \
MIN_SCORE=0.0 \
bash scripts/build_uadetrac_det_db.sh
```

The generated keys look like:

```text
UA-DETRAC/train/MVI_20011/img1/000001.txt
```

and values contain `x,y,w,h,score` proposals.

## 5. Convert the DAB-Deformable-DETR pretrain correctly

The original HNCD one-class loader defaults to the COCO **person** classifier row, which is suitable for pedestrian datasets but not UA-DETRAC. This package therefore provides a UA-specific converter that maps the COCO **car** classifier row to the single UA-DETRAC `vehicle` head.

Prepare the checkpoint once:

```bash
SOURCE_CHECKPOINT=/path/to/dab_deformable_detr_coco.pth \
OUTPUT_CHECKPOINT=./pretrains/dab_deformable_detr_coco_uadetrac.pth \
bash scripts/prepare_uadetrac_pretrain.sh
```

The converter supports:

```text
91-row COCO layout -> car index 3
80-row contiguous COCO layout -> car index 2
```

The UA config defaults to:

```yaml
PRETRAINED_MODEL: ./pretrains/dab_deformable_detr_coco_uadetrac.pth
```

## 6. Validate setup

```bash
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
SPLITS="train test" \
bash scripts/check_uadetrac_setup.sh
```

The checker validates numeric frames, GT parsing, GT-to-image consistency, valid boxes, track counts, and proposal DB coverage.

Synthetic data-path smoke test:

```bash
bash scripts/smoke_test_uadetrac.sh
```

Expected final line:

```text
[OK] HNCD-MOTR UA-DETRAC smoke test passed.
```

## 7. Train

Eight GPUs:

```bash
NUM_GPUS=8 \
GPU_IDS=0,1,2,3,4,5,6,7 \
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
PRETRAINED_MODEL=./pretrains/dab_deformable_detr_coco_uadetrac.pth \
OUTPUTS_DIR=./outputs/hncd_uadetrac \
bash scripts/train_uadetrac.sh
```

Single-GPU debug run:

```bash
NUM_GPUS=1 \
GPU_IDS=0 \
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
PRETRAINED_MODEL=./pretrains/dab_deformable_detr_coco_uadetrac.pth \
OUTPUTS_DIR=./outputs/hncd_uadetrac \
bash scripts/train_uadetrac.sh
```

The schedule in `configs/train_uadetrac_hncd.yaml` is an operational starting point, **not a claimed optimized UA-DETRAC schedule**.

## 8. Submit tracker files

Assume:

```text
./outputs/hncd_uadetrac/checkpoint_41.pth
```

Run:

```bash
SUBMIT_DIR=./outputs/hncd_uadetrac \
MODEL_PATH=checkpoint_41.pth \
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
SPLIT=test \
GPU_ID=0 \
bash scripts/submit_uadetrac.sh
```

Tracker rows:

```text
frame,id,x,y,width,height,score,-1,-1,-1
```

Original numeric frame IDs are preserved.

## 9. Evaluate with native TrackEval

```bash
EVAL_DIR=./outputs/hncd_uadetrac \
MODEL_PATH=checkpoint_41.pth \
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
SPLIT=test \
GPU_ID=0 \
bash scripts/evaluate_uadetrac.sh
```

The custom TrackEval adapter evaluates the single `vehicle` class with HOTA, CLEAR, and Identity metrics. Main output:

```text
outputs/hncd_uadetrac/test/checkpoint_41_tracker/vehicle_summary.txt
outputs/hncd_uadetrac/test/checkpoint_41_tracker/hncd_uadetrac_metrics.json
```

Existing per-sequence runtime statistics from `Submitter` are aggregated during evaluation.

## 10. Adaptation files

Core:

```text
configs/train_uadetrac_hncd.yaml
data/uadetrac.py
data/__init__.py
data/seq_dataset.py
main.py
uadetrac_submit_engine.py
uadetrac_eval_engine.py
```

TrackEval:

```text
TrackEval/trackeval/datasets/uadetrac.py
TrackEval/trackeval/datasets/__init__.py
TrackEval/scripts/run_uadetrac.py
```

Preparation and validation:

```text
tools/convert_uadetrac_xml_to_mot.py
tools/build_uadetrac_det_db.py
tools/convert_uadetrac_dab_pretrain.py
tools/check_uadetrac_setup.py
tools/smoke_test_uadetrac.py
```

Launchers:

```text
scripts/convert_uadetrac.sh
scripts/build_uadetrac_det_db.sh
scripts/prepare_uadetrac_pretrain.sh
scripts/check_uadetrac_setup.sh
scripts/smoke_test_uadetrac.sh
scripts/train_uadetrac.sh
scripts/submit_uadetrac.sh
scripts/evaluate_uadetrac.sh
```

## 11. Validation status

The branch includes Python/shell static CI checks and a synthetic smoke-test script. No UA-DETRAC benchmark number is claimed until real-data proposal generation, training, and evaluation are run on the target server/GPU environment.
