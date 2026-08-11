# HNCD-MOTR: Complete UA-DETRAC Adaptation

This branch adds a complete single-class UA-DETRAC pipeline to the HNCD-MOTR codebase while preserving the proposal-driven MOTRv2 + HNCD architecture.

## 1. Design choice

UA-DETRAC is treated as a **single `vehicle` class**. Internally the configuration keeps:

```yaml
DATASET: DanceTrack
DATASET_ADAPTER: UADETRAC
DATASET_NAME: UA-DETRAC
```

This is intentional: the original HNCD model and criterion already implement a one-class head for `DanceTrack`. `DATASET_ADAPTER` selects the UA-DETRAC loader and UA-specific submit/evaluation path without changing the HNCD loss, query updater, or one-class classifier semantics.

The default UA-DETRAC HNCD configuration uses the proposal-driven variant:

```yaml
ARCH: motrv2
USE_PROPOSALS: True
NUM_DET_QUERIES: 10
MERGE_DET_TRACK_LAYER: 0
```

A detector proposal database is therefore required for a faithful HNCD-MOTR/MOTRv2-style experiment.

## 2. Target dataset layout

The adapter accepts `UA-DETRAC`, `UADETRAC`, or `UA_DETRAC` as the dataset directory name. The canonical layout is:

```text
/home/zsj/data/datasets/
└── UA-DETRAC/
    ├── train/
    │   └── MVI_XXXX/
    │       ├── img1/
    │       │   ├── 000001.jpg
    │       │   └── ...
    │       ├── gt/
    │       │   └── gt.txt
    │       └── seqinfo.ini
    └── test/
        └── MVI_XXXX/
            ├── img1/
            ├── gt/gt.txt          # required for local evaluation
            └── seqinfo.ini
```

Supported GT layouts:

```text
# Standard MOTChallenge
frame,id,x,y,width,height,mark,class,visibility[,unused]

# Class-first native8
frame id class x y width height flag

# Compact single-class
frame id x y width height [mark]
```

Set `UADETRAC_GT_FORMAT` to `auto`, `mot`, `native8`, or `simple` if needed.

## 3. Convert official UA-DETRAC XML

If starting from the official image directories and XML annotations:

```bash
cd /data/zsj/workspace/multi-object_tracking/HNCD-MOTR_baseline

IMAGES_ROOT=/path/to/official/train/images \
ANNOTATIONS_ROOT=/path/to/official/train/xml \
OUTPUT_ROOT=/home/zsj/data/datasets/UA-DETRAC \
SPLIT=train \
LINK_MODE=symlink \
bash scripts/convert_uadetrac.sh
```

Run again for the evaluation split:

```bash
IMAGES_ROOT=/path/to/official/test/images \
ANNOTATIONS_ROOT=/path/to/official/test/xml \
OUTPUT_ROOT=/home/zsj/data/datasets/UA-DETRAC \
SPLIT=test \
LINK_MODE=symlink \
bash scripts/convert_uadetrac.sh
```

The converter writes standard MOTChallenge GT rows:

```text
frame,id,x,y,w,h,1,1,visibility
```

where `visibility = 1 - truncation_ratio` when the XML provides a truncation ratio.

## 4. Build the MOTRv2 detector proposal database

HNCD-MOTR's proposal-driven variant requires cached detector proposals. This package does **not** silently substitute ground-truth boxes for detector proposals.

Prepare one detector txt per UA-DETRAC sequence. Supported detector rows are:

```text
# Standard MOT detector output
frame,id,x,y,w,h,score[, ...]

# Compact detector output
frame,x,y,w,h,score
```

Detection files may be stored as any of:

```text
<DETECTIONS_ROOT>/<sequence>.txt
<DETECTIONS_ROOT>/<split>/<sequence>.txt
<DETECTIONS_ROOT>/<sequence>/det/det.txt
<DETECTIONS_ROOT>/<split>/<sequence>/det/det.txt
```

Build the HNCD proposal JSON:

```bash
DATA_ROOT=/home/zsj/data/datasets \
DETECTIONS_ROOT=/path/to/uadetrac_detector_outputs \
OUTPUT=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
SPLITS="train test" \
MIN_SCORE=0.0 \
bash scripts/build_uadetrac_det_db.sh
```

The generated database uses keys such as:

```text
UA-DETRAC/train/MVI_20011/img1/000001.txt
```

and stores `x,y,w,h,score` proposals.

## 5. Validate the dataset and proposals

```bash
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
SPLITS="train test" \
bash scripts/check_uadetrac_setup.sh
```

The checker validates:

- numeric frame IDs and image availability;
- GT parsing and GT-to-image consistency;
- valid positive-size boxes;
- track counts;
- proposal DB coverage for every frame when required.

Run the built-in synthetic smoke test:

```bash
bash scripts/smoke_test_uadetrac.sh
```

Expected final line:

```text
[OK] HNCD-MOTR UA-DETRAC smoke test passed.
```

## 6. Pretrained detector checkpoint

The HNCD implementation loads the official DAB-Deformable-DETR R50 COCO checkpoint through the existing `load_pretrained_model()` path. Because UA-DETRAC is a single-class experiment, the existing one-class classifier conversion is reused.

Set the checkpoint path at launch time:

```bash
PRETRAINED_MODEL=/path/to/dab_deformable_detr.pth
```

Do not leave `/path/to/dab_deformable_detr.pth` from the YAML unchanged.

## 7. Train HNCD-MOTR on UA-DETRAC

Eight GPUs:

```bash
cd /data/zsj/workspace/multi-object_tracking/HNCD-MOTR_baseline

NUM_GPUS=8 \
GPU_IDS=0,1,2,3,4,5,6,7 \
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
PRETRAINED_MODEL=/path/to/dab_deformable_detr.pth \
OUTPUTS_DIR=./outputs/hncd_uadetrac \
bash scripts/train_uadetrac.sh
```

Single-GPU debug run:

```bash
NUM_GPUS=1 \
GPU_IDS=0 \
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
PRETRAINED_MODEL=/path/to/dab_deformable_detr.pth \
OUTPUTS_DIR=./outputs/hncd_uadetrac \
bash scripts/train_uadetrac.sh
```

The default training schedule in `configs/train_uadetrac_hncd.yaml` is an operational starting point inherited from the existing adaptation; it is **not claimed to be an optimized UA-DETRAC schedule**.

## 8. Generate tracker files

Assume the checkpoint is:

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

Tracker rows are serialized as:

```text
frame,id,x,y,width,height,score,-1,-1,-1
```

Original numeric UA-DETRAC frame IDs are preserved.

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

The custom UA-DETRAC TrackEval adapter evaluates the single class `vehicle` using:

```text
HOTA
CLEAR
Identity
```

The main summary file is:

```text
outputs/hncd_uadetrac/test/checkpoint_41_tracker/vehicle_summary.txt
```

and a JSON copy is written to:

```text
outputs/hncd_uadetrac/test/checkpoint_41_tracker/hncd_uadetrac_metrics.json
```

The existing runtime statistics generated by `Submitter` are also aggregated during evaluation.

## 10. Files added or modified

Core adaptation:

```text
configs/train_uadetrac_hncd.yaml
data/uadetrac.py
data/__init__.py
data/seq_dataset.py
main.py
uadetrac_submit_engine.py
uadetrac_eval_engine.py
```

Evaluation:

```text
TrackEval/trackeval/datasets/uadetrac.py
TrackEval/trackeval/datasets/__init__.py
TrackEval/scripts/run_uadetrac.py
```

Preparation / validation:

```text
tools/convert_uadetrac_xml_to_mot.py
tools/build_uadetrac_det_db.py
tools/check_uadetrac_setup.py
tools/smoke_test_uadetrac.py
```

Launchers:

```text
scripts/convert_uadetrac.sh
scripts/build_uadetrac_det_db.sh
scripts/check_uadetrac_setup.sh
scripts/smoke_test_uadetrac.sh
scripts/train_uadetrac.sh
scripts/submit_uadetrac.sh
scripts/evaluate_uadetrac.sh
```

## 11. What is and is not validated

The package includes static checks and a synthetic data-path smoke test. It does **not** claim UA-DETRAC benchmark performance until the adapted model is actually trained and evaluated on the real dataset and real detector proposal database.
