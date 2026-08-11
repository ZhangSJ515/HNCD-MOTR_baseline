# HNCD-MOTR

![HNCD-MOTR](./assets/overview.png)

**HNCD-MOTR** is an end-to-end query-based multi-object tracker with **Hard Negative Confusion-aware Denoising (HNCD)**. HNCD is a training-time denoising mechanism that constructs target-centered hard negative denoising queries from spatially proximate objects. By exposing the tracker to local confusion cases during training, HNCD improves association behavior in crowded scenes while keeping the inference procedure unchanged.

The main implementation uses a proposal-anchored tracker with temporal memory. It combines MOTRv2-style detector proposals for target discovery with MeMOTR-style temporal interaction for long-term tracking. HNCD is applied only during training and is removed during inference.

## Installation

```shell
conda create -n HNCD python=3.10
conda activate HNCD
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib pyyaml scipy tqdm tensorboard
pip install opencv-python
```

You also need to compile the Deformable Attention CUDA ops:

```shell
cd ./models/ops/
sh make.sh

# Optional test
python test.py
```

## Data

Put the unzipped CrowdHuman dataset into:

```text
DATADIR/CrowdHuman/images/
```

Then generate the ground-truth files by running:

```shell
python ./data/gen_crowdhuman_gts.py
```

For the proposal-anchored tracker, cached YOLOX detector outputs are required as `det_db.json`. The path is set by the `DET_DB` field in the YAML config.

The expected dataset structure is:

```text
DATADIR/
  ├── DanceTrack/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── SportsMOT/
  │ ├── train/
  │ ├── test/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── CrowdHuman/
  │ ├── images/
  │ │ ├── train/
  │ │ └── val/
  │ └── gts/
  │   ├── train/
  │   └── val/
  └── det_db.json
```

## UA-DETRAC adaptation

The `feature/uadetrac-support` branch adds a complete HNCD-MOTR pipeline for single-class UA-DETRAC, including dataset conversion/loading, MOTRv2 proposal DB preparation, vehicle-specific DAB pretrain conversion, training, submission, native TrackEval evaluation, setup checks, and a synthetic smoke test.

Canonical layout:

```text
/home/zsj/data/datasets/UA-DETRAC/
├── train/<sequence>/{img1,gt/gt.txt,seqinfo.ini}
└── test/<sequence>/{img1,gt/gt.txt,seqinfo.ini}
```

`UA-DETRAC`, `UADETRAC`, and `UA_DETRAC` directory aliases are accepted. All valid targets are mapped to the single `vehicle` class.

Prepare/validate data:

```shell
# Convert official XML annotations.
IMAGES_ROOT=/path/to/images \
ANNOTATIONS_ROOT=/path/to/xml \
OUTPUT_ROOT=/home/zsj/data/datasets/UA-DETRAC \
SPLIT=train \
bash scripts/convert_uadetrac.sh

# Build HNCD/MOTRv2 proposal JSON from detector txt outputs.
DATA_ROOT=/home/zsj/data/datasets \
DETECTIONS_ROOT=/path/to/detector/results \
OUTPUT=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
bash scripts/build_uadetrac_det_db.sh

# Convert the DAB COCO classifier from car -> single vehicle class.
SOURCE_CHECKPOINT=/path/to/dab_deformable_detr_coco.pth \
OUTPUT_CHECKPOINT=./pretrains/dab_deformable_detr_coco_uadetrac.pth \
bash scripts/prepare_uadetrac_pretrain.sh

# Validate data + proposal coverage.
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
bash scripts/check_uadetrac_setup.sh

bash scripts/smoke_test_uadetrac.sh
```

Train:

```shell
NUM_GPUS=8 \
GPU_IDS=0,1,2,3,4,5,6,7 \
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
PRETRAINED_MODEL=./pretrains/dab_deformable_detr_coco_uadetrac.pth \
OUTPUTS_DIR=./outputs/hncd_uadetrac \
bash scripts/train_uadetrac.sh
```

Evaluate:

```shell
EVAL_DIR=./outputs/hncd_uadetrac \
MODEL_PATH=checkpoint_41.pth \
DATA_ROOT=/home/zsj/data/datasets \
DET_DB=/home/zsj/data/datasets/uadetrac_yolox_det_db.json \
SPLIT=test \
bash scripts/evaluate_uadetrac.sh
```

UA-DETRAC tracker results are written in standard MOTChallenge format and the native TrackEval adapter reports HOTA, CLEAR, and Identity metrics for `vehicle`. See [`docs/uadetrac_adaptation.md`](docs/uadetrac_adaptation.md) for the complete workflow and format details.

## Pretrain

We initialize the original pedestrian/dance models with the official DAB-Deformable-DETR R50 checkpoint pretrained on COCO. You can download the checkpoint used in the original experiments [here](https://drive.google.com/file/d/17FxIGgIZJih8LWkGdlIOe9ZpVZ9IRxSj/view?usp=sharing). For UA-DETRAC, first run `scripts/prepare_uadetrac_pretrain.sh` so the one-class head is initialized from the COCO **car** classifier rather than the pedestrian classifier.

## Scripts on DanceTrack

All commands below assume 4 GPUs. We use gradient checkpointing (`--use-checkpoint`) to fit on 24 GB GPUs. You can remove `--use-checkpoint` if you have sufficient memory.

### Training

```shell
python -m torch.distributed.run --nproc_per_node=4 main.py \
    --use-distributed \
    --config-path ./configs/train_dancetrack.yaml \
    --outputs-dir ./outputs/hncd \
    --use-checkpoint \
    --arch memotr \
    --use-proposals
```

### Validation

Evaluate a specific checkpoint on the validation set:

```shell
python main.py \
    --mode eval \
    --eval-mode specific \
    --eval-model checkpoint.pth \
    --eval-dir ./outputs/hncd \
    --eval-threads 4 \
    --config-path ./configs/train_dancetrack.yaml
```

### Test-set submission

Generate test-set submission results:

```shell
python -m torch.distributed.run --nproc_per_node=4 main.py \
    --mode submit \
    --submit-dir ./outputs/hncd \
    --submit-model checkpoint.pth \
    --use-distributed \
    --config-path ./configs/train_dancetrack.yaml
```

## Scripts on SportsMOT and other datasets

You can replace the `--config-path` in the DanceTrack scripts with the corresponding dataset config, for example:

```text
./configs/train_sportsmot.yaml
```

The main implementation still uses:

```text
--arch memotr --use-proposals
```

## Results

### Multi-Object Tracking on the DanceTrack test set

| Method    | HOTA | DetA | AssA | IDF1 | MOTA | Checkpoint                                                                                            |
| --------- | ---: | ---: | ---: | ---: | ---: | ----------------------------------------------------------------------------------------------------- |
| HNCD-MOTR | 71.8 | 83.5 | 61.9 | 74.5 | 92.5 | [Google Drive](https://drive.google.com/file/d/13XjwQwKs6juhuXntZWM0LU6quawh7VIp/view?usp=drive_link) |

### Multi-Object Tracking on the SportsMOT test set

| Method    | HOTA | DetA | AssA | IDF1 | MOTA | Checkpoint                                                                                            |
| --------- | ---: | ---: | ---: | ---: | ---: | ----------------------------------------------------------------------------------------------------- |
| HNCD-MOTR | 71.7 | 84.2 | 61.2 | 73.4 | 92.5 | [Google Drive](https://drive.google.com/file/d/1H58SpBb_DcjvAvEPTo5BXnAki89UUxMG/view?usp=drive_link) |
