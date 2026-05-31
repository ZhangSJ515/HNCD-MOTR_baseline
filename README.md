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

## Pretrain

We initialize the model with the official DAB-Deformable-DETR R50 checkpoint pretrained on COCO. You can download the checkpoint used in our experiments [here](https://drive.google.com/file/d/17FxIGgIZJih8LWkGdlIOe9ZpVZ9IRxSj/view?usp=sharing), and place it at the root of this project directory.

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

