from __future__ import annotations

import json
import os
from os import path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.uadetrac import resolve_uadetrac_root
from log.logger import Logger
from models import build_model
from models.utils import load_checkpoint
from submit_engine import Submitter
from utils.utils import (
    distributed_rank,
    distributed_world_size,
    is_distributed,
    yaml_to_dict,
)


class UADETRACSubmitter(Submitter):
    """Submitter that preserves UA-DETRAC frame IDs and MOTChallenge rows."""

    def __init__(self, *args, min_track_area: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_track_area = float(min_track_area)

    def filter_by_area(self, tracks, thresh=None):
        threshold = self.min_track_area if thresh is None else float(thresh)
        keep = tracks.area > threshold
        return tracks[keep]

    def write_results(self, tracks_result, frame_idx: int):
        # ``frame_idx`` is the zero-based loader position. SeqDataset exposes the
        # original numeric frame ID so conversion is robust to non-padded names.
        frame_id = int(self.dataset.frame_ids[frame_idx])
        result_path = os.path.join(self.predict_dir, f"{self.seq_name_safe}.txt")
        with open(result_path, "a", encoding="utf-8") as file:
            for index in range(len(tracks_result)):
                x1, y1, x2, y2 = tracks_result.boxes[index].tolist()
                width = x2 - x1
                height = y2 - y1
                if tracks_result.scores.ndim > 1:
                    score = float(torch.max(tracks_result.scores[index]).item())
                else:
                    score = float(tracks_result.scores[index].item())
                file.write(
                    f"{frame_id},{int(tracks_result.ids[index].item())},"
                    f"{x1:.6f},{y1:.6f},{width:.6f},{height:.6f},"
                    f"{score:.6f},-1,-1,-1\n"
                )


def _load_det_db(config: dict, train_config: dict):
    use_proposals = bool(train_config.get("USE_PROPOSALS", True))
    if not use_proposals:
        return None

    det_db_path = config.get("DET_DB") or train_config.get("DET_DB")
    if not det_db_path or str(det_db_path).lower() == "none":
        raise ValueError(
            "HNCD-MOTR UA-DETRAC is configured with USE_PROPOSALS=True but DET_DB is empty. "
            "Build the proposal DB first or explicitly train/evaluate with USE_PROPOSALS=False."
        )
    det_db_path = os.path.expanduser(str(det_db_path))
    if not os.path.isfile(det_db_path):
        raise FileNotFoundError(f"UA-DETRAC proposal DB not found: {det_db_path}")
    with open(det_db_path, "r", encoding="utf-8") as file:
        return json.load(file)


def submit_uadetrac(config: dict):
    if config.get("SUBMIT_DIR") is None:
        raise ValueError("SUBMIT_DIR must be set for UA-DETRAC submission.")
    if config.get("SUBMIT_MODEL") is None:
        raise ValueError("SUBMIT_MODEL must be set for UA-DETRAC submission.")
    if config.get("SUBMIT_DATA_SPLIT") is None:
        raise ValueError("SUBMIT_DATA_SPLIT must be set for UA-DETRAC submission.")

    outputs_dir = path.join(config["SUBMIT_DIR"], config["SUBMIT_DATA_SPLIT"])
    logger = Logger(logdir=outputs_dir, only_main=True)
    logger.show(head="UA-DETRAC submit configs:", log=config)
    logger.write(log=config, filename="config.yaml", mode="w")

    train_config_path = path.join(config["SUBMIT_DIR"], "train", "config.yaml")
    if not path.isfile(train_config_path):
        raise FileNotFoundError(
            f"Training config was not found at {train_config_path}. "
            "SUBMIT_DIR must point to the training output directory."
        )
    train_config = yaml_to_dict(train_config_path)
    train_config["DATASET"] = "DanceTrack"  # one-class model/criterion compatibility
    train_config["DATASET_ADAPTER"] = "UADETRAC"
    train_config["DATASET_NAME"] = config.get("DATASET_NAME", "UA-DETRAC")

    model = build_model(config=train_config)
    checkpoint_path = path.join(config["SUBMIT_DIR"], config["SUBMIT_MODEL"])
    if not path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    load_checkpoint(model=model, path=checkpoint_path)

    data_root = os.path.abspath(os.path.expanduser(config["DATA_ROOT"]))
    dataset_root = resolve_uadetrac_root(data_root, config.get("DATASET_NAME"))
    split_dir = path.join(dataset_root, config["SUBMIT_DATA_SPLIT"])
    if not path.isdir(split_dir):
        raise FileNotFoundError(split_dir)

    sequence_names = sorted(
        name
        for name in os.listdir(split_dir)
        if path.isdir(path.join(split_dir, name, "img1"))
    )
    if not sequence_names:
        raise RuntimeError(f"No UA-DETRAC sequences found under {split_dir}")

    if is_distributed():
        model = DDP(
            module=model,
            device_ids=[distributed_rank()],
            find_unused_parameters=True,
        )
        sequence_names = [
            sequence_name
            for index, sequence_name in enumerate(sequence_names)
            if index % distributed_world_size() == distributed_rank()
        ]

    det_db = _load_det_db(config, train_config)

    for sequence_name in sequence_names:
        submitter = UADETRACSubmitter(
            dataset_name="UA-DETRAC",
            split_dir=split_dir,
            seq_name=sequence_name,
            outputs_dir=outputs_dir,
            model=model,
            use_dab=train_config["USE_DAB"],
            det_score_thresh=config["DET_SCORE_THRESH"],
            track_score_thresh=config["TRACK_SCORE_THRESH"],
            result_score_thresh=config["RESULT_SCORE_THRESH"],
            miss_tolerance=config["MISS_TOLERANCE"],
            use_motion=config["USE_MOTION"],
            motion_min_length=config["MOTION_MIN_LENGTH"],
            motion_max_length=config["MOTION_MAX_LENGTH"],
            motion_lambda=config["MOTION_LAMBDA"],
            visualize=config.get("VISUALIZE", False),
            det_db=det_db,
            save_embeddings_dir=config.get("SAVE_EMBEDDINGS_DIR"),
            min_track_area=config.get("MIN_TRACK_AREA", 0.0),
        )
        submitter.run()
