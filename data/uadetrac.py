from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image

import data.transforms as T
from .mot import MOTDataset


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_UADETRAC_ALIASES = ("UA-DETRAC", "UADETRAC", "UA_DETRAC")


def _split_fields(line: str) -> List[str]:
    return [field for field in re.split(r"[,\s]+", line.strip()) if field]


def _normalize_key(path: str) -> str:
    return path.replace("\\", "/")


def _parse_proposal_line(item) -> Tuple[float, float, float, float, float]:
    if isinstance(item, str):
        fields = _split_fields(item)
    elif isinstance(item, (list, tuple)):
        fields = list(item)
    else:
        raise TypeError(f"Unsupported proposal item type: {type(item)}")
    if len(fields) != 5:
        raise ValueError(f"Proposal must be x,y,w,h,score, got: {fields}")
    return tuple(map(float, fields))


def _small_integer(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number.is_integer() and -1 <= int(number) <= 20


def resolve_uadetrac_root(data_root: str, dataset_name: str | None = None) -> str:
    root = os.path.abspath(os.path.expanduser(data_root))
    names = []
    if dataset_name:
        names.append(dataset_name)
    names.extend(_UADETRAC_ALIASES)
    for name in dict.fromkeys(names):
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return candidate
    # Keep the canonical path in the error message even before data is prepared.
    return os.path.join(root, "UA-DETRAC")


class UADETRAC(MOTDataset):
    """UA-DETRAC adapter for HNCD-MOTR training.

    The model is single-class: every valid UA-DETRAC target is mapped to label 0.

    Supported GT layouts:
      * MOTChallenge: frame,id,x,y,w,h,mark,class,visibility[,unused]
      * native8:      frame id class x y w h flag
      * simple:       frame id x y w h [mark]

    ``UADETRAC_GT_FORMAT`` may be ``auto``, ``mot``, ``native8`` or ``simple``.
    """

    def __init__(self, config: dict, split: str, transform):
        super().__init__(config=config, split=split, transform=transform)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported UA-DETRAC split: {split}")

        self.config = config
        self.transform = transform
        self.data_root = os.path.abspath(os.path.expanduser(config["DATA_ROOT"]))
        self.dataset_root = resolve_uadetrac_root(
            self.data_root, config.get("DATASET_NAME")
        )
        self.split_dir = os.path.join(self.dataset_root, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(
                f"UA-DETRAC split directory does not exist: {self.split_dir}"
            )

        self.gt_format = str(config.get("UADETRAC_GT_FORMAT", "auto")).lower()
        if self.gt_format not in {"auto", "mot", "native8", "simple"}:
            raise ValueError(f"Unsupported UADETRAC_GT_FORMAT: {self.gt_format}")
        self.filter_gt_by_mark = bool(config.get("UADETRAC_FILTER_GT_BY_MARK", True))
        self.min_visibility = float(config.get("UADETRAC_MIN_VISIBILITY", 0.0))

        self.sample_steps = list(config["SAMPLE_STEPS"])
        self.sample_intervals = list(config["SAMPLE_INTERVALS"])
        self.sample_modes = list(config["SAMPLE_MODES"])
        self.sample_lengths = list(config["SAMPLE_LENGTHS"])

        self.sample_stage = 0
        self.sample_begin_frames: List[Tuple[str, int]] = []
        self.sample_length = 1
        self.sample_mode = "random_interval"
        self.sample_interval = 1

        self.gts: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
        self.frame_paths: Dict[str, Dict[int, str]] = defaultdict(dict)
        self.frame_ids: Dict[str, List[int]] = {}
        self.vid_idx: Dict[str, int] = {}
        self.idx_vid: Dict[int, str] = {}

        self.use_proposals = bool(config.get("USE_PROPOSALS", True))
        self.det_db = None
        det_db_path = config.get("DET_DB")
        if self.use_proposals and det_db_path and str(det_db_path).lower() != "none":
            det_db_path = os.path.expanduser(str(det_db_path))
            if not os.path.isfile(det_db_path):
                raise FileNotFoundError(
                    f"UA-DETRAC proposal DB does not exist: {det_db_path}"
                )
            with open(det_db_path, "r", encoding="utf-8") as file:
                raw_db = json.load(file)
            self.det_db = {_normalize_key(str(key)): value for key, value in raw_db.items()}

        self._load_sequences()
        self.set_epoch(0)

    def _parse_gt(self, fields: List[str]):
        layout = self.gt_format
        if layout == "auto":
            if len(fields) >= 9:
                layout = "mot"
            elif len(fields) == 8 and _small_integer(fields[2]):
                layout = "native8"
            else:
                layout = "simple"

        if layout == "mot":
            if len(fields) < 8:
                raise ValueError("MOT GT rows require at least 8 columns")
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            mark = float(fields[6])
            visibility = float(fields[8]) if len(fields) >= 9 else 1.0
        elif layout == "native8":
            if len(fields) < 8:
                raise ValueError("native8 GT rows require 8 columns")
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = map(float, fields[3:7])
            mark = float(fields[7])
            visibility = 1.0
        elif layout == "simple":
            if len(fields) < 6:
                raise ValueError("simple GT rows require at least 6 columns")
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            mark = float(fields[6]) if len(fields) >= 7 else 1.0
            visibility = 1.0
        else:
            raise ValueError(layout)

        return frame_id, track_id, x, y, width, height, mark, visibility

    def _load_sequences(self) -> None:
        sequence_names = sorted(
            name
            for name in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, name))
        )

        for sequence_name in sequence_names:
            sequence_dir = os.path.join(self.split_dir, sequence_name)
            image_dir = os.path.join(sequence_dir, "img1")
            gt_path = os.path.join(sequence_dir, "gt", "gt.txt")
            if not os.path.isdir(image_dir) or not os.path.isfile(gt_path):
                continue

            for image_name in os.listdir(image_dir):
                stem, extension = os.path.splitext(image_name)
                if extension.lower() in _IMAGE_EXTENSIONS and stem.isdigit():
                    self.frame_paths[sequence_name][int(stem)] = os.path.join(
                        image_dir, image_name
                    )

            with open(gt_path, "r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    fields = _split_fields(line)
                    try:
                        (
                            frame_id,
                            track_id,
                            x,
                            y,
                            width,
                            height,
                            mark,
                            visibility,
                        ) = self._parse_gt(fields)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid UA-DETRAC GT row {gt_path}:{line_number}: {line.strip()}"
                        ) from exc

                    if track_id < 0 or width <= 0 or height <= 0:
                        continue
                    if self.filter_gt_by_mark and mark <= 0:
                        continue
                    if visibility < self.min_visibility:
                        continue

                    self.gts[sequence_name][frame_id].append(
                        [track_id, 0, x, y, width, height]
                    )

            valid_frames = sorted(
                set(self.frame_paths[sequence_name]) & set(self.gts[sequence_name])
            )
            if not valid_frames:
                self.gts.pop(sequence_name, None)
                self.frame_paths.pop(sequence_name, None)
                continue

            self.frame_ids[sequence_name] = valid_frames
            video_index = len(self.vid_idx)
            self.vid_idx[sequence_name] = video_index
            self.idx_vid[video_index] = sequence_name

        if not self.vid_idx:
            raise RuntimeError(
                f"No trainable UA-DETRAC sequences found under {self.split_dir}. "
                "Training splits must contain img1/ and gt/gt.txt."
            )

    def _proposal_key_candidates(self, image_path: str) -> List[str]:
        path = Path(image_path).resolve()
        candidates = []
        for root in [Path(self.data_root), Path(self.dataset_root), Path(self.split_dir)]:
            try:
                relative = path.relative_to(root.resolve())
            except ValueError:
                continue
            candidates.append(_normalize_key(str(relative.with_suffix(".txt"))))

        # Canonical and alias-prefixed forms are common in detector DB exports.
        relative_split = _normalize_key(str(path.relative_to(Path(self.split_dir).resolve()).with_suffix(".txt")))
        split_name = Path(self.split_dir).name
        for alias in _UADETRAC_ALIASES:
            candidates.append(f"{alias}/{split_name}/{relative_split}")

        return list(dict.fromkeys(candidates))

    def _get_proposals(self, image_path: str):
        if self.det_db is None:
            return []
        items = None
        for key in self._proposal_key_candidates(image_path):
            if key in self.det_db:
                items = self.det_db[key]
                break
        if items is None:
            return []

        proposals = []
        for item in items:
            x, y, width, height, score = _parse_proposal_line(item)
            if width > 0 and height > 0:
                proposals.append((x, y, width, height, score))
        return proposals

    def __len__(self):
        return len(self.sample_begin_frames)

    def set_epoch(self, epoch: int):
        self.sample_stage = sum(epoch >= step for step in self.sample_steps)
        self.sample_length = self.sample_lengths[
            min(self.sample_stage, len(self.sample_lengths) - 1)
        ]
        self.sample_mode = self.sample_modes[
            min(self.sample_stage, len(self.sample_modes) - 1)
        ]
        self.sample_interval = self.sample_intervals[
            min(self.sample_stage, len(self.sample_intervals) - 1)
        ]

        self.sample_begin_frames = []
        for sequence_name, frame_ids in self.frame_ids.items():
            n_starts = max(0, len(frame_ids) - self.sample_length + 1)
            for begin_position in range(n_starts):
                self.sample_begin_frames.append((sequence_name, begin_position))

    def sample_frames_idx(self, vid: str, begin_pos: int) -> List[int]:
        if self.sample_mode != "random_interval":
            raise ValueError(f"Unsupported sample mode: {self.sample_mode}")

        frame_ids = self.frame_ids[vid]
        if self.sample_length <= 1:
            return [frame_ids[begin_pos]]

        remaining = len(frame_ids) - 1 - begin_pos
        max_interval = max(
            1,
            min(self.sample_interval, remaining // (self.sample_length - 1)),
        )
        interval = random.randint(1, max_interval)
        return [
            frame_ids[begin_pos + interval * index]
            for index in range(self.sample_length)
        ]

    def get_single_frame(self, vid: str, idx: int):
        image_path = self.frame_paths[vid][idx]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        info = {
            "boxes": [],
            "ids": [],
            "labels": [],
            "areas": [],
            "frame_idx": torch.as_tensor(idx),
            "scores": [],
            "unnorm_img": torch.as_tensor([height, width]),
        }

        id_offset = self.vid_idx[vid] * 100000
        for track_id, class_id, x, y, box_width, box_height in self.gts[vid][idx]:
            info["boxes"].append([x, y, box_width, box_height])
            info["ids"].append(track_id + id_offset)
            info["labels"].append(class_id)
            info["areas"].append(box_width * box_height)
            info["scores"].append(1.0)

        for x, y, box_width, box_height, score in self._get_proposals(image_path):
            info["boxes"].append([x, y, box_width, box_height])
            info["ids"].append(-1)
            info["labels"].append(-1)
            info["areas"].append(box_width * box_height)
            info["scores"].append(score)

        info["boxes"] = torch.as_tensor(info["boxes"], dtype=torch.float32).reshape(-1, 4)
        info["ids"] = torch.as_tensor(info["ids"], dtype=torch.long)
        info["labels"] = torch.as_tensor(info["labels"], dtype=torch.long)
        info["areas"] = torch.as_tensor(info["areas"], dtype=torch.float32)
        info["scores"] = torch.as_tensor(info["scores"], dtype=torch.float32)
        if len(info["boxes"]) > 0:
            info["boxes"][:, 2:] += info["boxes"][:, :2]
        return image, info

    def get_multi_frames(self, vid: str, idxs: Sequence[int]):
        items = [self.get_single_frame(vid, frame_id) for frame_id in idxs]
        images, infos = zip(*items)
        return list(images), list(infos)

    def __getitem__(self, item: int):
        vid, begin_position = self.sample_begin_frames[item]
        frame_ids = self.sample_frames_idx(vid, begin_position)
        images, infos = self.get_multi_frames(vid, frame_ids)
        if self.transform is not None:
            images, infos = self.transform(images, infos)

        proposals = []
        gt_infos = []
        for target in infos:
            num_gt = int((target["labels"] >= 0).sum().item())
            proposals.append(
                torch.cat(
                    [target["boxes"][num_gt:], target["scores"][num_gt:, None]],
                    dim=1,
                )
            )
            gt_infos.append(
                {
                    "labels": target["labels"][:num_gt],
                    "ids": target["ids"][:num_gt],
                    "boxes": target["boxes"][:num_gt],
                    "areas": target["areas"][:num_gt],
                    "frame_idx": target["frame_idx"],
                    "unnorm_img": target["unnorm_img"],
                }
            )

        return {"imgs": images, "infos": gt_infos, "proposals": proposals}


def transforms_for_train(coco_size=False, overflow_bbox=False, reverse_clip=0.0):
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]
    return T.MultiCompose(
        [
            T.MultiRandomHorizontalFlip(),
            T.MultiRandomSelect(
                T.MultiRandomResize(sizes=scales, max_size=1536),
                T.MultiCompose(
                    [
                        T.MultiRandomResize([400, 500, 600] if coco_size else [800, 1000, 1200]),
                        T.MultiRandomCrop(
                            min_size=384 if coco_size else 800,
                            max_size=600 if coco_size else 1200,
                            overflow_bbox=overflow_bbox,
                        ),
                        T.MultiRandomResize(sizes=scales, max_size=1536),
                    ]
                ),
            ),
            T.MultiHSV(),
            T.MultiCompose(
                [
                    T.MultiToTensor(),
                    T.MultiNormalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            ),
            T.MultiReverseClip(reverse=reverse_clip),
        ]
    )


def transforms_for_eval():
    return T.MultiCompose(
        [
            T.MultiRandomResize(sizes=[800], max_size=1536),
            T.MultiCompose(
                [
                    T.MultiToTensor(),
                    T.MultiNormalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            ),
        ]
    )


def build(config: dict, split: str):
    transform = (
        transforms_for_train(
            coco_size=config["COCO_SIZE"],
            overflow_bbox=config["OVERFLOW_BBOX"],
            reverse_clip=config["REVERSE_CLIP"],
        )
        if split == "train"
        else transforms_for_eval()
    )
    return UADETRAC(config=config, split=split, transform=transform)
