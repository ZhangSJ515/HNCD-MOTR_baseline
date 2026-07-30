from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image

import data.transforms as T
from .mot import MOTDataset


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _split_fields(line: str) -> List[str]:
    return [x for x in re.split(r"[,\s]+", line.strip()) if x]


def _normalize_key(path: str) -> str:
    return path.replace("\\", "/")


def _parse_proposal_line(line) -> Tuple[float, float, float, float, float]:
    if isinstance(line, str):
        parts = _split_fields(line)
    elif isinstance(line, (list, tuple)):
        parts = list(line)
    else:
        raise TypeError(f"Unsupported proposal item type: {type(line)}")
    if len(parts) != 5:
        raise ValueError(f"Proposal must be x,y,w,h,score, got: {parts}")
    return tuple(map(float, parts))


class AirMot(MOTDataset):
    """AirMOT adapter for HNCD-MOTR.

    Expected GT format per line:
        frame_id track_id class_id x y w h flag

    Class IDs:
        0 airplane
        1 person
        2 baggage_tug
        3 follow_me_vehicle
    """

    def __init__(self, config: dict, split: str, transform):
        super().__init__(config=config, split=split, transform=transform)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported AirMOT split: {split}")

        self.config = config
        self.transform = transform
        self.dataset_name = config["DATASET"]
        self.data_root = os.path.abspath(config["DATA_ROOT"])
        self.split_dir = os.path.join(self.data_root, self.dataset_name, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(self.split_dir)

        self.filter_gt_by_flag = bool(config.get("FILTER_GT_BY_FLAG", False))
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
            with open(det_db_path, "r", encoding="utf-8") as f:
                self.det_db = json.load(f)

        self._load_sequences()
        self.set_epoch(0)

    def _load_sequences(self) -> None:
        seq_names = sorted(
            name for name in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, name))
        )
        for seq_name in seq_names:
            seq_dir = os.path.join(self.split_dir, seq_name)
            img_dir = os.path.join(seq_dir, "img1")
            gt_path = os.path.join(seq_dir, "gt", "gt.txt")
            if not os.path.isdir(img_dir) or not os.path.isfile(gt_path):
                continue

            for image_name in os.listdir(img_dir):
                stem, ext = os.path.splitext(image_name)
                if ext.lower() in _IMAGE_EXTENSIONS and stem.isdigit():
                    self.frame_paths[seq_name][int(stem)] = os.path.join(img_dir, image_name)

            with open(gt_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    parts = _split_fields(line)
                    if len(parts) < 8:
                        raise ValueError(f"Invalid AirMOT GT {gt_path}:{line_no}: {line.strip()}")
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    class_id = int(float(parts[2]))
                    x, y, w, h = map(float, parts[3:7])
                    flag = float(parts[7])
                    if class_id not in {0, 1, 2, 3}:
                        raise ValueError(f"Invalid AirMOT class {class_id} at {gt_path}:{line_no}")
                    if self.filter_gt_by_flag and flag <= 0:
                        continue
                    if w <= 0 or h <= 0:
                        continue
                    self.gts[seq_name][frame_id].append([track_id, class_id, x, y, w, h])

            valid_frames = sorted(
                set(self.frame_paths[seq_name].keys()) & set(self.gts[seq_name].keys())
            )
            if not valid_frames:
                self.gts.pop(seq_name, None)
                self.frame_paths.pop(seq_name, None)
                continue
            self.frame_ids[seq_name] = valid_frames
            self.vid_idx[seq_name] = len(self.vid_idx)
            self.idx_vid[self.vid_idx[seq_name]] = seq_name

        if not self.vid_idx:
            raise RuntimeError(f"No valid AirMOT sequences found under {self.split_dir}")

    def _proposal_key_candidates(self, image_path: str) -> List[str]:
        rel_data_root = _normalize_key(os.path.relpath(image_path, self.data_root))
        rel_split_root = _normalize_key(os.path.relpath(image_path, self.split_dir))
        candidates = [
            os.path.splitext(rel_data_root)[0] + ".txt",
            os.path.splitext(rel_split_root)[0] + ".txt",
        ]
        if rel_data_root.startswith("AirMot/"):
            candidates.append(
                os.path.splitext("AirMOT/" + rel_data_root[len("AirMot/"):])[0] + ".txt"
            )
        return list(dict.fromkeys(candidates))

    def _get_proposals(self, image_path: str):
        if self.det_db is None:
            return []
        lines = None
        for key in self._proposal_key_candidates(image_path):
            if key in self.det_db:
                lines = self.det_db[key]
                break
        if lines is None:
            return []
        proposals = []
        for line in lines:
            x, y, w, h, score = _parse_proposal_line(line)
            if w > 0 and h > 0:
                proposals.append((x, y, w, h, score))
        return proposals

    def __len__(self):
        return len(self.sample_begin_frames)

    def set_epoch(self, epoch: int):
        self.sample_stage = sum(epoch >= step for step in self.sample_steps)
        self.sample_length = self.sample_lengths[min(self.sample_stage, len(self.sample_lengths) - 1)]
        self.sample_mode = self.sample_modes[min(self.sample_stage, len(self.sample_modes) - 1)]
        self.sample_interval = self.sample_intervals[min(self.sample_stage, len(self.sample_intervals) - 1)]

        self.sample_begin_frames = []
        for seq_name, frame_ids in self.frame_ids.items():
            for start_pos in range(max(0, len(frame_ids) - self.sample_length + 1)):
                self.sample_begin_frames.append((seq_name, start_pos))

    def sample_frames_idx(self, vid: str, begin_pos: int) -> List[int]:
        if self.sample_mode != "random_interval":
            raise ValueError(f"Unsupported sample mode: {self.sample_mode}")
        frame_ids = self.frame_ids[vid]
        if self.sample_length <= 1:
            return [frame_ids[begin_pos]]
        remain = len(frame_ids) - 1 - begin_pos
        max_interval = max(1, min(self.sample_interval, remain // (self.sample_length - 1)))
        interval = random.randint(1, max_interval)
        return [frame_ids[begin_pos + interval * i] for i in range(self.sample_length)]

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
        for track_id, class_id, x, y, w, h in self.gts[vid][idx]:
            info["boxes"].append([x, y, w, h])
            info["ids"].append(track_id + id_offset)
            info["labels"].append(class_id)
            info["areas"].append(w * h)
            info["scores"].append(1.0)

        for x, y, w, h, score in self._get_proposals(image_path):
            info["boxes"].append([x, y, w, h])
            info["ids"].append(-1)
            info["labels"].append(-1)
            info["areas"].append(w * h)
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
        vid, begin_pos = self.sample_begin_frames[item]
        frame_ids = self.sample_frames_idx(vid, begin_pos)
        images, infos = self.get_multi_frames(vid, frame_ids)
        if self.transform is not None:
            images, infos = self.transform(images, infos)

        proposals = []
        gt_infos = []
        for target in infos:
            num_gt = int((target["labels"] >= 0).sum().item())
            proposals.append(
                torch.cat([target["boxes"][num_gt:], target["scores"][num_gt:, None]], dim=1)
            )
            gt_infos.append({
                "labels": target["labels"][:num_gt],
                "ids": target["ids"][:num_gt],
                "boxes": target["boxes"][:num_gt],
                "areas": target["areas"][:num_gt],
                "frame_idx": target["frame_idx"],
                "unnorm_img": target["unnorm_img"],
            })
        return {"imgs": images, "infos": gt_infos, "proposals": proposals}


def transforms_for_train(coco_size=False, overflow_bbox=False, reverse_clip=0.0):
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]
    return T.MultiCompose([
        T.MultiRandomHorizontalFlip(),
        T.MultiRandomSelect(
            T.MultiRandomResize(sizes=scales, max_size=1536),
            T.MultiCompose([
                T.MultiRandomResize([400, 500, 600] if coco_size else [800, 1000, 1200]),
                T.MultiRandomCrop(
                    min_size=384 if coco_size else 800,
                    max_size=600 if coco_size else 1200,
                    overflow_bbox=overflow_bbox,
                ),
                T.MultiRandomResize(sizes=scales, max_size=1536),
            ]),
        ),
        T.MultiHSV(),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        T.MultiReverseClip(reverse=reverse_clip),
    ])


def transforms_for_eval():
    return T.MultiCompose([
        T.MultiRandomResize(sizes=[800], max_size=1333),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ])


def build(config: dict, split: str):
    transform = transforms_for_train(
        coco_size=config["COCO_SIZE"],
        overflow_bbox=config["OVERFLOW_BBOX"],
        reverse_clip=config["REVERSE_CLIP"],
    ) if split == "train" else transforms_for_eval()
    return AirMot(config=config, split=split, transform=transform)
