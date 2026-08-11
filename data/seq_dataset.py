# Copyright (c) Ruopeng Gao. All Rights Reserved.
from __future__ import annotations

import os
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _normalize_key(path: str) -> str:
    return path.replace("\\", "/")


def _parse_proposal(item):
    if isinstance(item, str):
        fields = [field for field in item.replace(",", " ").split() if field]
    elif isinstance(item, (list, tuple)):
        fields = list(item)
    else:
        raise TypeError(f"Unsupported proposal item type: {type(item)}")
    if len(fields) != 5:
        raise ValueError(f"Proposal must be x,y,w,h,score, got {fields}")
    return tuple(map(float, fields))


class SeqDataset(Dataset):
    """Single-sequence inference loader used by submit/evaluation engines.

    Compared with the original loader, this version:
      * sorts image frames numerically rather than lexicographically;
      * exposes original numeric ``frame_ids``;
      * supports common image extensions;
      * resolves proposal DB keys relative to data/split/sequence roots instead
        of a machine-specific hard-coded prefix.
    """

    def __init__(self, seq_dir: str, det_db=None):
        self.seq_dir = os.path.abspath(seq_dir)
        self.det_db = (
            {_normalize_key(str(key)): value for key, value in det_db.items()}
            if isinstance(det_db, dict)
            else None
        )

        if "BDD100K" in self.seq_dir:
            image_dir = Path(self.seq_dir)
        else:
            image_dir = Path(self.seq_dir) / "img1"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Sequence image directory does not exist: {image_dir}")

        frame_items = []
        for image_path in image_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            try:
                frame_id = int(image_path.stem)
            except ValueError:
                # Keep BDD100K compatibility for non-numeric frame names.
                frame_id = None
            frame_items.append((frame_id, image_path))

        if not frame_items:
            raise RuntimeError(f"No images found under {image_dir}")

        if all(frame_id is not None for frame_id, _ in frame_items):
            frame_items.sort(key=lambda item: (item[0], item[1].name))
            self.frame_ids = [int(item[0]) for item in frame_items]
        else:
            frame_items.sort(key=lambda item: item[1].name)
            self.frame_ids = list(range(1, len(frame_items) + 1))

        self.image_paths = [str(item[1]) for item in frame_items]

        self.image_height = 800
        self.image_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    @staticmethod
    def load(path):
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _proposal_key_candidates(self, image_path: str):
        path = Path(image_path).resolve()
        seq_root = Path(self.seq_dir).resolve()
        split_root = seq_root.parent
        dataset_root = split_root.parent
        data_root = dataset_root.parent

        candidates = []
        for root in [data_root, dataset_root, split_root, seq_root]:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            candidates.append(_normalize_key(str(relative.with_suffix(".txt"))))

        # UA-DETRAC aliases are accepted because detector databases frequently
        # use a different spelling than the local dataset directory.
        try:
            relative_split = _normalize_key(str(path.relative_to(split_root).with_suffix(".txt")))
            for alias in ("UA-DETRAC", "UADETRAC", "UA_DETRAC"):
                candidates.append(f"{alias}/{split_root.name}/{relative_split}")
        except ValueError:
            pass

        return list(dict.fromkeys(candidates))

    def _get_proposals(self, image_path: str, image_width: int, image_height: int):
        if not self.det_db:
            return torch.zeros((0, 5), dtype=torch.float32)

        items = None
        for key in self._proposal_key_candidates(image_path):
            if key in self.det_db:
                items = self.det_db[key]
                break
        if items is None:
            return torch.zeros((0, 5), dtype=torch.float32)

        proposals = []
        for item in items:
            left, top, width, height, score = _parse_proposal(item)
            if width <= 0 or height <= 0:
                continue
            proposals.append(
                [
                    (left + width / 2.0) / image_width,
                    (top + height / 2.0) / image_height,
                    width / image_width,
                    height / image_height,
                    score,
                ]
            )
        return (
            torch.as_tensor(proposals, dtype=torch.float32).reshape(-1, 5)
            if proposals
            else torch.zeros((0, 5), dtype=torch.float32)
        )

    def process_image(self, image, info, frame_idx=None):
        height, width = image.shape[:2]
        proposals = self._get_proposals(info, width, height)

        ori_image = image.copy()
        scale = self.image_height / min(height, width)
        if max(height, width) * scale > self.image_width:
            scale = self.image_width / max(height, width)
        target_h = int(height * scale)
        target_w = int(width * scale)
        image = cv2.resize(image, (target_w, target_h))
        image = F.normalize(F.to_tensor(image), self.mean, self.std)
        return image, ori_image, proposals

    def __getitem__(self, item):
        image = self.load(self.image_paths[item])
        info = self.image_paths[item]
        return self.process_image(image=image, info=info, frame_idx=self.frame_ids[item]), info

    def __len__(self):
        return len(self.image_paths)
