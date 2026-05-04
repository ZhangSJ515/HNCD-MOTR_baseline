# @Author       : Ruopeng Gao
# @Date         : 2022/8/30
import json
import os
from math import floor
from random import randint
import copy

import torch
from PIL import Image
import data.transforms as T
from data.utils import FixedMotRandomShift

# from typing import List
# from torch.utils.data import Dataset
from .mot import MOTDataset
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


def is_crowd(ann):
    return "extra" in ann and "ignore" in ann["extra"] and ann["extra"]["ignore"] == 1


class DanceTrack(MOTDataset):
    def __init__(self, config: dict, split: str, transform):
        super(DanceTrack, self).__init__(
            config=config, split=split, transform=transform
        )

        self.config = config
        self.transform = transform
        self.dataset_name = config["DATASET"]
        assert split == "train" or split == "test", f"Split {split} is not supported!"
        self.split_dir = os.path.join(config["DATA_ROOT"], self.dataset_name, split)
        assert os.path.exists(self.split_dir), f"Dir {self.split_dir} is not exist."

        # Sampling setting.
        self.sample_steps: list = config["SAMPLE_STEPS"]
        self.sample_intervals: list = config["SAMPLE_INTERVALS"]
        self.sample_modes: list = config["SAMPLE_MODES"]
        self.sample_lengths: list = config["SAMPLE_LENGTHS"]
        self.sample_stage = None
        self.sample_begin_frames = None
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        self.sample_vid_tmax = None

        self.gts = defaultdict(lambda: defaultdict(list))
        self.vid_idx = dict()
        self.idx_vid = dict()

        self.det_db = None
        # NOTE proposal
        self.use_proposals = config.get("USE_PROPOSALS", True)
        if self.use_proposals and config["DET_DB"] != "None":
            with open(config["DET_DB"]) as f:
                self.det_db = json.load(f)

        for vid in os.listdir(self.split_dir):
            gt_path = os.path.join(self.split_dir, vid, "gt", "gt.txt")
            for line in open(gt_path):
                # gt per line: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1
                # https://github.com/DanceTrack/DanceTrack
                t, i, *xywh, a, b, c = line.strip().split(",")[:9]
                t, i, a, b, c = map(int, (t, i, a, b, c))
                x, y, w, h = map(float, xywh)
                assert a == b == c == 1, f"Check Digit ERROR!"
                self.gts[vid][t].append([i, x, y, w, h])

        vids = list(self.gts.keys())

        for vid in vids:
            self.vid_idx[vid] = len(self.vid_idx)
            self.idx_vid[self.vid_idx[vid]] = vid

        # CrowdHuman
        self.ch_dir = Path(config["DATA_ROOT"]) / "CrowdHuman"
        self.ch_indices = []
        if config["APPEND_CROWD"]:
            for line in open(self.ch_dir / f"annotation_trainval.odgt"):
                datum = json.loads(line)
                boxes = [ann["fbox"] for ann in datum["gtboxes"] if not is_crowd(ann)]
                self.ch_indices.append((datum["ID"], boxes))

        self.set_epoch(0)

        return

    def load_crowd(self, index):
        ID, boxes = self.ch_indices[index]
        boxes = copy.deepcopy(boxes)
        img = Image.open(self.ch_dir / "Images" / f"{ID}.jpg")

        w, h = img._size
        n_gts = len(boxes)
        scores = [1.0 for _ in range(len(boxes))]
        if self.det_db is not None and f"crowdhuman/train_image/{ID}.txt" in self.det_db:
            for line in self.det_db[f"crowdhuman/train_image/{ID}.txt"]:
                *box, s = map(float, line.split(","))
                boxes.append(box)
                scores.append(s)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        areas = boxes[..., 2:].prod(-1)
        boxes[:, 2:] += boxes[:, :2]

        target = {
            "boxes": boxes,
            "scores": torch.as_tensor(scores),
            "labels": torch.cat(
                [
                    torch.zeros((n_gts,), dtype=torch.long),
                    torch.full((len(boxes) - n_gts,), -1, dtype=torch.long),
                ]
            ),
            "iscrowd": torch.zeros((n_gts,), dtype=torch.bool),
            "image_id": torch.tensor([0]),
            "area": areas,
            "obj_ids": torch.cat(
                [
                    torch.arange(n_gts),
                    torch.full((len(boxes) - n_gts,), -1, dtype=torch.long),
                ]
            ),
            "size": torch.as_tensor([h, w]),
            "orig_size": torch.as_tensor([h, w]),
            "dataset": "CrowdHuman",
        }
        rs = FixedMotRandomShift(self.sample_length)
        return rs([img], [target])

    def __getitem__(self, item):
        # item = 50000
        if item < len(self.sample_begin_frames):
            vid, begin_frame = self.sample_begin_frames[item]
            frame_idxs = self.sample_frames_idx(vid=vid, begin_frame=begin_frame)
            imgs, infos = self.get_multi_frames(vid=vid, idxs=frame_idxs)
        else:
            imgs, infos = self.load_crowd(item - len(self.sample_begin_frames))
            for info in infos:
                info["areas"] = info.pop("area")
                info["ids"] = info.pop("obj_ids")
                info["frame_idx"] = info.pop("image_id")
                info["unnorm_img"] = info.pop("orig_size")
        if self.transform is not None:
            imgs, infos = self.transform(imgs, infos)

        # NOTE proposal
        proposals = []
        gt_infos = []
        for target_i in infos:
            n_gt = (target_i["labels"] == 0).sum().item()
            # print(target_i['boxes'][n_gt:].shape, target_i['scores'][n_gt:, None].shape)

            proposals.append(
                torch.cat(
                    [target_i["boxes"][n_gt:], target_i["scores"][n_gt:, None]], dim=1
                )
            )
            gt_infos.append(
                {
                    "labels": target_i["labels"][:n_gt],
                    "ids": target_i["ids"][:n_gt],
                    "boxes": target_i["boxes"][:n_gt],
                    "areas": target_i["areas"][:n_gt],
                    "frame_idx": target_i["frame_idx"],
                    "unnorm_img": target_i["unnorm_img"],
                }
            )
        # print(self.gts)
        return {"imgs": imgs, "infos": gt_infos, "proposals": proposals}

    def __len__(self):
        assert (
            self.sample_begin_frames is not None
        ), "Please use set_epoch to init DanceTrack Dataset."
        return len(self.sample_begin_frames) + len(self.ch_indices)

    def sample_frames_idx(self, vid: int, begin_frame: int) -> list[int]:
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample length is less than 2."
            remain_frames = self.sample_vid_tmax[vid] - begin_frame
            max_interval = floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            frame_idxs = [begin_frame + interval * i for i in range(self.sample_length)]
            return frame_idxs
        else:
            raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

    def set_epoch(self, epoch: int):
        self.sample_begin_frames = list()
        self.sample_vid_tmax = dict()
        self.sample_stage = 0
        for step in self.sample_steps:
            if epoch >= step:
                self.sample_stage += 1
        assert self.sample_stage < len(self.sample_steps) + 1
        self.sample_length = self.sample_lengths[
            min(len(self.sample_lengths) - 1, self.sample_stage)
        ]
        self.sample_mode = self.sample_modes[
            min(len(self.sample_modes) - 1, self.sample_stage)
        ]
        self.sample_interval = self.sample_intervals[
            min(len(self.sample_intervals) - 1, self.sample_stage)
        ]
        for vid in self.vid_idx.keys():
            t_min = min(self.gts[vid].keys())
            t_max = max(self.gts[vid].keys())
            self.sample_vid_tmax[vid] = t_max
            for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                self.sample_begin_frames.append((vid, t))

        return

    def get_single_frame(self, vid: str, idx: int):
        img_path = os.path.join(
            self.split_dir,
            vid,
            "img1",
            f"{idx:08d}.jpg" if self.dataset_name == "DanceTrack" else f"{idx:06d}.jpg",
        )
        img = Image.open(img_path)
        info = {}
        ids_offset = self.vid_idx[vid] * 100000

        # 真值：
        info["boxes"] = list()
        info["ids"] = list()
        info["labels"] = list()
        info["areas"] = list()
        info["frame_idx"] = torch.as_tensor(idx)
        info["scores"] = list()

        for i, *xywh in self.gts[vid][idx]:
            info["boxes"].append(list(map(float, xywh)))
            info["areas"].append(xywh[2] * xywh[3])  # area = w * h
            info["ids"].append(i + ids_offset)
            info["labels"].append(0)  # DanceTrack, all people.
            info["scores"].append(1.0)

        # NOTE proposal
        if self.det_db is not None:
            txt_key = os.path.join(
                self.dataset_name,
                self.split_dir.split("/")[-1],
                vid,
                "img1",
                f"{idx:08d}.txt" if self.dataset_name == 'DanceTrack' else f"{idx:06d}.txt",
            )
            for line in self.det_db[txt_key]:
                *box, s = map(float, line.split(","))
                info["boxes"].append(box)
                info["scores"].append(s)
                info["ids"].append(-1)
                info["labels"].append(-1)
                info["areas"].append(box[2] * box[3])

        info["boxes"] = torch.as_tensor(info["boxes"])
        info["areas"] = torch.as_tensor(info["areas"])
        info["ids"] = torch.as_tensor(info["ids"])
        info["labels"] = torch.as_tensor(info["labels"])
        info["scores"] = torch.as_tensor(info["scores"])
        # xywh to x1y1x2y2
        if len(info["boxes"]) > 0:
            info["boxes"][:, 2:] += info["boxes"][:, :2]
        else:
            info["boxes"] = torch.zeros((0, 4))
            info["ids"] = torch.zeros((0,), dtype=torch.long)
            info["labels"] = torch.zeros((0,), dtype=torch.long)

        return img, info

    def get_multi_frames(self, vid: str, idxs: list[int]):
        return zip(*[self.get_single_frame(vid=vid, idx=i) for i in idxs])


def transfroms_for_train(coco_size: bool = False, overflow_bbox: bool = False, reverse_clip: bool = False):
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]  # from MOTR
    return T.MultiCompose([
        T.MultiRandomHorizontalFlip(),
        T.MultiRandomSelect(
            T.MultiRandomResize(sizes=scales, max_size=1536),
            T.MultiCompose([
                T.MultiRandomResize([400, 500, 600] if coco_size else [800, 1000, 1200]),
                T.MultiRandomCrop(
                    min_size=384 if coco_size else 800,
                    max_size=600 if coco_size else 1200,
                    overflow_bbox=overflow_bbox
                ),
                T.MultiRandomResize(sizes=scales, max_size=1536)
            ])
        ),
        T.MultiHSV(),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        T.MultiReverseClip(reverse=reverse_clip)
    ])


def transforms_for_eval():
    return T.MultiCompose([
        T.MultiRandomResize(sizes=[800], max_size=1333),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ])


def build(config: dict, split: str):
    if split == "train":
        return DanceTrack(
            config=config,
            split=split,
            transform=transfroms_for_train(
                coco_size=config["COCO_SIZE"],
                overflow_bbox=config["OVERFLOW_BBOX"],
                reverse_clip=config["REVERSE_CLIP"]
            )
        )
    elif split == "test":
        return DanceTrack(config=config, split=split, transform=transforms_for_eval())
    else:
        raise ValueError(f"Data split {split} is not supported for DanceTrack dataset.")
