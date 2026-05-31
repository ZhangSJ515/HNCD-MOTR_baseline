# @Author       : Based on DanceTrack implementation
# @Description  : AICity22 Dataset for vehicle tracking

import json
import os
import cv2
from math import floor
from random import randint

import torch
from PIL import Image
import data.transforms as T

from .mot import MOTDataset
from collections import defaultdict

# Suppress FFmpeg warnings for MSMPEG4v2 codec
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
cv2.setLogLevel(0)


class AICity22(MOTDataset):
    def __init__(self, config: dict, split: str, transform):
        super(AICity22, self).__init__(
            config=config, split=split, transform=transform
        )

        self.config = config
        self.transform = transform
        self.dataset_name = config["DATASET"]
        assert split == "train" or split == "val" or split == "test", f"Split {split} is not supported!"

        # AICity22 dataset path
        self.data_root = os.path.join(config["DATA_ROOT"], "AICity22_Track1_MTMC_Tracking")
        if split == "val":
            self.split_dir = os.path.join(self.data_root, "validation")
        else:
            self.split_dir = os.path.join(self.data_root, split)
        assert os.path.exists(self.split_dir), f"Dir {self.split_dir} is not exist."

        # Sampling setting
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
        self.vid_idx = dict()  # vid (scenario/camera) -> idx
        self.idx_vid = dict()  # idx -> vid

        # Video captures cache
        self.video_caps = {}
        # Video info: {vid: {"path": str, "total_frames": int}}
        self.video_info = {}

        self.det_db = None
        # NOTE proposal
        if config["DET_DB"] != "None":
            with open(config["DET_DB"]) as f:
                self.det_db = json.load(f)

        # Parse dataset structure: scenario/camera
        for scenario in os.listdir(self.split_dir):
            scenario_path = os.path.join(self.split_dir, scenario)
            if not os.path.isdir(scenario_path):
                continue

            for camera in os.listdir(scenario_path):
                camera_path = os.path.join(scenario_path, camera)
                if not os.path.isdir(camera_path):
                    continue

                gt_path = os.path.join(camera_path, "gt", "gt.txt")
                video_path = os.path.join(camera_path, "vdo.avi")

                if not os.path.exists(gt_path) or not os.path.exists(video_path):
                    continue

                vid = f"{scenario}/{camera}"

                # Store video info
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                self.video_info[vid] = {
                    "path": video_path,
                    "total_frames": total_frames
                }

                # Parse ground truth
                # Format: frame, ID, left, top, width, height, 1, -1, -1, -1
                for line in open(gt_path):
                    parts = line.strip().split(",")[:10]
                    t, i = int(parts[0]), int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    self.gts[vid][t].append([i, x, y, w, h])

        vids = list(self.gts.keys())
        for vid in vids:
            self.vid_idx[vid] = len(self.vid_idx)
            self.idx_vid[self.vid_idx[vid]] = vid

        self.set_epoch(0)
        return

    def __getitem__(self, item):
        vid, begin_frame = self.sample_begin_frames[item]
        frame_idxs = self.sample_frames_idx(vid=vid, begin_frame=begin_frame)
        imgs, infos = self.get_multi_frames(vid=vid, idxs=frame_idxs)

        if self.transform is not None:
            imgs, infos = self.transform(imgs, infos)

        # NOTE proposal
        proposals = []
        gt_infos = []
        for target_i in infos:
            n_gt = (target_i["labels"] == 0).sum().item()
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
        return {"imgs": imgs, "infos": gt_infos, "proposals": proposals}

    def __len__(self):
        assert (
            self.sample_begin_frames is not None
        ), "Please use set_epoch to init AICity22 Dataset."
        return len(self.sample_begin_frames)

    def sample_frames_idx(self, vid: str, begin_frame: int) -> list[int]:
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
            if len(self.gts[vid]) == 0:
                continue
            t_min = min(self.gts[vid].keys())
            t_max = max(self.gts[vid].keys())
            self.sample_vid_tmax[vid] = t_max
            for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                self.sample_begin_frames.append((vid, t))
        return

    def _get_video_cap(self, vid: str):
        """Get or create video capture for the given video id."""
        if vid not in self.video_caps:
            video_path = self.video_info[vid]["path"]
            self.video_caps[vid] = cv2.VideoCapture(video_path)
        return self.video_caps[vid]

    def get_single_frame(self, vid: str, idx: int):
        """Read a single frame from video and its annotations."""
        # Read frame from video
        cap = self._get_video_cap(vid)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)  # Frame index is 1-based in gt.txt
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError(f"Failed to read frame {idx} from video {vid}")

        # Convert BGR to RGB and create PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        info = {}
        ids_offset = self.vid_idx[vid] * 100000

        info["boxes"] = list()
        info["ids"] = list()
        info["labels"] = list()
        info["areas"] = list()
        info["frame_idx"] = torch.as_tensor(idx)
        info["scores"] = list()

        for i, *xywh in self.gts[vid][idx]:
            info["boxes"].append(list(map(float, xywh)))
            info["areas"].append(xywh[2] * xywh[3])
            info["ids"].append(i + ids_offset)
            info["labels"].append(0)  # All vehicles
            info["scores"].append(1.0)

        # NOTE proposal: use GT as proposals (simulating detection results)
        # This adds GT boxes again with labels=-1 to be used as proposals
        if self.det_db is not None:
            scenario, camera = vid.split("/")
            # Key format: AICity22/split/scenario/camera/frame.txt
            split_name = "val" if "validation" in self.split_dir else self.split_dir.split("/")[-1]
            txt_key = f"AICity22/{split_name}/{scenario}/{camera}/{idx:06d}.txt"
            if txt_key in self.det_db:
                for line in self.det_db[txt_key]:
                    *box, s = map(float, line.strip().split(","))
                    info["boxes"].append(box)
                    info["scores"].append(s)
                    info["ids"].append(-1)
                    info["labels"].append(-1)
                    info["areas"].append(box[2] * box[3])
        else:
            # Use GT as proposals when no external detection DB
            for i, *xywh in self.gts[vid][idx]:
                info["boxes"].append(list(map(float, xywh)))
                info["areas"].append(xywh[2] * xywh[3])
                info["ids"].append(-1)  # -1 indicates proposal
                info["labels"].append(-1)  # -1 indicates proposal
                info["scores"].append(1.0)

        info["boxes"] = torch.as_tensor(info["boxes"]) if info["boxes"] else torch.zeros((0, 4))
        info["areas"] = torch.as_tensor(info["areas"]) if info["areas"] else torch.zeros((0,))
        info["ids"] = torch.as_tensor(info["ids"]) if info["ids"] else torch.zeros((0,), dtype=torch.long)
        info["labels"] = torch.as_tensor(info["labels"]) if info["labels"] else torch.zeros((0,), dtype=torch.long)
        info["scores"] = torch.as_tensor(info["scores"]) if info["scores"] else torch.zeros((0,))

        # xywh to x1y1x2y2
        if len(info["boxes"]) > 0:
            info["boxes"][:, 2:] += info["boxes"][:, :2]

        return img, info

    def get_multi_frames(self, vid: str, idxs: list[int]):
        return zip(*[self.get_single_frame(vid=vid, idx=i) for i in idxs])

    def __del__(self):
        """Release all video captures when dataset is deleted."""
        for cap in self.video_caps.values():
            if cap is not None:
                cap.release()


def transfroms_for_train(coco_size: bool = False, overflow_bbox: bool = False, reverse_clip: bool = False):
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
        return AICity22(
            config=config,
            split=split,
            transform=transfroms_for_train(
                coco_size=config["COCO_SIZE"],
                overflow_bbox=config["OVERFLOW_BBOX"],
                reverse_clip=config["REVERSE_CLIP"]
            )
        )
    elif split == "val" or split == "test":
        return AICity22(config=config, split=split, transform=transforms_for_eval())
    else:
        raise ValueError(f"Data split {split} is not supported for AICity22 dataset.")
