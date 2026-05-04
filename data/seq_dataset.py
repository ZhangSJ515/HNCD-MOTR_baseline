# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import cv2

import torchvision.transforms.functional as F

from torch.utils.data import Dataset
import torch


class SeqDataset(Dataset):
    def __init__(self, seq_dir: str, det_db):
        self.seq_dir = seq_dir
        self.det_db = det_db

        if "BDD100K" in seq_dir:
            image_paths = sorted(os.listdir(seq_dir))
            image_paths = [os.path.join(seq_dir, _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
            self.image_paths = image_paths
        else:
            image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
            image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
            self.image_paths = image_paths

        self.image_height = 800
        self.image_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        return

    @staticmethod
    def load(path):
        image = cv2.imread(path)
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def process_image(self, image, info, frame_idx=None):
        proposals = []
        h, w = image.shape[:2]

        prefix = '/home/y23_zhanghaoyu/subject/datasets/'
        key = info[len(prefix):].replace('.jpg', '.txt') if len(info) > len(prefix) else ""

        if self.det_db and key in self.det_db:
            for line in self.det_db[key]:
                l, t, p_w, p_h, s = list(map(float, line.strip().split(',')))
                proposals.append([(l + p_w / 2) / w,
                                  (t + p_h / 2) / h,
                                  p_w / w,
                                  p_h / h,
                                  s])
        proposals = torch.as_tensor(proposals).reshape(-1, 5) if proposals else torch.zeros((0, 5))
        ori_image = image.copy()
        scale = self.image_height / min(h, w)
        if max(h, w) * scale > self.image_width:
            scale = self.image_width / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        image = cv2.resize(image, (target_w, target_h))
        image = F.normalize(F.to_tensor(image), self.mean, self.std)
        return image, ori_image, proposals

    def __getitem__(self, item):
        image = self.load(self.image_paths[item])
        info = self.image_paths[item]
        return self.process_image(image=image, info=info), info

    def __len__(self):
        return len(self.image_paths)
