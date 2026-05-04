import random
import torch
import cv2
import copy
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import PIL.Image
from math import floor
from utils.box_ops import box_xyxy_to_cxcywh


def filter_by_keep_idxs(info: dict, keep_idxs: torch.Tensor) -> dict:
    for field in ["boxes", "scores", "labels", "ids"]:
        if field in info:
            # if len(keep_idxs) != len(info[field]):
            #     print('not equals')
            info[field] = info[field][keep_idxs]
    return info

class MultiCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, infos):
        for t in self.transforms:
            images, infos = t(images, infos)
        return images, infos


class MultiRandomSelect:
    def __init__(self, transform1, transform2, p: float = 0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(self, imgs, infos):
        if random.random() < self.p:
            return self.transform1(imgs, infos)
        return self.transform2(imgs, infos)


class MultiRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, imgs: list, infos: list):
        if random.random() < self.p:
            def hflip(img: PIL.Image, info: dict):
                img = F.hflip(img)
                w, h = img.size
                if len(info["boxes"]) > 0:
                    info["boxes"] = (info["boxes"][:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1])
                                     + torch.as_tensor([w, 0, w, 0]))
                return img, info

            imgs, infos = zip(*[hflip(img, info) for img, info in zip(imgs, infos)])
        return imgs, infos


class MultiRandomResize:
    def __init__(self, sizes: list | tuple, max_size=None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, imgs, infos):
        new_size = random.choice(self.sizes)

        def resize(img, info, size, max_size):
            def get_new_hw(current_wh, new_short_size, new_long_max_size):
                w, h = current_wh
                if new_long_max_size is not None:
                    min_wh, max_wh = float(min(w, h)), float(max(w, h))
                    if max_wh / min_wh * new_short_size > new_long_max_size:
                        new_short_size = int(floor(new_long_max_size * min_wh / max_wh))
                if w < h:
                    new_w = new_short_size
                    new_h = int(round(new_short_size * h / w))
                else:
                    new_h = new_short_size
                    new_w = int(round(new_short_size * w / h))
                return new_h, new_w

            if isinstance(size, (list, tuple)):
                new_hw = size[::-1]
            else:
                new_hw = get_new_hw(img.size, size, max_size)
            resized_img = F.resize(img, new_hw)
            ratio_w, ratio_h = (float(s) / float(origin_s) for s, origin_s in zip(resized_img.size, img.size))

            if len(info["boxes"]) > 0:
                info["boxes"] = info["boxes"] * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
                info["areas"] = info["areas"] * ratio_w * ratio_h
            return resized_img, info

        return zip(*[resize(img, info, new_size, self.max_size) for img, info in zip(imgs, infos)])


class MultiToTensor:
    def __call__(self, imgs, infos):
        tensor_imgs = list(map(F.to_tensor, imgs))
        return tensor_imgs, infos


class MultiNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs, infos):
        def normalize(img: torch.Tensor, info):
            info["unnorm_img"] = img.clone()
            img = F.normalize(img, mean=self.mean, std=self.std)
            h, w = img.shape[-2:]
            if len(info["boxes"]) > 0:
                info["boxes"] = box_xyxy_to_cxcywh(info["boxes"])
                info["boxes"] = info["boxes"] / torch.as_tensor([w, h, w, h])
            return img, info

        return zip(*[normalize(img, info) for img, info in zip(imgs, infos)])


class MultiRandomCrop:
    def __init__(self, min_size: int, max_size: int, overflow_bbox: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.overflow_bbox = overflow_bbox

    def __call__(self, imgs, infos):
        crop_w = random.randint(self.min_size, min(imgs[0].width, self.max_size))
        crop_h = random.randint(self.min_size, min(imgs[0].height, self.max_size))
        crop_ijhw = T.RandomCrop.get_params(imgs[0], (crop_h, crop_w))

        def crop(img, info, ijhw):
            cropped_img = F.crop(img, *ijhw)
            i, j, h, w = ijhw
            if len(info["boxes"]) > 0:
                info["boxes"] = info["boxes"] - torch.as_tensor([j, i, j, i])
                max_wh = torch.as_tensor([w, h])
                boxes = info["boxes"].clone().reshape(-1, 2, 2)
                boxes = torch.min(boxes, max_wh)
                boxes = boxes.clamp(min=0)
                keep_idxs = torch.all(boxes[:, 1, :] > boxes[:, 0, :], dim=1)
                info["boxes"] = boxes.reshape(-1, 4)
                info = filter_by_keep_idxs(info, keep_idxs)
            return cropped_img, info

        return zip(*[crop(img, info, crop_ijhw) for img, info in zip(imgs, infos)])


class MultiRandomShift:
    def __init__(self, max_shift: int = 50):
        self.max_shift = max_shift

    def __call__(self, imgs: list, infos: list):
        res_imgs, res_infos = [], []
        n_frames = len(imgs)
        w, h = imgs[0].size
        x_shift = (self.max_shift * torch.rand(1)).ceil()
        x_shift *= (torch.randn(1) > 0.0).int() * 2 - 1
        y_shift = (self.max_shift * torch.rand(1)).ceil()
        y_shift *= (torch.randn(1) > 0.0).int() * 2 - 1
        res_imgs.append(imgs[0])
        res_infos.append(infos[0])

        def shift(image, info, region, target_h, target_w):
            cropped_image = F.crop(image, *region)
            cropped_image = F.resize(cropped_image, [target_h, target_w])
            top, left, height, width = region
            if "boxes" in info:
                info["boxes"] = info["boxes"] - torch.as_tensor([left, top, left, top])
                scale = torch.as_tensor([target_w / width, target_h / height] * 2)
                info["boxes"] *= scale
                max_wh = torch.as_tensor([target_w, target_h])
                boxes = info["boxes"].clone().reshape(-1, 2, 2)
                boxes = torch.min(boxes, max_wh)
                boxes = boxes.clamp(min=0)
                keep_idxs = torch.all(boxes[:, 1, :] > boxes[:, 0, :], dim=1)
                info["boxes"] = boxes.reshape(-1, 4)
                info = filter_by_keep_idxs(info, keep_idxs)
            return cropped_image, info

        for i in range(1, n_frames):
            y_min = max(0, -y_shift[0].item())
            y_max = min(h, h - y_shift[0].item())
            x_min = max(0, -x_shift[0].item())
            x_max = max(w, w - x_shift[0].item())
            prev_img = res_imgs[i - 1].copy()
            prev_info = copy.deepcopy(res_infos[i - 1])
            shift_region = (int(y_min), int(x_min), int(y_max - y_min), int(x_max - x_min))
            img_i, info_i = shift(prev_img, prev_info, shift_region, h, w)
            res_imgs.append(img_i)
            res_infos.append(info_i)

        if torch.randn(1)[0].item() > 0:
            res_imgs.reverse()
            res_infos.reverse()

        return res_imgs, res_infos


class MultiHSV:
    def __init__(self, hgain=5, sgain=30, vgain=30):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, imgs, infos):
        hsv_augs = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
        hsv_augs *= np.random.randint(0, 2, 3)
        hsv_augs = hsv_augs.astype(np.int16)

        def hsv(img, info):
            img = np.array(img)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
            img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
            img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)
            return cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2RGB), info

        return zip(*[hsv(img, info) for img, info in zip(imgs, infos)])


class MultiReverseClip:
    def __init__(self, reverse: bool = False):
        self.reverse = reverse

    def __call__(self, imgs, infos):
        if random.random() < self.reverse:
            imgs = list(imgs)
            infos = list(infos)
            imgs.reverse()
            infos.reverse()
        return imgs, infos
