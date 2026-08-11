from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from ._base_dataset import _BaseDataset
from .. import _timing, utils
from ..utils import TrackEvalException


class UADETRAC(_BaseDataset):
    """TrackEval adapter for converted UA-DETRAC sequences.

    Ground-truth rows may use standard MOTChallenge format or the compact
    layouts supported by ``data/uadetrac.py``. Tracker rows use standard
    MOTChallenge output or ``frame,id,x,y,w,h,score``.
    """

    @staticmethod
    def get_default_dataset_config():
        code_path = utils.get_code_path()
        return {
            "GT_FOLDER": os.path.join(code_path, "data/gt/uadetrac/test"),
            "TRACKERS_FOLDER": os.path.join(code_path, "data/trackers/uadetrac"),
            "OUTPUT_FOLDER": None,
            "TRACKERS_TO_EVAL": None,
            "CLASSES_TO_EVAL": ["vehicle"],
            "SPLIT_TO_EVAL": "test",
            "PRINT_CONFIG": True,
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_SUB_FOLDER": "",
            "TRACKER_DISPLAY_NAMES": None,
            "SEQMAP_FILE": None,
            "FILTER_GT_BY_MARK": True,
            "MIN_VISIBILITY": 0.0,
            "GT_FORMAT": "auto",
        }

    def __init__(self, config=None):
        super().__init__()
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.gt_fol = self.config["GT_FOLDER"]
        self.tracker_fol = self.config["TRACKERS_FOLDER"]
        self.output_fol = self.config["OUTPUT_FOLDER"] or self.tracker_fol
        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]
        self.filter_gt_by_mark = self.config["FILTER_GT_BY_MARK"]
        self.min_visibility = float(self.config["MIN_VISIBILITY"])
        self.gt_format = self.config["GT_FORMAT"]

        self.should_classes_combine = False
        self.use_super_categories = False
        self.valid_classes = ["vehicle"]
        self.class_name_to_class_id = {"vehicle": 0}
        self.class_list = [
            cls.lower() if cls.lower() in self.valid_classes else None
            for cls in self.config["CLASSES_TO_EVAL"]
        ]
        if not all(self.class_list):
            raise TrackEvalException(
                f"Invalid UA-DETRAC class. Valid classes are {self.valid_classes}."
            )

        self.seq_list = self._get_sequence_list()
        if not self.seq_list:
            raise TrackEvalException(
                f"No UA-DETRAC sequences were found in GT folder: {self.gt_fol}"
            )
        self.seq_lengths = {seq: self._infer_sequence_length(seq) for seq in self.seq_list}

        self.tracker_list = self._get_tracker_list()
        if self.config["TRACKER_DISPLAY_NAMES"] is None:
            self.tracker_to_disp = {
                tracker: (tracker if tracker else "HNCD-MOTR")
                for tracker in self.tracker_list
            }
        elif len(self.config["TRACKER_DISPLAY_NAMES"]) == len(self.tracker_list):
            self.tracker_to_disp = dict(
                zip(self.tracker_list, self.config["TRACKER_DISPLAY_NAMES"])
            )
        else:
            raise TrackEvalException("Tracker files and tracker display names do not match.")

        for tracker in self.tracker_list:
            for seq in self.seq_list:
                tracker_file = self._tracker_file(tracker, seq)
                if not os.path.isfile(tracker_file):
                    raise TrackEvalException(f"Tracker file not found: {tracker_file}")

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_sequence_list(self):
        seqmap = self.config.get("SEQMAP_FILE")
        if seqmap and str(seqmap).lower() != "none":
            if not os.path.isfile(seqmap):
                raise TrackEvalException(f"Sequence map does not exist: {seqmap}")
            names = []
            with open(seqmap, "r", encoding="utf-8") as file:
                for line_index, line in enumerate(file):
                    name = line.strip().split(",")[0]
                    if line_index == 0 and name.lower() in {"name", "seqname"}:
                        continue
                    if name:
                        names.append(name)
            return names

        root = Path(self.gt_fol)
        if not root.is_dir():
            raise TrackEvalException(f"GT folder does not exist: {root}")
        return sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and (path / "gt" / "gt.txt").is_file()
        )

    def _infer_sequence_length(self, seq):
        sequence_dir = Path(self.gt_fol) / seq
        image_max_frame = 0
        image_dir = sequence_dir / "img1"
        if image_dir.is_dir():
            for image_path in image_dir.iterdir():
                if not image_path.is_file():
                    continue
                try:
                    image_max_frame = max(image_max_frame, int(image_path.stem))
                except ValueError:
                    continue

        gt_file = sequence_dir / "gt" / "gt.txt"
        gt_max_frame = 0
        with gt_file.open("r", encoding="utf-8") as file:
            for line in file:
                fields = self._split_fields(line)
                if fields:
                    gt_max_frame = max(gt_max_frame, int(float(fields[0])))

        max_frame = max(image_max_frame, gt_max_frame)
        if max_frame <= 0:
            raise TrackEvalException(f"No valid frames found in {sequence_dir}")
        return max_frame

    def _get_tracker_list(self):
        configured = self.config["TRACKERS_TO_EVAL"]
        if configured is not None:
            return list(configured)

        if all(
            os.path.isfile(os.path.join(self.tracker_fol, seq + ".txt"))
            for seq in self.seq_list
        ):
            return [""]

        return sorted(
            name
            for name in os.listdir(self.tracker_fol)
            if os.path.isdir(os.path.join(self.tracker_fol, name))
        )

    def _tracker_file(self, tracker, seq):
        return os.path.join(
            self.tracker_fol,
            tracker,
            self.tracker_sub_fol,
            seq + ".txt",
        )

    @staticmethod
    def _split_fields(line):
        return [field for field in re.split(r"[,\s]+", line.strip()) if field]

    @staticmethod
    def _small_integer(value):
        try:
            number = float(value)
        except ValueError:
            return False
        return number.is_integer() and -1 <= int(number) <= 20

    def _parse_gt(self, fields):
        layout = self.gt_format
        if layout == "auto":
            if len(fields) >= 9:
                layout = "mot"
            elif len(fields) == 8 and self._small_integer(fields[2]):
                layout = "native8"
            else:
                layout = "simple"

        if layout == "mot":
            if len(fields) < 8:
                raise ValueError("MOT rows require at least 8 columns")
            frame = int(float(fields[0]))
            obj_id = int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            mark = float(fields[6])
            visibility = float(fields[8]) if len(fields) >= 9 else 1.0
        elif layout == "native8":
            if len(fields) < 8:
                raise ValueError("native8 rows require 8 columns")
            frame = int(float(fields[0]))
            obj_id = int(float(fields[1]))
            x, y, width, height = map(float, fields[3:7])
            mark = float(fields[7])
            visibility = 1.0
        elif layout == "simple":
            if len(fields) < 6:
                raise ValueError("simple rows require at least 6 columns")
            frame = int(float(fields[0]))
            obj_id = int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            mark = float(fields[6]) if len(fields) >= 7 else 1.0
            visibility = 1.0
        else:
            raise ValueError(f"Unsupported GT_FORMAT: {layout}")

        return frame, obj_id, x, y, width, height, mark, visibility

    @staticmethod
    def _parse_tracker(fields):
        if len(fields) < 6:
            raise ValueError("Tracker rows require at least 6 columns")
        frame = int(float(fields[0]))
        obj_id = int(float(fields[1]))
        x, y, width, height = map(float, fields[2:6])
        score = float(fields[6]) if len(fields) >= 7 else 1.0
        return frame, obj_id, x, y, width, height, score

    def _load_raw_file(self, tracker, seq, is_gt):
        file_path = (
            os.path.join(self.gt_fol, seq, "gt", "gt.txt")
            if is_gt
            else self._tracker_file(tracker, seq)
        )

        num_timesteps = self.seq_lengths[seq]
        ids = [[] for _ in range(num_timesteps)]
        classes = [[] for _ in range(num_timesteps)]
        dets = [[] for _ in range(num_timesteps)]
        confidences = [[] for _ in range(num_timesteps)]

        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                fields = self._split_fields(line)
                try:
                    if is_gt:
                        frame, obj_id, x, y, width, height, mark, visibility = self._parse_gt(fields)
                        if self.filter_gt_by_mark and mark <= 0:
                            continue
                        if visibility < self.min_visibility:
                            continue
                        score = 1.0
                    else:
                        frame, obj_id, x, y, width, height, score = self._parse_tracker(fields)
                except (TypeError, ValueError) as exc:
                    raise TrackEvalException(
                        f"Invalid UA-DETRAC row at {file_path}:{line_number}: {line.strip()}"
                    ) from exc

                if frame < 1 or frame > num_timesteps:
                    raise TrackEvalException(
                        f"Frame {frame} in {file_path} is outside [1, {num_timesteps}]"
                    )
                if obj_id < 0 or width <= 0 or height <= 0:
                    continue

                timestep = frame - 1
                ids[timestep].append(obj_id)
                classes[timestep].append(0)
                dets[timestep].append([x, y, x + width, y + height])
                if not is_gt:
                    confidences[timestep].append(score)

        ids = [np.asarray(values, dtype=int) for values in ids]
        classes = [np.asarray(values, dtype=int) for values in classes]
        dets = [np.asarray(values, dtype=float).reshape((-1, 4)) for values in dets]

        if is_gt:
            return {
                "seq": seq,
                "gt_ids": ids,
                "gt_classes": classes,
                "gt_dets": dets,
                "num_timesteps": num_timesteps,
            }

        return {
            "seq": seq,
            "tracker_ids": ids,
            "tracker_classes": classes,
            "tracker_dets": dets,
            "tracker_confidences": [
                np.asarray(values, dtype=float) for values in confidences
            ],
            "num_timesteps": num_timesteps,
        }

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        cls_id = self.class_name_to_class_id[cls]
        data = {
            key: [None] * raw_data["num_timesteps"]
            for key in [
                "gt_ids",
                "tracker_ids",
                "gt_dets",
                "tracker_dets",
                "tracker_confidences",
                "similarity_scores",
            ]
        }
        data["seq"] = raw_data["seq"]

        unique_gt_ids = set()
        unique_tracker_ids = set()
        num_gt_dets = 0
        num_tracker_dets = 0

        for timestep in range(raw_data["num_timesteps"]):
            gt_mask = raw_data["gt_classes"][timestep] == cls_id
            tracker_mask = raw_data["tracker_classes"][timestep] == cls_id

            data["gt_ids"][timestep] = raw_data["gt_ids"][timestep][gt_mask]
            data["gt_dets"][timestep] = raw_data["gt_dets"][timestep][gt_mask]
            data["tracker_ids"][timestep] = raw_data["tracker_ids"][timestep][tracker_mask]
            data["tracker_dets"][timestep] = raw_data["tracker_dets"][timestep][tracker_mask]
            data["tracker_confidences"][timestep] = raw_data["tracker_confidences"][timestep][tracker_mask]
            data["similarity_scores"][timestep] = raw_data["similarity_scores"][timestep][gt_mask, :][:, tracker_mask]

            unique_gt_ids.update(data["gt_ids"][timestep].tolist())
            unique_tracker_ids.update(data["tracker_ids"][timestep].tolist())
            num_gt_dets += len(data["gt_ids"][timestep])
            num_tracker_dets += len(data["tracker_ids"][timestep])

        gt_id_map = {obj_id: index for index, obj_id in enumerate(sorted(unique_gt_ids))}
        tracker_id_map = {
            obj_id: index for index, obj_id in enumerate(sorted(unique_tracker_ids))
        }
        for timestep in range(raw_data["num_timesteps"]):
            data["gt_ids"][timestep] = np.asarray(
                [gt_id_map[obj_id] for obj_id in data["gt_ids"][timestep]],
                dtype=np.int32,
            )
            data["tracker_ids"][timestep] = np.asarray(
                [tracker_id_map[obj_id] for obj_id in data["tracker_ids"][timestep]],
                dtype=np.int32,
            )

        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]

        self._check_unique_ids(data)
        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        return self._calculate_box_ious(
            gt_dets_t,
            tracker_dets_t,
            box_format="x0y0x1y1",
        )
