#!/usr/bin/env python3
"""Validate UA-DETRAC layout, annotations, and optional proposal coverage."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALIASES = ("UA-DETRAC", "UADETRAC", "UA_DETRAC")


def split_fields(line):
    return [field for field in re.split(r"[,\s]+", line.strip()) if field]


def resolve_root(data_root: Path):
    for name in ALIASES:
        candidate = data_root / name
        if candidate.is_dir():
            return candidate
    return data_root / "UA-DETRAC"


def parse_gt(fields, layout):
    if layout == "auto":
        if len(fields) >= 9:
            layout = "mot"
        elif len(fields) == 8:
            layout = "native8"
        else:
            layout = "simple"
    if layout == "mot":
        frame, obj_id = int(float(fields[0])), int(float(fields[1]))
        x, y, w, h = map(float, fields[2:6])
        mark = float(fields[6])
        visibility = float(fields[8]) if len(fields) >= 9 else 1.0
    elif layout == "native8":
        frame, obj_id = int(float(fields[0])), int(float(fields[1]))
        x, y, w, h = map(float, fields[3:7])
        mark = float(fields[7])
        visibility = 1.0
    else:
        frame, obj_id = int(float(fields[0])), int(float(fields[1]))
        x, y, w, h = map(float, fields[2:6])
        mark = float(fields[6]) if len(fields) >= 7 else 1.0
        visibility = 1.0
    return frame, obj_id, x, y, w, h, mark, visibility


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--gt-format", default="auto", choices=["auto", "mot", "native8", "simple"])
    parser.add_argument("--filter-gt-by-mark", action="store_true")
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--det-db")
    parser.add_argument("--require-det-db", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    dataset_root = resolve_root(data_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    det_db = None
    if args.det_db:
        det_path = Path(args.det_db).expanduser().resolve()
        if not det_path.is_file():
            raise FileNotFoundError(det_path)
        with det_path.open("r", encoding="utf-8") as file:
            det_db = {str(key).replace("\\", "/"): value for key, value in json.load(file).items()}
    elif args.require_det_db:
        raise ValueError("--require-det-db was set but --det-db is empty")

    total_sequences = total_frames = total_boxes = total_tracks = 0
    missing_proposal_frames = []

    for split in args.splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            print(f"[WARN] split not found: {split_dir}")
            continue

        for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            image_dir = sequence_dir / "img1"
            if not image_dir.is_dir():
                raise FileNotFoundError(image_dir)
            images = {}
            for image_path in image_dir.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    try:
                        images[int(image_path.stem)] = image_path
                    except ValueError:
                        raise ValueError(f"UA-DETRAC frame name must be numeric: {image_path}")
            if not images:
                raise RuntimeError(f"No images in {image_dir}")

            gt_path = sequence_dir / "gt" / "gt.txt"
            boxes = 0
            track_ids = set()
            gt_frames = set()
            if gt_path.is_file():
                with gt_path.open("r", encoding="utf-8") as file:
                    for line_number, line in enumerate(file, start=1):
                        if not line.strip():
                            continue
                        fields = split_fields(line)
                        try:
                            frame, obj_id, x, y, w, h, mark, visibility = parse_gt(fields, args.gt_format)
                        except Exception as exc:
                            raise ValueError(
                                f"Invalid GT row {gt_path}:{line_number}: {line.strip()}"
                            ) from exc
                        if frame not in images:
                            raise ValueError(
                                f"GT references missing image: {sequence_dir.name} frame {frame}"
                            )
                        if obj_id < 0 or w <= 0 or h <= 0:
                            continue
                        if args.filter_gt_by_mark and mark <= 0:
                            continue
                        if visibility < args.min_visibility:
                            continue
                        boxes += 1
                        track_ids.add(obj_id)
                        gt_frames.add(frame)

            if split == "train" and not gt_path.is_file():
                raise FileNotFoundError(gt_path)

            if det_db is not None:
                for frame, image_path in images.items():
                    key = f"UA-DETRAC/{split}/{sequence_dir.name}/img1/{image_path.stem}.txt"
                    if key not in det_db:
                        missing_proposal_frames.append(key)

            total_sequences += 1
            total_frames += len(images)
            total_boxes += boxes
            total_tracks += len(track_ids)
            print(
                f"[OK] {split}/{sequence_dir.name}: frames={len(images)}, "
                f"gt_boxes={boxes}, tracks={len(track_ids)}"
            )

    if total_sequences == 0:
        raise RuntimeError("No UA-DETRAC sequences found.")
    if det_db is not None and missing_proposal_frames:
        raise RuntimeError(
            f"Proposal DB misses {len(missing_proposal_frames)} frame keys; "
            f"first: {missing_proposal_frames[:10]}"
        )

    print(
        f"UA-DETRAC setup OK: sequences={total_sequences}, frames={total_frames}, "
        f"gt_boxes={total_boxes}, tracks(sum per seq)={total_tracks}"
    )


if __name__ == "__main__":
    main()
