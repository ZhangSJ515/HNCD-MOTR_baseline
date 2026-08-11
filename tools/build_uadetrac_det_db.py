#!/usr/bin/env python3
"""Build the HNCD-MOTR proposal JSON database from UA-DETRAC detector files.

Expected detector input is one text file per sequence. Supported rows:
  * MOT detector: frame,id,x,y,w,h,score[, ...]
  * compact:      frame,x,y,w,h,score

The output JSON uses keys such as:
  UA-DETRAC/train/MVI_20011/img1/000001.txt
with values containing ``x,y,w,h,score`` proposal rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALIASES = ("UA-DETRAC", "UADETRAC", "UA_DETRAC")


def split_fields(line: str):
    return [field for field in re.split(r"[,\s]+", line.strip()) if field]


def resolve_dataset_root(data_root: Path):
    for name in ALIASES:
        candidate = data_root / name
        if candidate.is_dir():
            return candidate
    return data_root / "UA-DETRAC"


def find_detection_file(detections_root: Path, split: str, sequence: str):
    candidates = [
        detections_root / f"{sequence}.txt",
        detections_root / split / f"{sequence}.txt",
        detections_root / sequence / "det" / "det.txt",
        detections_root / split / sequence / "det" / "det.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def parse_detection(fields):
    if len(fields) >= 7:
        frame = int(float(fields[0]))
        x, y, width, height = map(float, fields[2:6])
        score = float(fields[6])
    elif len(fields) == 6:
        frame = int(float(fields[0]))
        x, y, width, height, score = map(float, fields[1:6])
    else:
        raise ValueError("Detector row requires 6 or more columns")
    return frame, x, y, width, height, score


def image_map(sequence_dir: Path):
    image_dir = sequence_dir / "img1"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    result = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                result[int(path.stem)] = path
            except ValueError:
                continue
    if not result:
        raise RuntimeError(f"No numeric frame images found under {image_dir}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--detections-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    dataset_root = resolve_dataset_root(data_root)
    detections_root = Path(args.detections_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not detections_root.is_dir():
        raise FileNotFoundError(detections_root)

    database = {}
    total_proposals = 0
    missing_sequences = []

    for split in args.splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            if args.strict:
                raise FileNotFoundError(split_dir)
            print(f"[WARN] split does not exist, skipped: {split_dir}")
            continue

        for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            sequence = sequence_dir.name
            detector_file = find_detection_file(detections_root, split, sequence)
            if detector_file is None:
                missing_sequences.append(f"{split}/{sequence}")
                if args.strict:
                    raise FileNotFoundError(
                        f"No detector file found for {split}/{sequence} under {detections_root}"
                    )
                print(f"[WARN] no detector file for {split}/{sequence}")
                continue

            frames = image_map(sequence_dir)
            by_frame = defaultdict(list)
            with detector_file.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    fields = split_fields(line)
                    try:
                        frame, x, y, width, height, score = parse_detection(fields)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid detector row {detector_file}:{line_number}: {line.strip()}"
                        ) from exc
                    if frame not in frames or width <= 0 or height <= 0 or score < args.min_score:
                        continue
                    by_frame[frame].append((x, y, width, height, score))

            sequence_proposals = 0
            for frame, image_path in frames.items():
                key = f"UA-DETRAC/{split}/{sequence}/img1/{image_path.stem}.txt"
                proposals = sorted(by_frame.get(frame, []), key=lambda item: item[4], reverse=True)
                database[key] = [
                    f"{x:.6f},{y:.6f},{width:.6f},{height:.6f},{score:.8f}"
                    for x, y, width, height, score in proposals
                ]
                sequence_proposals += len(proposals)
            total_proposals += sequence_proposals
            print(
                f"[OK] {split}/{sequence}: frames={len(frames)}, proposals={sequence_proposals}"
            )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(database, file, ensure_ascii=False)

    print(
        f"Wrote {len(database)} frame keys / {total_proposals} proposals to {output}"
    )
    if missing_sequences:
        print(f"Missing detector files for {len(missing_sequences)} sequences: {missing_sequences[:20]}")


if __name__ == "__main__":
    main()
