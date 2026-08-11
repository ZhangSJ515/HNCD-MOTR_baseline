#!/usr/bin/env python3
"""Convert official UA-DETRAC XML annotations into HNCD-MOTR/MOT layout."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def frame_number(path: Path):
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Cannot infer frame number from image name: {path}")
    return int(numbers[-1])


def find_sequence_images(images_root: Path, sequence_name: str):
    sequence_dir = images_root / sequence_name
    candidates = []
    if sequence_dir.is_dir():
        candidates.extend(
            path
            for path in sequence_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not candidates:
        candidates.extend(
            path
            for path in images_root.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.stem.startswith(sequence_name)
            )
        )
    return sorted(candidates, key=lambda path: (frame_number(path), natural_key(path)))


def install_image(source: Path, destination: Path, mode: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def parse_sequence(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    sequence_name = root.attrib.get("name") or xml_path.stem
    rows = []
    for frame_element in root.findall("frame"):
        frame_id = int(frame_element.attrib["num"])
        target_list = frame_element.find("target_list")
        if target_list is None:
            continue
        for target in target_list.findall("target"):
            object_id = int(target.attrib["id"])
            box = target.find("box")
            if box is None:
                continue
            x = float(box.attrib["left"])
            y = float(box.attrib["top"])
            width = float(box.attrib["width"])
            height = float(box.attrib["height"])
            if width <= 0 or height <= 0:
                continue

            visibility = 1.0
            attribute = target.find("attribute")
            if attribute is not None:
                truncation = attribute.attrib.get("truncation_ratio")
                if truncation is not None:
                    try:
                        visibility = max(0.0, min(1.0, 1.0 - float(truncation)))
                    except ValueError:
                        visibility = 1.0

            # Standard MOTChallenge row: frame,id,x,y,w,h,mark,class,visibility
            rows.append(
                (frame_id, object_id, x, y, width, height, 1, 1, visibility)
            )
    rows.sort(key=lambda row: (row[0], row[1]))
    return sequence_name, rows


def write_seqinfo(
    path: Path,
    name: str,
    frame_rate: int,
    length: int,
    width: int,
    height: int,
    extension: str,
):
    path.write_text(
        "[Sequence]\n"
        f"name={name}\n"
        "imDir=img1\n"
        f"frameRate={frame_rate}\n"
        f"seqLength={length}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        f"imExt={extension}\n",
        encoding="utf-8",
    )


def convert_sequence(
    xml_path: Path,
    images_root: Path,
    output_split: Path,
    fps: int,
    link_mode: str,
):
    sequence_name, rows = parse_sequence(xml_path)
    image_paths = find_sequence_images(images_root, sequence_name)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found for sequence {sequence_name} under {images_root}"
        )

    image_by_frame = {frame_number(path): path for path in image_paths}
    sequence_dir = output_split / sequence_name
    image_dir = sequence_dir / "img1"
    gt_dir = sequence_dir / "gt"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    first_image = image_paths[0]
    extension = first_image.suffix.lower()
    with Image.open(first_image) as image:
        width, height = image.size

    for source_frame_id, source_path in sorted(image_by_frame.items()):
        destination = image_dir / f"{source_frame_id:06d}{extension}"
        install_image(source_path, destination, link_mode)

    missing_frames = sorted({row[0] for row in rows} - set(image_by_frame))
    if missing_frames:
        raise ValueError(
            f"Sequence {sequence_name} has annotations without images for "
            f"{len(missing_frames)} frames; first missing frames: {missing_frames[:10]}"
        )

    gt_lines = [
        f"{frame_id},{object_id},{x:.6f},{y:.6f},{box_width:.6f},{box_height:.6f},"
        f"{mark},{class_id},{visibility:.6f}\n"
        for (
            frame_id,
            object_id,
            x,
            y,
            box_width,
            box_height,
            mark,
            class_id,
            visibility,
        ) in rows
    ]
    (gt_dir / "gt.txt").write_text("".join(gt_lines), encoding="utf-8")
    write_seqinfo(
        sequence_dir / "seqinfo.ini",
        name=sequence_name,
        frame_rate=fps,
        length=max(image_by_frame),
        width=width,
        height=height,
        extension=extension,
    )
    return sequence_name, len(image_paths), len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--annotations-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink", "hardlink"],
        default="symlink",
    )
    args = parser.parse_args()

    images_root = Path(args.images_root).expanduser().resolve()
    annotations_root = Path(args.annotations_root).expanduser().resolve()
    output_split = Path(args.output_root).expanduser().resolve() / args.split
    if not images_root.is_dir():
        raise FileNotFoundError(images_root)
    if not annotations_root.is_dir():
        raise FileNotFoundError(annotations_root)
    output_split.mkdir(parents=True, exist_ok=True)

    xml_paths = sorted(annotations_root.glob("*.xml"), key=natural_key)
    if not xml_paths:
        raise FileNotFoundError(f"No XML files found in {annotations_root}")

    total_frames = 0
    total_boxes = 0
    for xml_path in xml_paths:
        sequence_name, frames, boxes = convert_sequence(
            xml_path=xml_path,
            images_root=images_root,
            output_split=output_split,
            fps=args.fps,
            link_mode=args.link_mode,
        )
        total_frames += frames
        total_boxes += boxes
        print(f"[OK] {sequence_name}: frames={frames}, boxes={boxes}")

    print(
        f"Converted {len(xml_paths)} sequences to {output_split}: "
        f"frames={total_frames}, boxes={total_boxes}"
    )


if __name__ == "__main__":
    main()
