#!/usr/bin/env python3
"""Convert a COCO DAB-Deformable-DETR checkpoint for one-class UA-DETRAC.

The existing HNCD one-class pretrain loader defaults to the COCO person row,
which is appropriate for pedestrian datasets but not UA-DETRAC. This utility
rewrites every classifier tensor whose first dimension is a COCO class layout
so UA-DETRAC starts from the COCO ``car`` classifier row.

Supported classifier layouts:
  * 91-row COCO-with-gaps layout: car index = 3
  * 80-row contiguous COCO layout: car index = 2
  * 1-row checkpoint: unchanged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert_classifier_tensor(name: str, tensor: torch.Tensor):
    if "class_embed" not in name or tensor.ndim == 0:
        return tensor, False

    rows = tensor.shape[0]
    if rows == 91:
        return tensor[3:4].clone(), True
    if rows == 80:
        return tensor[2:3].clone(), True
    if rows == 1:
        return tensor, False
    return tensor, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    checkpoint = torch.load(source, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            "Expected a DAB-Deformable-DETR checkpoint dict containing a 'model' state dict."
        )

    state_dict = checkpoint["model"]
    converted = []
    for name in list(state_dict.keys()):
        value = state_dict[name]
        if not torch.is_tensor(value):
            continue
        new_value, changed = convert_classifier_tensor(name, value)
        if changed:
            state_dict[name] = new_value
            converted.append((name, tuple(value.shape), tuple(new_value.shape)))

    if not converted:
        one_class_heads = [
            name
            for name, value in state_dict.items()
            if "class_embed" in name and torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1
        ]
        if not one_class_heads:
            raise RuntimeError(
                "No 91-row/80-row COCO class_embed tensor was found. "
                "Check that --source is the intended DAB-Deformable-DETR COCO checkpoint."
            )
        print("Checkpoint already has one-class classifier tensors; copied unchanged.")

    checkpoint.setdefault("meta", {})
    if isinstance(checkpoint["meta"], dict):
        checkpoint["meta"].update(
            {
                "uadetrac_pretrain_conversion": "COCO car -> UA-DETRAC vehicle",
                "source_checkpoint": str(source),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    print(f"Saved UA-DETRAC DAB pretrain to: {output}")
    for name, before, after in converted:
        print(f"  {name}: {before} -> {after} (COCO car row)")


if __name__ == "__main__":
    main()
