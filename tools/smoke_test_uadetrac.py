#!/usr/bin/env python3
"""Smoke-test the HNCD-MOTR UA-DETRAC adaptation without real benchmark data.

This test validates:
  * UA-DETRAC training loader and one-class labels;
  * proposal DB key resolution;
  * inference SeqDataset numeric frame order/proposals;
  * custom TrackEval adapter preprocessing on an exact toy tracker.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_sequence(dataset_root: Path, split: str, sequence: str):
    sequence_dir = dataset_root / split / sequence
    image_dir = sequence_dir / "img1"
    gt_dir = sequence_dir / "gt"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for frame_id in (1, 2, 10):
        image = Image.new("RGB", (96, 64), color=(frame_id * 10, 20, 30))
        image.save(image_dir / f"{frame_id:06d}.jpg")
        rows.append(
            f"{frame_id},1,{10 + frame_id:.2f},12.00,20.00,16.00,1,1,1.0\n"
        )
    (gt_dir / "gt.txt").write_text("".join(rows), encoding="utf-8")
    (sequence_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={sequence}\n"
        "imDir=img1\n"
        "frameRate=25\n"
        "seqLength=10\n"
        "imWidth=96\n"
        "imHeight=64\n"
        "imExt=.jpg\n",
        encoding="utf-8",
    )
    return sequence_dir


def _make_proposal_db(root: Path, sequence: str):
    database = {}
    for split in ("train", "test"):
        for frame_id in (1, 2, 10):
            key = f"UA-DETRAC/{split}/{sequence}/img1/{frame_id:06d}.txt"
            database[key] = [
                f"{10 + frame_id:.2f},12.0,20.0,16.0,0.95"
            ]
    output = root / "uadetrac_yolox_det_db.json"
    output.write_text(json.dumps(database), encoding="utf-8")
    return output


def _test_training_loader(data_root: Path, det_db: Path, sequence: str):
    from data.uadetrac import UADETRAC

    config = {
        "DATASET": "DanceTrack",
        "DATASET_ADAPTER": "UADETRAC",
        "DATASET_NAME": "UA-DETRAC",
        "DATA_ROOT": str(data_root),
        "SAMPLE_STEPS": [1],
        "SAMPLE_INTERVALS": [1, 1],
        "SAMPLE_MODES": ["random_interval", "random_interval"],
        "SAMPLE_LENGTHS": [2, 2],
        "USE_PROPOSALS": True,
        "DET_DB": str(det_db),
        "UADETRAC_GT_FORMAT": "auto",
        "UADETRAC_FILTER_GT_BY_MARK": True,
        "UADETRAC_MIN_VISIBILITY": 0.0,
    }
    dataset = UADETRAC(config=config, split="train", transform=None)
    assert sequence in dataset.frame_ids
    assert dataset.frame_ids[sequence] == [1, 2, 10]
    sample = dataset[0]
    assert set(sample) == {"imgs", "infos", "proposals"}
    assert len(sample["imgs"]) == 2
    for target in sample["infos"]:
        assert target["labels"].numel() == 1
        assert int(target["labels"][0]) == 0
    for proposals in sample["proposals"]:
        assert proposals.shape == (1, 5)


def _test_inference_loader(sequence_dir: Path):
    from data.seq_dataset import SeqDataset

    sequence = sequence_dir.name
    split = sequence_dir.parent.name
    database = {
        f"UA-DETRAC/{split}/{sequence}/img1/{frame_id:06d}.txt": [
            f"{10 + frame_id:.2f},12.0,20.0,16.0,0.95"
        ]
        for frame_id in (1, 2, 10)
    }
    dataset = SeqDataset(seq_dir=str(sequence_dir), det_db=database)
    assert dataset.frame_ids == [1, 2, 10], dataset.frame_ids
    (_, _, proposals), _ = dataset[0]
    assert proposals.shape == (1, 5)
    assert np.isclose(float(proposals[0, 4]), 0.95)


def _test_trackeval(dataset_root: Path, temporary_root: Path, sequence: str):
    trackeval_root = REPO_ROOT / "TrackEval"
    if str(trackeval_root) not in sys.path:
        sys.path.insert(0, str(trackeval_root))
    import trackeval

    tracker_dir = temporary_root / "tracker"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    tracker_rows = []
    for frame_id in (1, 2, 10):
        tracker_rows.append(
            f"{frame_id},1,{10 + frame_id:.2f},12.00,20.00,16.00,0.99,-1,-1,-1\n"
        )
    (tracker_dir / f"{sequence}.txt").write_text(
        "".join(tracker_rows), encoding="utf-8"
    )

    adapter = trackeval.datasets.UADETRAC(
        {
            "GT_FOLDER": str(dataset_root / "test"),
            "TRACKERS_FOLDER": str(tracker_dir),
            "OUTPUT_FOLDER": str(tracker_dir),
            "TRACKERS_TO_EVAL": [""],
            "CLASSES_TO_EVAL": ["vehicle"],
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_SUB_FOLDER": "",
            "PRINT_CONFIG": False,
            "FILTER_GT_BY_MARK": True,
            "MIN_VISIBILITY": 0.0,
            "GT_FORMAT": "auto",
        }
    )
    raw = adapter.get_raw_seq_data("", sequence)
    processed = adapter.get_preprocessed_seq_data(raw, "vehicle")
    assert processed["num_gt_dets"] == 3
    assert processed["num_tracker_dets"] == 3
    assert processed["num_gt_ids"] == 1
    assert processed["num_tracker_ids"] == 1


def main():
    with tempfile.TemporaryDirectory(prefix="hncd_uadetrac_smoke_") as tmp:
        temporary_root = Path(tmp)
        data_root = temporary_root / "datasets"
        dataset_root = data_root / "UA-DETRAC"
        sequence = "MVI_SMOKE"
        train_sequence = _make_sequence(dataset_root, "train", sequence)
        _make_sequence(dataset_root, "test", sequence)
        det_db = _make_proposal_db(data_root, sequence)

        _test_training_loader(data_root, det_db, sequence)
        _test_inference_loader(train_sequence)
        _test_trackeval(dataset_root, temporary_root, sequence)

    print("[OK] HNCD-MOTR UA-DETRAC smoke test passed.")


if __name__ == "__main__":
    main()
