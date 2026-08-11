from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from torch.utils import tensorboard as tb

from data.uadetrac import resolve_uadetrac_root
from eval_engine import aggregate_runtime_stats
from utils.utils import yaml_to_dict


def _read_metric_summary(metric_path: str) -> dict:
    if not os.path.isfile(metric_path):
        raise FileNotFoundError(f"TrackEval summary was not generated: {metric_path}")
    with open(metric_path, "r", encoding="utf-8") as file:
        names = file.readline().strip().split()
        values = file.readline().strip().split()
    if not names or len(names) != len(values):
        raise RuntimeError(f"Invalid TrackEval summary: {metric_path}")
    return {name: float(value) for name, value in zip(names, values)}


def evaluate_uadetrac(config: dict):
    eval_split = config["EVAL_DATA_SPLIT"]
    eval_dir = config["EVAL_DIR"]
    if eval_dir is None:
        raise ValueError("EVAL_DIR must point to the HNCD-MOTR training output directory.")

    outputs_dir = os.path.join(eval_dir, eval_split)
    os.makedirs(outputs_dir, exist_ok=True)
    eval_states_path = os.path.join(outputs_dir, "eval_states.yaml")
    eval_states = yaml_to_dict(eval_states_path) if os.path.exists(eval_states_path) else {"NEXT_INDEX": 0}
    writer = tb.SummaryWriter(log_dir=os.path.join(outputs_dir, "tb"))

    if config["EVAL_MODE"] == "specific":
        if config["EVAL_MODEL"] is None:
            raise ValueError("EVAL_MODEL must be set in specific evaluation mode.")
        metrics = eval_uadetrac_model(
            model=config["EVAL_MODEL"],
            eval_dir=eval_dir,
            data_root=config["DATA_ROOT"],
            data_split=eval_split,
            threads=config["EVAL_THREADS"],
            port=config.get("EVAL_PORT") or 22701,
            config_path=config["CONFIG_PATH"],
            filter_gt_by_mark=config.get("UADETRAC_FILTER_GT_BY_MARK", True),
            min_visibility=config.get("UADETRAC_MIN_VISIBILITY", 0.0),
            gt_format=config.get("UADETRAC_GT_FORMAT", "auto"),
            save_embeddings_dir=config.get("SAVE_EMBEDDINGS_DIR"),
        )
        print("===> UA-DETRAC metrics:", metrics)
    elif config["EVAL_MODE"] == "continue":
        for index in range(int(eval_states["NEXT_INDEX"]), 10000):
            model = f"checkpoint_{index}.pth"
            checkpoint = os.path.join(eval_dir, model)
            if not os.path.isfile(checkpoint):
                continue
            summary = os.path.join(
                eval_dir,
                eval_split,
                model.rsplit(".", 1)[0] + "_tracker",
                "vehicle_summary.txt",
            )
            if os.path.isfile(summary):
                metrics = _read_metric_summary(summary)
            else:
                metrics = eval_uadetrac_model(
                    model=model,
                    eval_dir=eval_dir,
                    data_root=config["DATA_ROOT"],
                    data_split=eval_split,
                    threads=config["EVAL_THREADS"],
                    port=config.get("EVAL_PORT") or 22701,
                    config_path=config["CONFIG_PATH"],
                    filter_gt_by_mark=config.get("UADETRAC_FILTER_GT_BY_MARK", True),
                    min_visibility=config.get("UADETRAC_MIN_VISIBILITY", 0.0),
                    gt_format=config.get("UADETRAC_GT_FORMAT", "auto"),
                    save_embeddings_dir=config.get("SAVE_EMBEDDINGS_DIR"),
                )
            for name, value in metrics.items():
                writer.add_scalar(name, value, global_step=index)
            eval_states["NEXT_INDEX"] = index + 1
            with open(eval_states_path, "w", encoding="utf-8") as file:
                yaml.dump(eval_states, file, allow_unicode=True)
    else:
        raise ValueError(f"Unsupported EVAL_MODE: {config['EVAL_MODE']}")

    with open(eval_states_path, "w", encoding="utf-8") as file:
        yaml.dump(eval_states, file, allow_unicode=True)
    writer.close()


def eval_uadetrac_model(
    model: str,
    eval_dir: str,
    data_root: str,
    data_split: str,
    threads: int,
    port: int,
    config_path: str,
    filter_gt_by_mark: bool = True,
    min_visibility: float = 0.0,
    gt_format: str = "auto",
    save_embeddings_dir: str | None = None,
):
    print(f"===> Running UA-DETRAC checkpoint '{model}'")

    submit_command = [
        sys.executable,
        "main.py",
        "--mode",
        "submit",
        "--submit-dir",
        eval_dir,
        "--submit-model",
        model,
        "--data-root",
        data_root,
        "--submit-data-split",
        data_split,
        "--config-path",
        config_path,
    ]
    if save_embeddings_dir:
        submit_command.extend(
            [
                "--save-embeddings-dir",
                os.path.join(save_embeddings_dir, model.rsplit(".", 1)[0]),
            ]
        )

    if threads > 1:
        submit_command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={threads}",
            f"--master_port={port}",
            "main.py",
            "--mode",
            "submit",
            "--submit-dir",
            eval_dir,
            "--submit-model",
            model,
            "--data-root",
            data_root,
            "--submit-data-split",
            data_split,
            "--config-path",
            config_path,
            "--use-distributed",
        ]
        if save_embeddings_dir:
            submit_command.extend(
                [
                    "--save-embeddings-dir",
                    os.path.join(save_embeddings_dir, model.rsplit(".", 1)[0]),
                ]
            )

    subprocess.run(submit_command, check=True)

    split_output = os.path.join(eval_dir, data_split)
    tracker_dir = os.path.join(split_output, "tracker")
    tracker_output = os.path.join(
        split_output, model.rsplit(".", 1)[0] + "_tracker"
    )
    if os.path.exists(tracker_output):
        shutil.rmtree(tracker_output)
    if not os.path.isdir(tracker_dir):
        raise RuntimeError(f"Submission tracker directory was not generated: {tracker_dir}")
    shutil.move(tracker_dir, tracker_output)

    runtime_metrics = {}
    runtime_dir = os.path.join(split_output, "runtime_stats")
    if os.path.isdir(runtime_dir):
        runtime_output = os.path.join(
            split_output, model.rsplit(".", 1)[0] + "_runtime_stats"
        )
        if os.path.exists(runtime_output):
            shutil.rmtree(runtime_output)
        shutil.move(runtime_dir, runtime_output)
        runtime_metrics = aggregate_runtime_stats(runtime_output)

    dataset_root = resolve_uadetrac_root(data_root)
    gt_dir = os.path.join(dataset_root, data_split)
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(gt_dir)

    trackeval_script = Path(__file__).resolve().parent / "TrackEval" / "scripts" / "run_uadetrac.py"
    command = [
        sys.executable,
        str(trackeval_script),
        "--GT_FOLDER",
        gt_dir,
        "--TRACKERS_FOLDER",
        tracker_output,
        "--OUTPUT_FOLDER",
        tracker_output,
        "--TRACKER_SUB_FOLDER",
        "",
        "--OUTPUT_SUB_FOLDER",
        "",
        "--CLASSES_TO_EVAL",
        "vehicle",
        "--METRICS",
        "HOTA",
        "CLEAR",
        "Identity",
        "--USE_PARALLEL",
        "True",
        "--NUM_PARALLEL_CORES",
        "8",
        "--PLOT_CURVES",
        "False",
        "--PRINT_ONLY_COMBINED",
        "True",
        "--FILTER_GT_BY_MARK",
        str(bool(filter_gt_by_mark)),
        "--MIN_VISIBILITY",
        str(float(min_visibility)),
        "--GT_FORMAT",
        str(gt_format),
    ]
    subprocess.run(command, check=True)

    metric_path = os.path.join(tracker_output, "vehicle_summary.txt")
    metrics = _read_metric_summary(metric_path)
    metrics.update(runtime_metrics)

    with open(
        os.path.join(tracker_output, "hncd_uadetrac_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    return metrics
