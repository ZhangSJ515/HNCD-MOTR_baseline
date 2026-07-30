# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Main Function.
import os
import argparse
import torch.distributed
import torch.backends.cuda
import torch.backends.cudnn

from utils.utils import distributed_rank
from utils.utils import yaml_to_dict
from configs.utils import update_config
import torch


def parse_option():
    parser = argparse.ArgumentParser(
        "Network training and evaluation script.",
        add_help=True,
    )

    parser.add_argument("--git-version", type=str)

    # About system, Like GPUs:
    parser.add_argument(
        "--available-gpus",
        type=str,
        help="Available GPUs, like '0,1,2,3'.",
    )
    # default=None is intentional. update_config only applies non-None CLI
    # values, so omitting a flag preserves the YAML value.
    parser.add_argument(
        "--use-distributed",
        action="store_true",
        default=None,
        help="Use distributed training.",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        default=None,
        help="Use gradient checkpoint to save GPU memory.",
    )
    parser.add_argument("--checkpoint-level", type=int)
    # For torchrun compatibility
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training (automatically set by torchrun)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training (automatically set by torchrun)",
    )

    # Running mode, Training? Evaluation? or ?
    parser.add_argument("--mode", type=str, help="Running mode.")

    # Only For **Result Submit Process**:
    parser.add_argument("--submit-dir", type=str)
    parser.add_argument("--submit-model", type=str)
    parser.add_argument("--submit-data-split", type=str)

    # Only For **Model Eval Process**:
    parser.add_argument("--eval-dir", type=str)
    parser.add_argument("--eval-mode", type=str)
    parser.add_argument("--eval-model", type=str)
    parser.add_argument("--eval-threads", type=int)
    parser.add_argument("--eval-port", type=int)
    parser.add_argument("--eval-data-split", type=str)
    parser.add_argument(
        "--save-embeddings-dir",
        type=str,
        help="Optional directory for exporting per-frame track embeddings during eval/submit.",
    )

    # Pretrained Model Load:
    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Pretrained model path.",
    )
    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume checkpoint path.",
    )
    parser.add_argument(
        "--resume-scheduler",
        type=str,
        help="Whether resume the training scheduler.",
    )

    # About Paths:
    parser.add_argument(
        "--config-path",
        type=str,
        help="Config file path.",
        default="./configs/train_dancetrack.yaml",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        help="Dataset root dir.",
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--outputs-dir", type=str)

    # Data:
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--coco-size", type=str)
    parser.add_argument("--overflow-bbox", type=str)
    parser.add_argument("--reverse-clip", type=float)
    parser.add_argument("--use-motsynth", type=str)
    parser.add_argument("--use-crowdhuman", type=str)
    parser.add_argument("--motsynth-rate", type=float)
    parser.add_argument("--sample-steps", type=int, nargs="*")
    parser.add_argument("--sample-lengths", type=int, nargs="*")

    # Training setting:
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-points", type=float)
    parser.add_argument("--lr-backbone", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr-drop-milestones", type=int, nargs="*")

    # Submit setting:
    parser.add_argument("--miss-tolerance", type=float)
    parser.add_argument("--min-track-area", type=float)
    parser.add_argument("--inference-min-size", type=int)
    parser.add_argument("--inference-max-size", type=int)

    # Model setting:
    parser.add_argument("--num-det-queries", type=int)
    parser.add_argument("--merge-det-track-layer", type=int)

    # Training augmentation:
    parser.add_argument("--tp-drop-rate", type=float)
    parser.add_argument("--fp-insert-rate", type=float)

    # Proposal toggle. Omission preserves USE_PROPOSALS from YAML.
    parser.add_argument(
        "--use-proposals",
        action="store_true",
        default=None,
        help="Enable MOTRv2-style detector proposals.",
    )
    parser.add_argument("--det-db", type=str)

    # Architecture switch:
    parser.add_argument(
        "--arch",
        type=str,
        choices=["memotr", "motrv2"],
        default=None,
        help="Architecture preset: memotr (TIM) or motrv2 (QIMv2).",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        default=None,
    )

    return parser.parse_args()


def main(config: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["AVAILABLE_GPUS"]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if config["USE_DISTRIBUTED"]:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(distributed_rank())

    # AirMOT has a native four-class annotation/result format and requires
    # proposal-path handling that differs from MOTChallenge datasets.
    if config.get("DATASET") == "AirMot":
        from airmot_engine import (
            evaluate_airmot,
            train_airmot,
        )
        from airmot_submit_engine import submit_airmot

        if config["MODE"] == "train":
            train_airmot(config=config)
        elif config["MODE"] == "submit":
            submit_airmot(config=config)
        elif config["MODE"] == "eval":
            evaluate_airmot(config=config)
        else:
            raise ValueError(
                f"Unsupported mode '{config['MODE']}'"
            )
        return

    from train_engine import train
    from submit_engine import submit
    from eval_engine import evaluate

    if config["MODE"] == "train":
        train(config=config)
    elif config["MODE"] == "submit":
        submit(config=config)
    elif config["MODE"] == "eval":
        evaluate(config=config)
    else:
        raise ValueError(f"Unsupported mode '{config['MODE']}'")


if __name__ == "__main__":
    opt = parse_option()
    cfg = yaml_to_dict(opt.config_path)

    # Merge parser options and YAML config, then run.
    merged_config = update_config(config=cfg, option=opt)
    merged_config["CONFIG_PATH"] = opt.config_path
    main(config=merged_config)
