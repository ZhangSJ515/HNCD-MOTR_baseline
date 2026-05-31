# @Author       : Ruopeng Gao
# @Date         : 2022/11/21

import os
import json
import shutil
import yaml

from torch.utils import tensorboard as tb

from utils.utils import yaml_to_dict


def evaluate(config: dict):
    eval_split = config["EVAL_DATA_SPLIT"]
    eval_dir = config["EVAL_DIR"]
    if config["EVAL_PORT"] is not None:
        port = config["EVAL_PORT"]
    else:
        port = 22701
    outputs_dir = os.path.join(eval_dir, eval_split)
    os.makedirs(outputs_dir, exist_ok=True)
    eval_states_path = os.path.join(outputs_dir, "eval_states.yaml")
    if os.path.exists(eval_states_path):
        eval_states: dict = yaml_to_dict(eval_states_path)
    else:
        eval_states: dict = {
            "NEXT_INDEX": 0,
        }
    # Tensorboard Setting
    tb_writer = tb.SummaryWriter(
        log_dir=os.path.join(outputs_dir, "tb")
    )

    use_proposals = config.get("USE_PROPOSALS", True)

    if config["EVAL_MODE"] == "specific":
        if config["EVAL_MODEL"] is None:
            raise ValueError("--eval-model should not be None.")
        metrics = eval_model(model=config["EVAL_MODEL"], eval_dir=eval_dir,
                             data_root=config['DATA_ROOT'], dataset_name=config["DATASET"], data_split=eval_split,
                             threads=config["EVAL_THREADS"], port=port, config_path=config["CONFIG_PATH"],
                             use_proposals=use_proposals,
                             save_embeddings_dir=config.get("SAVE_EMBEDDINGS_DIR"))
    elif config["EVAL_MODE"] == "continue":
        init_index = eval_states["NEXT_INDEX"]
        for i in range(init_index, 10000):
            model = "checkpoint_" + str(i) + ".pth"
            if os.path.exists(os.path.join(eval_dir, model)):
                if os.path.exists(os.path.join(eval_dir, eval_split, model.split(".")[0] + "_tracker",
                                               "pedestrian_summary.txt")):
                    pass
                else:
                    metrics = eval_model(
                        model=model, eval_dir=eval_dir,
                        data_root=config["DATA_ROOT"], dataset_name=config["DATASET"], data_split=eval_split,
                        threads=config["EVAL_THREADS"], port=port, config_path=config["CONFIG_PATH"],
                        use_proposals=use_proposals,
                        save_embeddings_dir=config.get("SAVE_EMBEDDINGS_DIR")
                    )
                    metrics_to_tensorboard(writer=tb_writer, metrics=metrics, epoch=i)
                eval_states["NEXT_INDEX"] = i + 1
                with open(eval_states_path, mode="w") as f:
                    yaml.dump(eval_states, f, allow_unicode=True)
    else:
        raise ValueError(f"Eval mode '{config['EVAL_MODE']}' is not supported.")

    with open(eval_states_path, mode="w") as f:
        yaml.dump(eval_states, f, allow_unicode=True)

    return


def eval_model(model: str, eval_dir: str, data_root: str, dataset_name: str, data_split: str, threads: int, port: int,
               config_path: str, use_proposals: bool = True, save_embeddings_dir: str = None):
    print(f"===>  Running checkpoint '{model}'")

    use_proposals_flag = "--use-proposals" if use_proposals else ""
    save_embeddings_flag = ""
    if save_embeddings_dir:
        model_embeddings_dir = os.path.join(save_embeddings_dir, model.split(".")[0])
        save_embeddings_flag = f"--save-embeddings-dir {model_embeddings_dir}"
    if threads > 1:
        os.system(f"python -m torch.distributed.run --nproc_per_node={str(threads)} --master_port={port} "
                  f"main.py --mode submit --submit-dir {eval_dir} --submit-model {model} "
                  f"--data-root {data_root} --submit-data-split {data_split} "
                  f"--use-distributed --config-path {config_path} {use_proposals_flag} {save_embeddings_flag}")
    else:
        os.system(f"python main.py --mode submit --submit-dir {eval_dir} --submit-model {model} "
                  f"--data-root {data_root} --submit-data-split {data_split} --config-path {config_path} "
                  f"{use_proposals_flag} {save_embeddings_flag}")

    # 将结果移动到对应的文件夹
    tracker_dir = os.path.join(eval_dir, data_split, "tracker")
    tracker_mv_dir = os.path.join(eval_dir, data_split, model.split(".")[0] + "_tracker")
    os.system(f"mv {tracker_dir} {tracker_mv_dir}")

    runtime_metrics = {}
    runtime_dir = os.path.join(eval_dir, data_split, "runtime_stats")
    if os.path.isdir(runtime_dir):
        runtime_mv_dir = os.path.join(eval_dir, data_split, model.split(".")[0] + "_runtime_stats")
        if os.path.exists(runtime_mv_dir):
            shutil.rmtree(runtime_mv_dir)
        shutil.move(runtime_dir, runtime_mv_dir)
        runtime_metrics = aggregate_runtime_stats(runtime_mv_dir)

    # 进行指标计算
    data_dir = os.path.join(data_root, dataset_name)
    if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
        gt_dir = os.path.join(data_dir, data_split)
    elif "MOT17" in dataset_name:
        gt_dir = os.path.join(data_dir, "images", data_split)
    else:
        raise NotImplementedError(f"Eval Engine DO NOT support dataset '{dataset_name}'")
    if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
        os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                  f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                  f"--SEQMAP_FILE {os.path.join(data_dir, f'{data_split}_seqmap.txt')} "
                  f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                  f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                  f"--TRACKERS_FOLDER {tracker_mv_dir}")
    elif "MOT17" in dataset_name:
        if "mot15" in data_split:
            os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                      f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                      f"--SEQMAP_FILE {os.path.join(data_dir, f'{data_split}_seqmap.txt')} "
                      f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                      f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                      f"--TRACKERS_FOLDER {tracker_mv_dir} --BENCHMARK MOT15")
        else:
            os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                      f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                      f"--SEQMAP_FILE {os.path.join(data_dir, f'{data_split}_seqmap.txt')} "
                      f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                      f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                      f"--TRACKERS_FOLDER {tracker_mv_dir} --BENCHMARK MOT17")
    else:
        raise NotImplementedError(f"Do not support this Dataset name: {dataset_name}")

    metric_path = os.path.join(tracker_mv_dir, "pedestrian_summary.txt")
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    metrics.update(runtime_metrics)
    return metrics


def aggregate_runtime_stats(runtime_dir: str) -> dict:
    stat_paths = [
        os.path.join(runtime_dir, name)
        for name in os.listdir(runtime_dir)
        if name.endswith(".json")
    ]
    if not stat_paths:
        return {}

    total_frames = 0
    total_time_sec = 0.0
    model_time_sec = 0.0
    track_time_sec = 0.0
    result_time_sec = 0.0
    per_sequence = []

    for stat_path in stat_paths:
        with open(stat_path, "r", encoding="utf-8") as f:
            stat = json.load(f)
        per_sequence.append(stat)
        total_frames += int(stat.get("num_frames", 0))
        total_time_sec += float(stat.get("total_time_sec", 0.0))
        model_time_sec += float(stat.get("model_time_sec", 0.0))
        track_time_sec += float(stat.get("track_time_sec", 0.0))
        result_time_sec += float(stat.get("result_time_sec", 0.0))

    if total_frames == 0 or total_time_sec <= 0:
        return {}

    summary = {
        "NUM_FRAMES": float(total_frames),
        "FPS": float(total_frames / total_time_sec),
        "AVG_TOTAL_MS": float(total_time_sec / total_frames * 1000.0),
        "AVG_MODEL_MS": float(model_time_sec / total_frames * 1000.0),
        "AVG_TRACK_MS": float(track_time_sec / total_frames * 1000.0),
        "AVG_RESULT_MS": float(result_time_sec / total_frames * 1000.0),
    }

    with open(os.path.join(runtime_dir, "runtime_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "per_sequence": sorted(per_sequence, key=lambda x: x.get("seq_name", "")),
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "===>  Runtime summary: "
        f"FPS={summary['FPS']:.2f}, "
        f"avg_total={summary['AVG_TOTAL_MS']:.2f} ms, "
        f"avg_model={summary['AVG_MODEL_MS']:.2f} ms, "
        f"avg_track={summary['AVG_TRACK_MS']:.2f} ms, "
        f"avg_result={summary['AVG_RESULT_MS']:.2f} ms"
    )
    return summary


def metrics_to_tensorboard(writer: tb.SummaryWriter, metrics: dict, epoch: int):
    for k, v in metrics.items():
        writer.add_scalar(tag=k, scalar_value=v, global_step=epoch)
    return
