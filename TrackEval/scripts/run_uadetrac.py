#!/usr/bin/env python3

import argparse
import os
import sys
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import trackeval  # noqa: E402


_LIST_SETTINGS = {
    "TRACKERS_TO_EVAL",
    "TRACKER_DISPLAY_NAMES",
    "CLASSES_TO_EVAL",
    "METRICS",
}


def _parse_args(config):
    parser = argparse.ArgumentParser()
    for setting, default in config.items():
        if isinstance(default, list) or setting in _LIST_SETTINGS:
            parser.add_argument("--" + setting, nargs="+")
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__

    for setting, value in args.items():
        if value is None:
            continue
        default = config[setting]
        if isinstance(default, bool):
            if value == "True":
                config[setting] = True
            elif value == "False":
                config[setting] = False
            else:
                raise ValueError(
                    f"Boolean argument --{setting} must be True or False, got {value}."
                )
        elif isinstance(default, int):
            config[setting] = int(value)
        elif isinstance(default, float):
            config[setting] = float(value)
        else:
            config[setting] = value
    return config


def main():
    freeze_support()

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config["DISPLAY_LESS_PROGRESS"] = False
    default_dataset_config = trackeval.datasets.UADETRAC.get_default_dataset_config()
    default_metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
    }

    config = {
        **default_eval_config,
        **default_dataset_config,
        **default_metrics_config,
    }
    config = _parse_args(config)

    eval_config = {
        key: value for key, value in config.items() if key in default_eval_config
    }
    dataset_config = {
        key: value for key, value in config.items() if key in default_dataset_config
    }
    metrics_config = {
        key: value for key, value in config.items() if key in default_metrics_config
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.UADETRAC(dataset_config)]
    metrics_list = []
    for metric_class in [
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity,
    ]:
        if metric_class.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric_class(metrics_config))
    if not metrics_list:
        raise ValueError("No evaluation metrics were selected.")

    evaluator.evaluate(dataset_list, metrics_list)


if __name__ == "__main__":
    main()
