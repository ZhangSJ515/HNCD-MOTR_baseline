# @Author       : Ruopeng Gao
# @Date         : 2022/7/6
# @Description  : Data operators, such as data read, dataset, dataloader.

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from .dancetrack import build as build_dancetrack
from .airmot import build as build_airmot
from .uadetrac import build as build_uadetrac
from .mot17 import build as build_mot17
from .bdd100k import build as build_bbd100k
from .mot import MOTDataset
from .utils import collate_fn
from utils.utils import is_distributed


_UADETRAC_ALIASES = {"UA-DETRAC", "UADETRAC", "UA_DETRAC"}


def build_dataset(config: dict, split: str) -> MOTDataset:
    # UA-DETRAC is a one-class MOT dataset.  ``DATASET`` is intentionally kept
    # as DanceTrack in the UA-DETRAC config so the original HNCD model/criterion
    # retain their one-class heads without invasive changes.  DATASET_ADAPTER
    # selects the actual loader.
    dataset_adapter = config.get("DATASET_ADAPTER", config["DATASET"])
    if dataset_adapter in _UADETRAC_ALIASES:
        return build_uadetrac(config=config, split=split)

    if config["DATASET"] == "DanceTrack":
        return build_dancetrack(config=config, split=split)
    elif config["DATASET"] == "SportsMOT":
        return build_dancetrack(config=config, split=split)
    elif config["DATASET"] == "AirMot":
        return build_airmot(config=config, split=split)
    elif config["DATASET"] == "MOT17":
        return build_mot17(config=config, split=split)
    elif config["DATASET"] == "MOT17_SPLIT":
        return build_mot17(config=config, split=split)
    elif config["DATASET"] == "BDD100K":
        return build_bbd100k(config=config, split=split)
    raise ValueError(f"Dataset {config['DATASET']} is not supported!")


def build_sampler(dataset: MOTDataset, shuffle: bool):
    if is_distributed():
        return DistributedSampler(dataset=dataset, shuffle=shuffle)
    return RandomSampler(dataset) if shuffle else SequentialSampler(dataset)


def build_dataloader(dataset: MOTDataset, sampler, batch_size: int, num_workers: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
