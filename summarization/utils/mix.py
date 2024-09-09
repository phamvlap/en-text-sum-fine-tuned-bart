import random
import numpy as np
import torch

from pathlib import Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dir(dir_path: str) -> None:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def make_dirs(config: dict, dir_names: list[str]) -> None:
    for dir_name in dir_names:
        dir_path = config[dir_name]
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_weights_file_path(config: dict, epoch: str) -> str:
    return f"{config['model_dir']}/{config['model_basename']}{epoch}.pt"


def get_list_weights_file_paths(config: dict) -> None | list[Path]:
    model_dir = config["model_dir"]
    model_basename = config["model_basename"]
    weights_files = list(Path(model_dir).glob(pattern=f"{model_basename}*.pt"))
    if len(weights_files) == 0:
        return None
    return sorted(weights_files)
