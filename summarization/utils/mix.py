import random
import numpy as np
import torch

from pathlib import Path

from config.config import get_config

config = get_config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dir(dir_path: str) -> None:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def make_dirs(config: dict, dir_names: list[str]) -> None:
    for dir_name in dir_names:
        dir_path = get_dir_path(config[dir_name])
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_weights_file_path(config: dict, epoch: str) -> str:
    return f"{get_dir_path(dir_name=config['model_dir'])}/{config['model_basename']}{epoch}.pt"


def get_list_weights_file_paths(config: dict) -> None | list[Path]:
    model_dir = get_dir_path(dir_name=config["model_dir"])
    model_basename = config["model_basename"]
    weights_files = list(Path(model_dir).glob(pattern=f"{model_basename}*.pt"))
    if len(weights_files) == 0:
        return None
    return sorted(weights_files)


def noam_lr(
    model_size: int,
    step: int,
    warmup_steps: int,
    factor: float = 1.0,
) -> float:
    step = max(step, 1)
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )


def get_dir_path(dir_name: str) -> str:
    if config["base_dir"] is None or config["base_dir"] == "":
        return f"./{dir_name}"
    return f"{config['base_dir']}/{dir_name}"


def get_dataset_path(filename: str) -> str:
    if config["dataset_dir"] is None or config["dataset_dir"] == "":
        return f"./{filename}"
    return f"{config['dataset_dir']}/{filename}"
