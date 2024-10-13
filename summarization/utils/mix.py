import yaml
import types
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from .path import make_dir


def write_to_csv(
    columns: list[str],
    data: list[list],
    file_path: str | Path,
) -> pd.DataFrame:
    obj = {}
    for i, column in enumerate(columns):
        obj[column] = data[i]

    df = pd.DataFrame(obj)

    file_path = str(file_path)
    dir_path = file_path.rsplit("/", 1)[0]
    make_dir(dir_path=dir_path)

    df.to_csv(file_path, index=False)

    return df


def get_constants_from_module(module: object) -> dict:
    constants = vars(module)
    super_keys = ["__module__", "__init__", "__dict__", "__weakref__", "__doc__"]

    normal_constants = {}
    for key, value in constants.items():
        if key not in super_keys and not isinstance(value, types.FunctionType):
            normal_constants[key] = constants[key]

    return normal_constants


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["eps"] = float(config["eps"])
    config["eta_min"] = float(config["eta_min"])
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def is_torch_cuda_available() -> bool:
    return torch.cuda.is_available()


def write_to_yaml(data: dict, file_path: str | Path) -> None:
    file_path = str(file_path)
    parent_dir = file_path.rsplit("/", 1)[0]
    make_dir(parent_dir)

    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
