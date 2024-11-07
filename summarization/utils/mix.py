import yaml
import types
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from datetime import datetime
from pytz import timezone
from typing import Any

from bart.constants import SETTING_CONFIG_FILE


def make_dir(dir_path: str) -> None:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def write_to_csv(
    columns: list[str],
    data: list[list[int | float | str]],
    file_path: str | Path,
) -> pd.DataFrame:
    obj = {}
    for i, column in enumerate(columns):
        obj[column] = data[i]

    df = pd.DataFrame(obj)

    file_path = str(file_path)
    dir_path_splits = file_path.rsplit("/", 1)
    if len(dir_path_splits) > 1:
        make_dir(dir_path=dir_path_splits[0])

    df.to_csv(file_path, index=False)

    return df


def get_constants_from_module(module: object) -> dict[str, Any]:
    constants = vars(module)
    super_keys = ["__module__", "__init__", "__dict__", "__weakref__", "__doc__"]

    normal_constants = {}
    for key, value in constants.items():
        if key not in super_keys and not isinstance(value, types.FunctionType):
            normal_constants[key] = constants[key]

    return normal_constants


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path: str) -> dict[str, Any]:
    if not ensure_exist_path(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["eps"] = float(config["eps"])
    config["betas"] = tuple(config["betas"])
    config["eta_min"] = float(config["eta_min"])
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def is_torch_cuda_available() -> bool:
    return torch.cuda.is_available()


def write_to_yaml(data: dict[str, Any], file_path: str | Path) -> None:
    file_path = str(file_path)
    dir_path_splits = file_path.rsplit("/", 1)
    if len(dir_path_splits) > 1:
        make_dir(dir_path=dir_path_splits[0])

    for key in ["special_tokens", "betas", "rouge_keys"]:
        if key in data:
            data[key] = list(data[key])

    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def get_current_time(to_string: bool = False) -> datetime | str:
    timezone_name = "Asia/Ho_Chi_Minh"
    tz = timezone(timezone_name)

    now_date = datetime.now(tz)

    if to_string:
        formatted_date = now_date.strftime("%H-%M-%S_%m-%d-%Y")
        return formatted_date
    return now_date


def print_once(config: dict, text: str) -> None:
    if is_torch_cuda_available():
        if config["use_ddp"]:
            if config["rank"] == 0:
                print(text)
        else:
            print(text)
    else:
        print(text)


def update_setting_config(new_config: dict[str, Any]) -> dict[str, Any]:
    config = load_config(SETTING_CONFIG_FILE)

    config = {**config, **new_config}
    write_to_yaml(config, SETTING_CONFIG_FILE)

    return config


def ensure_exist_path(path: str | Path) -> bool:
    path = str(path)
    if not Path(path).exists():
        return False
    return True
