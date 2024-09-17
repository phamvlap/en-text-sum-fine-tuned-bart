import types
import pandas as pd
import torch.nn as nn

from pathlib import Path
from .path import make_dir


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
