from pathlib import Path
from typing import Optional


def join_path(base_dir: str, sub_path: str) -> str:
    return str(Path(base_dir) / sub_path)


def make_dir(dir_path: str) -> None:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def make_dirs(config: dict, dir_names: list[str]) -> None:
    for dir_name in dir_names:
        dir_path = config[dir_name]
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_weights_file_path(
    model_basedir: str,
    model_basename: str,
    epoch: int,
) -> Optional[str]:
    file_path = join_path(
        base_dir=model_basedir,
        sub_path=f"{model_basename}{epoch}.pt",
    )
    if not Path(file_path).exists():
        return None
    return file_path


def get_list_weights_file_paths(config: dict) -> None | list[Path]:
    model_dir = config["model_dir"]
    model_basename = config["model_basename"]
    weights_files = list(Path(model_dir).glob(pattern=f"{model_basename}*.pt"))
    if len(weights_files) == 0:
        return None
    return sorted(weights_files)
