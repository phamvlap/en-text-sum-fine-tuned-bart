import wandb

from typing import Optional

from .path import make_dir

DEFAULT_LOG_DIR = "wandb-logs"


def format_logs(logs: dict) -> dict:
    new_logs = {}
    eval_prefix = "eval_"
    eval_prefix_length = len(eval_prefix)

    for key, value in logs.items():
        if key.startswith(eval_prefix):
            new_logs["val/" + key[eval_prefix_length:]] = value
        else:
            new_logs["train/" + key] = value

    return new_logs


class WandbLogger:
    def __init__(self, project_name: str, config: dict, **kwargs) -> None:
        log_dir = kwargs.get("log_dir", DEFAULT_LOG_DIR)
        make_dir(log_dir)
        wandb_key = kwargs.get("key", None)

        args = {**kwargs}
        if "key" in args:
            del args["key"]
        if "log_dir" in args:
            del args["log_dir"]

        wandb.login(key=wandb_key)
        self.wb_run = wandb.init(
            project=project_name,
            config=config,
            dir=log_dir,
            **args,
        )

    def log(self, logs: dict, step: Optional[int] = None) -> None:
        logs = format_logs(logs)
        if step is None:
            self.wb_run.log(logs)
        else:
            self.wb_run.log(logs, step=step)

    def finish(self) -> None:
        self.wb_run.finish()
