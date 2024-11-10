import wandb

from typing import Optional, Any

from .mix import make_dir

DEFAULT_LOG_DIR = "wandb-logs"
VALID_PREFIX_KEY = "eval_"


def format_logs(logs: dict[str, int | float]) -> dict[str, int | float]:
    new_logs = {}
    eval_prefix_length = len(VALID_PREFIX_KEY)

    for key, value in logs.items():
        if key.startswith(VALID_PREFIX_KEY):
            new_logs["val/" + key[eval_prefix_length:]] = value
        else:
            new_logs["train/" + key] = value

    return new_logs


class WandbLogger:
    def __init__(self, project_name: str, config: dict[str, Any], **kwargs) -> None:
        log_dir = kwargs.get("log_dir", DEFAULT_LOG_DIR)
        make_dir(log_dir)
        wandb_key = kwargs.get("key", None)

        args = {**kwargs}
        for key in ["key", "log_dir"]:
            if key in args:
                del args[key]

        wandb.login(key=wandb_key)
        self.wb_run = wandb.init(
            project=project_name,
            config=config,
            dir=log_dir,
            **args,
        )

    def log(self, logs: dict[str, int | float], step: Optional[int] = None) -> None:
        logs = format_logs(logs)
        if step is None:
            self.wb_run.log(logs)
        else:
            self.wb_run.log(logs, step=step)

    def finish(self) -> None:
        self.wb_run.finish()
