import wandb

from typing import Optional

from bart.model import FineTunedBartForGenerationConfig
from .path import make_dir
from ..trainer import TrainingConfig

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
    def __init__(
        self,
        project_name: str,
        trainer_args: TrainingConfig,
        model_config: FineTunedBartForGenerationConfig,
        **kwargs,
    ) -> None:
        combined_config = {
            "model_config": model_config.__dict__,
            "trainer_args": trainer_args.__dict__,
        }
        log_dir = kwargs.get("log_dir", DEFAULT_LOG_DIR)
        make_dir(log_dir)
        wandb_key = kwargs.get("wandb_key", None)

        args = {**kwargs}
        if "wandb_key" in args:
            del args["wandb_key"]

        wandb.login(key=wandb_key)
        self.wb_run = wandb.init(
            project=project_name,
            config=combined_config,
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
