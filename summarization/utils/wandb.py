import wandb

from typing import Optional


class WandbWriter:
    def __init__(
        self,
        project: str,
        config: dict,
        log_dir: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.project = project
        self.config = config
        wandb.init(
            project=self.project,
            config=self.config,
            dir=log_dir,
            notes=notes,
        )

    def log(self, data: dict) -> None:
        wandb.log(data)
