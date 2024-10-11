import torch.optim as optim

from typing import Optional

NOAM_DECAY = "noam"
COSINE_ANNEALING = "cosine"


# Noam decay formula found at https://arxiv.org/pdf/1706.03762
def noam_decay(
    model_size: int,
    step: int,
    warmup_steps: int,
    factor: float = 1.0,
) -> float:
    step = max(step, 1)
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )


# Cosine annealing with warm restarts formula found at https://arxiv.org/pdf/1608.03983
def get_lr_scheduler(
    optimizer: optim.Optimizer,
    config: dict,
) -> Optional[optim.lr_scheduler.LRScheduler]:
    lr_scheduler_type = config["lr_scheduler"].strip().lower()
    lr_scheduler = None

    if lr_scheduler_type == NOAM_DECAY:
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: noam_decay(
                model_size=config["d_model"],
                step=step,
                warmup_steps=config["warmup_steps"],
            ),
        )
    elif lr_scheduler_type == COSINE_ANNEALING:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=config["T_0"],
            T_mult=config["T_mult"],
            eta_min=config["eta_min"],
        )

    return lr_scheduler
