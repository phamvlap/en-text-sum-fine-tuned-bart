import torch.nn as nn
import torch.optim as optim

ADAM = "adam"
ADAMW = "adamw"


def get_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    optimizer_alg = config["optimizer"].strip().lower()

    if optimizer_alg == ADAM:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
        )
    elif optimizer_alg == ADAMW:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(
            f"Unknown optimizer algorithm: {optimizer_alg}. Only support {ADAM} and {ADAMW}."
        )

    return optimizer
