import torch.nn as nn
import torch.optim as optim

ADAM = "adam"
ADAMW = "adamw"


def get_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    optimizer_alg = config["optimizer"].strip().lower()

    no_decay = ["bias"]
    grouped_parameters = [
        {
            "params": [
                p for n, p in model.parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p for n, p in model.parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_alg == ADAM:
        optimizer = optim.Adam(
            grouped_parameters,
            lr=config["lr"],
            betas=config["betas"],
            eps=config["eps"],
        )
    elif optimizer_alg == ADAMW:
        optimizer = optim.AdamW(
            grouped_parameters,
            lr=config["lr"],
            betas=config["betas"],
            eps=config["eps"],
        )
    else:
        raise ValueError(
            f"Unknown optimizer algorithm: {optimizer_alg}. Only support {ADAM} and {ADAMW}."
        )

    return optimizer
