import torch

from dataclasses import dataclass
from typing import Literal, Optional

from bart.constants import RougeKey


@dataclass
class TrainingArguments:
    device: torch.device
    seq_length: int
    num_epochs: int
    model_dir: str
    model_basename: str
    initial_epoch: int = 0
    initial_global_step: int = 0
    eval_every_n_steps: int = 5000
    save_every_n_steps: int = 5000
    beam_size: Optional[int] = None
    topk: int = 1
    log_examples: bool = True
    logging_steps: int = 100
    rouge_keys: list[str] | tuple[str] = (
        RougeKey.ROUGE_1,
        RougeKey.ROUGE_2,
        RougeKey.ROUGE_L,
    )
    use_stemmer: bool = True
    accumulate: Literal["best", "avg"] = "best"
    max_grad_norm: Optional[float] = None
    f16_precision: bool = True
