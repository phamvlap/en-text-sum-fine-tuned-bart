import torch

from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple

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
    eval_bert_score: bool = True
    rescale: bool = True
    log_examples: bool = True
    logging_steps: int = 100
    rouge_keys: List[str] | Tuple[str] = field(
        default_factory=lambda: [
            RougeKey.ROUGE_1,
            RougeKey.ROUGE_2,
            RougeKey.ROUGE_L,
        ]
    )
    use_stemmer: bool = True
    truncation: bool = True
    accumulate: Literal["best", "avg"] = "best"
    max_grad_norm: Optional[float] = None
    f16_precision: bool = True
    use_ddp: bool = False
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    world_size: Optional[int] = None
    max_eval_steps: int = 100
    max_train_steps: int = 5000


def has_length(dataset: Dataset) -> bool:
    try:
        return len(dataset) is not None
    except TypeError:
        return False
