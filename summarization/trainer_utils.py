import torch

from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple
from pathlib import Path

from bart.constants import RougeKey


@dataclass
class TrainingArguments:
    device: torch.device
    src_seq_length: int
    tgt_seq_length: int
    num_epochs: int
    checkpoint_dir: str
    model_basename: str
    initial_epoch: int = 0
    initial_global_step: int = 0
    eval_every_n_steps: int = 1000
    save_every_n_steps: int = 5000
    beam_size: int = 3
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
    accumulate: Literal["best", "avg"] = "best"
    max_grad_norm: Optional[float] = None
    f16_precision: bool = True
    use_ddp: bool = False
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    world_size: Optional[int] = None
    max_eval_steps: int = 100
    max_train_steps: int = -1
    max_saved_checkpoints: int = 2
    bart_tokenizer_dir: str = "tokenizer-bart"
    show_eval_progress: bool = False
    push_to_hub: bool = True
    hub_repo_name: str = "text-summarization-finetuned-bart"


def has_length(dataset: Dataset) -> bool:
    try:
        return len(dataset) is not None
    except TypeError:
        return False


def sorted_checkpoints(
    output_dir: str,
    checkpoint_prefix: str,
) -> list[str]:
    checkpoints = [
        str(path) for path in Path(output_dir).glob(pattern=f"{checkpoint_prefix}*.pt")
    ]

    return sorted(checkpoints)


def rotate_checkpoints(
    checkpoints: list[str],
    max_saved_total: Optional[int] = None,
) -> None:
    if max_saved_total is None or max_saved_total < 0:
        return

    num_deleted_checkpoints = max(0, len(checkpoints) - max_saved_total)
    deleted_checkpoints = checkpoints[:num_deleted_checkpoints]

    for checkpoint in deleted_checkpoints:
        if Path(checkpoint).exists():
            Path(checkpoint).unlink()


def get_last_checkpoint(output_dir: str, checkpoint_prefix: str) -> Optional[str]:
    checkpoints = sorted_checkpoints(
        output_dir=output_dir,
        checkpoint_prefix=checkpoint_prefix,
    )
    return checkpoints[-1] if len(checkpoints) > 0 else None


def get_checkpoint_path(checkpoint_dir: str, model_basename: str, step: int) -> str:
    return str(Path(checkpoint_dir) / f"{model_basename}_{step}.pt")
