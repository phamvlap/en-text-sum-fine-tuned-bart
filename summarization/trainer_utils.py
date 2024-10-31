import torch

from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple
from pathlib import Path

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
    max_train_steps: int = -1
    greater_checking: bool = False
    checked_metric: str = "loss"
    max_saved_checkpoints: int = 2


def has_length(dataset: Dataset) -> bool:
    try:
        return len(dataset) is not None
    except TypeError:
        return False


def sorted_checkpoints(
    output_dir: str,
    checkpoint_prefix: str,
    best_checkpoint: Optional[str] = None,
) -> list[str]:
    globbed_checkpoints = [
        str(path) for path in Path(output_dir).glob(pattern=f"{checkpoint_prefix}*.pt")
    ]

    checkpoints = globbed_checkpoints
    if (
        best_checkpoint is not None
        and str(Path(best_checkpoint)) in globbed_checkpoints
    ):
        best_checkpoint_path = str(Path(best_checkpoint))
        best_checkpoint_index = globbed_checkpoints.index(best_checkpoint_path)
        globbed_checkpoints.pop(best_checkpoint_index)
        checkpoints = globbed_checkpoints + [best_checkpoint_path]

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


def determine_best_metric_value(
    metric_scores: dict[str, int | float],
    checked_metric: str,
    greater_checking: bool,
    best_metric_value: int | float,
    output_dir: str,
    checkpoint_prefix: str,
    step: int,
) -> tuple[Optional[float], Optional[str]]:
    checked_metric = checked_metric.lower()

    if checked_metric not in metric_scores:
        raise ValueError(
            f"{checked_metric} not found in metric_scores, keys availability {', '.join(list(metric_scores.keys()))}"
        )

    is_new_best_metric = False

    if greater_checking:
        if metric_scores[checked_metric] > best_metric_value:
            is_new_best_metric = True
    else:
        if metric_scores[checked_metric] < best_metric_value:
            is_new_best_metric = True

    new_best_metric_value = None
    best_checkpoint_path = None

    if is_new_best_metric:
        new_best_metric_value = metric_scores[checked_metric]
        best_checkpoint_path = str(Path(output_dir) / f"{checkpoint_prefix}_{step}.pt")

    return new_best_metric_value, best_checkpoint_path
