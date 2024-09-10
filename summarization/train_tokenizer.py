import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from transformers import BartTokenizer

from .tokenizer import CustomBartTokenizer, save_tokenizer


def train_tokenizer(
    dataset: Dataset | pd.DataFrame,
    vocab_size: int,
    special_tokens: list[str],
    min_freq: int,
    model_type: str,
    bart_tokenizer_path: str | Path,
    config: dict,
    show_progress: bool = False,
) -> BartTokenizer:
    bart_tokenizer = CustomBartTokenizer(
        dataset=dataset,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_freq=min_freq,
        model_type=model_type,
    )

    bart_tokenizer = bart_tokenizer.train(
        config=config,
        show_progress=show_progress,
    )

    save_tokenizer(
        bart_tokenizer=bart_tokenizer,
        bart_tokenizer_dir=bart_tokenizer_path,
    )

    return bart_tokenizer
