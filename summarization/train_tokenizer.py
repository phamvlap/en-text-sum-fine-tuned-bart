from pathlib import Path
from torch.utils.data import Dataset
from tokenizers import Tokenizer


from .tokenizer import HuggingfaceAPITokenizer, save_tokenizer


def train_tokenizer(
    dataset: Dataset,
    vocab_size: int,
    special_tokens: list[str],
    min_freq: int,
    model_type: str,
    tokenizer_path: str | Path,
) -> Tokenizer:
    tokenizer = HuggingfaceAPITokenizer(
        dataset=dataset,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_freq=min_freq,
        model_type=model_type,
    )

    tokenizer = tokenizer.train()
    save_tokenizer(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
    )

    return tokenizer
