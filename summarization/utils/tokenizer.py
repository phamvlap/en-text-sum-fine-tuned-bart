import shutil
import pandas as pd

from pathlib import Path
from typing import Generator
from torch.utils.data import Dataset
from transformers import BartTokenizer
from tokenizers import ByteLevelBPETokenizer

from bart.constants import TokenizerType, SpecialToken
from .mix import get_constants_from_module, make_dir, ensure_exist_path


class CustomBartTokenizer:
    def __init__(
        self,
        dataset: Dataset | pd.DataFrame,
        vocab_size: int,
        special_tokens: list[str],
        min_freq: int,
        model_type: str,
    ) -> None:
        self.dataset_iterator = self._get_iterator(dataset)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.min_freq = min_freq
        self.model_type = model_type.strip().lower()

    def _get_iterator(
        self,
        dataset: Dataset | pd.DataFrame,
    ) -> Generator[str, None, None]:
        for item in dataset:
            yield item

    def train(self, show_progress: bool = True) -> BartTokenizer:
        print(f"Training tokenizer with model type: {self.model_type}...")

        if self.model_type == TokenizerType.BYTE_LEVEL_BPE:
            tokenizer = ByteLevelBPETokenizer(
                end_of_word_suffix=SpecialToken.BYTE_LEVEL_BPE_SUFFIX,
            )
            tokenizer.train_from_iterator(
                self.dataset_iterator,
                vocab_size=self.vocab_size,
                min_frequency=self.min_freq,
                special_tokens=self.special_tokens,
                show_progress=show_progress,
            )
        else:
            raise ValueError(
                f"{self.model_type} not supported. Please choose from {', '.join(get_constants_from_module(module=TokenizerType).values())}"
            )

        tmp_tokenizer_dir = "tokenizer-tmp"

        if Path(tmp_tokenizer_dir).exists():
            shutil.rmtree(tmp_tokenizer_dir)

        make_dir(dir_path=tmp_tokenizer_dir)

        tokenizer.model.save(tmp_tokenizer_dir)

        bart_tokenizer = BartTokenizer(
            vocab_file=f"{tmp_tokenizer_dir}/vocab.json",
            merges_file=f"{tmp_tokenizer_dir}/merges.txt",
            clean_up_tokenization_spaces=False,
        )

        if Path(tmp_tokenizer_dir).exists():
            shutil.rmtree(tmp_tokenizer_dir)

        return bart_tokenizer


def load_tokenizer(bart_tokenizer_dir: str) -> BartTokenizer:
    if not ensure_exist_path(bart_tokenizer_dir):
        raise ValueError(f"Tokenizer path {bart_tokenizer_dir} not found.")

    bart_tokenizer = BartTokenizer.from_pretrained(str(bart_tokenizer_dir))

    return bart_tokenizer


def save_tokenizer(
    bart_tokenizer: BartTokenizer,
    bart_tokenizer_dir: str | Path,
) -> None:
    bart_tokenizer.save_pretrained(str(bart_tokenizer_dir))
