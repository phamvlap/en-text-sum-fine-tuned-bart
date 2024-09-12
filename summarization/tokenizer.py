import shutil
import pandas as pd

from torch.utils.data import Dataset
from pathlib import Path
from typing import Generator

from tokenizers import Tokenizer, ByteLevelBPETokenizer
from tokenizers.normalizers import Sequence, Lowercase, Normalizer
from tokenizers.pre_tokenizers import Whitespace, PreTokenizer
from tokenizers.models import WordLevel, BPE, WordPiece, Model
from tokenizers.trainers import WordLevelTrainer, BpeTrainer, WordPieceTrainer, Trainer

from transformers import BartTokenizer

from bart.constants import TokenizerType, SpecialToken
from .utils.mix import get_dir_path


class CustomBartTokenizer:
    def __init__(
        self,
        dataset: Dataset | pd.DataFrame,
        vocab_size: int,
        special_tokens: list[str],
        min_freq: int,
        model_type: str,
    ) -> None:
        self.dataset = self.__get_iterator(dataset)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.min_freq = min_freq
        self.model_type = model_type

    def __get_iterator(
        self,
        dataset: Dataset | pd.DataFrame,
    ) -> Generator[str, None, None]:
        for item in dataset:
            yield item

    def train(self, config: dict, show_progress: bool = False) -> BartTokenizer:
        if self.model_type == TokenizerType.BYTE_LEVEL_BPE:
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train_from_iterator(
                self.dataset,
                vocab_size=self.vocab_size,
                min_frequency=self.min_freq,
                special_tokens=self.special_tokens,
                show_progress=show_progress,
            )
        else:
            normalizer, pre_tokenizer, model, trainer = self.__setup_tokenizer(
                show_progress=show_progress,
            )

            tokenizer = Tokenizer(model)
            tokenizer.normalizer = normalizer
            tokenizer.pre_tokenizer = pre_tokenizer

            tokenizer.train_from_iterator(iterator=self.dataset, trainer=trainer)

        tmp_tokenizer_dir = get_dir_path(dir_name=config["tokenizer_tmp_dir"])
        Path(tmp_tokenizer_dir).mkdir(parents=True, exist_ok=True)

        tokenizer.model.save(tmp_tokenizer_dir)

        bart_tokenizer = BartTokenizer(
            vocab_file=f"{tmp_tokenizer_dir}/vocab.json",
            merges_file=f"{tmp_tokenizer_dir}/merges.txt",
        )

        if Path(tmp_tokenizer_dir).exists():
            shutil.rmtree(tmp_tokenizer_dir)

        return bart_tokenizer

    def __setup_tokenizer(
        self,
        show_progress: bool = False,
    ) -> tuple[Normalizer, PreTokenizer, Model, Trainer]:
        if self.model_type == TokenizerType.WORD_LEVEL:
            normalizer = Sequence([Lowercase()])
            pre_tokenizer = Whitespace()
            model = WordLevel(unk_token=SpecialToken.UNK)

            trainer = WordLevelTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_freq,
                special_tokens=self.special_tokens,
                show_progress=show_progress,
            )
        elif self.model_type == TokenizerType.BPE:
            normalizer = Sequence([Lowercase()])
            pre_tokenizer = Whitespace()
            model = BPE(unk_token=SpecialToken.UNK)

            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_freq,
                special_tokens=self.special_tokens,
                show_progress=show_progress,
            )
        elif self.model_type == TokenizerType.WORD_PIECE:
            normalizer = Sequence([Lowercase()])
            pre_tokenizer = Whitespace()
            model = WordPiece(unk_token=SpecialToken.UNK)

            trainer = WordPieceTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_freq,
                special_tokens=self.special_tokens,
                show_progress=show_progress,
            )

        return normalizer, pre_tokenizer, model, trainer


def load_tokenizer(bart_tokenizer_dir: str) -> BartTokenizer:
    full_dir_path = get_dir_path(dir_name=bart_tokenizer_dir)
    if not Path(full_dir_path).exists():
        raise ValueError(f"Tokenizer path {full_dir_path} not found.")

    bart_tokenizer = BartTokenizer.from_pretrained(str(full_dir_path))

    return bart_tokenizer


def save_tokenizer(
    bart_tokenizer: BartTokenizer,
    bart_tokenizer_dir: str | Path,
) -> None:
    bart_tokenizer.save_pretrained(get_dir_path(dir_name=bart_tokenizer_dir))
