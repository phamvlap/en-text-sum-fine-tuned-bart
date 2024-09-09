from torch.utils.data import Dataset
from pathlib import Path
from typing import Generator

from tokenizers import Tokenizer
from tokenizers.normalizers import Sequence, Lowercase, Normalizer
from tokenizers.pre_tokenizers import Whitespace, PreTokenizer
from tokenizers.models import WordLevel, BPE, WordPiece, Model
from tokenizers.trainers import WordLevelTrainer, BpeTrainer, WordPieceTrainer, Trainer

from bart.constants import TokenizerType, SpecialToken
from utils.mix import make_dir


class HuggingfaceAPITokenizer:
    def __init__(
        self,
        dataset: Dataset,
        vocab_size: int,
        special_tokens: list[str],
        min_freq: int,
        model_type: str,
    ) -> None:
        self.dataset = self.get_iterator(dataset)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.min_freq = min_freq
        self.model_type = model_type

    def get_iterator(self, dataset: Dataset) -> Generator[str, None, None]:
        for item in dataset:
            yield item

    def train(self) -> Tokenizer:
        normalizer, pre_tokenizer, model, trainer = self.__setup_tokenizer()

        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.train_from_iterator(iterator=self.dataset, trainer=trainer)

        return tokenizer

    def __setup_tokenizer(self) -> tuple[Normalizer, PreTokenizer, Model, Trainer]:
        if self.model_type == TokenizerType.WORD_LEVEL:
            normalizer = Sequence([Lowercase()])
            pre_tokenizer = Whitespace()
            model = WordLevel(unk_token=SpecialToken.UNK)

            trainer = WordLevelTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_freq,
                special_tokens=self.special_tokens,
            )
        elif self.model_type == TokenizerType.BPE:
            normalizer = Sequence([Lowercase()])
            pre_tokenizer = Whitespace()
            model = BPE(unk_token=SpecialToken.UNK)

            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_freq,
                special_tokens=self.special_tokens,
            )
        elif self.model_type == TokenizerType.WORD_PIECE:
            normalizer = Sequence([Lowercase()])
            pre_tokenizer = Whitespace()
            model = WordPiece(unk_token=SpecialToken.UNK)

            trainer = WordPieceTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_freq,
                special_tokens=self.special_tokens,
            )

        return normalizer, pre_tokenizer, model, trainer


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    if not Path(tokenizer_path).exists():
        raise ValueError(f"Tokenizer path {tokenizer_path} not found.")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, tokenizer_path: str) -> None:
    tokenizer_dir = tokenizer_path.rsplit("/", 1)[0]
    make_dir(tokenizer_dir)
    tokenizer.save(tokenizer_path)
