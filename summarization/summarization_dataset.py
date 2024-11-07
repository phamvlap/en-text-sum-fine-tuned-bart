import pandas as pd
import torch

from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer
from typing import Literal, Optional, Any

from bart.constants import SpecialToken
from .preprocess import load_dataset
from .trainer_utils import has_length


class SummarizationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BartTokenizer,
        text_src: str,
        text_tgt: str,
        src_seq_length: int,
        tgt_seq_length: int,
        attach_text: bool = False,
    ) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.text_src = text_src
        self.text_tgt = text_tgt
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(SpecialToken.EOS)
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.attach_text = attach_text

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        row = self.df.iloc[idx]

        text_src = row[self.text_src]
        text_tgt = row[self.text_tgt]

        src_tokens = torch.cat(
            [
                Tensor([self.bos_token_id]),
                Tensor(self.tokenizer.encode(text_src)),
                Tensor([self.eos_token_id]),
            ]
        )
        tgt_tokens = torch.cat(
            [
                Tensor([self.bos_token_id]),
                Tensor(self.tokenizer.encode(text_tgt)),
            ]
        )
        labels = torch.cat(
            [
                Tensor(self.tokenizer.encode(text_tgt)),
                Tensor([self.eos_token_id]),
            ]
        )

        if len(src_tokens) > self.src_seq_length:
            src_tokens = torch.cat(
                [
                    src_tokens[: self.src_seq_length - 1],
                    Tensor([self.eos_token_id]),
                ]
            )
        if len(tgt_tokens) > self.tgt_seq_length:
            tgt_tokens = tgt_tokens[: self.tgt_seq_length]
        if len(labels) > self.tgt_seq_length:
            labels = torch.cat(
                [
                    labels[: self.tgt_seq_length - 1],
                    Tensor([self.eos_token_id]),
                ]
            )

        src_tokens = src_tokens.type(torch.int32)
        tgt_tokens = tgt_tokens.type(torch.int32)
        # labels must be of type long (torch.int64) for CrossEntropyLoss
        labels = labels.type(torch.int64)

        assert src_tokens.size(-1) <= self.src_seq_length
        assert tgt_tokens.size(-1) <= self.tgt_seq_length
        assert labels.size(-1) <= self.tgt_seq_length

        data = {
            "encoder_input": src_tokens,
            "decoder_input": tgt_tokens,
            "labels": labels,
        }

        if self.attach_text:
            data["text_src"] = text_src
            data["text_tgt"] = text_tgt

        return data


def get_summarization_dataset(
    data_files_path: dict[str, str],
    split: Literal["train", "val", "test"],
    tokenizer: BartTokenizer,
    text_src: str,
    text_tgt: str,
    src_seq_length: int,
    tgt_seq_length: int,
    attach_text: bool = False,
) -> SummarizationDataset:
    if split not in ["train", "val", "test"]:
        raise ValueError(f"split must be one of ['train', 'val', 'test'], got {split}")

    dataset = load_dataset(path=data_files_path[split])

    return SummarizationDataset(
        df=dataset,
        tokenizer=tokenizer,
        text_src=text_src,
        text_tgt=text_tgt,
        src_seq_length=src_seq_length,
        tgt_seq_length=tgt_seq_length,
        attach_text=attach_text,
    )


def collate_fn(
    batch: list[dict[str, Tensor | str]],
    tokenizer: BartTokenizer,
) -> dict[str, list[Tensor]]:
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

    src_batch, tgt_batch, label_batch = [], [], []
    for item in batch:
        src_batch.append(item["encoder_input"])
        tgt_batch.append(item["decoder_input"])
        label_batch.append(item["labels"])

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_token_id,
    )
    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=pad_token_id,
    )
    label_batch = pad_sequence(
        label_batch,
        batch_first=True,
        padding_value=pad_token_id,
    )

    return {
        "encoder_input": src_batch,
        "decoder_input": tgt_batch,
        "labels": label_batch,
    }


def get_dataloader(
    tokenizer: BartTokenizer,
    split: Literal["train", "val", "test"],
    batch_size: int,
    shuffle: bool,
    config: dict[str, Any],
) -> DataLoader:
    if split not in ["train", "val", "test"]:
        raise ValueError(f"split must be one of ['train', 'val', 'test'], got {split}")

    dataset = get_summarization_dataset(
        data_files_path=config["data_files_path"],
        split=split,
        tokenizer=tokenizer,
        text_src=config["text_src"],
        text_tgt=config["text_tgt"],
        src_seq_length=config["src_seq_length"],
        tgt_seq_length=config["tgt_seq_length"],
        attach_text=config["attach_text"],
    )

    sampler = _get_sampler(dataset=dataset, config=config)

    is_shuffle = sampler is None and shuffle

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        collate_fn=lambda batch: collate_fn(batch=batch, tokenizer=tokenizer),
        pin_memory=True,
        sampler=sampler,
        num_workers=config["num_workers"] if config["num_workers"] > 0 else 0,
    )

    return dataloader


def _get_sampler(dataset: Dataset, config: dict[str, Any]) -> Optional[Sampler]:
    if not has_length(dataset):
        return None
    sampler = None
    if config["use_ddp"]:
        sampler = DistributedSampler(
            dataset,
            num_replicas=config["world_size"],
            rank=config["rank"],
            shuffle=True,
            seed=config["seed"],
        )
    return sampler
