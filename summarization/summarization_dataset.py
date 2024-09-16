import pandas as pd
import torch

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer
from typing import Literal

from bart.constants import SpecialToken
from .preprocessing import load_dataset
from .utils.tokenizer import load_tokenizer


class SummarizationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BartTokenizer,
        text_src: str,
        text_tgt: str,
        seq_length: int,
    ) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.text_src = text_src
        self.text_tgt = text_tgt
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(SpecialToken.EOS)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
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
        label = torch.cat(
            [
                Tensor(self.tokenizer.encode(text_tgt)),
                Tensor([self.eos_token_id]),
            ]
        )

        if len(src_tokens) > self.seq_length:
            src_tokens = torch.cat(
                [
                    src_tokens[: self.seq_length - 1],
                    Tensor([self.eos_token_id]),
                ]
            )
        if len(tgt_tokens) > self.seq_length:
            tgt_tokens = tgt_tokens[: self.seq_length]
        if len(label) > self.seq_length:
            label = torch.cat(
                [
                    label[: self.seq_length - 1],
                    Tensor([self.eos_token_id]),
                ]
            )

        src_tokens = src_tokens.type(torch.int64)
        tgt_tokens = tgt_tokens.type(torch.int64)
        label = label.type(torch.int64)

        assert src_tokens.size(-1) <= self.seq_length
        assert tgt_tokens.size(-1) <= self.seq_length
        assert label.size(-1) <= self.seq_length

        return {
            "src": src_tokens,
            "tgt": tgt_tokens,
            "label": label,
        }


def get_summarization_dataset(
    data_files_path: dict[str, str],
    split: Literal["train", "val", "test"],
    tokenizer: BartTokenizer,
    text_src: str,
    text_tgt: str,
    seq_length: int,
) -> SummarizationDataset:
    dataset = load_dataset(path=data_files_path[split])

    return SummarizationDataset(
        df=dataset,
        tokenizer=tokenizer,
        text_src=text_src,
        text_tgt=text_tgt,
        seq_length=seq_length,
    )


def collate_fn(batch: list, tokenizer: BartTokenizer) -> dict:
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

    src_batch, tgt_batch, label_batch = [], [], []
    for item in batch:
        src_batch.append(item["src"])
        tgt_batch.append(item["tgt"])
        label_batch.append(item["label"])

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
        "src": src_batch,
        "tgt": tgt_batch,
        "label": label_batch,
    }


def get_dataloader(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    tokenizer = load_tokenizer(bart_tokenizer_dir=config["tokenizer_bart_dir"])

    train_dataset = get_summarization_dataset(
        data_files_path=config["data_files_path"],
        split="train",
        tokenizer=tokenizer,
        text_src=config["text_src"],
        text_tgt=config["text_tgt"],
        seq_length=config["seq_length"],
    )
    val_dataset = get_summarization_dataset(
        data_files_path=config["data_files_path"],
        tokenizer=tokenizer,
        split="val",
        text_src=config["text_src"],
        text_tgt=config["text_tgt"],
        seq_length=config["seq_length"],
    )
    test_dataset = get_summarization_dataset(
        data_files_path=config["data_files_path"],
        tokenizer=tokenizer,
        split="test",
        text_src=config["text_src"],
        text_tgt=config["text_tgt"],
        seq_length=config["seq_length"],
    )

    batch_size_train = config["batch_size_train"]
    batch_size_val = config["batch_size_val"]
    batch_size_test = config["batch_size_test"]

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch=batch, tokenizer=tokenizer),
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch=batch, tokenizer=tokenizer),
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_test,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch=batch, tokenizer=tokenizer),
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )
