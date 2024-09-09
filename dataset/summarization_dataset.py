import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer

from bart.constants import SpecialToken
from summarization.tokenizer import load_tokenizer


def load_dataset(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file {path} not found")
    return pd.read_csv(path)


class SummarizationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        config: dict,
    ) -> None:
        super().__init__()
        self.df = df
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.text_src = config["text_src"]
        self.text_tgt = config["text_tgt"]
        self.bos_token_id = self.tokenizer_src.token_to_id(SpecialToken.BOS)
        self.eos_token_id = self.tokenizer_src.token_to_id(SpecialToken.EOS)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        text_src = row[self.text_src]
        text_tgt = row[self.text_tgt]

        src_tokens = (
            [self.bos_token_id]
            + self.tokenizer_src.encode(text_src).ids
            + [self.eos_token_id]
        )
        tgt_tokens = [self.bos_token_id] + self.tokenizer_tgt.encode(text_tgt).ids
        label = self.tokenizer_tgt.encode(text_tgt).ids + [self.eos_token_id]

        return {
            "src": src_tokens,
            "tgt": tgt_tokens,
            "label": label,
        }


def collate_fn(batch: list, tokenizer_src: Tokenizer) -> dict:
    pad_token_id = tokenizer_src.token_to_id(SpecialToken.PAD)

    src_batch, tgt_batch, label_batch = [], [], []
    for item in batch:
        src = torch.tensor(item["src"], dtype=torch.int64)
        tgt = torch.tensor(item["tgt"], dtype=torch.int64)
        label = torch.tensor(item["label"], dtype=torch.int64)

        src_batch.append(src)
        tgt_batch.append(tgt)
        label_batch.append(label)

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
    tokenizer_src = load_tokenizer(config["tokenizer_src_path"])
    tokenizer_tgt = load_tokenizer(config["tokenizer_tgt_path"])

    train_ds = load_dataset(path=config["train_ds_path"])
    val_ds = load_dataset(path=config["val_ds_path"])
    test_ds = load_dataset(path=config["test_ds_path"])

    batch_size_train = config["batch_size_train"]
    batch_size_val = config["batch_size_val"]
    batch_size_test = config["batch_size_test"]

    train_dataset = SummarizationDataset(
        df=train_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        config=config,
    )
    val_dataset = SummarizationDataset(
        df=val_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        config=config,
    )
    test_dataset = SummarizationDataset(
        df=test_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        config=config,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch=batch, tokenizer_src=tokenizer_src),
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch=batch, tokenizer_src=tokenizer_src),
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_test,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch=batch, tokenizer_src=tokenizer_src),
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )
