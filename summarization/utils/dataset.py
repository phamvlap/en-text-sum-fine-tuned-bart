import pandas as pd

from transformers import BartTokenizer
from typing import Any


def remove_rows_by_invalid_seq_length(
    df: pd.DataFrame,
    tokenizer: BartTokenizer,
    config: dict,
    src_seq_length: int,
    tgt_seq_length: int,
) -> pd.DataFrame:
    if min(src_seq_length, tgt_seq_length) < 0:
        raise ValueError(
            f"src_seq_length and tgt_seq_length must be greater than or equal to zero, got src_seq_length={src_seq_length} and tgt_seq_length={tgt_seq_length}."
        )
    if src_seq_length < tgt_seq_length:
        raise ValueError(
            f"src_seq_length must be greater or equal than to tgt_seq_length, got src_seq_length={src_seq_length} and tgt_seq_length={tgt_seq_length}."
        )
    if any(
        [
            feature not in df.columns
            for feature in [config["text_src"], config["text_tgt"]]
        ]
    ):
        raise ValueError(
            f"{config['text_src']} or {config['text_tgt']} not found in dataset."
        )

    df_size = len(df)
    is_valid_rows = [True] * df_size

    for i in range(df_size):
        row = df.iloc[i]
        source_text = row[config["text_src"]]
        target_text = row[config["text_tgt"]]
        source_token_length = len(tokenizer.encode(source_text))
        target_token_length = len(tokenizer.encode(target_text))
        is_valid_rows[i] = (
            source_token_length + 2 <= src_seq_length
            and target_token_length + 1 <= tgt_seq_length
        )

    return df[is_valid_rows].reset_index(drop=True)


def truncate_exceeded_length(
    df: pd.DataFrame,
    tokenizer: BartTokenizer,
    config: dict[str, Any],
    seq_length: int,
) -> pd.DataFrame:
    source_field = config["text_src"]
    target_field = config["text_tgt"]

    if seq_length < 0:
        raise ValueError(
            f"seq_length must be greater than or equal to zero, got src_seq_length={seq_length}."
        )
    if any([feature not in df.columns for feature in [source_field, target_field]]):
        raise ValueError(f"{source_field} or {target_field} not found in dataset.")

    for idx in range(len(df)):
        row = df.iloc[idx]
        source_text = row[source_field]
        target_text = row[target_field]

        source_tokens = tokenizer.encode(source_text)
        target_tokens = tokenizer.encode(target_text)

        if len(source_tokens) > seq_length:
            source_tokens = source_tokens[:seq_length]
            df.loc[idx, source_field] = tokenizer.decode(
                source_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        if len(target_tokens) > seq_length:
            target_tokens = target_tokens[:seq_length]
            df.loc[idx, target_field] = tokenizer.decode(
                target_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

    return df


def retain_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    dropped_columns = []
    for column in df.columns:
        if column not in columns:
            dropped_columns.append(column)
    df = df.drop(columns=dropped_columns).reset_index(drop=True)
    return df
