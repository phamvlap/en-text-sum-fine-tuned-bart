import re
import contractions
import pandas as pd

from transformers import BartTokenizer
from typing import Any


def remove_urls(text: str) -> str:
    return re.sub(r"http[s]?:\/\/\S+|www\.\S+", "", text, flags=re.MULTILINE)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


def process_punctuations(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s\.,-]", " ", text)
    text = re.sub(r"([\.,-])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def process_en_text(text: str) -> str:
    text = str(text).lower()
    text = contractions.fix(text)

    text = remove_urls(text)
    text = remove_html_tags(text)
    text = process_punctuations(text)

    return text


def remove_rows_by_exceeded_length(
    df: pd.DataFrame,
    tokenizer: BartTokenizer,
    config: dict,
    seq_length: int,
) -> pd.DataFrame:
    source_field = config["text_src"]
    target_field = config["text_tgt"]

    if seq_length < 0:
        raise ValueError(
            f"seq_length must be greater than or equal to zero, got seq_length={seq_length}."
        )
    if any([feature not in df.columns for feature in [source_field, target_field]]):
        raise ValueError(f"{source_field} or {target_field} not found in dataset.")

    df_size = len(df)
    is_valid_rows = [True] * df_size

    for i in range(df_size):
        row = df.iloc[i]
        source_text = row[source_field]
        target_text = row[target_field]

        source_token_length = len(tokenizer.encode(source_text))
        target_token_length = len(tokenizer.encode(target_text))

        is_valid_rows[i] = (
            source_token_length + 2 <= seq_length
            and target_token_length + 1 <= seq_length
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
            f"seq_length must be greater than or equal to zero, got seq_length={seq_length}."
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
