import re
import contractions
import pandas as pd
import numpy as np

from transformers import BartTokenizer

from bart.constants import SentenceContractions


def clean_source_feature(text: str) -> str:
    cleaned_text = re.sub(r"\n", "", text)
    cleaned_text = re.sub(r"[\.;],", ".", cleaned_text)
    cleaned_text = re.sub(r",{2,}", ",", cleaned_text)
    cleaned_text = re.sub(r"\.{2,}", ".", cleaned_text)
    cleaned_text = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned_text)
    cleaned_text = cleaned_text + "."

    return cleaned_text


def clean_target_feature(summary: str) -> str:
    cleaned_summary = re.sub(r"\n", "", summary)
    cleaned_summary = re.sub(r"\.,", ". ", cleaned_summary)
    cleaned_summary = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned_summary)
    cleaned_summary = cleaned_summary + "."

    return cleaned_summary


def remove_urls(text: str) -> str:
    return re.sub(r"http[s]?:\/\/\S+|www\.\S+", "", text, flags=re.MULTILINE)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


def process_punctuations(text: str) -> str:
    text = re.sub(r"([^\w\s\.,-])", "", text)
    text = re.sub(r"([\.,-])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def process_en_text(text: str, conditions: list[str] = []) -> str:
    if SentenceContractions.LOWERCASE in conditions:
        text = str(text).lower()
    if SentenceContractions.CONTRACTIONS in conditions:
        text = contractions.fix(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = process_punctuations(text)

    return text


def process_features(
    df: pd.DataFrame,
    features: list[str],
    conditions: list[str] = [],
) -> pd.DataFrame:
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"{feature} not found in dataset.")

    processed_df = df.copy()

    for feature in features:
        processed_df.loc[:, feature] = processed_df[feature].map(
            lambda text: process_en_text(text=text, conditions=conditions)
        )
        processed_df[feature] = processed_df[feature].replace("", np.nan)

    df = processed_df.dropna().drop_duplicates().reset_index(drop=True)

    return df


def _is_valid_text(text: str, special_chars: list[str]) -> bool:
    i = 0
    while i < len(text):
        if text[i] not in special_chars:
            return True
        i += 1
    return False


def remove_all_invalid_text(
    df: pd.DataFrame,
    features: list[str],
    special_chars: list[str],
) -> pd.DataFrame:
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"{feature} not found in dataset.")

    df_size = len(df)
    is_valid_rows = [True] * df_size

    for i in range(df_size):
        row = df.iloc[i]
        is_valid_rows[i] = all(
            [
                _is_valid_text(text=row[feature], special_chars=special_chars)
                for feature in features
            ]
        )
    return df[is_valid_rows].reset_index(drop=True)


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


def retain_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    dropped_columns = []
    for column in df.columns:
        if column not in columns:
            dropped_columns.append(column)
    df = df.drop(columns=dropped_columns).reset_index(drop=True)
    return df
