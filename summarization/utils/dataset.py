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


def replace_in_series(
    series: pd.Series,
    to_replace: str | int | float,
    value: str | int | float,
) -> pd.Series:
    return series.replace(to_replace=to_replace, value=value)


def process_features(
    df: pd.DataFrame,
    features: list[str],
    conditions: list[str] = [],
) -> pd.DataFrame:
    processed_df = df.copy()

    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"{feature} not found in dataset.")

    for feature in features:
        processed_df.loc[:, feature] = processed_df[feature].map(
            lambda text: process_en_text(text=text, conditions=conditions)
        )
        processed_df[feature] = replace_in_series(
            series=processed_df[feature],
            to_replace="",
            value=np.nan,
        )
    processed_df = processed_df.dropna().drop_duplicates().reset_index(drop=True)

    return processed_df


def remove_rows_by_invalid_seq_length(
    df: pd.DataFrame,
    tokenizer: BartTokenizer,
    config: dict,
    max_seq_length: int,
    min_seq_length: int = 0,
) -> pd.DataFrame:
    if min(min_seq_length, max_seq_length) < 0:
        raise ValueError(
            "min_seq_length and max_seq_length must be greater than or equal to zero."
        )
    if min_seq_length >= max_seq_length:
        raise ValueError("min_seq_length must be less than to max_seq_length.")
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
            min(source_token_length, target_token_length) >= min_seq_length
            and max(source_token_length, target_token_length) <= max_seq_length
        )

    return df[is_valid_rows]


def retain_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    dropped_columns = []
    for column in df.columns:
        if column not in columns:
            dropped_columns.append(column)
    df = df.drop(columns=dropped_columns).reset_index(drop=True)
    return df
