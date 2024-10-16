import re
import contractions
import pandas as pd

from transformers import BartTokenizer

from bart.constants import SentenceContractions


def remove_urls(text: str) -> str:
    return re.sub(r"http[s]?:\/\/\S+|www\.\S+", "", text, flags=re.MULTILINE)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


def handle_punctuation(text: str) -> str:
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"([,.;?!\(\)\[\]\{\}])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def handle_en_text(text: str, conditions: list[str] = []) -> str:
    if SentenceContractions.LOWERCASE in conditions:
        text = str(text).lower()
    if SentenceContractions.CONTRACTIONS in conditions:
        text = contractions.fix(text)

    text = remove_urls(text)
    text = remove_html_tags(text)
    text = handle_punctuation(text)

    return text


def handle_feature(
    df: pd.DataFrame,
    feature: str,
    conditions: list[str] = [],
) -> pd.DataFrame:
    if feature not in df.columns:
        raise ValueError(f"{feature} not found in dataset.")
    df[feature] = df[[feature]].map(lambda s: handle_en_text(s, conditions))
    return df


def handle_dataset_features(
    df: pd.DataFrame,
    features: list[str],
    conditions: list[str] = [],
) -> pd.DataFrame:
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"{feature} not found in dataset.")

    for feature in features:
        df = handle_feature(df=df, feature=feature, conditions=conditions)

    return df


def clean_dataset(
    df: pd.DataFrame,
    features: list[str],
    conditions: list[str] = [],
) -> pd.DataFrame:
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df = handle_dataset_features(df=df, features=features, conditions=conditions)
    return df


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


def preprocess_abstract_feature(abstract: str) -> str:
    abstract = re.sub(r"[\n]", " ", abstract)
    abstract = re.sub(r"\.,", ". ", abstract)

    return abstract


def preprocess_article_feature(article: str) -> str:
    article = re.sub(r"[\n]+", "\n", article)
    article = re.sub(r"[\.;]\n,", ". ", article)
    article = re.sub(r"[\n]", " ", article)
    article = re.sub(r"[\.]+", ".", article)
    article = re.sub(r"\.,", ".", article)

    return article
