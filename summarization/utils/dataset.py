import re
import contractions
import pandas as pd


def remove_urls(text: str) -> str:
    return re.sub(r"http[s]?:\/\/\S+|www\.\S+", "", text, flags=re.MULTILINE)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


def handle_punctuation(text: str) -> str:
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"([,.;?!\(\)\[\]\{\}])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def handle_en_text(text: str) -> str:
    text = str(text).lower()
    text = contractions.fix(text)

    text = remove_urls(text)
    text = remove_html_tags(text)
    text = handle_punctuation(text)

    return text


def handle_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    df[feature] = df[[feature]].map(lambda s: handle_en_text(s))
    return df


def handle_dataset_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for feature in features:
        if feature not in df.columns:
            continue
        df = handle_feature(df=df, feature=feature)
    return df


def clean_dataset(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df = handle_dataset_features(df=df, features=features)
    return df
