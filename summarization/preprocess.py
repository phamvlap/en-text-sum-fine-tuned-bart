import pandas as pd

from pathlib import Path
from typing import Optional
from datasets import load_dataset as load_dataset_remote, load_dataset_builder

from bart.constants import DEFAULT_TRAIN_VAL_TEST_RATIO, SentenceContractions
from .utils.tokenizer import load_tokenizer
from .utils.dataset import (
    clean_dataset,
    remove_rows_by_invalid_seq_length,
    retain_columns,
    preprocess_abstract_feature,
    preprocess_article_feature,
)
from .utils.path import make_dir
from .utils.seed import set_seed


def load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset file {path} not found")
        return pd.read_csv(path)

    ds_builder = load_dataset_builder(path)
    splits = ds_builder.info.splits
    if "train" not in splits:
        raise ValueError("Dataset has no train split")

    ds = load_dataset_remote(path)
    df = pd.DataFrame()
    for split in list(splits.keys()):
        df = pd.concat([df, ds[split].to_pandas()], ignore_index=True)
    df = df.reset_index(drop=True)

    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_size: Optional[float] = None,
    val_size: Optional[float] = None,
    test_size: Optional[float] = None,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_size is None and val_size is None and test_size is None:
        train_size, val_size, test_size = DEFAULT_TRAIN_VAL_TEST_RATIO
    elif (
        (train_size is None and val_size is None)
        or (val_size is None and test_size is None)
        or (test_size is None and train_size is None)
    ):
        raise ValueError(
            "train_size, val_size, and test_size must be least two specified values or all None"
        )
    elif train_size is None:
        train_size = 1.0 - val_size - test_size
    elif val_size is None:
        val_size = 1.0 - train_size - test_size
    elif test_size is None:
        test_size = 1.0 - train_size - val_size

    if float(train_size) + float(val_size) + float(test_size) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal to 1.0")

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    train_end = int(float(train_size) * len(df))
    val_end = train_end + int(float(val_size) * len(df))

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


def get_data(config: dict) -> None:
    print("Getting data...")
    # External data source
    raw_data_file_path = config["raw_data_file_path"]
    raw_df = load_dataset(path=raw_data_file_path)
    raw_df = raw_df.astype(str)

    raw_df[config["text_src"]] = raw_df.apply(
        lambda row: preprocess_article_feature(row[config["original_text_src"]]),
        axis=1,
    )
    raw_df[config["text_tgt"]] = raw_df.apply(
        lambda row: preprocess_abstract_feature(row[config["original_text_tgt"]]),
        axis=1,
    )

    features = [config["text_src"], config["text_tgt"]]
    retained_df = retain_columns(df=raw_df, columns=features)

    output_data_file_path = config["data_files_path"]["raw"]
    make_dir(dir_path=config["dataset_dir"])
    retained_df.to_csv(output_data_file_path, index=False)

    print("Getting data done!")
    print(f"Extracted data saved at {output_data_file_path}")


def preprocess(config: dict) -> None:
    set_seed(seed=config["seed"])
    print("Preprocessing dataset...")

    raw_data_file_path = config["data_files_path"]["raw"]
    df = load_dataset(path=raw_data_file_path)

    tokenizer = None
    tokenizer = load_tokenizer(bart_tokenizer_dir=config["tokenizer_bart_dir"])

    conditions = []
    if config["lowercase"]:
        conditions.append(SentenceContractions.LOWERCASE)
    if config["contractions"]:
        conditions.append(SentenceContractions.CONTRACTIONS)

    features = [config["text_src"], config["text_tgt"]]
    cleaned_df = clean_dataset(df=df, features=features, conditions=conditions)

    num_keep_tokens = 2
    valid_df = remove_rows_by_invalid_seq_length(
        df=cleaned_df,
        tokenizer=tokenizer,
        config=config,
        max_seq_length=config["seq_length"] - num_keep_tokens,
        min_seq_length=0,
    )

    if config["is_sampling"]:
        if (
            config["num_samples"] is not None
            and config["num_samples"] > 0
            and config["num_samples"] < len(valid_df)
        ):
            valid_df = valid_df.sample(n=config["num_samples"]).reset_index(drop=True)

    train_df, val_df, test_df = train_val_test_split(
        df=valid_df,
        train_size=config["train_size"],
        val_size=config["val_size"],
        test_size=config["test_size"],
        shuffle=True,
    )

    data_files_path = config["data_files_path"]
    make_dir(dir_path=config["dataset_dir"])

    train_df.to_csv(data_files_path["train"], index=False)
    val_df.to_csv(data_files_path["val"], index=False)
    test_df.to_csv(data_files_path["test"], index=False)

    print("Preprocessing dataset done!")
    print(f"Datasets saved at directory: {config['dataset_dir']}")
    print(f"Length of dataset: {len(valid_df)}")
    print(f"Length of train dataset: {len(train_df)}")
    print(f"Length of val dataset: {len(val_df)}")
    print(f"Length of test dataset: {len(test_df)}")
