import pandas as pd

from typing import Optional
from datasets import load_dataset as load_dataset_remote, load_dataset_builder

from bart.constants import DEFAULT_TRAIN_VAL_TEST_RATIO
from .utils.tokenizer import load_tokenizer
from .utils.dataset import (
    retain_columns,
    truncate_exceeded_length,
    remove_rows_by_invalid_seq_length,
)
from .utils.seed import set_seed
from .utils.mix import make_dir, ensure_exist_path


def load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        if not ensure_exist_path(path):
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


def prepare_dataset(config: dict) -> None:
    print("Preparing data...")
    # External data source
    datasource_path = config["datasource_path"]

    df = load_dataset(path=datasource_path)

    src_field, tgt_field = config["text_src"], config["text_tgt"]
    original_src_field, original_tgt_field = (
        config["original_text_src"],
        config["original_text_tgt"],
    )

    df.loc[:, src_field] = df[original_src_field].map(lambda text: text)
    df.loc[:, tgt_field] = df[original_tgt_field].map(lambda summary: summary)

    features = [src_field, tgt_field]
    prepared_df = retain_columns(df=df, columns=features)

    output_data_file_path = config["data_files_path"]["raw"]
    make_dir(dir_path=config["dataset_dir"])
    prepared_df.to_csv(output_data_file_path, index=False)

    print("Preprocessing dataset done!")
    print(f"Shape of dataset: {prepared_df.shape}")
    print(f"Extracted data saved at {output_data_file_path}")


def split_dataset(config: dict) -> None:
    print("Splitting dataset...")
    set_seed(seed=config["seed"])

    raw_data_file_path = config["data_files_path"]["raw"]
    df = load_dataset(path=raw_data_file_path)

    tokenizer = None
    tokenizer = load_tokenizer(bart_tokenizer_dir=config["tokenizer_bart_dir"])

    valid_df = df.copy()

    if config["remove_invalid_length"]:
        valid_df = remove_rows_by_invalid_seq_length(
            df=valid_df,
            tokenizer=tokenizer,
            config=config,
            src_seq_length=config["src_seq_length"],
            tgt_seq_length=config["tgt_seq_length"],
        )
    if config["truncate_exceeded_length"]:
        valid_df = truncate_exceeded_length(
            df=valid_df,
            tokenizer=tokenizer,
            config=config,
            seq_length=config["seq_length"],
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
        shuffle=config["shuffle"],
    )

    data_files_path = config["data_files_path"]
    make_dir(dir_path=config["dataset_dir"])

    train_df.to_csv(data_files_path["train"], index=False)
    val_df.to_csv(data_files_path["val"], index=False)
    test_df.to_csv(data_files_path["test"], index=False)

    print("Splitting dataset done!")
    print(f"Datasets saved at directory: {config['dataset_dir']}")
    print(f"Length of dataset: {len(valid_df)}")
    print(f"Length of train dataset: {len(train_df)}")
    print(f"Length of val dataset: {len(val_df)}")
    print(f"Length of test dataset: {len(test_df)}")
