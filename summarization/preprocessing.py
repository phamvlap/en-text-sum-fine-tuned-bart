import pandas as pd

from pathlib import Path

from .utils.dataset import clean_dataset
from .utils.path import make_dir


def load_dataset(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file {path} not found")
    return pd.read_csv(path)


def preprocessing(config: dict) -> None:
    raw_data_files_path = config["raw_data_files_path"]

    raw_train_ds = load_dataset(path=raw_data_files_path["train"])
    raw_val_ds = load_dataset(path=raw_data_files_path["val"])
    raw_test_ds = load_dataset(path=raw_data_files_path["test"])

    features = [config["text_src"], config["text_tgt"]]

    train_ds = clean_dataset(df=raw_train_ds, features=features)
    val_ds = clean_dataset(df=raw_val_ds, features=features)
    test_ds = clean_dataset(df=raw_test_ds, features=features)

    data_files_path = config["data_files_path"]
    make_dir(dir_path=config["dataset_dir"])

    train_ds.to_csv(data_files_path["train"], index=False)
    val_ds.to_csv(data_files_path["val"], index=False)
    test_ds.to_csv(data_files_path["test"], index=False)

    print("Preprocessing dataset done!")
    print("Datasets saved at directory: {}".format(config["dataset_dir"]))
    print("Length of train dataset: {}".format(len(train_ds)))
    print("Length of val dataset: {}".format(len(val_ds)))
    print("Length of test dataset: {}".format(len(test_ds)))
