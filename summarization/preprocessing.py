import pandas as pd

from pathlib import Path

from .utils.dataset import handle_dataset_features
from .utils.mix import make_dir


def load_dataset(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file {path} not found")
    return pd.read_csv(path)


def preprocessing(config: dict) -> None:
    raw_train_ds = load_dataset(path=config["raw_train_ds_path"])
    raw_val_ds = load_dataset(path=config["raw_val_ds_path"])
    raw_test_ds = load_dataset(path=config["raw_test_ds_path"])

    features = [config["text_src"], config["text_tgt"]]

    train_ds = handle_dataset_features(df=raw_train_ds, features=features)
    val_ds = handle_dataset_features(df=raw_val_ds, features=features)
    test_ds = handle_dataset_features(df=raw_test_ds, features=features)

    make_dir(dir_path=config["dataset_dir"])

    train_ds.to_csv(config["train_ds_path"], index=False)
    val_ds.to_csv(config["val_ds_path"], index=False)
    test_ds.to_csv(config["test_ds_path"], index=False)

    print("Preprocessing done!")
    print("Datasets saved at directory: {}".format(config["dataset_dir"]))
    print("Length of train dataset: {}".format(len(train_ds)))
    print("Length of val dataset: {}".format(len(val_ds)))
    print("Length of test dataset: {}".format(len(test_ds)))
