import pandas as pd
from transformers import BartTokenizer

from .preprocessing import load_dataset
from .tokenizer import CustomBartTokenizer, save_tokenizer
from .utils.dataset import clean_dataset


def train(config: dict) -> BartTokenizer:
    ds = load_dataset(path=config["tokenizer_train_ds_path"])
    if config["shared_vocab"]:
        ds_train = pd.concat([ds[config["text_src"]], ds[config["text_tgt"]]], axis=0)
    else:
        ds_train = ds[config["text_src"]]

    ds_train = clean_dataset(df=ds_train, features=config["text_src"])

    bart_tokenizer = CustomBartTokenizer(
        dataset=ds_train,
        vocab_size=config["vocab_size"],
        special_tokens=config["special_tokens"],
        min_freq=config["min_freq"],
        model_type=config["model_type"],
    )

    bart_tokenizer = bart_tokenizer.train(
        config=config,
        show_progress=config["show_progress"],
    )

    save_tokenizer(
        bart_tokenizer=bart_tokenizer,
        bart_tokenizer_dir=config["tokenizer_bart_dir"],
    )

    print("Training tokenizer done!")
    print(f"Trained tokenizer saved at directory: {config['tokenizer_bart_dir']}")
    print(f"Vocab size: {config['vocab_size']}")

    return bart_tokenizer
