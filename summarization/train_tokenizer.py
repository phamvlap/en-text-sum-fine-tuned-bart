import pandas as pd
from transformers import BartTokenizer

from .preprocess import load_dataset
from .utils.tokenizer import CustomBartTokenizer, save_tokenizer


def train_tokenizer(config: dict) -> BartTokenizer:
    ds = load_dataset(path=config["tokenizer_train_ds_path"])
    src_feature = config["text_src"]
    tgt_feature = config["text_tgt"]

    if config["shared_vocab"]:
        ds_train = pd.concat([ds[src_feature], ds[tgt_feature]], axis=0)
    else:
        ds_train = ds[src_feature]

    ds_train = pd.DataFrame(ds_train, columns=[src_feature])

    bart_tokenizer = CustomBartTokenizer(
        dataset=ds_train[src_feature],
        vocab_size=config["vocab_size"],
        special_tokens=config["special_tokens"],
        min_freq=config["min_freq"],
        model_type=config["model_type"],
    )

    bart_tokenizer = bart_tokenizer.train(show_progress=config["show_progress"])

    save_tokenizer(
        bart_tokenizer=bart_tokenizer,
        bart_tokenizer_dir=config["tokenizer_bart_dir"],
    )

    print("Training tokenizer done!")
    print(f"Trained tokenizer saved at directory: {config['tokenizer_bart_dir']}")
    print(f"Vocab size: {bart_tokenizer.vocab_size}")

    return bart_tokenizer
