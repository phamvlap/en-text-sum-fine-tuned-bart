from typing import Any

from summarization.train_tokenizer import train_tokenizer
from summarization.utils.mix import update_setting_config


def main(config: dict):
    train_tokenizer(config)


def parse_args() -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train tokenizer for summarization task"
    )
    parser.add_argument(
        "--shared_vocab",
        action="store_true",
        dest="shared_vocab",
        help="share vocabulary between source and target text (default: False)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=False,
        default=50000,
        help="size of vocabulary (default: 50000)",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        required=False,
        default=2,
        help="minium frequency for merging tokens (default: 2)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="byte_level_bpe",
        help="type of tokenizer model (default: byte_level_bpe)",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        dest="show_progress",
        help="enable progress during training tokenizer (default: False)",
    )
    parser.set_defaults(
        shared_vocab=False,
        show_progress=False,
    )

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    config = update_setting_config(new_config=args)

    main(config)
