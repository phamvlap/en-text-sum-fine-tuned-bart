from typing import Any

from summarization.preprocess import split_dataset
from summarization.utils.mix import update_setting_config


def main(config: dict):
    split_dataset(config)


def parse_args() -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Splitting dataset for summarization task",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        dest="shuffle",
        help="shuffle dataset before spliting (default: False)",
    )
    parser.add_argument(
        "--remove_invalid_text",
        action="store_true",
        dest="remove_invalid_text",
        help="remove text with special characters (only keep text contains ['.', ',', '-']) (default: False)",
    )
    parser.add_argument(
        "--remove_invalid_length",
        action="store_true",
        dest="remove_invalid_length",
        help="remove rows with invalid length text (default: False)",
    )
    parser.add_argument(
        "--src_seq_length",
        type=int,
        required=True,
        default=768,
        help="maximum length of input sequence (default: 768)",
    )
    parser.add_argument(
        "--tgt_seq_length",
        type=int,
        required=True,
        default=256,
        help="maximum length of output sequence (default: 256)",
    )
    parser.add_argument(
        "--sampling",
        action="store_true",
        dest="is_sampling",
        help="get samples from dataset (default: False)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=1000,
        help="number of samples from dataset (default: 1000)",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        required=False,
        default=0.75,
        help="ratio of train dataset (default: 0.75)",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        required=False,
        default=0.1,
        help="ratio of validation dataset (default: 0.1)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        required=False,
        default=0.15,
        help="ratio of test dataset (default: 0.15)",
    )
    parser.set_defaults(
        shuffle=False,
        remove_invalid_text=False,
        remove_invalid_length=False,
        is_sampling=False,
    )

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    config = update_setting_config(new_config=args)

    main(config)
