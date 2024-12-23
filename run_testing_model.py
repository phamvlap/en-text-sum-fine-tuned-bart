from typing import Any

from summarization.test import test
from summarization.utils.mix import update_setting_config


def main(config: dict):
    test(config)


def parse_args() -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Train fine-tuned BART model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        default="facebook/bart-base",
        help="name or path of model (default: facebook/bart-base)",
    )
    parser.add_argument(
        "--batch_size_test",
        type=int,
        required=False,
        default=8,
        help="batch size of dataset for testing (default: 8)",
    )
    parser.add_argument(
        "--tgt_seq_length",
        type=int,
        required=True,
        default=256,
        help="maximum length of output sequence (default: 256)",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        required=False,
        default=100,
        help="maximum examples for each evaluating iterator (default: 100)",
    )
    parser.add_argument(
        "--show_eval_progress",
        action="store_true",
        dest="show_eval_progress",
        help="show progress during evaluating (default: False)",
    )
    parser.add_argument(
        "--use_stemmer",
        action="store_true",
        dest="use_stemmer",
        help="use stemmer for computing ROUGE scores (default: False)",
    )
    parser.add_argument(
        "--accumulate",
        type=str,
        required=False,
        default="avg",
        help="accumulate type for training (best, avg) (default: avg)",
    )
    parser.add_argument(
        "--log_examples",
        action="store_true",
        dest="log_examples",
        help="log examples during evaluating (default: False)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        required=False,
        default=1000,
        help="number of steps to log examples (default: 1000)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        required=False,
        default=3,
        help="beam size for decoding (default: 3)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=False,
        default=2,
        help="top k result returned for decoding (default: 2)",
    )
    parser.add_argument(
        "--statistic_dir",
        type=str,
        required=False,
        default="statistics",
        help="directory to save statistics (default: statistics)",
    )
    parser.set_defaults(
        show_eval_progress=False,
        use_stemmer=False,
        log_examples=False,
    )

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    config = update_setting_config(new_config=args)

    main(config)
