from typing import Any

from summarization.preprocess import preprocess
from summarization.utils.mix import update_setting_config


def main(config: dict):
    preprocess(config)


def parse_args() -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess data for summarization task",
    )
    parser.add_argument(
        "--datasource_path",
        type=str,
        required=True,
        help="path to source data",
    )
    parser.add_argument(
        "--text_src",
        type=str,
        required=False,
        default="article",
        help="key of source text (default: article)",
    )
    parser.add_argument(
        "--text_tgt",
        type=str,
        required=False,
        default="abstract",
        help="key of target text (default: abstract)",
    )
    parser.add_argument(
        "--original_text_src",
        type=str,
        required=False,
        default="text",
        help="key of original source text (default: text)",
    )
    parser.add_argument(
        "--original_text_tgt",
        type=str,
        required=False,
        default="headline",
        help="key of original target text (default: headline)",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        dest="lowercase",
        help="apply to lowercase text (default: False)",
    )
    parser.add_argument(
        "--contractions",
        action="store_true",
        dest="contractions",
        help="apply to contractions text (default: False)",
    )
    parser.set_defaults(
        lowercase=False,
        contractions=True,
    )

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    config = update_setting_config(new_config=args)

    main(config)
