from summarization.preprocess import preprocess, get_data
from summarization.utils.mix import load_config


def run_getting_data(config: dict):
    get_data(config)


def main(config: dict):
    preprocess(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess data for summarization task"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (.yaml)",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
