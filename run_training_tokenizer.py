from summarization.train_tokenizer import train_tokenizer
from summarization.utils.mix import load_config


def main(config: dict):
    train_tokenizer(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train tokenizer for summarization task"
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
