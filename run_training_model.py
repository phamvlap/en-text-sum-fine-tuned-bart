from summarization.utils.mix import load_config
from summarization.train import main as run_training_model


def main(config: dict):
    run_training_model(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fine-tuned BART model")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (.yaml)",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
