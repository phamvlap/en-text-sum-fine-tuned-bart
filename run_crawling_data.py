from typing import Any

from summarization.crawl_data import main as run_crawling_data


def main(config: dict):
    run_crawling_data(config)


def parse_args() -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Crawl data from specific URL and save to CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="data/data.csv",
        help="path to save data (default: data/data.csv)",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="https://vietnamnet.vn/en/",
        help="URL to crawl data (default: https://vietnamnet.vn/en/)",
    )
    parser.add_argument(
        "--total",
        type=int,
        required=False,
        default=10,
        help="total number of news to crawl (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        dest="verbose",
        help="show progress during crawling (default: False)",
    )

    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    args = parse_args()

    main(args)
