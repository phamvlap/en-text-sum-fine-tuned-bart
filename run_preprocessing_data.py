from summarization.preprocess import preprocess


def run_preprocessing_data(config: dict) -> None:
    preprocess(config)
