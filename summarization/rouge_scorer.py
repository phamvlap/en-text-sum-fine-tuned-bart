from torchmetrics.text.rouge import ROUGEScore

from bart.constants import RougeKey

all_rouge_keys = (
    RougeKey.ROUGE_1,
    RougeKey.ROUGE_2,
    RougeKey.ROUGE_L,
)


class RougeScorer:
    def __init__(
        self,
        rouge_keys: list[str] | tuple[str] = all_rouge_keys,
        use_stemmer: bool = True,
        normalizer_function: callable = None,
        tokenizer_function: callable = None,
        accumulate: str = "best",
    ) -> None:
        self.rouge_scorer = ROUGEScore(
            use_stemmer=use_stemmer,
            normalizer=normalizer_function,
            tokenizer=tokenizer_function,
            accumulate=accumulate,
            rouge_keys=tuple(rouge_keys),
        )

    def calculate(self, preds: list[str] | str, targets: list[str] | str) -> dict:
        return self.rouge_scorer(preds, targets)
