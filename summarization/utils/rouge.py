import torch

from torch import Tensor
from torchmetrics.text.rouge import ROUGEScore
from transformers import BartTokenizer
from typing import Literal, Callable, Optional

from bart.model import FineTunedBartForGeneration
from ..summarization_dataset import SummarizationDataset
from bart.constants import RougeKey
from .eval import greedy_search_decode, beam_search_decode


class RougeScorer:
    def __init__(
        self,
        rouge_keys: Optional[list[str] | tuple[str]] = None,
        use_stemmer: bool = True,
        normalizer_function: Optional[Callable] = None,
        tokenizer_function: Optional[Callable] = None,
        accumulate: Literal["best", "avg"] = "best",
    ) -> None:
        all_rouge_keys = [
            RougeKey.ROUGE_1,
            RougeKey.ROUGE_2,
            RougeKey.ROUGE_L,
        ]
        rouge_keys = rouge_keys if rouge_keys is not None else all_rouge_keys

        self.rouge_scorer = ROUGEScore(
            use_stemmer=use_stemmer,
            normalizer=normalizer_function,
            tokenizer=tokenizer_function,
            accumulate=accumulate,
            rouge_keys=tuple(rouge_keys),
        )

    def compute(
        self,
        preds: list[str] | str,
        targets: list[str] | str,
    ) -> dict[str, Tensor]:
        return self.rouge_scorer(preds, targets)


def format_rouge_score(pure_rouge_score: dict[str, Tensor]) -> dict[str, float]:
    rouge_score = {}
    for key in pure_rouge_score.keys():
        rouge_type = key.split("_")[0].replace("rouge", "")
        rouge_key = f"ROUGE@{rouge_type}"
        if rouge_key not in rouge_score.keys():
            rouge_score[rouge_key] = round(
                pure_rouge_score[f"rouge{rouge_type}_fmeasure"].item() * 100,
                2,
            )
    return rouge_score


@torch.no_grad()
def compute_dataset_rouge(
    model: FineTunedBartForGeneration,
    dataset: SummarizationDataset,
    tokenizer: BartTokenizer,
    seq_length: int,
    device: torch.device,
    beam_size: Optional[int] = None,
    topk: int = 1,
    log_examples: bool = True,
    logging_steps: int = 100,
    use_stemmer: bool = True,
    rouge_keys: Optional[list[str] | tuple[str]] = None,
    normalizer_function: Optional[Callable] = None,
    accumulate: Literal["best", "avg"] = "best",
) -> dict[str, float]:
    pred_text_list = []
    target_text_list = []

    # Set model to evaluation mode
    model.eval()

    rouge_scorer = RougeScorer(
        rouge_keys=rouge_keys,
        use_stemmer=use_stemmer,
        normalizer_function=normalizer_function,
        tokenizer_function=tokenizer.tokenize,
        accumulate=accumulate,
    )

    print("Computing ROUGE Score...")

    for idx, data in enumerate(dataset):
        encoder_input = data["encoder_input"]
        labels = data["labels"]

        if beam_size is not None and beam_size > 1:
            cands = beam_search_decode(
                model=model,
                beam_size=beam_size,
                input_ids=encoder_input,
                tokenizer=tokenizer,
                seq_length=seq_length,
                device=device,
                topk=topk,
            )
            pred_tokens = cands[0]
        else:
            pred_tokens = greedy_search_decode(
                model=model,
                input_ids=encoder_input,
                tokenizer=tokenizer,
                seq_length=seq_length,
                device=device,
            )

        src_tokens = encoder_input.detach().cpu().numpy()
        tgt_tokens = labels.detach().cpu().numpy()
        if isinstance(pred_tokens, Tensor):
            pred_tokens = pred_tokens.detach().cpu().numpy()

        src_text = tokenizer.decode(src_tokens, skip_special_tokens=True)
        tgt_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)

        pred_text_list.append(pred_text)
        target_text_list.append(tgt_text)

        if log_examples and idx % logging_steps == 0:
            rouge_score = rouge_scorer.compute(preds=pred_text, targets=tgt_text)

            print(f"EXAMPLE: {idx}")
            print(f"SOURCE TEXT: {src_text}")
            print(f"TARGET TEXT: {tgt_text}")
            print(f"PREDICTED TEXT: {pred_text}")

            rouge_score = format_rouge_score(rouge_score)
            print("ROUGE SCORE:")
            for key, value in rouge_score.items():
                print(f"{key}: {value}")

    rouge_score = rouge_scorer.compute(preds=pred_text_list, targets=target_text_list)
    rouge_score = format_rouge_score(rouge_score)

    # Set model back to train mode
    model.train()

    return rouge_score
