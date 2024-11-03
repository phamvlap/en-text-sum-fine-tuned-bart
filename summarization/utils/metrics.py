import torch

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
from transformers import BartTokenizer
from typing import Literal, Callable, Optional, List, Tuple, Sequence
from tqdm import tqdm

from bart.model import FineTunedBartForGeneration
from bart.constants import RougeKey
from .eval import greedy_search_decode, beam_search_decode
from ..summarization_dataset import SummarizationDataset


class RougeScorer:
    def __init__(
        self,
        use_stemmer: bool = True,
        rouge_keys: Optional[List[str] | Tuple[str]] = None,
        normalizer_function: Optional[Callable] = None,
        tokenizer_function: Optional[Callable] = None,
        accumulate: Literal["best", "avg"] = "avg",
    ) -> None:
        all_rouge_keys = [
            RougeKey.ROUGE_1,
            RougeKey.ROUGE_2,
            RougeKey.ROUGE_L,
        ]
        if rouge_keys is not None:
            for key in rouge_keys:
                if key not in all_rouge_keys:
                    raise ValueError(
                        f"Key {key} not found in {', '.join(all_rouge_keys)}"
                    )
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
        target: list[str] | str,
    ) -> dict[str, Tensor]:
        return self.rouge_scorer(preds, target)


class BertScorer:
    def __init__(
        self,
        max_length: int,
        model_name_or_path: str = "roberta-large",
        truncation: bool = True,
        rescale: bool = True,
        accumulate: Literal["best", "avg"] = "avg",
    ) -> None:
        self.bert_scorer = BERTScore(
            model_name_or_path=model_name_or_path,
            max_length=max_length,
            truncation=truncation,
            rescale_with_baseline=rescale,
        )
        self.accumulate = accumulate

    def compute(
        self,
        preds: Sequence[str] | str,
        target: Sequence[str] | str,
    ) -> dict[str, Tensor]:
        score = self.bert_scorer(preds, target)
        score_summary = {}

        for key, value in score.items():
            if self.accumulate == "avg":
                score_summary[key] = value.mean()
            elif self.accumulate == "best":
                score_summary[key] = value.max()

        return score_summary


def format_rouge_score(pure_rouge_score: dict[str, Tensor]) -> dict[str, float]:
    rouge_score = {}
    for key in pure_rouge_score.keys():
        rouge_type = key.split("_")[0].replace("rouge", "")
        rouge_key = f"ROUGE@{rouge_type}"
        if rouge_key not in rouge_score.keys():
            rouge_score[rouge_key] = pure_rouge_score[
                f"rouge{rouge_type}_fmeasure"
            ].item()
    return rouge_score


def format_bert_score(pure_bert_score: dict[str, Tensor]) -> dict[str, float]:
    key_mapping = {
        "precision": "P-BERT",
        "recall": "R-BERT",
        "f1": "F-BERT",
    }
    bert_score = {}

    for key in pure_bert_score.keys():
        mapped_key = key_mapping[key]
        bert_score[mapped_key] = pure_bert_score[key].item()

    return bert_score


@torch.no_grad()
def compute_rouge_bert_score(
    model: FineTunedBartForGeneration | DDP,
    dataset: SummarizationDataset,
    tokenizer: BartTokenizer,
    seq_length: int,
    device: torch.device,
    beam_size: Optional[int] = None,
    topk: int = 1,
    eval_bert_score: bool = True,
    rescale: bool = True,
    log_examples: bool = True,
    logging_steps: int = 100,
    use_stemmer: bool = True,
    rouge_keys: Optional[List[str] | Tuple[str]] = None,
    normalizer_function: Optional[Callable] = None,
    truncation: bool = True,
    accumulate: Literal["best", "avg"] = "avg",
    use_ddp: bool = False,
    rank: Optional[int] = None,
    local_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> dict[str, float]:
    pred_text_list = []
    target_text_list = []

    # Set model to evaluation mode
    model.eval()

    assert (
        local_rank is not None and rank is not None if use_ddp else True
    ), "local_rank and rank must be not None if use DDP"

    rouge_scorer = RougeScorer(
        use_stemmer=use_stemmer,
        rouge_keys=rouge_keys,
        normalizer_function=normalizer_function,
        tokenizer_function=tokenizer.tokenize,
        accumulate=accumulate,
    )

    bert_scorer = None
    if eval_bert_score:
        bert_scorer = BertScorer(
            max_length=seq_length,
            truncation=truncation,
            rescale=rescale,
            accumulate=accumulate,
        )

    if max_steps is None:
        total_iters = len(dataset)
    else:
        if use_ddp:
            if world_size is None:
                raise ValueError(
                    "world_size must be not None if use_ddp is True and max_steps is not None"
                )
            else:
                if max_steps % world_size != 0:
                    raise ValueError(
                        f"max_steps must be divisible by world_size = {world_size}"
                    )
                max_steps = max_steps // world_size
        total_iters = min(len(dataset), max_steps)

    if use_ddp:
        dataset_iterator = tqdm(
            dataset,
            desc=f"[GPU-{rank}] Computing metrics",
            disable=local_rank != 0,
            total=total_iters,
        )
    else:
        dataset_iterator = tqdm(
            dataset,
            desc="Computing metrics",
            total=total_iters,
        )

    for idx, data in enumerate(dataset_iterator):
        if idx >= total_iters:
            break

        encoder_input = data["encoder_input"]
        labels = data["labels"]
        cand_list: list[Tensor] = []
        cand_text_list: list[str] = []

        if beam_size is not None and beam_size > 1:
            cands = beam_search_decode(
                model=model,
                beam_size=beam_size,
                input_ids=encoder_input,
                tokenizer=tokenizer,
                seq_length=seq_length,
                topk=topk,
                device=device,
                use_ddp=use_ddp,
                local_rank=local_rank,
            )
            cand_list = cands
            pred_tokens = cands[0]
        else:
            pred_tokens = greedy_search_decode(
                model=model,
                input_ids=encoder_input,
                tokenizer=tokenizer,
                seq_length=seq_length,
                device=device,
                use_ddp=use_ddp,
                local_rank=local_rank,
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

        if len(cand_list) > 0:
            for cand in cand_list:
                if isinstance(cand, Tensor):
                    cand = cand.detach().cpu().numpy()
                cand_text = tokenizer.decode(cand, skip_special_tokens=True)
                cand_text_list.append(cand_text)

        if log_examples and idx % logging_steps == 0:
            rouge_score = rouge_scorer.compute(preds=pred_text, target=tgt_text)
            bert_score = None
            if bert_scorer is not None:
                bert_score = bert_scorer.compute(preds=pred_text, target=tgt_text)

            print(f"EXAMPLE: {idx}")
            print(f"SOURCE TEXT: {src_text}")
            print(f"TARGET TEXT: {tgt_text}")
            if len(cand_text_list) > 0:
                print("CANDIDATE TEXTS:")
                for i, cand_text in enumerate(cand_text_list):
                    print(f"CANDIDATE TEXT {i}: {cand_text}")
            else:
                print(f"PREDICTED TEXT: {pred_text}")

            rouge_score = format_rouge_score(rouge_score)
            if bert_score is not None:
                bert_score = format_bert_score(bert_score)

            print("ROUGE SCORE:")
            for key, value in rouge_score.items():
                print(f"{key}: {value}")
            if bert_score is not None:
                print("BERT SCORE:")
                for key, value in bert_score.items():
                    print(f"{key}: {value}")

    rouge_score = rouge_scorer.compute(preds=pred_text_list, target=target_text_list)
    rouge_score = format_rouge_score(rouge_score)
    bert_score = None
    if bert_scorer is not None:
        bert_score = bert_scorer.compute(preds=pred_text_list, target=target_text_list)
        bert_score = format_bert_score(bert_score)

    scores = {
        **rouge_score,
    }
    if bert_score is not None:
        scores = {
            **rouge_score,
            **bert_score,
        }
    # Set model back to train mode
    model.train()

    return scores
