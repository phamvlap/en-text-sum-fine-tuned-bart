import torch

from torch import Tensor
from transformers import BartTokenizer
from typing import Any

from bart.model import FineTunedBartForGeneration
from bart.constants import SpecialToken, SentenceContractions
from .utils.dataset import process_en_text
from .utils.eval import greedy_search_decode, beam_search_decode


class Summarizer:
    def __init__(
        self,
        model: FineTunedBartForGeneration,
        tokenizer: BartTokenizer,
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        self.bos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
        self.eos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.EOS)
        self.pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

        self.model.eval()

    def summarize(
        self,
        text: str,
        max_pred_seq_length: int = 100,
        nums: int = 1,
    ) -> str | list[str]:
        input_text = self._preprocess_input_text(text=text)
        encoder_input = self._encode_input_text(text=input_text)

        with torch.no_grad():
            if self.config["beam_size"] is not None and self.config["beam_size"] > 0:
                cand_list = beam_search_decode(
                    model=self.model,
                    beam_size=self.config["beam_size"],
                    input_ids=encoder_input,
                    tokenizer=self.tokenizer,
                    seq_length=max_pred_seq_length,
                    device=self.device,
                    topk=nums,
                )
                cand_list = [cand.detach().cpu().numpy() for cand in cand_list]
                cand_text_list = [
                    self.tokenizer.decode(
                        cand,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    for cand in cand_list
                ]
                cand_text_list = [
                    self._postprocess_output_text(text=cand_text)
                    for cand_text in cand_text_list
                ]
                return cand_text_list
            else:
                pred_tokens = greedy_search_decode(
                    model=self.model,
                    input_ids=encoder_input,
                    tokenizer=self.tokenizer,
                    seq_length=max_pred_seq_length,
                    device=self.device,
                )
                pred_tokens = pred_tokens.detach().cpu().numpy()
                pred_text = self.tokenizer.decode(
                    pred_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                pred_text = self._postprocess_output_text(text=pred_text)
                return pred_text

    def _preprocess_input_text(self, text: str) -> str:
        conditions = []

        if self.config["lowercase"]:
            conditions.append(SentenceContractions.LOWERCASE)
        if self.config["contractions"]:
            conditions.append(SentenceContractions.CONTRACTIONS)

        text = process_en_text(text=text, conditions=conditions)
        return text

    def _encode_input_text(self, text: str) -> Tensor:
        input_tokens = self.tokenizer.encode(text)
        encoder_input = torch.cat(
            [
                Tensor([self.bos_token_id]),
                Tensor(input_tokens),
                Tensor([self.eos_token_id]),
            ]
        ).type(torch.int32)
        return encoder_input

    def _postprocess_output_text(self, text: str) -> str:
        output_text = text.replace(SpecialToken.BOS, "")
        output_text = text.replace(SpecialToken.EOS, "")
        output_text = output_text.strip()

        return output_text
