import torch

from torch import Tensor
from transformers import BartTokenizer, PretrainedConfig
from typing import Optional
from pathlib import Path

from bart.model import (
    FineTunedBartForConditionalGeneration,
    build_bart_model,
)
from bart.constants import SpecialToken
from .utils.tokenizer import load_tokenizer
from .utils.dataset import process_en_text
from .utils.eval import greedy_search_decode, beam_search_decode
from .utils.mix import is_torch_cuda_available


class Summarizer:
    def __init__(
        self,
        model: FineTunedBartForConditionalGeneration,
        tokenizer: BartTokenizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.bos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
        self.eos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.EOS)
        self.pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

        self.model.eval()

    def summarize(
        self,
        text: str,
        max_pred_seq_length: int = 100,
        nums: int = 1,
        beam_size: Optional[int] = None,
    ) -> str | list[str]:
        input_text = self._preprocess_input_text(text=text)
        encoder_input = self._encode_input_text(text=input_text)

        with torch.no_grad():
            if beam_size is not None and beam_size > 0:
                cand_list = beam_search_decode(
                    model=self.model,
                    beam_size=beam_size,
                    input_ids=encoder_input,
                    tokenizer=self.tokenizer,
                    seq_length=max_pred_seq_length,
                    device=self.device,
                    topk=nums,
                    use_ddp=False,
                )
                cand_list = [cand.detach().cpu().numpy() for cand in cand_list]
                cand_text_list = [
                    self.tokenizer.decode(
                        cand,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
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
                    use_ddp=False,
                )
                pred_tokens = pred_tokens.detach().cpu().numpy()
                pred_text = self.tokenizer.decode(
                    pred_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                pred_text = self._postprocess_output_text(text=pred_text)
                return pred_text

    def _preprocess_input_text(self, text: str) -> str:
        text = process_en_text(text=text)
        return text

    def _encode_input_text(self, text: str) -> Tensor:
        input_tokens = self.tokenizer.encode(text)
        encoder_input = torch.cat(
            [
                Tensor([self.bos_token_id]),
                Tensor(input_tokens),
                Tensor([self.eos_token_id]),
            ]
        ).type(torch.int64)
        return encoder_input

    def _postprocess_output_text(self, text: str) -> str:
        output_text = text.replace(SpecialToken.BOS, "")
        output_text = text.replace(SpecialToken.EOS, "")
        output_text = output_text.strip()

        return output_text


def build_summarizer(model_path: str, tokenizer_path: str) -> Summarizer:
    device = torch.device("cuda" if is_torch_cuda_available() else "cpu")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(bart_tokenizer_dir=tokenizer_path)

    print(f"Loading model {model_path}...")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path {model_path} not exists.")

    torch.serialization.add_safe_globals([PretrainedConfig])
    checkpoint_states = torch.load(
        model_path,
        weights_only=True,
        map_location=device,
    )

    required_keys = [
        "model_state_dict",
        "config",
    ]
    for key in required_keys:
        if key not in checkpoint_states.keys():
            raise ValueError(f"Missing key {key} in checkpoint states.")

    print("Building model...")
    bart_model_config = checkpoint_states["config"]
    model_name = f"facebook/{bart_model_config._name_or_path}"
    model = build_bart_model(model_name_or_path=model_name, config=bart_model_config)
    model.load_state_dict(checkpoint_states["model_state_dict"])
    model.to(device=device)

    print("Building summarizer...")
    summarizer = Summarizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    return summarizer
