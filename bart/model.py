import torch.nn as nn

from torch import Tensor, LongTensor
from transformers import BartConfig, BartModel, BartTokenizer

from .constants import SpecialToken


def get_bart_config(config: dict, tokenizer: BartTokenizer) -> BartConfig:
    bos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)
    eos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.EOS)

    bart_config = BartConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        encoder_attention_heads=config["encoder_attention_heads"],
        decoder_attention_heads=config["decoder_attention_heads"],
        encoder_ffn_dim=config["encoder_ffn_dim"],
        decoder_ffn_dim=config["decoder_ffn_dim"],
        activation_function=config["activation_function"],
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        activation_dropout=config["activation_dropout"],
        classifier_dropout=config["classifier_dropout"],
        max_position_embeddings=config["max_position_embeddings"],
        init_std=config["init_std"],
        encoder_layerdrop=config["encoder_layerdrop"],
        decoder_layerdrop=config["decoder_layerdrop"],
        scale_embedding=config["scale_embedding"],
        num_beams=config["num_beams"],
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        forced_bos_token_id=bos_token_id,
        forced_eos_token_id=eos_token_id,
    )

    return bart_config


class FinetuneBartModel(nn.Module):
    def __init__(self, config: BartConfig, tokenizer: BartTokenizer) -> None:
        super().__init__()
        self.bart_model = BartModel(config)
        self.proj = nn.Linear(config.d_model, tokenizer.vocab_size)

    def forward(self, **kwargs) -> Tensor:
        # output.last_hidden_state (batch_size, seq_len, d_model)
        # logits (batch_size, seq_len, vocab_size)
        out = self.bart_model(**kwargs)
        logits = self.proj(out.last_hidden_state)
        return logits

    def encode(
        self,
        input_ids: LongTensor,
        attention_mask: Tensor,
    ) -> Tensor:
        return self.bart_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def decode(
        self,
        input_ids: LongTensor,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        return self.bart_model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

    def out(self, x: Tensor) -> Tensor:
        return self.proj(x)


def build_bart_model(
    config: dict,
    tokenizer: BartTokenizer,
) -> FinetuneBartModel:
    bart_config = get_bart_config(config=config, tokenizer=tokenizer)
    bart_model = FinetuneBartModel(config=bart_config, tokenizer=tokenizer)
    return bart_model
