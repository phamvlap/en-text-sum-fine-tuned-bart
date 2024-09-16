import torch.nn as nn

from torch import Tensor, LongTensor
from transformers import BartConfig, BartModel
from dataclasses import dataclass
from typing import Literal


@dataclass
class FinetuneBartModelConfig:
    seq_length: int
    device: Literal["cpu", "cuda"]
    vocab_size: int
    d_model: int
    encoder_layers: int
    decoder_layers: int
    encoder_attention_heads: int
    decoder_attention_heads: int
    encoder_ffn_dim: int
    decoder_ffn_dim: int
    activation_function: Literal["gelu", "relu", "silu", "gelu_new"] = "gelu"
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    classifier_dropout: float = 0.1
    init_std: float = 0.02
    encoder_layerdrop: float = 0.2
    decoder_layerdrop: float = 0.2
    scale_embedding: bool = True
    num_beams: int = 4
    bos_token_id: int
    pad_token_id: int
    eos_token_id: int


class FinetuneBartModel(nn.Module):
    def __init__(self, config: BartConfig) -> None:
        super().__init__()
        self.bart_model = BartModel(config)
        self.proj = nn.Linear(config.d_model, config.vocab_size)

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


def build_bart_model(config: FinetuneBartModelConfig) -> FinetuneBartModel:
    bart_config = BartConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        encoder_attention_heads=config.encoder_attention_heads,
        decoder_attention_heads=config.decoder_attention_heads,
        encoder_ffn_dim=config.encoder_ffn_dim,
        decoder_ffn_dim=config.decoder_ffn_dim,
        activation_function=config.activation_function,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        activation_dropout=config.activation_dropout,
        classifier_dropout=config.classifier_dropout,
        max_position_embeddings=config.seq_length,
        init_std=config.init_std,
        encoder_layerdrop=config.encoder_layerdrop,
        decoder_layerdrop=config.decoder_layerdrop,
        scale_embedding=config.scale_embedding,
        num_beams=config.num_beams,
        bos_token_id=config.bos_token_id,
        pad_token_id=config.pad_token_id,
        eos_token_id=config.eos_token_id,
        forced_bos_token_id=config.bos_token_id,
        forced_eos_token_id=config.eos_token_id,
    )

    bart_model = FinetuneBartModel(config=bart_config)

    return bart_model
