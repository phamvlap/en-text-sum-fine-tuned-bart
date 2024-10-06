import torch.nn as nn

from torch import Tensor
from transformers import BartConfig, BartModel
from dataclasses import dataclass
from typing import Literal


@dataclass
class FineTunedBartForGenerationConfig:
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
    bos_token_id: int
    pad_token_id: int
    eos_token_id: int
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


class FineTunedBartForGeneration(nn.Module):
    def __init__(self, config: BartConfig) -> None:
        super().__init__()
        self.bart_model = BartModel(config)
        self.linear_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        encoder_input_ids: Tensor,
        encoder_attn_mask: Tensor,
        decoder_input_ids: Tensor,
        decoder_attn_mask: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            encoder_input_ids: encoder input tensor, shape `(batch_size, seq_length)`
            encoder_attn_mask: encoder attention mask tensor, shape `(batch_size, seq_length)`
            decoder_input_ids: decoder input tensor, shape `(batch_size, tgt_seq_length)`
            decoder_attn_mask: decoder attention mask tensor, shape `(batch_size, tgt_seq_length)`
            **kwargs: additional arguments
        Returns:
            Tensor: `(batch_size, seq_len, vocab_size)`
        """
        output = self.bart_model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attn_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attn_mask,
            **kwargs,
        )
        # output.last_hidden_state (batch_size, seq_len, d_model)
        logits = self.linear_proj(output.last_hidden_state)
        return logits

    def encode(
        self,
        encoder_input_ids: Tensor,
        encoder_attn_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            encoder_input_ids: encoder input tensor, shape `(batch_size, seq_length)`
            encoder_attn_mask: encoder attention mask tensor, shape `(batch_size, seq_length)`
        Returns:
            Tensor: `(batch_size, seq_length, d_model)`
        """
        return self.bart_model.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attn_mask,
        ).last_hidden_state

    def decode(
        self,
        decoder_input_ids: Tensor,
        decoder_attn_mask: Tensor,
        encoder_output: Tensor,
        encoder_attn_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            decoder_input_ids: decoder input tensor, shape `(batch_size, seq_length)`
            decoder_attn_mask: decoder input tensor, shape `(batch_size, seq_length)`
            encoder_output: encoder output tensor, shape `(batch_size, encoder_seq_length, d_model)`
            encoder_attn_mask: encoder attention mask tensor, shape `(batch_size, encoder_seq_length)`
        Returns:
            Tensor: `(batch_size, seq_length, d_model)`
        """
        return self.bart_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attn_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attn_mask,
        ).last_hidden_state

    def proj(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor, shape `(batch_size, seq_length, d_model)`
        Returns:
            Tensor: output tensor, shape `(batch_size, seq_length, vocab_size)`
        """
        return self.linear_proj(x)


def build_bart_model(
    config: FineTunedBartForGenerationConfig,
) -> FineTunedBartForGeneration:
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

    bart_model = FineTunedBartForGeneration(config=bart_config)

    return bart_model
