import torch.nn as nn

from torch import Tensor
from transformers import BartModel, PretrainedConfig, BartConfig
from typing import Optional


class FineTunedBartForGeneration(nn.Module):
    def __init__(
        self,
        model_path: str,
        config: Optional[BartConfig | PretrainedConfig] = None,
    ) -> None:
        super().__init__()
        self.bart_model = BartModel.from_pretrained(model_path, config=config)
        self.config = self.bart_model.config
        self.linear_proj = nn.Linear(self.config.d_model, self.config.vocab_size)

    def get_config(self) -> PretrainedConfig | BartConfig:
        return self.config

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
    model_path: str,
    config: Optional[BartConfig | PretrainedConfig] = None,
) -> FineTunedBartForGeneration:
    bart_model = FineTunedBartForGeneration(model_path, config=config)

    return bart_model
