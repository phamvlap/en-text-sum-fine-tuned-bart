import torch.nn as nn

from torch import Tensor
from transformers import BartForConditionalGeneration, PretrainedConfig, BartConfig
from typing import Optional
from dataclasses import dataclass

PRE_TRAINED_BART_MODELS = ["facebook/bart-base"]


@dataclass
class FineTunedBartForConditionalGenerationConfig:
    vocab_size: int = 50265
    d_model: int = 768
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_ffn_dim: int = 3072
    decoder_ffn_dim: int = 3072
    activation_function: str = "gelu"
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    classifier_dropout: float = 0.1
    max_position_embeddings: int = 512
    init_std: float = 0.02
    encoder_layerdrop: float = 0.1
    decoder_layerdrop: float = 0.1
    scale_embedding: bool = True
    num_beams: int = 4


class FineTunedBartForConditionalGeneration(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "facebook/bart-base",
        config: Optional[FineTunedBartForConditionalGenerationConfig] = None,
    ) -> None:
        """
        Args:
            model_name_or_path: model name, default: "facebook/bart-base"
            config: FineTunedBartForConditionalGenerationConfig optional
        """
        super().__init__()
        if model_name_or_path not in PRE_TRAINED_BART_MODELS:
            raise ValueError(
                f"Supported models: {', '.join(PRE_TRAINED_BART_MODELS)}, got {model_name_or_path}"
            )

        self.config = BartConfig.from_pretrained(model_name_or_path)

        if config is not None:
            self.config.vocab_size = config.vocab_size
            self.config.d_model = config.d_model
            self.config.encoder_layers = config.encoder_layers
            self.config.decoder_layers = config.decoder_layers
            self.config.encoder_attention_heads = config.encoder_attention_heads
            self.config.decoder_attention_heads = config.decoder_attention_heads
            self.config.encoder_ffn_dim = config.encoder_ffn_dim
            self.config.decoder_ffn_dim = config.decoder_ffn_dim
            self.config.activation_function = config.activation_function
            self.config.dropout = config.dropout
            self.config.attention_dropout = config.attention_dropout
            self.config.activation_dropout = config.activation_dropout
            self.config.classifier_dropout = config.classifier_dropout
            self.config.max_position_embeddings = config.max_position_embeddings
            self.config.init_std = config.init_std
            self.config.encoder_layerdrop = config.encoder_layerdrop
            self.config.decoder_layerdrop = config.decoder_layerdrop
            self.config.scale_embedding = config.scale_embedding
            self.config.num_beams = config.num_beams

        self.bart_model = BartForConditionalGeneration.from_pretrained(
            model_name_or_path,
            config=self.config,
        )
        self.linear_proj = nn.Linear(self.config.d_model, self.config.vocab_size)

    def get_config(self) -> PretrainedConfig:
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
            encoder_input_ids: encoder input tensor, shape `(batch_size, src_seq_length)`
            encoder_attn_mask: encoder attention mask tensor, shape `(batch_size, src_seq_length)`
            decoder_input_ids: decoder input tensor, shape `(batch_size, tgt_seq_length)`
            decoder_attn_mask: decoder attention mask tensor, shape `(batch_size, tgt_seq_length)`
            **kwargs: additional arguments
        Returns:
            Tensor: shape `(batch_size, tgt_seq_length, vocab_size)`
        """
        output = self.bart_model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attn_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attn_mask,
            **kwargs,
        )
        logits = output.logits
        return logits

    def encode(
        self,
        encoder_input_ids: Tensor,
        encoder_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            encoder_input_ids: encoder input tensor, shape `(batch_size, src_seq_length)`
            encoder_attn_mask: encoder attention mask tensor, shape `(batch_size, src_seq_length)`
        Returns:
            Tensor: shape `(batch_size, src_seq_length, d_model)`
        """
        return self.bart_model.get_encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attn_mask,
        ).last_hidden_state

    def decode(
        self,
        decoder_input_ids: Tensor,
        decoder_attn_mask: Optional[Tensor] = None,
        encoder_output: Optional[Tensor] = None,
        encoder_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            decoder_input_ids: decoder input tensor, shape `(batch_size, tgt_seq_length)`
            decoder_attn_mask: decoder input tensor, shape `(batch_size, tgt_seq_length)`
            encoder_output: encoder output tensor, shape `(batch_size, src_seq_length, d_model)`
            encoder_attn_mask: encoder attention mask tensor, shape `(batch_size, src_seq_length)`
        Returns:
            Tensor: shape `(batch_size, tgt_seq_length, d_model)`
        """
        return self.bart_model.get_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attn_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attn_mask,
        ).last_hidden_state

    def proj(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor, shape `(batch_size, tgt_seq_length, d_model)`
        Returns:
            Tensor: output tensor, shape `(batch_size, tgt_seq_length, vocab_size)`
        """
        return self.linear_proj(x)


def build_bart_model(
    model_name_or_path: str = "facebook/bart-base",
    config: Optional[FineTunedBartForConditionalGenerationConfig] = None,
) -> FineTunedBartForConditionalGeneration:
    bart_model = FineTunedBartForConditionalGeneration(
        model_name_or_path=model_name_or_path,
        config=config,
    )

    return bart_model
