import torch.nn as nn

from torch import Tensor
from transformers import BartForConditionalGeneration, PretrainedConfig
from typing import Literal, Optional
from dataclasses import dataclass

PRE_TRAINED_BART_MODELS = ["facebook/bart-base", "facebook/bart-large"]


@dataclass
class ModelArguments:
    model_name_or_path: Literal["facebook/bart-base", "facebook/bart-large"]
    config_name_or_path: Optional[
        Literal["facebook/bart-base", "facebook/bart-large"]
    ] = None


class FineTunedBartForGeneration(nn.Module):
    def __init__(
        self,
        model_args: ModelArguments,
        config: Optional[PretrainedConfig] = None,
    ) -> None:
        """
        Args:
            model_args: ModelArguments
            config: PretrainedConfig optional
        """
        super().__init__()
        if model_args.model_name_or_path not in PRE_TRAINED_BART_MODELS:
            raise ValueError(
                f"Supported models: {', '.join(PRE_TRAINED_BART_MODELS)}, got {model_args.model_name_or_path}"
            )
        if config is None:
            if (
                model_args.config_name_or_path is not None
                and model_args.config_name_or_path not in PRE_TRAINED_BART_MODELS
            ):
                raise ValueError(
                    f"Supported model configs: {', '.join(PRE_TRAINED_BART_MODELS)}, got {model_args.config_name_or_path}"
                )
            if (
                model_args.config_name_or_path is not None
                and model_args.model_name_or_path != model_args.config_name_or_path
            ):
                raise ValueError(
                    f"Model {model_args.model_name_or_path} and config {model_args.config_name_or_path} incompatible"
                )

            if model_args.config_name_or_path is None:
                model_args.config_name_or_path = model_args.model_name_or_path

            self.config = PretrainedConfig.from_pretrained(
                model_args.config_name_or_path
            )
        else:
            self.config = config

        self.bart_model = BartForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
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
    model_args: ModelArguments,
    config: Optional[PretrainedConfig] = None,
) -> FineTunedBartForGeneration:
    bart_model = FineTunedBartForGeneration(model_args=model_args, config=config)

    return bart_model
