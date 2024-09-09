import torch.nn as nn

from torch import Tensor
from tokenizers import ByteLevelBPETokenizer
from transformers import BartConfig, BartModel


def get_bart_config(config: dict, tokenizer: ByteLevelBPETokenizer) -> BartConfig:
    bart_config = BartConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config["d_model"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        encoder_attention_heads=config["encoder_attention_heads"],
        decoder_attention_heads=config["decoder_attention_heads"],
        encoder_ffn_dim=config["encoder_ffn_dim"],
        decoder_ffn_dim=config["decoder_ffn_dim"],
        activation_function=config["activation_function"],
        dropout=config["dropout"],
        max_position_embeddings=config["max_position_embeddings"],
        init_std=config["init_std"],
        scale_embedding=config["scale_embedding"],
        num_beams=config["num_beams"],
        bos_token_id=config["bos_token_id"],
        pad_token_id=config["pad_token_id"],
        eos_token_id=config["eos_token_id"],
    )

    return bart_config


class FinetuneBartModel(nn.Module):
    def __init__(self, config: BartConfig, tokenizer: ByteLevelBPETokenizer) -> None:
        super().__init__()
        self.bart_model = BartModel(config)
        self.proj = nn.Linear(config.d_model, tokenizer.get_vocab_size())

    def forward(self, **kwargs) -> Tensor:
        # output.last_hidden_state (batch_size, seq_len, d_model)
        # logits (batch_size, seq_len, vocab_size)
        out = self.bart_model(**kwargs)
        logits = self.proj(out.last_hidden_state)
        return logits


def build_bart_model(
    config: dict,
    tokenizer: ByteLevelBPETokenizer,
) -> FinetuneBartModel:
    bart_config = get_bart_config(config, tokenizer)
    bart_model = FinetuneBartModel(bart_config, tokenizer)
    return bart_model
