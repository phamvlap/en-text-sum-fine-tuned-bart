import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as Func
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from typing import Optional
from tqdm import tqdm

from bart.model import FineTunedBartForGeneration
from bart.constants import SpecialToken
from .metric_tracker import MetricTracker


def create_encoder_mask(encoder_input: Tensor, pad_token_id: int) -> Tensor:
    """
    Args:
        encoder_input: input tensor, shape `(1, seq_length)`
        pad_token_id: id of padding token
    Returns:
        Tensor: masked tensor, shape `(1, seq_length)`
    """
    return (encoder_input != pad_token_id).type_as(encoder_input)


def create_decoder_mask(decoder_input: Tensor, pad_token_id: int) -> Tensor:
    """
    Args:
        decoder_input: input tensor, shape `(1, seq_length)`
        pad_token_id: id of padding token
    Returns:
        Tensor: masked tensor, shape `(1, seq_length)`
    """
    return (decoder_input != pad_token_id).type_as(decoder_input)


def greedy_search_decode(
    model: FineTunedBartForGeneration | DDP,
    input_ids: Tensor,
    tokenizer: BartTokenizer,
    seq_length: int,
    device: torch.device,
    use_ddp: bool = False,
    local_rank: Optional[int] = None,
) -> Tensor:
    """
    Args:
        model: model of FinetuneBartModel
        input_ids: input tensor, shape `(seq_length,)`
        tokenizer: tokenizer of BartTokenizer
        seq_length: maximum sequence length
        device: torch.device
        use_ddp: whether to use DDP
        local_rank: local rank if use DDP
    Returns:
        Tensor: output tensor, shape `(seq_length,)`
    """
    bos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
    eos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.EOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

    # input_ids (1, seq_length)
    assert (
        local_rank is not None if use_ddp else True
    ), "local_rank must be not None if use DDP"

    if use_ddp:
        model.to(local_rank)
        input_ids = input_ids.unsqueeze(dim=0).to(local_rank)
    else:
        model.to(device)
        input_ids = input_ids.unsqueeze(dim=0).to(device)

    # encoder_attention_mask (1, seq_length)
    encoder_attention_mask = create_encoder_mask(
        encoder_input=input_ids,
        pad_token_id=pad_token_id,
    )

    # decoder_input (1, 1)
    decoder_input = torch.empty(1, 1).fill_(value=bos_token_id).type_as(input_ids)

    for _ in range(seq_length):
        decoder_attention_mask = create_decoder_mask(
            decoder_input=decoder_input,
            pad_token_id=pad_token_id,
        )

        # logits (1, decoder_input.size(1), vocab_size)
        logits = model(
            encoder_input_ids=input_ids,
            encoder_attn_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input,
            decoder_attn_mask=decoder_attention_mask,
        )

        # last_logits (1, vocab_size)
        last_logits = logits[:, -1, :]
        next_token = torch.argmax(input=last_logits, dim=-1)

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).fill_(value=next_token.item()).type_as(input_ids),
            ],
            dim=1,
        )

        if next_token.item() == eos_token_id:
            break

    output = decoder_input.squeeze(dim=0)
    return output


# Calculate length penalty, formula in (Wu et al., 2016)
def length_penalty(length: int, alpha: float = 0.6) -> float:
    return (5 * length) ** alpha / (5 + 1) ** alpha


def beam_search_decode(
    model: FineTunedBartForGeneration | DDP,
    beam_size: int,
    input_ids: Tensor,
    tokenizer: BartTokenizer,
    seq_length: int,
    device: torch.device,
    topk: int = 1,
    use_ddp: bool = False,
    local_rank: Optional[int] = None,
) -> list[Tensor]:
    """
    Args:
        model: model of FinetuneBartModel
        beam_size: size of beam
        input_ids: input tensor, shape `(seq_length,)`
        tokenizer: tokenizer of BartTokenizer
        seq_length: maximum sequence length
        device: torch.device
        topk: top k best candidates returned (default: 1)
        use_ddp: whether to use DDP
        local_rank: local rank if use DDP
    Returns:
        list[Tensor]: list of output tensors, each tensor with shape `(seq_length,)`
    """
    bos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
    eos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.EOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

    # input_ids (1, seq_length)
    assert (
        local_rank is not None if use_ddp else True
    ), "local_rank must be not None if use DDP"

    if use_ddp:
        model.to(local_rank)
        input_ids = input_ids.unsqueeze(dim=0).to(local_rank)
    else:
        model.to(device)
        input_ids = input_ids.unsqueeze(dim=0).to(device)

    # encoder_attention_mask (1, seq_length)
    encoder_attention_mask = create_encoder_mask(
        encoder_input=input_ids,
        pad_token_id=pad_token_id,
    )

    # Initialize decoder input with only <s> token (1, 1)
    decoder_input = torch.empty(1, 1).fill_(value=bos_token_id).type_as(input_ids)

    # Candidate list ccontaints tuples of (cand, log_score)
    candidates = [(decoder_input, 0.0)]

    while True:
        if all(
            [
                cand.size(1) == seq_length or cand[0][-1].item() == eos_token_id
                for cand, _ in candidates
            ]
        ):
            break

        new_candidates = []
        for cand, score in candidates:
            if cand.size(1) == seq_length or cand[0][-1].item() == eos_token_id:
                new_candidates.append((cand, score))
                continue

            # Create attention mask for decoder input, shape (1, cand.size(1))
            decoder_attention_mask = create_decoder_mask(
                decoder_input=cand,
                pad_token_id=pad_token_id,
            )

            # logits (1, cand.size(1), vocab_size)
            logits = model(
                encoder_input_ids=input_ids,
                encoder_attn_mask=encoder_attention_mask,
                decoder_input_ids=cand,
                decoder_attn_mask=decoder_attention_mask,
            )

            # last_logits (1, vocab_size)
            last_logits = logits[:, -1, :]

            # Get the next token probabilities
            # norm_probs (1, vocab_size)
            norm_probs = Func.log_softmax(last_logits, dim=-1) / length_penalty(
                length=cand.size(1)
            )

            # Get top k probabilities and indices
            # topk_probs (1, beam_size)
            # topk_indices (1, beam_size)
            topk_probs, topk_indices = torch.topk(input=norm_probs, k=beam_size, dim=-1)

            for i in range(beam_size):
                # next_token (1, 1)
                next_token = topk_indices[0][i].unsqueeze(dim=0).unsqueeze(dim=0)
                next_token = next_token.type_as(input_ids)
                next_token_score = topk_probs[0][i].item()

                # cand (1, cand.size(1))
                new_candidate = torch.cat([cand, next_token], dim=1)

                new_candidates.append((new_candidate, score + next_token_score))

        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_size]

    candidates = candidates[:topk]
    outputs = [cand[0].squeeze(dim=0) for cand, _ in candidates]

    return outputs


@torch.no_grad()
def evaluate(
    model: FineTunedBartForGeneration | DDP,
    val_dataloader: DataLoader,
    tokenizer: BartTokenizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    use_ddp: bool = False,
    rank: Optional[int] = None,
    local_rank: Optional[int] = None,
) -> MetricTracker:
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

    assert (
        local_rank is not None and rank is not None if use_ddp else True
    ), "local_rank and rank must be not None if use DDP"

    if use_ddp:
        model.to(local_rank)
    else:
        model.to(device=device)

    # Set model to evaluation mode
    model.eval()
    eval_metric_tracker = MetricTracker()

    if use_ddp:
        val_iterator = tqdm(
            val_dataloader,
            desc=f"[GPU-{rank}] Evaluating model",
            disable=local_rank != 0,
        )
    else:
        val_iterator = tqdm(
            val_dataloader,
            desc="Evaluating model",
        )

    for batch in val_iterator:
        # encoder_input (batch_size, seq_length)
        # decoder_input (batch_size, seq_length)
        # labels (batch_size, seq_length)
        if use_ddp:
            encoder_input = batch["encoder_input"].to(local_rank)
            decoder_input = batch["decoder_input"].to(local_rank)
            labels = batch["labels"].to(local_rank)
        else:
            encoder_input = batch["encoder_input"].to(device=device)
            decoder_input = batch["decoder_input"].to(device=device)
            labels = batch["labels"].to(device=device)

        encoder_attention_mask = create_encoder_mask(
            encoder_input=encoder_input,
            pad_token_id=pad_token_id,
        ).to(torch.int64)
        decoder_attention_mask = create_decoder_mask(
            decoder_input=decoder_input,
            pad_token_id=pad_token_id,
        ).to(torch.int64)

        # logits (batch_size, seq_length, vocab_size)
        logits = model(
            encoder_input_ids=encoder_input,
            encoder_attn_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input,
            decoder_attn_mask=decoder_attention_mask,
        )

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        eval_metric_tracker.update(loss=loss.item())

        val_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    # Set model back to training mode
    model.train()

    return eval_metric_tracker
