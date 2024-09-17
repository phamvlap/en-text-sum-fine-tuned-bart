import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as Func
from torch.utils.data import DataLoader

from transformers import BartTokenizer
from tqdm import tqdm

from bart.model import FinetuneBartModel
from bart.constants import SpecialToken
from .statistics import Statistics


def greedy_search_decode(
    model: FinetuneBartModel,
    source: Tensor,
    tokenizer: BartTokenizer,
    seq_length: int,
    device: torch.device,
) -> Tensor:
    bos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
    eos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.EOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

    # source (1, seq_length)
    source = source.unsqueeze(dim=0).to(device=device)

    # src_attention_mask (1, seq_length)
    src_attention_mask = (source != pad_token_id).type_as(source).to(device=device)
    # encoder_output (1, seq_length, d_model)
    encoder_output = model.encode(
        input_ids=source,
        attention_mask=src_attention_mask,
    )

    # decoder_input (1, 1)
    decoder_input = (
        torch.empty(1, 1).fill_(value=bos_token_id).type_as(source).to(device=device)
    )

    for _ in range(seq_length):
        decoder_attention_mask = (
            (decoder_input != pad_token_id).type_as(source).to(device=device)
        )

        decoder_output = model.decode(
            input_ids=decoder_input,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=src_attention_mask,
        )

        logits = model.out(decoder_output[:, -1, :])  # logits (1, vocab_size)

        next_token = torch.argmax(input=logits, dim=-1)

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1)
                .fill_(value=next_token.item())
                .type_as(source)
                .to(device=device),
            ],
            dim=1,
        )

        if next_token.item() == eos_token_id:
            break

    output = decoder_input.squeeze(dim=0)
    return output


# Calculate length penalty, formular in (Wu et al., 2016)
def length_penalty(length: int, alpha: float = 0.6) -> float:
    return (5 * length) ** alpha / (5 + 1) ** alpha


def beam_search_decode(
    model: FinetuneBartModel,
    beam_size: int,
    source: Tensor,
    tokenizer: BartTokenizer,
    seq_length: int,
    device: torch.device,
) -> Tensor:
    bos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
    eos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.EOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

    # source (1, seq_length)
    source = source.unsqueeze(dim=0).to(device=device)

    # src_attention_mask (1, seq_length)
    src_attention_mask = (source != pad_token_id).type_as(source).to(device=device)
    # encoder_output (1, seq_length, d_model)
    encoder_output = model.encode(
        input_ids=source,
        attention_mask=src_attention_mask,
    )

    # Initialize decoder input with only <s> token (1, 1)
    decoder_input = (
        torch.empty(1, 1).fill_(value=bos_token_id).type_as(source).to(device=device)
    )

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

            # Create attention mask for decoder input
            decoder_attention_mask = (
                (cand != pad_token_id).type_as(source).to(device=device)
            )

            # decoder_output (1, cand.size(1), d_model)
            decoder_output = model.decode(
                input_ids=cand,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=src_attention_mask,
            )

            # Get the last token logits
            # logits (1, vocab_size)
            logits = model.out(decoder_output[:, -1, :])

            # Get the next token probabilities
            # norm_probs (1, vocab_size)
            norm_probs = Func.log_softmax(logits, dim=-1) / length_penalty(
                length=cand.size(1)
            )

            # Get top k probabilities and indices
            # topk_probs (1, beam_size)
            # topk_indices (1, beam_size)
            topk_probs, topk_indices = torch.topk(input=norm_probs, k=beam_size, dim=-1)

            for i in range(beam_size):
                # next_token (1, 1)
                next_token = topk_indices[0][i].unsqueeze(dim=0).unsqueeze(dim=0)
                next_token_score = topk_probs[0][i].item()

                # cand (1, cand.size(1))
                new_candidate = torch.cat([cand, next_token], dim=1)

                new_candidates.append((new_candidate, score + next_token_score))

        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_size]

    return candidates[0][0].squeeze(dim=0)


@torch.no_grad()
def evaluate(
    model: FinetuneBartModel,
    val_dataloader: DataLoader,
    tokenizer: BartTokenizer,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> Statistics:
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)
    model.to(device=device)

    # Set model to evaluation mode
    model.eval()

    eval_stats = Statistics(
        vocab_size=tokenizer.vocab_size,
        ignore_index=pad_token_id,
        device=device,
    )

    batch_iterator = tqdm(val_dataloader, desc="Evaluating model ...")

    for batch in batch_iterator:
        """
        encoder_input (batch_size, seq_length)
        decoder_input (batch_size, seq_length)
        label (batch_size, seq_length)
        logits (batch_size, seq_length, vocab_size)
        pred (batch_size, seq_length)
        """
        encoder_input = batch["src"].to(device=device)
        decoder_input = batch["tgt"].to(device=device)
        label = batch["label"].to(device=device)

        src_attention_mask = (encoder_input != pad_token_id).to(
            device=device,
            dtype=torch.int64,
        )
        tgt_attention_mask = (decoder_input != pad_token_id).to(
            device=device,
            dtype=torch.int64,
        )

        logits = model(
            input_ids=encoder_input,
            attention_mask=src_attention_mask,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=tgt_attention_mask,
        )

        loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
        pred = torch.argmax(logits, dim=-1)

        eval_stats.update(
            loss=loss.item(),
            pred=pred.view(-1),
            target=label.view(-1),
        )

        batch_iterator.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
            }
        )

    # Set model back to training mode
    model.train()

    return eval_stats
