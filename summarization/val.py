import torch

from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from tqdm import tqdm

from bart.model import FinetuneBartModel
from bart.constants import SpecialToken
from .tokenizer import load_tokenizer
from .rouge_scorer import RougeScorer
from .utils.mix import set_seed


def greedy_search_decode(
    model: FinetuneBartModel,
    source: Tensor,
    source_mask: Tensor,
    tokenizer: BartTokenizer,
    seq_length: int,
    device: torch.device,
) -> Tensor:
    bos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
    eos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.EOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

    encoder_output = model.encode(input_ids=source, attention_mask=source_mask)

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
            encoder_hidden_states=encoder_output.last_hidden_state,
            encoder_attention_mask=source_mask,
        ).last_hidden_state

        logits = model.out(decoder_output[:, -1, :])

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


@torch.no_grad()
def validate(
    model: FinetuneBartModel,
    val_dataloader: DataLoader,
    config: dict,
    num_examples: int = 5,
) -> dict:
    set_seed(seed=config["seed"])
    device = torch.device(config["device"])

    model = model.to(device=device)
    model.eval()

    tokenizer = load_tokenizer(bart_tokenizer_dir=config["tokenizer_bart_dir"])
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

    rouge_scorer = RougeScorer(
        rouge_keys=config["rouge_keys"],
        use_stemmer=config["use_stemmer"],
        tokenizer_function=tokenizer.tokenize,
        accumulate=config["accumulate"],
    )

    logging_steps = len(val_dataloader) // num_examples
    count = 0

    target_texts = []
    pred_texts = []

    batch_iterator = tqdm(val_dataloader, desc="Validating model ...")

    for batch in batch_iterator:
        count += 1
        src = batch["src"].to(device=device)
        tgt = batch["tgt"].to(device=device)

        src_attention_mask = (src != pad_token_id).to(device=device, dtype=torch.int64)

        src_text = tokenizer.decode(
            src[0].detach().cpu().numpy(),
            skip_special_tokens=True,
        )
        tgt_text = tokenizer.decode(
            tgt[0].detach().cpu().numpy(),
            skip_special_tokens=True,
        )

        pred_ids = greedy_search_decode(
            model=model,
            source=src,
            source_mask=src_attention_mask,
            tokenizer=tokenizer,
            seq_length=config["max_sequence_length"],
            device=device,
        )

        pred_text = tokenizer.decode(
            pred_ids.detach().cpu().numpy(),
            skip_special_tokens=True,
        )

        tgt_tokens = tokenizer.tokenize(text=tgt_text)
        pred_tokens = tokenizer.tokenize(text=pred_text)

        target_texts.append(tgt_text)
        pred_texts.append(pred_text)

        if count % logging_steps == 0:
            print("SOURCE TEXT: {}".format(src_text))
            print("TARGET TEXT: {}".format(tgt_text))
            print("PREDICTED TEXT: {}".format(pred_text))
            print("TARGET TOKENS: {}".format(tgt_tokens))
            print("PREDICTED TOKENS: {}".format(pred_tokens))

            rouge_score = rouge_scorer.calculate(preds=pred_text, targets=tgt_text)

            print("ROUGE SCORE FOR EXAMPLE {}".format(count))
            for rouge_key in config["rouge_keys"]:
                print(
                    "ROUGE-{}: {}".format(
                        rouge_key.replace("rouge", ""),
                        round(rouge_score[f"{rouge_key}_fmeasure"].item() * 100, 2),
                    )
                )

    rouge_score = rouge_scorer.calculate(preds=pred_texts, targets=target_texts)

    simple_rouge_score = {}
    for rouge_key in config["rouge_keys"]:
        simple_rouge_score[rouge_key] = round(
            rouge_score[f"{rouge_key}_fmeasure"].item() * 100,
            2,
        )

    print("ROUGE SCORE FOR VALIDATION SET")
    for rouge_key in config["rouge_keys"]:
        print(
            "ROUGE-{}: {}".format(
                rouge_key.replace("rouge", ""),
                simple_rouge_score[rouge_key],
            )
        )

    return simple_rouge_score
