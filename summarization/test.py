import os
import pandas as pd
import torch
import torch.nn as nn

from transformers import PretrainedConfig

from bart.model import (
    build_bart_model,
    FineTunedBartForConditionalGeneration,
)
from bart.constants import SpecialToken
from .trainer_utils import get_last_checkpoint
from .summarization_dataset import get_dataloader
from .utils.seed import set_seed
from .utils.mix import make_dir, write_to_json
from .utils.eval import evaluate
from .utils.tokenizer import load_tokenizer
from .utils.metrics import compute_rouge_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test(config: dict) -> None:
    print("Testing model...")
    set_seed(seed=config["seed"])

    device = torch.device(config["device"])

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(bart_tokenizer_dir=config["tokenizer_bart_dir"])

    print("Getting dataloader...")
    test_dataloader = get_dataloader(
        tokenizer=tokenizer,
        split="test",
        batch_size=config["batch_size_test"],
        shuffle=False,
        config=config,
    )

    print("Building BART model...")
    last_checkpoint = get_last_checkpoint(
        output_dir=config["checkpoint_dir"],
        checkpoint_prefix=config["model_basename"],
    )
    if last_checkpoint is not None:
        print(f"Loading from checkpoint {last_checkpoint}")
        torch.serialization.add_safe_globals([PretrainedConfig])
        checkpoint_states = torch.load(
            last_checkpoint,
            weights_only=True,
            map_location=device,
        )
        required_keys = ["model_state_dict", "config"]
        for key in required_keys:
            if key not in checkpoint_states:
                raise ValueError(f"Missing key {key} in checkpoint states")
    else:
        raise ValueError("No checkpoint found")

    bart_config = checkpoint_states["config"]
    model: FineTunedBartForConditionalGeneration = build_bart_model(
        model_name_or_path=config["model_name_or_path"],
        config=bart_config,
    ).to(device=device)
    model.load_state_dict(checkpoint_states["model_state_dict"])

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.convert_tokens_to_ids(SpecialToken.PAD),
        label_smoothing=config["label_smoothing"],
    ).to(device=device)

    eval_result = evaluate(
        model=model,
        val_dataloader=test_dataloader,
        tokenizer=tokenizer,
        criterion=loss_fn,
        device=device,
        use_ddp=False,
        show_eval_progress=config["show_eval_progress"],
    )
    scores = compute_rouge_score(
        model=model,
        dataset=test_dataloader.dataset,
        tokenizer=tokenizer,
        seq_length=config["tgt_seq_length"],
        device=device,
        beam_size=config["beam_size"],
        topk=config["topk"],
        log_examples=config["log_examples"],
        logging_steps=config["logging_steps"],
        use_stemmer=config["use_stemmer"],
        rouge_keys=config["rouge_keys"],
        accumulate=config["accumulate"],
        use_ddp=False,
        max_steps=config["max_eval_steps"],
    )

    summary_metrics = {
        **eval_result,
        **scores,
    }
    df_metric_scores = pd.DataFrame(summary_metrics, index=[0])

    make_dir(config["statistic_dir"])
    file_path = f"{config['statistic_dir']}/test_summary_metrics.json"
    write_to_json(data=summary_metrics, file_path=file_path)

    print(f"Test result saved to {file_path}")
    print("TEST METRIC SCORES:")
    print(df_metric_scores)
