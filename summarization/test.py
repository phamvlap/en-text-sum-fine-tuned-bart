import torch
import torch.nn as nn

from bart.model import build_bart_model, FineTunedBartForGeneration
from bart.constants import SpecialToken
from .summarization_dataset import get_dataloader
from .utils.seed import set_seed
from .utils.mix import write_to_csv
from .utils.eval import evaluate
from .utils.path import get_list_weights_file_paths
from .utils.tokenizer import load_tokenizer
from .utils.metrics import compute_rouge_bert_score


def test(config: dict) -> None:
    print("Testing model...")
    set_seed(seed=config["seed"])

    device = torch.device(config["device"])

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(bart_tokenizer_dir=config["tokenizer_bart_dir"])

    print("Getting dataloaders...")
    test_dataloader = get_dataloader(
        tokenizer=tokenizer,
        split="test",
        batch_size=config["batch_size_test"],
        shuffle=False,
        config=config,
    )

    list_model_weight_files = get_list_weights_file_paths(config=config)
    if list_model_weight_files is not None:
        states = torch.load(list_model_weight_files[-1])
        required_keys = ["model_state_dict", "config"]
        for key in required_keys:
            if key not in states:
                raise ValueError(f"Missing key {key} in model state dict")
    else:
        raise ValueError("No model found.")

    print("Building BART model...")
    bart_config = states["config"]
    model: FineTunedBartForGeneration = build_bart_model(
        config["model_name_or_path"],
        config=bart_config,
    ).to(device=device)
    model.load_state_dict(states["model_state_dict"], weights_only=True)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.convert_tokens_to_ids(SpecialToken.PAD),
        label_smoothing=config["label_smoothing"],
    ).to(device=device)

    print("Evaluating model...")
    test_metric_tracker = evaluate(
        model=model,
        val_dataloader=test_dataloader,
        tokenizer=tokenizer,
        criterion=loss_fn,
        device=device,
        use_ddp=False,
    )
    scores = compute_rouge_bert_score(
        model=model,
        dataset=test_dataloader.dataset,
        tokenizer=tokenizer,
        seq_length=config["seq_length"],
        device=device,
        beam_size=config["beam_size"],
        topk=config["topk"],
        eval_bert_score=config["eval_bert_score"],
        rescale=config["rescale"],
        log_examples=config["log_examples"],
        logging_steps=config["logging_steps"],
        use_stemmer=config["use_stemmer"],
        rouge_keys=config["rouge_keys"],
        accumulate=config["accumulate"],
        use_ddp=False,
    )

    metric_scores = test_metric_tracker.compute()

    columns = list(metric_scores.keys()) + list(scores.keys())
    data = [[value] for value in metric_scores.values()] + [
        [value] for value in scores.values()
    ]

    df = write_to_csv(
        columns=columns,
        data=data,
        file_path=f"{config['statistic_dir']}/metric_scores.csv",
    )
    print("TEST METRIC SCORES:")
    print(df)
