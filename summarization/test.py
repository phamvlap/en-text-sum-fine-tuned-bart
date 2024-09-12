import torch

from bart.model import build_bart_model
from .tokenizer import load_tokenizer
from .summarization_dataset import get_dataloader
from .val import validate
from .utils.mix import set_seed, get_list_weights_file_paths, write_to_csv


def test(config: dict) -> None:
    set_seed(seed=config["seed"])

    tokenizer = load_tokenizer(bart_tokenizer_dir=config["tokenizer_bart_dir"])

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(config=config)

    model = build_bart_model(config=config, tokenizer=tokenizer)

    list_model_weight_files = get_list_weights_file_paths(config=config)
    if list_model_weight_files is not None:
        state = torch.load(list_model_weight_files[-1])
        model.load_state_dict(state["model_state_dict"])
    else:
        raise ValueError("No model found.")

    rouge_scores = validate(
        model=model,
        val_dataloader=val_dataloader,
        config=config,
    )

    df = write_to_csv(
        columns=rouge_scores.keys(),
        data=[[value] for value in rouge_scores.values()],
        file_path=f"{config['statistic_dir']}/rouge_scores.csv",
    )

    print(df)
