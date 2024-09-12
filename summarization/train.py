import json
import torch
import torch.optim as optim
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm

from bart.model import build_bart_model
from bart.constants import SpecialToken
from .summarization_dataset import get_dataloader
from .utils.mix import (
    set_seed,
    make_dirs,
    get_weights_file_path,
    get_list_weights_file_paths,
    noam_lr,
    get_dir_path,
)
from .tokenizer import load_tokenizer


def save_model(
    model_filepath: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        model_filepath,
    )


def save_model_config(config: dict, epoch: int) -> None:
    filepath = (
        get_dir_path(config=config, dir_name=config["model_dir"])
        + "/"
        + config["model_config_file"].format(epoch)
    )
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)


def train(config: dict) -> None:
    # Make directories
    make_dirs(
        config=config,
        dir_names=[
            "model_dir",
        ],
    )

    # Device
    device = torch.device(config["device"])
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")

    # Set seed
    set_seed(seed=config["seed"])

    # Load tokenizer
    tokenizer = load_tokenizer(bart_tokenizer_dir=config["tokenizer_bart_dir"])

    # Build BART model
    bart_model = build_bart_model(
        config=config,
        tokenizer=tokenizer,
    ).to(device=device)

    # Get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(config=config)

    # Optimizer
    optimizer = optim.Adam(
        params=bart_model.parameters(),
        lr=config["lr"],
        betas=config["betas"],
        eps=config["eps"],
    )

    # Learning rate scheduler
    if config["lr_scheduler"] == "noam":
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: noam_lr(
                model_size=config["d_model"],
                step=step,
                warmup_steps=config["warmup_steps"],
            ),
        )

    # Loss function
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        label_smoothing=config["label_smoothing"],
    ).to(device=device)

    initial_epoch = 0
    global_step = 0

    # Load preloaded model
    model_filename = None
    if config["preload"] == "latest":
        list_weights_file_paths = get_list_weights_file_paths(config=config)
        if list_weights_file_paths is not None:
            model_filename = list_weights_file_paths[-1]
    else:
        model_filename = get_weights_file_path(config=config, epoch=config["preload"])

    if model_filename is not None:
        state = torch.load(model_filename)

        bart_model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

        print("Loaded model from {}.".format(model_filename))
    else:
        print("No model loaded, training from scratch.")

    train_losses = []
    val_losses = []

    # Traing loop
    for epoch in range(initial_epoch, config["epochs"]):
        torch.cuda.empty_cache()

        # Train
        bart_model.train()
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Training epoch {epoch + 1}/{config['epochs']}",
        )

        for batch in batch_iterator:
            src = batch["src"].to(device=device)
            tgt = batch["tgt"].to(device=device)
            label = batch["label"].to(device=device)

            src_attention_mask = (src != pad_token_id).to(
                device=device,
                dtype=torch.int64,
            )
            tgt_attention_mask = (tgt != pad_token_id).to(
                device=device,
                dtype=torch.int64,
            )

            logits = bart_model(
                input_ids=src,
                attention_mask=src_attention_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_attention_mask,
            )

            # Compute loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
            train_losses.append(loss.item())
            batch_iterator.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                }
            )

            loss.backward()

            optimizer.step()
            if config["lr_scheduler"] == "noam":
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        with torch.no_grad():
            # Evaluate
            bart_model.eval()
            batch_iterator = tqdm(
                val_dataloader,
                desc=f"Validating epoch {epoch + 1}/{config['epochs']}",
            )

            for batch in batch_iterator:
                src = batch["src"].to(device=device)
                tgt = batch["tgt"].to(device=device)
                label = batch["label"].to(device=device)

                src_attention_mask = (src != pad_token_id).to(
                    device=device,
                    dtype=torch.int64,
                )
                tgt_attention_mask = (tgt != pad_token_id).to(
                    device=device,
                    dtype=torch.int64,
                )

                logits = bart_model(
                    input_ids=src,
                    attention_mask=src_attention_mask,
                    decoder_input_ids=tgt,
                    decoder_attention_mask=tgt_attention_mask,
                )

                loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
                val_losses.append(loss.item())
                batch_iterator.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                    }
                )

        # Save model
        save_model(
            model_filepath=get_weights_file_path(config=config, epoch=epoch),
            model=bart_model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
        )
        save_model_config(
            config=config,
            epoch=epoch,
        )

    # Statistic
    print("Train loss: {:.4f}".format(sum(train_losses) / len(train_losses)))
    print("Validation loss: {:.4f}".format(sum(val_losses) / len(val_losses)))
