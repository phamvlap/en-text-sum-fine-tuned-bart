import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from bart.model import build_bart_model, FinetuneBartModelConfig
from bart.constants import SpecialToken
from .summarization_dataset import get_dataloader
from .trainer import Trainer, TrainingConfig
from .utils.tokenizer import load_tokenizer
from .utils.seed import set_seed
from .utils.mix import noam_lr
from .utils.path import make_dirs, get_weights_file_path, get_list_weights_file_paths


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
    bos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)
    eos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.EOS)

    # Build BART model
    bart_model_config = FinetuneBartModelConfig(
        seq_length=config["seq_length"],
        device=config["device"],
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        encoder_attention_heads=config["encoder_attention_heads"],
        decoder_attention_heads=config["decoder_attention_heads"],
        encoder_ffn_dim=config["encoder_ffn_dim"],
        decoder_ffn_dim=config["decoder_ffn_dim"],
        activation_function=config["activation_function"],
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        activation_dropout=config["activation_dropout"],
        classifier_dropout=config["classifier_dropout"],
        init_std=config["init_std"],
        encoder_layerdrop=config["encoder_layerdrop"],
        decoder_layerdrop=config["decoder_layerdrop"],
        scale_embedding=config["scale_embedding"],
        num_beams=config["num_beams"],
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    bart_model = build_bart_model(config=bart_model_config).to(device=device)

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
    lr_scheduler = None
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
        model_filename = get_weights_file_path(
            model_basedir=config["model_dir"],
            model_basename=config["model_basename"],
            epoch=config["preload"],
        )

    if model_filename is not None:
        state = torch.load(model_filename)

        bart_model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

        print("Loaded model from {}.".format(model_filename))
    else:
        print("No model loaded, training from scratch.")

    writer = SummaryWriter(log_dir=config["log_dir"])

    training_config = TrainingConfig(
        device=device,
        seq_length=config["seq_length"],
        initial_epoch=initial_epoch,
        initial_global_step=global_step,
        num_epochs=config["epochs"],
        model_dir=config["model_dir"],
        model_basename=config["model_basename"],
        evaluating_steps=config["evaluating_steps"],
        beam_size=config["beam_size"],
        log_examples=config["log_examples"],
        logging_steps=config["logging_steps"],
        use_stemmer=config["use_stemmer"],
        accumulate=config["accumulate"],
    )

    trainer = Trainer(
        model=bart_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        config=training_config,
        bart_config=bart_model_config,
        writer=writer,
    )

    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
