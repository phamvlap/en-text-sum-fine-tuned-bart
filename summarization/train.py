import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from bart.model import build_bart_model, FineTunedBartForGenerationConfig
from bart.constants import SpecialToken
from .summarization_dataset import get_dataloader
from .trainer import Trainer
from .trainer_utils import TrainingArguments
from .utils.tokenizer import load_tokenizer
from .utils.seed import set_seed
from .utils.mix import count_parameters, is_torch_cuda_available
from .utils.path import make_dirs, get_weights_file_path, get_list_weights_file_paths
from .utils.optimizer import get_optimizer
from .utils.lr_scheduler import get_lr_scheduler
from .utils.wb_logger import WandbLogger


def ddp_setup(config: dict) -> None:
    config["rank"] = int(os.environ.get("RANK", -1))
    config["local_rank"] = int(os.environ.get("LOCAL_RANK", -1))
    config["world_size"] = int(os.environ.get("WORLD_SIZE", 0))
    config["use_ddp"] = config["rank"] >= 0

    if config["use_ddp"]:
        # Initialize the process group
        torch.cuda.set_device(config["local_rank"])
        # Use NCCL backend for communication between processes
        init_process_group(backend="nccl")


def train(config: dict) -> None:
    set_seed(seed=config["seed"])

    make_dirs(
        config=config,
        dir_names=[
            "model_dir",
        ],
    )

    device = torch.device(config["device"])
    if is_torch_cuda_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(bart_tokenizer_dir=config["tokenizer_bart_dir"])
    bos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.BOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)
    eos_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.EOS)

    print("Loading dataloaders...")
    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        split="train",
        batch_size=config["batch_size_train"],
        shuffle=True,
        config=config,
    )
    val_dataloader = get_dataloader(
        tokenizer=tokenizer,
        split="val",
        batch_size=config["batch_size_val"],
        shuffle=True,
        config=config,
    )

    initial_epoch = 0
    initial_global_step = 0
    model_filename = None
    checkpoint_states = None
    scaler_state_dict = None

    if config["preload"] == "latest":
        list_weights_file_paths = get_list_weights_file_paths(config=config)
        if list_weights_file_paths is not None:
            model_filename = list_weights_file_paths[-1]
    elif config["preload"] is not None:
        model_filename = get_weights_file_path(
            model_basedir=config["model_dir"],
            model_basename=config["model_basename"],
            epoch=config["preload"],
        )

    if model_filename is None:
        print("Starting training model from scratch")

        bart_model_config = FineTunedBartForGenerationConfig(
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
        bart_model = build_bart_model(config=bart_model_config)
        bart_model.to(device=device)
    else:
        print("Loading model from {}".format(model_filename))

        checkpoint_states = torch.load(model_filename, map_location=device)

        required_keys = [
            "model_state_dict",
            "optimizer_state_dict",
            "config",
        ]
        if config["lr_scheduler"] is not None:
            required_keys.append("lr_scheduler_state_dict")
        for key in required_keys:
            if key not in checkpoint_states.keys():
                raise ValueError(f"Missing key {key} in model state dict.")

        bart_model_config = checkpoint_states["config"]
        bart_model = build_bart_model(config=bart_model_config)
        bart_model.to(device=bart_model_config.device)
        bart_model.load_state_dict(checkpoint_states["model_state_dict"])

        if "epoch" in checkpoint_states:
            initial_epoch = checkpoint_states["epoch"] + 1
        if "global_step" in checkpoint_states:
            initial_global_step = checkpoint_states["global_step"]
        if "scaler_state_dict" in checkpoint_states:
            scaler_state_dict = checkpoint_states["scaler_state_dict"]

    original_bart_model = bart_model
    if config["use_ddp"]:
        bart_model = DDP(
            bart_model,
            device_ids=[config["local_rank"]],
            output_device=config["local_rank"],
            find_unused_parameters=True,
        )

    print(
        f"The model has {count_parameters(original_bart_model):,} trainable parameters."
    )

    # Optimizer
    optimizer = get_optimizer(model=original_bart_model, config=config)

    # Learning rate scheduler
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, config=config)

    if checkpoint_states is not None:
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint_states["lr_scheduler_state_dict"])

    # Loss function
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        label_smoothing=config["label_smoothing"],
    ).to(device=device)

    training_args = TrainingArguments(
        device=device,
        seq_length=config["seq_length"],
        initial_epoch=initial_epoch,
        initial_global_step=initial_global_step,
        num_epochs=config["epochs"],
        model_dir=config["model_dir"],
        model_basename=config["model_basename"],
        eval_every_n_steps=config["eval_every_n_steps"],
        save_every_n_steps=config["save_every_n_steps"],
        beam_size=config["beam_size"],
        topk=config["topk"],
        log_examples=config["log_examples"],
        logging_steps=config["logging_steps"],
        use_stemmer=config["use_stemmer"],
        accumulate=config["accumulate"],
        max_grad_norm=config["max_grad_norm"],
        f16_precision=config["f16_precision"],
        use_ddp=config["use_ddp"],
        rank=config["rank"] if config["use_ddp"] else None,
        local_rank=config["local_rank"] if config["use_ddp"] else None,
    )

    saved_config = {
        "model_config": bart_model_config.__dict__,
        "training_args": training_args.__dict__,
    }

    wb_logger = WandbLogger(
        project_name=config["wandb_project_name"],
        config=saved_config,
        key=config["wandb_key"],
        log_dir=config["wandb_log_dir"],
    )

    trainer = Trainer(
        model=bart_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        tokenizer=tokenizer,
        criterion=loss_fn,
        args=training_args,
        bart_config=bart_model_config,
        scaler_state_dict=scaler_state_dict,
        wb_logger=wb_logger,
    )

    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )


def main(config: dict) -> None:
    # Set up Distributed Data Parallel (DDP) is available
    ddp_setup(config)

    # Train model
    train(config)

    # Clean up DDP
    if config["use_ddp"]:
        destroy_process_group()
