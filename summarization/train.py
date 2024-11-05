import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from bart.model import ModelArguments, build_bart_model
from bart.constants import SpecialToken
from .summarization_dataset import get_dataloader
from .trainer import Trainer
from .trainer_utils import TrainingArguments, get_last_checkpoint
from .utils.tokenizer import load_tokenizer
from .utils.seed import set_seed
from .utils.mix import (
    count_parameters,
    is_torch_cuda_available,
    get_current_time,
    print_once,
)
from .utils.optimizer import get_optimizer, get_lr_scheduler
from .utils.wb_logger import WandbLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ddp_setup(config: dict) -> None:
    config["rank"] = int(os.environ.get("RANK", -1))
    config["local_rank"] = int(os.environ.get("LOCAL_RANK", -1))
    config["world_size"] = int(os.environ.get("WORLD_SIZE", 0))
    config["use_ddp"] = config["rank"] >= 0

    if config["use_ddp"]:
        # Initialize the process group
        torch.cuda.set_device(config["local_rank"])
        # Use NCCL backend for communication between processes
        init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=config["world_size"],
            rank=config["rank"],
        )

        # Devide the batch size by number of processes
        for key in ["batch_size_train", "batch_size_val"]:
            assert (
                config[key] % config["world_size"] == 0
            ), f"{key} should be divisible by world_size = {config['world_size']}"
            config[key] = config[key] // config["world_size"]


def train(config: dict) -> None:
    set_seed(seed=config["seed"])

    device = torch.device(config["device"])
    if is_torch_cuda_available():
        if config["use_ddp"]:
            if config["rank"] == 0:
                device_name = torch.cuda.get_device_name()
                device_count = torch.cuda.device_count()
                print(f"Running on GPU: {device_name} x {device_count}")
        else:
            print(f"Running on GPU: {torch.cuda.get_device_name()}")
    else:
        print("Running on CPU")

    print_once(config, "Loading tokenizer...")
    tokenizer = load_tokenizer(bart_tokenizer_dir=config["tokenizer_bart_dir"])
    pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)

    print_once(config, "Loading dataloaders...")
    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        split="train",
        batch_size=config["batch_size_train"],
        shuffle=config["shuffle_dataloader"],
        config=config,
    )
    val_dataloader = get_dataloader(
        tokenizer=tokenizer,
        split="val",
        batch_size=config["batch_size_val"],
        shuffle=config["shuffle_dataloader"],
        config=config,
    )

    config["f16_precision"] = config["f16_precision"] and is_torch_cuda_available()

    initial_epoch = 0
    initial_global_step = 0
    checkpoint_path = None
    checkpoint_states = None
    scaler_state_dict = None

    if config["resume_from_checkpoint"]:
        checkpoint_path = get_last_checkpoint(
            output_dir=config["checkpoint_dir"],
            checkpoint_prefix=config["model_basename"],
        )

    if checkpoint_path is None:
        model_name_or_path = config["model_name_or_path"]
        print_once(config, f"Training model from pre-trained {model_name_or_path}")
        model_args = ModelArguments(
            model_name_or_path=model_name_or_path,
            config_name_or_path=model_name_or_path,
        )
        bart_model = build_bart_model(model_args=model_args)
        bart_model_config = bart_model.get_config()
        bart_model.to(device=device)
    else:
        print_once(config, f"Loading model from checkpoint {checkpoint_path}")

        checkpoint_states = torch.load(checkpoint_path, map_location=device)

        required_keys = [
            "model_state_dict",
            "optimizer_state_dict",
            "config",
        ]
        if config["lr_scheduler"] is not None:
            required_keys.append("lr_scheduler_state_dict")
        for key in required_keys:
            if key not in checkpoint_states.keys():
                raise ValueError(f"Missing key {key} in checkpoint states.")

        bart_model_config = checkpoint_states["config"]
        model_name = f"facebook/{bart_model_config._name_or_path}"
        model_args = ModelArguments(model_name_or_path=model_name)
        bart_model = build_bart_model(model_args=model_args, config=bart_model_config)
        bart_model.load_state_dict(checkpoint_states["model_state_dict"])
        bart_model.to(device=device)

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

    print_once(
        config,
        f"The model has {count_parameters(original_bart_model):,} trainable parameters.",
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
        checkpoint_dir=config["checkpoint_dir"],
        model_basename=config["model_basename"],
        eval_every_n_steps=config["eval_every_n_steps"],
        save_every_n_steps=config["save_every_n_steps"],
        beam_size=config["beam_size"],
        topk=config["topk"],
        log_examples=config["log_examples"],
        logging_steps=config["logging_steps"],
        use_stemmer=config["use_stemmer"],
        truncation=config["truncation"],
        accumulate=config["accumulate"],
        max_grad_norm=config["max_grad_norm"],
        f16_precision=config["f16_precision"],
        use_ddp=config["use_ddp"],
        rank=config["rank"] if config["use_ddp"] else None,
        local_rank=config["local_rank"] if config["use_ddp"] else None,
        world_size=config["world_size"] if config["use_ddp"] else None,
        max_eval_steps=config["max_eval_steps"],
        max_train_steps=config["max_train_steps"],
    )

    wb_logger = None
    if config["is_logging_wandb"]:
        saved_config = {
            "model_config": bart_model_config.__dict__,
            "training_args": training_args.__dict__,
        }
        display_name = f"running_{get_current_time(to_string=True)}"

        wb_logger = WandbLogger(
            project_name=config["wandb_project_name"],
            config=saved_config,
            key=config["wandb_key"],
            log_dir=config["wandb_log_dir"],
            name=display_name,
        )

    trainer = Trainer(
        model=bart_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        tokenizer=tokenizer,
        criterion=loss_fn,
        args=training_args,
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
