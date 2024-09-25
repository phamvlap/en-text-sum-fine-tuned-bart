import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from transformers import BartTokenizer
from tqdm import tqdm
from dataclasses import dataclass
from typing import Literal, Optional

from bart.model import FinetuneBartModel, FinetuneBartModelConfig
from bart.constants import SpecialToken, RougeKey
from .utils.statistics import Statistics
from .utils.eval import evaluate
from .utils.rouge import compute_dataset_rouge
from .utils.path import get_weights_file_path, make_dir


@dataclass
class TrainingConfig:
    device: torch.device
    seq_length: int
    num_epochs: int
    model_dir: str
    model_basename: str
    wandb_project: str
    wandb_key: str
    wandb_notes: Optional[str] = None
    wandb_log_dir: Optional[str] = None
    initial_epoch: int = 0
    initial_global_step: int = 0
    evaluating_steps: int = 1000
    beam_size: Optional[int] = None
    log_examples: bool = True
    logging_steps: int = 100
    rouge_keys: list[str] | tuple[str] = (
        RougeKey.ROUGE_1,
        RougeKey.ROUGE_2,
        RougeKey.ROUGE_L,
    )
    use_stemmer: bool = True
    accumulate: Literal["best", "avg"] = "best"
    max_grad_norm: Optional[float] = None


class Trainer:
    def __init__(
        self,
        model: FinetuneBartModel,
        optimizer: optim.Optimizer,
        tokenizer: BartTokenizer,
        criterion: nn.CrossEntropyLoss,
        config: TrainingConfig,
        bart_config: FinetuneBartModelConfig,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        scaler_state_dict: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)
        self.config = config
        self.bart_config = bart_config
        self.train_stats = Statistics()
        # Automatic Mixed Precision
        self.scaler = None
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda")
            if scaler_state_dict is not None:
                self.scaler.load_state_dict(scaler_state_dict)
        # Wandb logger
        wandb.login(key=self.config.wandb_key)
        log_dir = (
            self.config.wandb_log_dir if self.config.wandb_log_dir else "wandb-logs"
        )
        make_dir(log_dir)
        self.run = wandb.init(
            project=self.config.wandb_project,
            config=self.bart_config.__dict__,
            dir=log_dir,
            notes=self.config.wandb_notes,
        )

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        # Set model to train mode
        self.model.train()

        global_step = self.config.initial_global_step

        # Traing loop
        for epoch in range(self.config.initial_epoch, self.config.num_epochs):
            # Empty cache
            torch.cuda.empty_cache()

            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Training epoch {epoch + 1}/{self.config.num_epochs}",
            )

            for batch in batch_iterator:
                # encoder_input (batch_size, seq_length)
                # decoder_input(batch_size, seq_length)
                # label (batch_size, seq_length)
                encoder_input = batch["src"].to(device=self.config.device)
                decoder_input = batch["tgt"].to(device=self.config.device)
                label = batch["label"].to(device=self.config.device)

                self.optimizer.zero_grad(set_to_none=True)

                src_attention_mask = (encoder_input != self.pad_token_id).to(
                    device=self.config.device,
                    dtype=torch.int64,
                )
                tgt_attention_mask = (decoder_input != self.pad_token_id).to(
                    device=self.config.device,
                    dtype=torch.int64,
                )

                with torch.autocast(
                    device_type=self.config.device.type,
                    dtype=torch.float16,
                    enabled=torch.cuda.is_available(),
                ):
                    # logits (batch_size, seq_length, vocab_size)
                    logits = self.model(
                        input_ids=encoder_input,
                        attention_mask=src_attention_mask,
                        decoder_input_ids=decoder_input,
                        decoder_attention_mask=tgt_attention_mask,
                    )

                    # Compute loss
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)), label.view(-1)
                    )

                if torch.cuda.is_available() and self.scaler is not None:
                    # Backpropagation
                    self.scaler.scale(loss).backward()
                    # Clip gradients norm
                    if (
                        self.config.max_grad_norm is not None
                        and self.config.max_grad_norm > 0
                    ):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            parameters=self.model.parameters(),
                            max_norm=self.config.max_grad_norm,
                        )
                    # Update weights and learning rate
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.train_stats.update(loss=loss.item())

                batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                global_step += 1

                self.run.log({"train/loss": loss.item()})

                if global_step % self.config.evaluating_steps == 0:
                    eval_stats = evaluate(
                        model=self.model,
                        val_dataloader=val_dataloader,
                        tokenizer=self.tokenizer,
                        criterion=self.criterion,
                        device=self.config.device,
                    )
                    rouge_score = compute_dataset_rouge(
                        model=self.model,
                        dataset=val_dataloader.dataset,
                        tokenizer=self.tokenizer,
                        seq_length=self.config.seq_length,
                        device=self.config.device,
                        beam_size=self.config.beam_size,
                        log_examples=self.config.log_examples,
                        logging_steps=self.config.logging_steps,
                        rouge_keys=self.config.rouge_keys,
                        use_stemmer=self.config.use_stemmer,
                        accumulate=self.config.accumulate,
                    )

                    self._update_metrics(
                        step=global_step,
                        eval_stats=eval_stats,
                        rouge_score=rouge_score,
                    )

            # Save model
            self._save_checkpoint(global_step=global_step, epoch=epoch)

        self.run.finish()

    def _update_metrics(
        self,
        step: int,
        eval_stats: Statistics,
        rouge_score: dict[str, float],
    ) -> None:
        train_stats_result = self.train_stats.compute()
        for key, value in train_stats_result.items():
            self.run.log({f"train/{key}": value})
        eval_stats_result = eval_stats.compute()
        for key, value in eval_stats_result.items():
            self.run.log({f"val/{key}": value})

        for key, value in rouge_score.items():
            self.run.log({f"val/{key}": value})

        # Reset statistics after writing to wandb
        self.train_stats.reset()

    def _save_checkpoint(self, global_step: int, epoch: int) -> None:
        model_filepath = get_weights_file_path(
            model_basedir=self.config.model_dir,
            model_basename=self.config.model_basename,
            epoch=epoch,
        )
        checkpoint_states = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.bart_config,
        }
        if torch.cuda.is_available() and self.scaler is not None:
            checkpoint_states["scaler_state_dict"] = self.scaler.state_dict()
        if self.lr_scheduler is not None:
            checkpoint_states["lr_scheduler_state_dict"] = (
                self.lr_scheduler.state_dict()
            )

        torch.save(checkpoint_states, model_filepath)
