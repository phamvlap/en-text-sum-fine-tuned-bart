import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from tqdm import tqdm
from dataclasses import dataclass
from typing import Literal

from bart.model import FinetuneBartModel, FinetuneBartModelConfig
from bart.constants import SpecialToken, RougeKey
from .utils.statistics import Statistics
from .utils.eval import evaluate
from .utils.rouge import compute_dataset_rouge
from .utils.path import get_weights_file_path


@dataclass
class TrainingConfig:
    device: torch.device
    seq_length: int
    num_epochs: int
    model_dir: str
    model_basename: str
    initial_epoch: int = 0
    initial_global_step: int = 0
    evaluating_steps: int = 1000
    beam_size: int | None = None
    log_examples: bool = True
    logging_steps: int = 100
    rouge_keys: list[str] | tuple[str] = (
        RougeKey.ROUGE_1,
        RougeKey.ROUGE_2,
        RougeKey.ROUGE_L,
    )
    use_stemmer: bool = True
    accumulate: Literal["best", "avg"] = "best"


class Trainer:
    def __init__(
        self,
        model: FinetuneBartModel,
        optimizer: optim.Optimizer,
        tokenizer: BartTokenizer,
        loss_fn: nn.CrossEntropyLoss,
        config: TrainingConfig,
        bart_config: FinetuneBartModelConfig,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
        writer: SummaryWriter | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)
        self.config = config
        self.bart_config = bart_config
        self.writer = writer
        self.train_stats = Statistics(
            vocab_size=tokenizer.vocab_size,
            ignore_index=self.pad_token_id,
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

                src_attention_mask = (encoder_input != self.pad_token_id).to(
                    device=self.config.device,
                    dtype=torch.int64,
                )
                tgt_attention_mask = (decoder_input != self.pad_token_id).to(
                    device=self.config.device,
                    dtype=torch.int64,
                )

                # logits (batch_size, seq_length, vocab_size)
                logits = self.model(
                    input_ids=encoder_input,
                    attention_mask=src_attention_mask,
                    decoder_input_ids=decoder_input,
                    decoder_attention_mask=tgt_attention_mask,
                )

                pred = torch.argmax(logits, dim=-1)

                # Compute loss
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
                batch_iterator.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                    }
                )

                # Backpropagation
                loss.backward()

                # Update weights and learning rate
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

                self.train_stats.update(
                    loss=loss.item(),
                    pred=pred.view(-1),
                    target=label.view(-1),
                )

                global_step += 1

                if self.writer is not None:
                    self.writer.add_scalar("train/train_loss", loss.item(), global_step)

                    if global_step % self.config.evaluating_steps == 0:
                        eval_stats = evaluate(
                            model=self.model,
                            val_dataloader=val_dataloader,
                            tokenizer=self.tokenizer,
                            loss_fn=self.loss_fn,
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

                        self._update_tensorboard(
                            step=global_step + 1,
                            eval_stats=eval_stats,
                            rouge_score=rouge_score,
                        )

                    self.writer.flush()

            # Save model
            self._save_model(global_step=global_step, epoch=epoch)

    def _update_tensorboard(
        self,
        step: int,
        eval_stats: Statistics,
        rouge_score: dict[str, float],
    ) -> None:
        if self.writer is None:
            return

        self.train_stats.write_to_tensorboard(
            writer=self.writer,
            mode="train",
            step=step,
        )
        eval_stats.write_to_tensorboard(
            writer=self.writer,
            mode="val",
            step=step,
        )

        for key, value in rouge_score.items():
            self.writer.add_scalar(f"val/{key}", value, step)

        # Reset statistics after writing to tensorboard
        self.train_stats.reset()

    def _save_model(self, global_step: int, epoch: int) -> None:
        model_filepath = get_weights_file_path(
            model_basedir=self.config.model_dir,
            model_basename=self.config.model_basename,
            epoch=epoch,
        )
        obj = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.bart_config,
        }
        if self.lr_scheduler is not None:
            obj["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(obj, model_filepath)
