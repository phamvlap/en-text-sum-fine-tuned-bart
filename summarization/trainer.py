import json
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from pathlib import Path
from tqdm import tqdm

from bart.model import FinetuneBartModel
from bart.constants import SpecialToken
from .utils.statistics import Statistics
from .utils.eval import evaluate
from .utils.path import get_weights_file_path, join_path


class Trainer:
    def __init__(
        self,
        model: FinetuneBartModel,
        optimizer: optim.Optimizer,
        tokenizer: BartTokenizer,
        loss_fn: nn.CrossEntropyLoss,
        initial_epoch: int,
        initial_global_step: int,
        num_epochs: int,
        args: dict,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
        writer: SummaryWriter | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)
        self.loss_fn = loss_fn
        self.initial_epoch = initial_epoch
        self.initial_global_step = initial_global_step
        self.num_epochs = num_epochs
        self.device = args["device"]
        self.args = args
        self.writer = writer
        self.train_stats = Statistics(
            vocab_size=tokenizer.vocab_size,
            ignore_index=self.pad_token_id,
        )

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        # Set model to train mode
        self.model.train()

        global_step = self.initial_global_step

        # Traing loop
        for epoch in range(self.initial_epoch, self.num_epochs):
            # Empty cache
            torch.cuda.empty_cache()

            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Training epoch {epoch + 1}/{self.num_epochs}",
            )

            for batch in batch_iterator:
                # encoder_input (batch_size, seq_length)
                # decoder_input(batch_size, seq_length)
                # label (batch_size, seq_length)
                encoder_input = batch["src"].to(device=self.device)
                decoder_input = batch["tgt"].to(device=self.device)
                label = batch["label"].to(device=self.device)

                src_attention_mask = (encoder_input != self.pad_token_id).to(
                    device=self.device,
                    dtype=torch.int64,
                )
                tgt_attention_mask = (decoder_input != self.pad_token_id).to(
                    device=self.device,
                    dtype=torch.int64,
                )

                # decoder_output (batch_size, seq_length, d_model)
                decoder_output = self.model(
                    input_ids=encoder_input,
                    attention_mask=src_attention_mask,
                    decoder_input_ids=decoder_input,
                    decoder_attention_mask=tgt_attention_mask,
                ).last_hidden_state

                # logits (batch_size, seq_length, vocab_size)
                # pred (batch_size, seq_length)
                logits = self.model.out(decoder_output)
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

                    if (global_step + 1) % self.args["evaluating_steps"] == 0:
                        eval_stats = evaluate(
                            model=self.model,
                            val_dataloade=val_dataloader,
                            tokenizer=self.tokenizer,
                            loss_fn=self.loss_fn,
                            device=self.device,
                        )
                        self._update_tensorboard(
                            step=global_step + 1,
                            eval_stats=eval_stats,
                        )

                    self.writer.flush()

            # Save model
            self._save_model(global_step=global_step, epoch=epoch)
            self._save_model_config(config_data=self.args["config_data"], epoch=epoch)

    def _update_tensorboard(self, step: int, eval_stats: Statistics) -> None:
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

        # Reset statistics after writing to tensorboard
        self.train_stats.reset()

    def _save_model(self, global_step: int, epoch: int) -> None:
        model_filepath = get_weights_file_path(
            model_basedir=self.args["model_dir"],
            model_basename=self.args["model_basename"],
            epoch=epoch,
        )
        obj = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": global_step,
        }
        if self.lr_scheduler is not None:
            obj["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(obj, model_filepath)

    def _save_model_config(self, config_data: dict, epoch: int) -> None:
        filepath = join_path(
            base_dir=self.args["model_dir"],
            sub_path=self.args["model_config_file"].format(epoch),
        )
        for key, value in config_data.items():
            if isinstance(value, Path):
                config_data[key] = str(value)
        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=4)
