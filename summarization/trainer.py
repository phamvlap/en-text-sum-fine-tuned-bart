import json
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from transformers import BartTokenizer
from pathlib import Path
from tqdm import tqdm

from bart.model import FinetuneBartModel
from bart.constants import SpecialToken
from .utils.path import get_weights_file_path, join_path


class Trainer:
    def __init__(
        self,
        model: FinetuneBartModel,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None,
        tokenizer: BartTokenizer,
        loss_fn: nn.CrossEntropyLoss,
        initial_epoch: int,
        initial_global_step: int,
        num_epochs: int,
        args: dict,
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

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        global_step = self.initial_global_step

        train_losses = []
        val_losses = []

        # Traing loop
        for epoch in range(self.initial_epoch, self.num_epochs):
            torch.cuda.empty_cache()

            # Train
            self.model.train()
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Training epoch {epoch + 1}/{self.num_epochs}",
            )

            for batch in batch_iterator:
                src = batch["src"].to(device=self.device)
                tgt = batch["tgt"].to(device=self.device)
                label = batch["label"].to(device=self.device)

                src_attention_mask = (src != self.pad_token_id).to(
                    device=self.device,
                    dtype=torch.int64,
                )
                tgt_attention_mask = (tgt != self.pad_token_id).to(
                    device=self.device,
                    dtype=torch.int64,
                )

                logits = self.model(
                    input_ids=src,
                    attention_mask=src_attention_mask,
                    decoder_input_ids=tgt,
                    decoder_attention_mask=tgt_attention_mask,
                )

                # Compute loss
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
                train_losses.append(loss.item())
                batch_iterator.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                    }
                )

                loss.backward()

                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                global_step += 1

            with torch.no_grad():
                # Evaluate
                self.model.eval()
                batch_iterator = tqdm(
                    val_dataloader,
                    desc=f"Validating epoch {epoch + 1}/{self.num_epochs}",
                )

                for batch in batch_iterator:
                    src = batch["src"].to(device=self.device)
                    tgt = batch["tgt"].to(device=self.device)
                    label = batch["label"].to(device=self.device)

                    src_attention_mask = (src != self.pad_token_id).to(
                        device=self.device,
                        dtype=torch.int64,
                    )
                    tgt_attention_mask = (tgt != self.pad_token_id).to(
                        device=self.device,
                        dtype=torch.int64,
                    )

                    logits = self.model(
                        input_ids=src,
                        attention_mask=src_attention_mask,
                        decoder_input_ids=tgt,
                        decoder_attention_mask=tgt_attention_mask,
                    )

                    loss = self.loss_fn(
                        logits.view(-1, logits.size(-1)),
                        label.view(-1),
                    )
                    val_losses.append(loss.item())
                    batch_iterator.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                        }
                    )

            # Save model
            self._save_model(global_step=global_step, epoch=epoch)
            self._save_model_config(config_data=self.args["config_data"], epoch=epoch)

        # Statistic
        print("Train loss: {:.4f}".format(sum(train_losses) / len(train_losses)))
        print("Validation loss: {:.4f}".format(sum(val_losses) / len(val_losses)))

    def _save_model(
        self,
        global_step: int,
        epoch: int,
    ) -> None:
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
