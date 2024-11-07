import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BartTokenizer
from tqdm import tqdm
from typing import Optional, Union
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

from bart.model import FineTunedBartForGeneration
from bart.constants import SpecialToken, SETTING_CONFIG_FILE
from .utils.eval import evaluate
from .utils.metrics import compute_rouge_bert_score
from .utils.mix import is_torch_cuda_available, make_dir, ensure_exist_path
from .utils.wb_logger import WandbLogger, VALID_PREFIX_KEY
from .trainer_utils import (
    TrainingArguments,
    determine_best_metric_value,
    sorted_checkpoints,
    rotate_checkpoints,
    get_checkpoint_path,
)
from .utils.meters import AverageMeter

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

HUGGINGFACE_BASE_URL = "https://huggingface.co"


class Trainer:
    def __init__(
        self,
        model: FineTunedBartForGeneration | DDP,
        optimizer: optim.Optimizer,
        tokenizer: BartTokenizer,
        criterion: nn.CrossEntropyLoss,
        args: TrainingArguments,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        scaler_state_dict: Optional[dict] = None,
        wb_logger: Optional[WandbLogger] = None,
        training_loss: Optional[AverageMeter] = None,
    ) -> None:
        self.model = model
        self.actual_model = self.get_actual_model(model)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)
        self.args = args
        self.bart_config = self.actual_model.get_config()
        self.wb_logger = wb_logger
        self.best_checkpoint = None
        self.best_score_value = (
            float("-inf") if self.args.greater_checking else float("inf")
        )
        self.hf_user = os.environ.get("HUGGINGFACE_USERNAME", None)
        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)

        # Automatic Mixed Precision
        # GradScaler: scales the loss and optimizer step
        self.scaler = None
        if self.args.f16_precision and is_torch_cuda_available():
            self.scaler = torch.amp.GradScaler("cuda")
            if scaler_state_dict is not None:
                self.scaler.load_state_dict(scaler_state_dict)

        # Set up training loss
        self.training_loss = training_loss
        if self.training_loss is None:
            self.training_loss = AverageMeter(
                name="training_loss",
                device=self.args.device,
            )

        make_dir(dir_path=self.args.checkpoint_dir)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        # Set model to train mode
        self.model.train()

        global_step = self.args.initial_global_step

        if self.args.max_train_steps > 0 and self.args.num_epochs > 0:
            logger.warning("max_train_steps is provided, it will override num_epochs")

        # Traing loop
        for epoch in range(self.args.initial_epoch, self.args.num_epochs):
            if (
                self.args.max_train_steps > 0
                and global_step > self.args.max_train_steps
            ):
                break

            # Empty cache
            torch.cuda.empty_cache()

            # Ensure shuffling work properly across multiple epochs
            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)

            if self.args.use_ddp:
                batch_iterator = tqdm(
                    train_dataloader,
                    desc=f"[GPU-{self.args.rank}] Training epoch {epoch + 1}/{self.args.num_epochs}",
                    disable=self.args.local_rank != 0,
                )
            else:
                batch_iterator = tqdm(
                    train_dataloader,
                    desc=f"Training epoch {epoch + 1}/{self.args.num_epochs}",
                )

            for batch in batch_iterator:
                # encoder_input (batch_size, src_seq_length)
                # decoder_input(batch_size, tgt_seq_length)
                # labels (batch_size, tgt_seq_length)
                if self.args.use_ddp:
                    encoder_input = batch["encoder_input"].to(self.args.local_rank)
                    decoder_input = batch["decoder_input"].to(self.args.local_rank)
                    labels = batch["labels"].to(self.args.local_rank)
                else:
                    encoder_input = batch["encoder_input"].to(self.args.device)
                    decoder_input = batch["decoder_input"].to(self.args.device)
                    labels = batch["labels"].to(self.args.device)

                self.optimizer.zero_grad(set_to_none=True)

                src_attention_mask = (encoder_input != self.pad_token_id).to(
                    dtype=torch.int32,
                )
                tgt_attention_mask = (decoder_input != self.pad_token_id).to(
                    dtype=torch.int32,
                )

                # Forward pass with autocast
                # Auto cast to float16 in certain parts of the model, while maintaining float32 precision in other parts
                with torch.autocast(
                    device_type=self.args.device.type,
                    dtype=torch.float16,
                    enabled=self.args.f16_precision and is_torch_cuda_available(),
                ):
                    # logits (batch_size, tgt_seq_length, vocab_size)
                    logits = self.model(
                        encoder_input_ids=encoder_input,
                        encoder_attn_mask=src_attention_mask,
                        decoder_input_ids=decoder_input,
                        decoder_attn_mask=tgt_attention_mask,
                    )

                    # Compute loss
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                    )

                if self.scaler is not None:
                    # Backpropagation
                    self.scaler.scale(loss).backward()
                    # Clip gradients norm
                    if (
                        self.args.max_grad_norm is not None
                        and self.args.max_grad_norm > 0
                    ):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            parameters=self.model.parameters(),
                            max_norm=self.args.max_grad_norm,
                        )
                    # Update weights and learning rate
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.lr_scheduler is not None:
                    if self.wb_logger is not None:
                        for idx, lr_value in enumerate(self.lr_scheduler.get_last_lr()):
                            self.wb_logger.log(
                                {f"learning_rate_{idx}": lr_value},
                                step=global_step,
                            )
                    self.lr_scheduler.step()

                if self.training_loss is not None:
                    self.training_loss.update(value=loss.item())

                batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                global_step += 1

                if self.wb_logger is not None:
                    self.wb_logger.log({"loss": loss.item()})

                if global_step % self.args.eval_every_n_steps == 0:
                    self._maybe_evaluate(
                        val_dataloader=val_dataloader,
                        step=global_step,
                    )

                # Save model
                if (
                    self._is_local_main_process()
                    and global_step % self.args.save_every_n_steps == 0
                ):
                    self._save_checkpoint(
                        global_step=global_step,
                        epoch=epoch,
                        step=global_step,
                    )

        if self.wb_logger is not None:
            self.wb_logger.finish()

    def _maybe_evaluate(self, val_dataloader: DataLoader, step: int) -> None:
        if self.args.use_ddp and self.training_loss is not None:
            self.training_loss.all_reduce()

        eval_result = evaluate(
            model=self.model,
            val_dataloader=val_dataloader,
            tokenizer=self.tokenizer,
            criterion=self.criterion,
            device=self.args.device,
            use_ddp=self.args.use_ddp,
            rank=self.args.rank,
            local_rank=self.args.local_rank,
            show_eval_progress=self.args.show_eval_progress,
        )
        scores = compute_rouge_bert_score(
            model=self.model,
            dataset=val_dataloader.dataset,
            tokenizer=self.tokenizer,
            seq_length=self.args.tgt_seq_length,
            device=self.args.device,
            beam_size=self.args.beam_size,
            topk=self.args.topk,
            eval_bert_score=self.args.eval_bert_score,
            rescale=self.args.rescale,
            log_examples=self.args.log_examples,
            logging_steps=self.args.logging_steps,
            rouge_keys=self.args.rouge_keys,
            use_stemmer=self.args.use_stemmer,
            truncation=self.args.truncation,
            accumulate=self.args.accumulate,
            use_ddp=self.args.use_ddp,
            rank=self.args.rank,
            local_rank=self.args.local_rank,
            world_size=self.args.world_size,
            max_steps=self.args.max_eval_steps,
        )

        self._log_evaluation_result(
            step=step,
            valid_result=eval_result,
            valid_scores=scores,
        )

        metric_scores = {
            **eval_result,
            **scores,
        }

        new_best_score_value, new_best_checkpoint = determine_best_metric_value(
            metric_scores=metric_scores,
            checked_metric=self.args.checked_metric,
            greater_checking=self.args.greater_checking,
            best_metric_value=self.best_score_value,
            output_dir=self.args.checkpoint_dir,
            checkpoint_prefix=self.args.model_basename,
            step=step,
        )

        if new_best_score_value is not None:
            self.best_score_value = new_best_score_value
        if new_best_checkpoint is not None:
            self.best_checkpoint = new_best_checkpoint

    def _log_evaluation_result(
        self,
        step: int,
        valid_result: dict[str, float],
        valid_scores: dict[str, float],
    ) -> None:
        if self.wb_logger is None:
            return

        logs: dict[str, float] = {}
        if self.training_loss is not None:
            logs["acc_loss"] = self.training_loss.average

        for key, value in valid_result.items():
            logs[VALID_PREFIX_KEY + f"acc_{key}"] = value

        for key, value in valid_scores.items():
            logs[VALID_PREFIX_KEY + key] = value

        self.wb_logger.log(logs=logs, step=step)

        # Reset metric tracker after writing to wandb
        if self.training_loss is not None:
            self.training_loss.reset()

    def _save_checkpoint(self, global_step: int, epoch: int, step: int) -> None:
        model_filepath = self._get_checkpoint_path(step=step)

        checkpoint_states = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.actual_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.bart_config,
        }
        if self.scaler is not None:
            checkpoint_states["scaler_state_dict"] = self.scaler.state_dict()
        if self.lr_scheduler is not None:
            checkpoint_states["lr_scheduler_state_dict"] = (
                self.lr_scheduler.state_dict()
            )

        torch.save(checkpoint_states, model_filepath)

        if self.args.push_to_hub and self._is_local_main_process():
            self._push_to_hub(step=step)

        # Get all checkpoints and sort them by ascending order
        checkpoint_sorted = sorted_checkpoints(
            output_dir=self.args.checkpoint_dir,
            checkpoint_prefix=self.args.model_basename,
            best_checkpoint=self.best_checkpoint,
        )

        # Remove old checkpoints
        rotate_checkpoints(
            checkpoints=checkpoint_sorted,
            max_saved_total=self.args.max_saved_checkpoints,
        )

    def _push_to_hub(self, step: int) -> None:
        if self.hf_user is None or self.hf_token is None:
            logger.warning(
                "Hugging Face username or token is not provided, skipped push to hub"
            )
            return

        login(token=self.hf_token)

        print("Pushing to hub...")
        hf_api = HfApi()

        hub_repo_id = f"{self.hf_user}/{self.args.hub_repo_name}"
        repo_url = f"{HUGGINGFACE_BASE_URL}/{self.hf_user}/{self.args.hub_repo_name}"

        # Create repo if it not exists
        if not hf_api.repo_exists(repo_id=hub_repo_id):
            repo_url = hf_api.create_repo(
                token=self.hf_token,
                repo_id=self.args.hub_repo_name,
                repo_type="model",
                private=True,
            )

        # Save config and tokenizer files
        tmp_dir = "archieved_stuff"

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        shutil.copytree("tokenizer-bart", tmp_dir)
        shutil.copy2(SETTING_CONFIG_FILE, tmp_dir)

        # Upload model to hub
        checkpoint_path = self._get_checkpoint_path(step=step)
        if self.best_checkpoint is not None and ensure_exist_path(checkpoint_path):
            checkpoint_name = checkpoint_path.split("/")[-1]
            hf_api.upload_file(
                path_or_fileobj=checkpoint_path,
                repo_id=hub_repo_id,
                commit_message=f"Upload model checkpoint at step {step}",
                path_in_repo=f"models/{checkpoint_name}",
            )

        # Upload config and tokenizer files to hub
        if step == self.args.save_every_n_steps and ensure_exist_path(tmp_dir):
            hf_api.upload_folder(
                repo_id=hub_repo_id,
                folder_path=tmp_dir,
                commit_message="Add config and tokenizer files",
            )

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        print(f"Pushed to {repo_url}")

    @staticmethod
    def get_actual_model(
        model: Union[nn.Module, FineTunedBartForGeneration, DDP],
    ) -> nn.Module:
        unwrapped_model = model
        if isinstance(model, DDP):
            unwrapped_model = model.module
        return unwrapped_model

    def _is_local_main_process(self) -> bool:
        if self.args.use_ddp:
            return self.args.local_rank == 0
        return True

    def _get_checkpoint_path(self, step: int) -> str:
        return get_checkpoint_path(
            checkpoint_dir=self.args.checkpoint_dir,
            model_basename=self.args.model_basename,
            step=step,
        )
