import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BartTokenizer
from tqdm import tqdm
from typing import Optional

from bart.model import FineTunedBartForGeneration
from bart.constants import SpecialToken
from .utils.eval import evaluate
from .utils.metrics import compute_rouge_bert_score
from .utils.path import get_weights_file_path
from .utils.mix import is_torch_cuda_available
from .utils.wb_logger import WandbLogger
from .trainer_utils import TrainingArguments
from .utils.metric_tracker import MetricTracker
from .utils.path import make_dir


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
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.PAD)
        self.args = args
        if isinstance(self.model, DDP):
            self.bart_config = self.model.module.get_config()
        else:
            self.bart_config = self.model.get_config()
        self.metric_tracker = MetricTracker()
        self.wb_logger = wb_logger

        # Automatic Mixed Precision
        # GradScaler: scales the loss and optimizer step
        self.scaler = None
        if self.args.f16_precision and is_torch_cuda_available():
            self.scaler = torch.amp.GradScaler("cuda")
            if scaler_state_dict is not None:
                self.scaler.load_state_dict(scaler_state_dict)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        # Set model to train mode
        self.model.train()

        global_step = self.args.initial_global_step

        # Traing loop
        for epoch in range(self.args.initial_epoch, self.args.num_epochs):
            # Empty cache
            torch.cuda.empty_cache()

            # Ensure shuffling work properly across multiple epochs
            if isinstance(train_dataloader.sampler, DistributedSampler):
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
                # encoder_input (batch_size, seq_length)
                # decoder_input(batch_size, seq_length)
                # labels (batch_size, seq_length)
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
                    dtype=torch.int64,
                )
                tgt_attention_mask = (decoder_input != self.pad_token_id).to(
                    dtype=torch.int64,
                )

                # Forward pass with autocast
                # Auto cast to float16 in certain parts of the model, while maintaining float32 precision in other parts
                with torch.autocast(
                    device_type=self.args.device.type,
                    dtype=torch.float16,
                    enabled=self.args.f16_precision and is_torch_cuda_available(),
                ):
                    # logits (batch_size, seq_length, vocab_size)
                    logits = self.model(
                        encoder_input_ids=encoder_input,
                        encoder_attn_mask=src_attention_mask,
                        decoder_input_ids=decoder_input,
                        decoder_attn_mask=tgt_attention_mask,
                    )

                    # Compute loss
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )

                if is_torch_cuda_available() and self.scaler is not None:
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

                self.metric_tracker.update(loss=loss.item())

                batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                global_step += 1

                if self.wb_logger is not None:
                    self.wb_logger.log({"loss": loss.item()})

                if global_step % self.args.eval_every_n_steps == 0:
                    eval_metric_tracker = evaluate(
                        model=self.model,
                        val_dataloader=val_dataloader,
                        tokenizer=self.tokenizer,
                        criterion=self.criterion,
                        device=self.args.device,
                        use_ddp=self.args.use_ddp,
                        rank=self.args.rank,
                        local_rank=self.args.local_rank,
                    )
                    scores = compute_rouge_bert_score(
                        model=self.model,
                        dataset=val_dataloader.dataset,
                        tokenizer=self.tokenizer,
                        seq_length=self.args.seq_length,
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

                    self._log_to_hub(
                        step=global_step,
                        eval_metric_tracker=eval_metric_tracker,
                        scores=scores,
                    )

                # Save model
                if (
                    self.args.local_rank == 0
                    and global_step % self.args.save_every_n_steps == 0
                ):
                    self._save_checkpoint(global_step=global_step, epoch=epoch)

        if self.wb_logger is not None:
            self.wb_logger.finish()

    def _log_to_hub(
        self,
        step: int,
        eval_metric_tracker: MetricTracker,
        scores: dict[str, float],
    ) -> None:
        metric_train_result = self.metric_tracker.compute()
        metric_eval_result = eval_metric_tracker.compute()

        if self.wb_logger is not None:
            logs = {
                **metric_train_result,
            }

            for key, value in metric_eval_result.items():
                logs[f"eval_{key}"] = value
            for key, value in scores.items():
                logs[f"eval_{key}"] = value

            self.wb_logger.log(logs=logs, step=step)

            # Reset metric tracker after writing to wandb
            self.metric_tracker.reset()

    def _save_checkpoint(self, global_step: int, epoch: int) -> None:
        make_dir(dir_path=self.args.model_dir)
        model_filepath = get_weights_file_path(
            model_basedir=self.args.model_dir,
            model_basename=self.args.model_basename,
            epoch=epoch,
        )
        model_state_dict = (
            self.model.module.state_dict()
            if self.args.use_ddp
            else self.model.state_dict()
        )
        checkpoint_states = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model_state_dict,
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
