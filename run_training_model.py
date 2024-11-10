from typing import Any

from summarization.utils.mix import update_setting_config
from summarization.train import main as run_training_model


def main(config: dict):
    run_training_model(config)


def parse_args() -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Train fine-tuned BART model")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        default="checkpoints",
        help="directory to save checkpoint (default: checkpoints)",
    )
    parser.add_argument(
        "--model_basename",
        type=str,
        required=False,
        default="bart_model",
        help="basename for name of the trained model (default: bart_model)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        default="facebook/bart-base",
        help="name or path of model (default: facebook/bart-base)",
    )
    parser.add_argument(
        "--attach_text",
        action="store_true",
        dest="attach_text",
        help="attach text when get item from dataset (default: False)",
    )
    parser.add_argument(
        "--batch_size_train",
        type=int,
        required=False,
        default=32,
        help="batch size of dataset for training (default: 32)",
    )
    parser.add_argument(
        "--batch_size_val",
        type=int,
        required=False,
        default=16,
        help="batch size of dataset for validating (default: 16)",
    )
    parser.add_argument(
        "--shuffle_dataloader",
        action="store_true",
        dest="shuffle_dataloader",
        help="shuffle dataset when get item from dataset (default: False)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=2,
        help="number of workers for dataloader (default: 2)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default="adamw",
        help="type of optimizer for training (adam, adamw) (default: adamw)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=0.3,
        help="learning rate for optimizer (default: 0.3)",
    )
    parser.add_argument(
        "--betas",
        type=str,
        required=False,
        default="0.9,0.98",
        help="betas for optimizer (default: 0.9,0.98)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        required=False,
        default=1e-5,
        help="epsilon for optimizer (default: 1e-5)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=False,
        default=0.0,
        help="weight decay for optimizer (default: 0.0)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        required=False,
        default="noam",
        help="type of learning rate for training (noam, cosine) (default: noam)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        required=False,
        default=4000,
        help="number of warmup steps for learning rate (default: 4000)",
    )
    parser.add_argument(
        "--T_0",
        type=int,
        required=False,
        default=10,
        help="T_0 for cosine annealing learning rate scheduler (default: 10)",
    )
    parser.add_argument(
        "--T_mult",
        type=int,
        required=False,
        default=2,
        help="T_mult for cosine annealing learning rate scheduler (default: 2)",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        required=False,
        default=1e-5,
        help="eta_min for cosine annealing learning rate scheduler (default: 1e-5)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        required=False,
        default=0.1,
        help="label smoothing for loss function (default: 0.1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=10,
        help="number of epochs for training (default: 10)",
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        required=False,
        default=1000,
        help="number of steps to evaluate model (default: 1000)",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        required=False,
        default=5000,
        help="number of steps to save checkpoint (default: 5000)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        required=False,
        default=1.0,
        help="maximum gradient for clipping (default: 1.0)",
    )
    parser.add_argument(
        "--f16_precision",
        action="store_true",
        dest="f16_precision",
        help="use float 16 precision for training (default: False)",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        required=False,
        default=100,
        help="maximum examples for each evaluating iterator (default: 100)",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        required=False,
        default=-1,
        help="maximum steps for each training iterator (default: -1)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        required=False,
        default=2,
        help="number of steps to accumulate gradients (default: 2)",
    )
    parser.add_argument(
        "--max_saved_checkpoints",
        type=int,
        required=False,
        default=2,
        help="maximum saved checkpoints (default: 2)",
    )
    parser.add_argument(
        "--show_eval_progress",
        action="store_true",
        dest="show_eval_progress",
        help="show progress during evaluating (default: False)",
    )
    parser.add_argument(
        "--src_seq_length",
        type=int,
        required=True,
        default=768,
        help="maximum length of input sequence (default: 768)",
    )
    parser.add_argument(
        "--tgt_seq_length",
        type=int,
        required=True,
        default=256,
        help="maximum length of output sequence (default: 256)",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        required=False,
        default=768,
        help="dimension of model (default: 768)",
    )
    parser.add_argument(
        "--use_stemmer",
        action="store_true",
        dest="use_stemmer",
        help="use stemmer for computing ROUGE scores (default: False)",
    )
    parser.add_argument(
        "--accumulate",
        type=str,
        required=False,
        default="best",
        help="accumulate type for training (best, avg) (default: best)",
    )
    parser.add_argument(
        "--eval_bert_score",
        action="store_true",
        dest="eval_bert_score",
        help="evaluate Bert score (default: False)",
    )
    parser.add_argument(
        "--rescale",
        action="store_true",
        dest="rescale",
        help="rescale when compute Bert score (default: False)",
    )
    parser.add_argument(
        "--log_examples",
        action="store_true",
        dest="log_examples",
        help="log examples during evaluating (default: False)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        required=False,
        default=1000,
        help="number of steps to log examples (default: 1000)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        required=False,
        default=3,
        help="beam size for decoding (default: 3)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=False,
        default=2,
        help="top k result returned for decoding (default: 2)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        dest="resume_from_checkpoint",
        help="resume training from checkpoint (default: Fasle)",
    )
    parser.add_argument(
        "--logging_wandb",
        action="store_true",
        dest="is_logging_wandb",
        help="whether log result to wandb (default: False)",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        required=False,
        default="en-text-sum-fine-tuned-bart",
        help="name of wandb project (default: en-text-sum-fine-tuned-bart)",
    )
    parser.add_argument(
        "--wandb_log_dir",
        type=str,
        required=False,
        default="wandb-logs",
        help="directory to save wandb logs (default: wandb-logs)",
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        required=False,
        help="key of wandb log (default: None)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        dest="push_to_hub",
        help="push model to huggingface hub (default: False)",
    )
    parser.add_argument(
        "--hub_repo_name",
        type=str,
        required=False,
        default="en-text-sum-fine-tuned-bart",
        help="name of hub repository (default: en-text-sum-fine-tuned-bart)",
    )

    parser.set_defaults(
        attach_text=False,
        shuffle_dataloader=False,
        f16_precision=False,
        show_eval_progress=False,
        use_stemmer=False,
        eval_bert_score=False,
        rescale=False,
        resume_from_checkpoint=False,
        log_examples=False,
        is_logging_wandb=False,
        push_to_hub=False,
    )

    args = parser.parse_args()
    args = vars(args)
    args["betas"] = list(map(float, args["betas"].split(",")))

    return args


if __name__ == "__main__":
    args = parse_args()
    config = update_setting_config(new_config=args, excepted_keys=["wandb_key"])

    main(config)
