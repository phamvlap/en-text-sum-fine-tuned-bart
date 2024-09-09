import torch
from pathlib import Path

from bart.constants import TokenId, SpecialToken


def get_config() -> dict:
    config = {}

    # General configs
    config["seed"] = 42
    config["max_length"] = 512
    config["model_dir"] = "models"
    config["model_basename"] = "bart_model_"

    # Dataset configs
    config["dataset_dir"] = "data"
    config["train_ds_path"] = Path(config["dataset_dir"], "train.csv")
    config["val_ds_path"] = Path(config["dataset_dir"], "val.csv")
    config["test_ds_path"] = Path(config["dataset_dir"], "test.csv")
    config["text_src"] = "document"
    config["text_tgt"] = "summary"

    # Tokenizer configs
    config["tokenizer_dir"] = "tokenizer"
    config["tokenizer_bart_dir"] = "tokenizer-bart"
    config["special_tokens"] = [
        SpecialToken.BOS,
        SpecialToken.EOS,
        SpecialToken.PAD,
        SpecialToken.MASK,
        SpecialToken.UNK,
    ]
    config["vocab_size_src"] = 30000
    config["vocab_size_tgt"] = 30000
    config["min_freq"] = 2
    config["model_type"] = "bpe"
    config["tokenizer_src_path"] = Path(config["tokenizer_dir"], "tokenizer_src.json")
    config["tokenizer_tgt_path"] = Path(config["tokenizer_dir"], "tokenizer_tgt.json")

    # Dataloader configs
    config["batch_size_train"] = 16
    config["batch_size_val"] = 16
    config["batch_size_test"] = 1

    # Learning configs
    config["lr"] = 0.5
    config["eps"] = 1e-9

    # Loss function configs
    config["label_smoothing"] = 0.1

    # Training configs
    config["epochs"] = 10
    config["preload"] = "latest"

    # BART configs
    config["d_model"] = 768
    config["encoder_layers"] = 6
    config["decoder_layers"] = 6
    config["encoder_attention_heads"] = 8
    config["decoder_attention_heads"] = 8
    config["encoder_ffn_dim"] = 2048
    config["decoder_ffn_dim"] = 2048
    config["activation_function"] = "relu"
    config["dropout"] = 0.1
    config["max_position_embeddings"] = config["max_length"]
    config["init_std"] = 0.02
    config["scale_embedding"] = True
    config["num_beams"] = 4
    config["bos_token_id"] = TokenId.BOS
    config["pad_token_id"] = TokenId.PAD
    config["eos_token_id"] = TokenId.EOS

    # Device
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return config
