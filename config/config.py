import torch

from bart.constants import SpecialToken, RougeKey


def get_config() -> dict:
    config = {}

    # General configs
    config["seed"] = 42
    config["base_dir"] = "trained"

    # Model configs
    config["model_dir"] = f"{config['base_dir']}/models"
    config["model_basename"] = "bart_model_"
    config["model_config_file"] = "model_config_{0}.json"

    # Dataset configs
    config["dataset_dir"] = "dataset"
    config["data_files_path"] = {
        "train": f"{config['dataset_dir']}/train.csv",
        "val": f"{config['dataset_dir']}/val.csv",
        "test": f"{config['dataset_dir']}/test.csv",
    }
    config["raw_dataset_dir"] = "raw/dataset/dir"
    config["raw_data_files_path"] = {
        "train": f"{config['raw_dataset_dir']}/train.csv",
        "val": f"{config['raw_dataset_dir']}/val.csv",
        "test": f"{config['raw_dataset_dir']}/test.csv",
    }
    config["text_src"] = "document"
    config["text_tgt"] = "summary"

    # Tokenizer configs
    config["tokenizer_train_ds_path"] = "path/to/train/tokenizer/file.csv"
    config["tokenizer_bart_dir"] = "tokenizer-bart"
    config["is_train_tokenizer"] = False
    config["special_tokens"] = [
        SpecialToken.BOS,
        SpecialToken.EOS,
        SpecialToken.PAD,
        SpecialToken.MASK,
        SpecialToken.UNK,
    ]
    config["shared_vocab"] = True
    config["vocab_size"] = 50000
    config["min_freq"] = 2
    config["model_type"] = "byte_level_bpe"
    config["show_progress"] = True

    # Dataloader configs
    config["batch_size_train"] = 32
    config["batch_size_val"] = 1
    config["batch_size_test"] = 1

    # Adam optimizer configs
    config["lr"] = 0.5
    config["betas"] = (0.9, 0.98)
    config["eps"] = 1e-9  # = 10 ** -9

    # Learning rate scheduler configs
    config["lr_scheduler"] = "noam"

    # NoamLR scheduler configs
    config["warmup_steps"] = 4000

    # Loss function configs
    config["label_smoothing"] = 0.1

    # Training configs
    config["epochs"] = 10
    config["preload"] = "latest"
    config["evaluating_steps"] = 1000
    config["log_dir"] = "logs"

    # BART configs
    config["seq_length"] = 512  # max length of input sequence
    config["d_model"] = 1024  # dimension of hidden layers
    config["encoder_layers"] = 6  # number of encoder layers
    config["decoder_layers"] = 6  # number of decoder layers
    config["encoder_attention_heads"] = 8  # number of encoder attention heads
    config["decoder_attention_heads"] = 8  # number of decoder attention heads
    config["encoder_ffn_dim"] = 2048  # dimension of feedforward network in encoder
    config["decoder_ffn_dim"] = 2048  # dimension of feedforward network in decoder
    config["activation_function"] = "gelu"  # 'gelu', 'relu', 'silu' or 'gelu_new'
    config["dropout"] = 0.1  # dropout rate for individual layer
    config["attention_dropout"] = 0.1  # dropout rate for attention layer
    config["activation_dropout"] = 0.1  # dropout rate after activation function
    config["classifier_dropout"] = 0.1  # dropout rate for classifier
    config["max_position_embeddings"] = config["seq_length"]
    config["init_std"] = 0.02  # standard deviation for initializing weight parameters
    config["encoder_layerdrop"] = 0.2  # layer dropout rate for entire encoder layers
    config["decoder_layerdrop"] = 0.2  # layer dropout rate for entire decoder layers
    config["scale_embedding"] = True  # scale embedding by sqrt(d_model)
    config["num_beams"] = 4  # beam search size

    # Rouge Score configs
    config["rouge_keys"] = [
        RougeKey.ROUGE_1,
        RougeKey.ROUGE_2,
        RougeKey.ROUGE_L,
    ]
    config["use_stemmer"] = True
    config["accumulate"] = "best"  # 'best' | 'avg'

    # Compute Rouge Score configs
    config["log_examples"] = True
    config["logging_steps"] = 1000

    # Beam search configs
    config["beam_size"] = 3

    # Statistics result configs
    config["statistic_dir"] = f"{config['base_dir']}/statistics"

    # Device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config
