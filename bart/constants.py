DEFAULT_TRAIN_VAL_TEST_RATIO = (0.75, 0.1, 0.15)
SETTING_CONFIG_FILE = "config/setting_config.yaml"
IGNORED_INDEX = -100


class SpecialToken:
    BOS = "<s>"  # begining of sentence
    EOS = "</s>"  # end of sentence
    PAD = "<pad>"
    UNK = "<unk>"
    MASK = "<mask>"
    BYTE_LEVEL_BPE_SUFFIX = "</eow>"  # end of word


class TokenizerType:
    BYTE_LEVEL_BPE = "byte_level_bpe"


class RougeKey:
    ROUGE_1 = "rouge1"
    ROUGE_2 = "rouge2"
    ROUGE_L = "rougeL"


class SentenceContractions:
    LOWERCASE = "lower"
    CONTRACTIONS = "contractions"
