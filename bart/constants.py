class SpecialToken:
    BOS = "<s>"  # begining of sentence
    EOS = "</s>"  # end of sentence
    PAD = "<pad>"
    UNK = "<unk>"
    MASK = "<mask>"
    BPE_SUFFIX = "</w>"
    BYTE_LEVEL_BPE_SUFFIX = "</w>"


class TokenizerType:
    BPE = "bpe"
    BYTE_LEVEL_BPE = "byte_level_bpe"


class RougeKey:
    ROUGE_1 = "rouge1"
    ROUGE_2 = "rouge2"
    ROUGE_L = "rougeL"
