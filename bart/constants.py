class SpecialToken:
    BOS = "<s>"  # begining of sentence
    EOS = "</s>"  # end of sentence
    PAD = "<pad>"
    UNK = "<unk>"
    MASK = "<mask>"


class TokenId:
    BOS = 0
    EOS = 1
    PAD = 2
    UNK = 3
    MASK = 4


class TokenizerType:
    WORD_LEVEL = "word_level"
    BPE = "bpe"
    WORD_PIECE = "word_piece"
