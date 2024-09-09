class SpecialToken:
    BOS = "<s>"  # begining of sentence
    EOS = "</s>"  # end of sentence
    PAD = "<pad>"
    UNK = "<unk>"
    MASK = "<mask>"


class TokenizerType:
    WORD_LEVEL = "word_level"
    BPE = "bpe"
    WORD_PIECE = "word_piece"
    BYTE_LEVEL_BPE = "byte_level_bpe"
