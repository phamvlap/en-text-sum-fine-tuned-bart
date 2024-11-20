# Automatic English Text Summarization

## Introduction

This project is a straightforward implementation of a text summarization using an abstractive approach. The project utilizes a fine-tuned BART model to generate the concise summaries of the input text.

## Training tokenizer

The tokenizer is trained on the WikiHow dataset after it cleaned, and it built base on Byte Level Byte-Pair Encoding (Byte Level BPE) algorithm. The tokenizer is trained using the Hugging Face tokenizers library.

The following parameters are used to train the tokenizer:
  - `vocab_size`: 50,265
  - `min_frequence`: 2
  - `special_tokens`:`<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`

The tokenizer trained saved in the `tokenizer-bart` directory.

## Preprocessing dataset

The dataset used to train the model is the [Wikihow](https://arxiv.org/abs/1810.09305) dataset. The dataset after cleaning can found [here](https://huggingface.co/datasets/phamvlap/wikihow).

## Fine-tuning BART model

The model is fine-tuned using the parameters reported on table below.

| Parameter | Value |
| --- | --- |
| dropout | 0.3 |
| `attention_dropout` | 0.3 |
| `activation+dropout` | 0.3 |
| `encoder_layer_dropout` | 0.1 |
| `decoder_layer_dropout` | 0.1 |
| `beam_size` | 4 |
| `learning_rate` | 0.3 |
| `learning_rate_scheduler` | Noam Decay |
| `warmup_steps` | 400 |
| `optimizer` | AdamW |
| `source_sequence_length` | 512 |
| `target_sequence_length` | 512 |
| `epochs` | 3 |
| `train_batch_size` | 48 |
| `validation_batch_size` | 16 |

The fine-tuned model is released on the [HuggingFace Hub](https://huggingface.co/phamvlap/text-summarization-finetuned-bart).

## Evaluation

The model is evaluated using the ROUGE metric. The result evaluation on `test` dataset reported on the table below.

| Metric | Value |
| --- | --- |
| ROUGE-1 | 27.232 |
| ROUGE-2 | 7.60 |
| ROUGE-L | 22.94 |
