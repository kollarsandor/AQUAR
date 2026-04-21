import sys
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import Normalizer
from pathlib import Path
import json
import re

GPT4_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def train_bpe_tokenizer(
    input_paths,
    vocab_size=32768,
    output_path="tokenizer.json",
    special_tokens=None,
    min_frequency=2,
    max_chars_per_doc=10000,
):
    if special_tokens is None:
        special_tokens = ["<|bos|>", "<|eos|>", "<|pad|>"]
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.normalizer = Normalizer.custom(0)
    tokenizer.pre_tokenizer = pre_tokenizers.Regex(GPT4_PAT)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    files = [str(p) for p in input_paths]
    tokenizer.train(files=files, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
        special_tokens=[
            ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
            ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
        ],
    )
    tokenizer.enable_padding(pad_token="<|pad|>", pad_id=tokenizer.token_to_id("<|pad|>"))
    tokenizer.enable_truncation(max_length=2048)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def load_tokenizer(path):
    tokenizer = Tokenizer.from_file(str(path))
    return tokenizer


def prepare_fineteweb_edu_for_tokenizer(
    data_path, output_path, max_chars_per_doc=10000, max_total_chars=2_000_000_000
):
    total_chars = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj["text"]
            except (json.JSONDecodeError, KeyError):
                text = line
            if len(text) > max_chars_per_doc:
                text = text[:max_chars_per_doc]
            if len(text) == 0:
                continue
            if total_chars + len(text) > max_total_chars:
                break
            fout.write(text + "\n")
            total_chars += len(text)
    return total_chars


def train_config_b_tokenizer(fineteweb_edu_path, output_path):
    prepared_path = str(Path(output_path).parent / "fineteweb_edu_prepared.txt")
    total = prepare_fineteweb_edu_for_tokenizer(
        fineteweb_edu_path, prepared_path, max_chars_per_doc=10000, max_total_chars=2_000_000_000
    )
    tokenizer = train_bpe_tokenizer(
        [prepared_path],
        vocab_size=32768,
        output_path=output_path,
        special_tokens=["<|bos|>", "<|eos|>", "<|pad|>"],
        min_frequency=2,
        max_chars_per_doc=10000,
    )
    return tokenizer


def train_config_a_tokenizer(text_corpus_path, output_path):
    tokenizer = train_bpe_tokenizer(
        [text_corpus_path],
        vocab_size=65536,
        output_path=output_path,
        special_tokens=["<|bos|>", "<|eos|>", "<|pad|>"],
        min_frequency=2,
        max_chars_per_doc=10000,
    )
    return tokenizer


def create_simple_tokenizer(vocab_size=65536, output_path="tokenizer_simple.json"):
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = Normalizer.custom(0)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.ByteLevel()
    special_tokens = ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
        special_tokens=[
            ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
            ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
        ],
    )
    tokenizer.enable_padding(pad_token="<|pad|>", pad_id=tokenizer.token_to_id("<|pad|>"))
    tokenizer.enable_truncation(max_length=2048)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer
