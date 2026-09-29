"""Byte-level BPE tokenizer trained from scratch on our own corpus.

Contract (docs/PLAN.md, Stage 1c):
- byte-level pre-tokenizer (GPT-2-style split regex, no prefix space), no normalizer, and a
  matching byte-level decoder, so decode(encode(s)) == s for any string;
- the trainer starts from the full 256-symbol byte alphabet, so no UNK token is needed;
- exactly one special token, EOT_TOKEN, which marks document boundaries. There is no BOS, PAD
  or UNK, and no post-processor inserts tokens;
- the total vocabulary, special token included, must equal the model's vocab_size exactly.

Raw text that literally contains the EOT spelling must never produce the EOT id. The
`encode_special_tokens` flag enforces this, but it is not persisted in tokenizer.json, so
tokenizers must be obtained via `train_tokenizer` or `load_tokenizer`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT_TOKEN = "<|endoftext|>"
SPECIAL_TOKENS = (EOT_TOKEN,)


def _enforce_raw_text_encoding(tok: Tokenizer) -> Tokenizer:
    tok.encode_special_tokens = True
    return tok


def train_tokenizer(
    texts: Iterable[str], vocab_size: int, min_frequency: int = 2, length: int | None = None
) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    actual = tok.get_vocab_size(with_added_tokens=True)
    if actual != vocab_size:
        # The trainer silently stops early when the corpus has too few mergeable pairs.
        raise ValueError(f"trained vocab has {actual} entries, required exactly {vocab_size}")
    return _enforce_raw_text_encoding(tok)


def load_tokenizer(path: str | Path) -> Tokenizer:
    return _enforce_raw_text_encoding(Tokenizer.from_file(str(path)))


def eot_id(tok: Tokenizer) -> int:
    token_id = tok.token_to_id(EOT_TOKEN)
    if token_id is None:
        raise ValueError(f"{EOT_TOKEN} is not in the vocabulary")
    return token_id


def encode_text(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def encode_documents(tok: Tokenizer, texts: Sequence[str]) -> list[list[int]]:
    """Encode documents for the training stream: each gets exactly one EOT appended."""
    eot = eot_id(tok)
    return [encoding.ids + [eot] for encoding in tok.encode_batch(list(texts))]


def decode(tok: Tokenizer, ids: Sequence[int]) -> str:
    return tok.decode(list(ids), skip_special_tokens=False)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corpus_token_stats(tok: Tokenizer, texts: Iterable[str], batch_size: int = 1000) -> dict[str, float]:
    """Token counts under our tokenizer. `tokens` excludes the per-document EOT."""
    documents = characters = utf8_bytes = tokens = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal tokens
        tokens += sum(len(e.ids) for e in tok.encode_batch(batch))
        batch.clear()

    for text in texts:
        documents += 1
        characters += len(text)
        utf8_bytes += len(text.encode("utf-8"))
        batch.append(text)
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()
    return {
        "documents": documents,
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "tokens": tokens,
        "tokens_with_eot": tokens + documents,
        "avg_tokens_per_document": tokens / documents if documents else 0.0,
        "bytes_per_token": utf8_bytes / tokens if tokens else 0.0,
        "characters_per_token": characters / tokens if tokens else 0.0,
    }
