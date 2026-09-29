"""Tokenized training storage (flat uint16 files) and a single-process batcher.

Storage: every document is encoded with our tokenizer and followed by exactly one EOT id (0);
documents are concatenated into one flat uint16 file per split. No padding. Every stored id is
checked to be in [0, vocab_size) when written AND when opened, which is why the training
forward pass can skip the per-step (GPU-synchronizing) target range check.

Batches: a window of seq_len + 1 contiguous tokens gives x = window[:-1], y = window[1:]
(targets pre-shifted exactly once, here; the model never shifts). Windows may cross document
boundaries, which the EOT token marks explicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tokenizers import Tokenizer

from gibc.tokenizer import encode_documents, eot_id

TOKEN_DTYPE = np.uint16
SPLITS = ("train", "val")


def write_token_file(tok: Tokenizer, texts: Iterable[str], out_path: Path, vocab_size: int,
                     batch_docs: int = 1000) -> dict[str, Any]:
    """Encode documents (one EOT after each) into a flat uint16 file. Returns counts and sha256."""
    if vocab_size > np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError(f"vocab_size {vocab_size} does not fit {np.dtype(TOKEN_DTYPE).name}")
    eot = eot_id(tok)
    digest = hashlib.sha256()
    documents = tokens = eot_count = 0
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with tmp_path.open("wb") as fh:
        batch: list[str] = []

        def flush() -> None:
            nonlocal documents, tokens, eot_count
            encoded = encode_documents(tok, batch)
            ids = np.fromiter((i for doc in encoded for i in doc), dtype=np.int64)
            if ids.size and (ids.min() < 0 or ids.max() >= vocab_size):
                raise ValueError(f"token id outside [0, {vocab_size}) in documents {documents}..")
            per_doc_eot = [doc.count(eot) for doc in encoded]
            if any(n != 1 or doc[-1] != eot for n, doc in zip(per_doc_eot, encoded)):
                raise ValueError("a document does not end with exactly one EOT")
            data = ids.astype(TOKEN_DTYPE).tobytes()
            fh.write(data)
            digest.update(data)
            documents += len(batch)
            tokens += ids.size
            eot_count += len(batch)
            batch.clear()

        for text in texts:
            batch.append(text)
            if len(batch) >= batch_docs:
                flush()
        if batch:
            flush()
    tmp_path.replace(out_path)
    return {"documents": documents, "tokens": tokens, "eot_tokens": eot_count,
            "bytes": out_path.stat().st_size, "sha256": digest.hexdigest()}


def prepare_token_splits(data_dir: Path, out_dir: Path, tokenizer_path: Path, vocab_size: int,
                         command: str) -> dict[str, Any]:
    """Tokenize an acquired sample's {train,val}.jsonl into out_dir/{train,val}.bin plus meta.json."""
    from gibc.acquire import read_jsonl_documents
    from gibc.tokenizer import file_sha256, load_tokenizer

    tok = load_tokenizer(tokenizer_path)
    if tok.get_vocab_size(with_added_tokens=True) != vocab_size:
        raise ValueError("tokenizer vocab size does not match the model config")
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in SPLITS:
        docs = (doc["text"] for doc in read_jsonl_documents(data_dir / f"{split}.jsonl"))
        splits[split] = write_token_file(tok, docs, out_dir / f"{split}.bin", vocab_size)
        open_token_file(out_dir / f"{split}.bin", vocab_size, splits[split]["tokens"])  # re-check from disk
    meta = {
        "format": "gibc.tokens.v1",
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "vocab_size": vocab_size,
        "eot_id": eot_id(tok),
        "eot_policy": "exactly one EOT appended after every document; documents concatenated; no BOS, no padding",
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "source_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "source_data_dir": data_dir.name,
        "splits": splits,
        "command": command,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta


def open_token_file(path: Path, vocab_size: int, expected_tokens: int | None = None) -> np.memmap:
    """Memory-map a token file read-only, re-checking length and id range once at open time."""
    tokens = np.memmap(path, dtype=TOKEN_DTYPE, mode="r")
    if expected_tokens is not None and tokens.size != expected_tokens:
        raise ValueError(f"{path}: {tokens.size} tokens, metadata says {expected_tokens}")
    if tokens.size and int(tokens.max()) >= vocab_size:  # uint16 cannot be negative
        raise ValueError(f"{path}: contains token id >= vocab_size {vocab_size}")
    return tokens


def load_token_meta(tokens_dir: Path) -> dict[str, Any]:
    return json.loads((tokens_dir / "meta.json").read_text(encoding="utf-8"))


def _windows_to_tensors(tokens: np.ndarray, starts: np.ndarray, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = np.stack([np.asarray(tokens[s : s + seq_len + 1], dtype=np.int64) for s in starts])
    chunk = torch.from_numpy(chunks)
    return chunk[:, :-1], chunk[:, 1:]


class Batcher:
    """Random contiguous windows with a dedicated, checkpointable RNG (numpy PCG64)."""

    def __init__(self, tokens: np.ndarray, seq_len: int, batch_size: int, seed: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_start = tokens.size - (seq_len + 1)  # inclusive
        if self.max_start < 0:
            raise ValueError(f"need at least {seq_len + 1} tokens, have {tokens.size}")
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """x, y: int64 CPU tensors of shape [batch_size, seq_len], with y == x shifted by one."""
        starts = self.rng.integers(0, self.max_start, size=self.batch_size, endpoint=True)
        return _windows_to_tensors(self.tokens, starts, self.seq_len)

    def state_dict(self) -> dict[str, Any]:
        return {"bit_generator": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["bit_generator"]


def file_sha256_streaming(path: Path, chunk_bytes: int = 1 << 24) -> str:
    """SHA-256 of a file read in 16 MiB chunks (never loads a multi-GB token file into RAM)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_token_files(tokens_dir: Path, meta: dict[str, Any]) -> None:
    """Recompute the train/val token-file hashes and compare with meta.json. Raises on any mismatch."""
    for split in SPLITS:
        path = tokens_dir / f"{split}.bin"
        actual = file_sha256_streaming(path)
        expected = meta["splits"][split]["sha256"]
        if actual != expected:
            raise ValueError(f"{path}: sha256 {actual} does not match meta.json {expected}; refusing to use it")


class SamplerExhausted(RuntimeError):
    pass


SHUFFLED_SAMPLER_VERSION = "shuffled_windows_v1"


class ShuffledWindowSampler:
    """One pass, without replacement, over the non-overlapping windows of a token stream.

    With N tokens and W = (N - 1) // seq_len windows, window i is tokens[seq_len*i : seq_len*i + seq_len + 1],
    so x = w[:-1] and y = w[1:]. Neighbouring windows share exactly one token (the last target of
    window i is the first context token of window i+1), every token position 1 .. W*seq_len is a
    prediction target exactly once, position 0 is context only, and the final (N - 1) - W*seq_len
    targets are dropped. Windows are visited in the order numpy PCG64(seed).permutation(W); the
    position in that order (cursor) is the only mutable state, so resume is exact. Running past the
    last window raises SamplerExhausted (no silent wraparound).
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, batch_size: int, seed: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.window_count = (tokens.size - 1) // seq_len
        if self.window_count < 1:
            raise ValueError(f"need at least {seq_len + 1} tokens, have {tokens.size}")
        self.dropped_targets = (tokens.size - 1) - self.window_count * seq_len
        self.order = np.random.Generator(np.random.PCG64(seed)).permutation(self.window_count)
        self.cursor = 0

    @property
    def remaining_windows(self) -> int:
        return self.window_count - self.cursor

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.remaining_windows < self.batch_size:
            raise SamplerExhausted(f"only {self.remaining_windows} unused windows left, batch needs {self.batch_size}")
        windows = self.order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return _windows_to_tensors(self.tokens, windows * self.seq_len, self.seq_len)

    def order_fingerprint(self) -> str:
        return hashlib.sha256(self.order.astype(np.int64).tobytes()).hexdigest()

    def consumed_fingerprint(self) -> str:
        """Hash of the window indices consumed so far: equal fingerprints = identical data sequence."""
        return hashlib.sha256(self.order[: self.cursor].astype(np.int64).tobytes()).hexdigest()

    def state_dict(self) -> dict[str, Any]:
        return {"version": SHUFFLED_SAMPLER_VERSION, "method": "numpy.random.Generator(PCG64(seed)).permutation(W)",
                "seed": self.seed, "seq_len": self.seq_len, "batch_size": self.batch_size,
                "window_count": self.window_count, "order_sha256": self.order_fingerprint(), "cursor": self.cursor}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        mine = self.state_dict()
        for key in ("version", "method", "seed", "seq_len", "batch_size", "window_count", "order_sha256"):
            if state[key] != mine[key]:
                raise ValueError(f"sampler {key} differs from checkpoint: {state[key]!r} vs {mine[key]!r}")
        if not 0 <= state["cursor"] <= self.window_count:
            raise ValueError(f"checkpoint cursor {state['cursor']} outside [0, {self.window_count}]")
        self.cursor = state["cursor"]


def fixed_validation_batches(tokens: np.ndarray, seq_len: int, batch_size: int,
                             n_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Evenly spaced, non-random windows: identical on every call, so measurements are comparable."""
    n_windows = batch_size * n_batches
    max_start = tokens.size - (seq_len + 1)
    if max_start < 0 or n_windows * (seq_len + 1) > tokens.size:
        raise ValueError(f"validation split too small for {n_windows} windows of {seq_len + 1} tokens")
    starts = np.linspace(0, max_start, n_windows).astype(np.int64)
    return [_windows_to_tensors(tokens, starts[i * batch_size : (i + 1) * batch_size], seq_len)
            for i in range(n_batches)]
