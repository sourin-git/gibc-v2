import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gibc.data import (
    Batcher,
    SamplerExhausted,
    ShuffledWindowSampler,
    file_sha256_streaming,
    fixed_validation_batches,
    open_token_file,
    prepare_token_splits,
    verify_token_files,
    write_token_file,
)
from gibc.tokenizer import EOT_TOKEN, decode, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = REPO_ROOT / "results" / "tokenizer" / "tokenizer.json"
VOCAB = 24_000

DOCS = [
    "First document about photosynthesis.",
    "Second one\nwith a newline and  spaces.",
    f"Third has a literal {EOT_TOKEN} inside raw text.",
    "",
    "Fifth: 中文 and emoji 👍🏽.",
]


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(TOKENIZER_PATH)


def test_write_token_file_uint16_one_eot_per_document(tok, tmp_path):
    path = tmp_path / "t.bin"
    info = write_token_file(tok, DOCS, path, VOCAB, batch_docs=2)
    tokens = np.fromfile(path, dtype=np.uint16)
    assert path.stat().st_size == 2 * tokens.size == info["bytes"]
    assert info["documents"] == len(DOCS) and info["tokens"] == tokens.size
    assert int(tokens.max()) < VOCAB
    # Exactly one EOT per document, at its end; splitting on EOT recovers every document exactly.
    assert int((tokens == 0).sum()) == len(DOCS) == info["eot_tokens"]
    assert tokens[-1] == 0
    boundaries = np.flatnonzero(tokens == 0)
    pieces = np.split(tokens, boundaries + 1)[:-1]
    assert [decode(tok, p[:-1].tolist()) for p in pieces] == DOCS
    assert not (tmp_path / "t.bin.tmp").exists()


def test_open_token_file_rejects_out_of_range_and_wrong_length(tmp_path):
    good = tmp_path / "good.bin"
    np.array([1, 2, VOCAB - 1, 0], dtype=np.uint16).tofile(good)
    assert open_token_file(good, VOCAB, expected_tokens=4).size == 4
    with pytest.raises(ValueError, match="metadata says"):
        open_token_file(good, VOCAB, expected_tokens=5)
    bad = tmp_path / "bad.bin"
    np.array([1, VOCAB, 0], dtype=np.uint16).tofile(bad)
    with pytest.raises(ValueError, match="vocab_size"):
        open_token_file(bad, VOCAB)


def test_prepare_token_splits_meta(tok, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}", encoding="utf-8")
    for split, docs in (("train", DOCS), ("val", DOCS[:2])):
        with (data / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for i, text in enumerate(docs):
                fh.write(json.dumps({"id": str(i), "url": None, "text": text}, ensure_ascii=False) + "\n")
    meta = prepare_token_splits(data, tmp_path / "tokens", TOKENIZER_PATH, VOCAB, command="test")
    assert meta["dtype"] == "uint16" and meta["eot_id"] == 0 and meta["vocab_size"] == VOCAB
    assert meta["splits"]["train"]["documents"] == len(DOCS) and meta["splits"]["val"]["documents"] == 2
    on_disk = json.loads((tmp_path / "tokens" / "meta.json").read_text(encoding="utf-8"))
    assert on_disk == meta


def corpus(n: int = 5000) -> np.ndarray:
    return (np.arange(n) % 997).astype(np.uint16)  # distinct neighbours make shift errors visible


def test_batcher_shapes_and_single_shift():
    tokens = corpus()
    x, y = Batcher(tokens, seq_len=16, batch_size=3, seed=0).next_batch()
    assert x.shape == y.shape == (3, 16) and x.dtype == y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is x shifted by exactly one
    for row_x, row_y in zip(x, y):
        start = int(np.flatnonzero(tokens == row_x[0].item())[0])  # values are unique within a period
        window = torch.from_numpy(tokens[start : start + 17].astype(np.int64))
        assert torch.equal(row_x, window[:-1]) and torch.equal(row_y, window[1:])


def test_batcher_deterministic_and_state_restorable():
    tokens = corpus()
    a, b = Batcher(tokens, 16, 4, seed=5), Batcher(tokens, 16, 4, seed=5)
    for _ in range(3):
        assert all(torch.equal(u, v) for u, v in zip(a.next_batch(), b.next_batch()))
    state = a.state_dict()
    expected = [a.next_batch() for _ in range(3)]
    c = Batcher(tokens, 16, 4, seed=999)
    c.load_state_dict(state)
    for got, want in zip((c.next_batch() for _ in range(3)), expected):
        assert all(torch.equal(u, v) for u, v in zip(got, want))
    assert not torch.equal(Batcher(tokens, 16, 4, seed=6).next_batch()[0], Batcher(tokens, 16, 4, seed=5).next_batch()[0])


def test_batcher_covers_last_valid_window():
    tokens = corpus(20)
    b = Batcher(tokens, seq_len=19, batch_size=8, seed=0)  # exactly one valid window
    x, y = b.next_batch()
    assert torch.equal(y[:, -1], torch.full((8,), int(tokens[-1])))
    with pytest.raises(ValueError):
        Batcher(tokens, seq_len=20, batch_size=1, seed=0)


# Shuffled one-pass window sampler -------------------------------------------------------------

def numbered(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.uint16)  # token value == position, so targets identify themselves


def drain(sampler) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    while sampler.remaining_windows >= sampler.batch_size:
        out.append(sampler.next_batch())
    return out


@pytest.mark.parametrize("n", [17, 33, 40, 49])
def test_windows_cover_every_target_once_and_drop_tail(n):
    seq = 8
    s = ShuffledWindowSampler(numbered(n), seq_len=seq, batch_size=1, seed=0)
    assert s.window_count == (n - 1) // seq
    assert s.dropped_targets == (n - 1) - s.window_count * seq
    batches = drain(s)
    targets = sorted(int(t) for _, y in batches for t in y.flatten())
    assert targets == list(range(1, s.window_count * seq + 1))  # no duplicates, no skipped full-window target
    for x, y in batches:
        start = int(x[0, 0])
        assert start % seq == 0  # window i starts at seq*i
        assert torch.equal(x[0], torch.arange(start, start + seq)) and torch.equal(y[0], torch.arange(start + 1, start + seq + 1))
    # Neighbouring windows share exactly one token: the last target of i is the first context token of i+1.
    by_start = {int(x[0, 0]): (x[0], y[0]) for x, y in batches}
    starts = sorted(by_start)
    for a, b in zip(starts, starts[1:]):
        (xa, ya), (xb, yb) = by_start[a], by_start[b]
        assert b == a + seq and int(xb[0]) == int(ya[-1])
        assert set(torch.cat([xa, ya]).tolist()) & set(torch.cat([xb, yb]).tolist()) == {b}
    assert 0 not in targets  # position zero is context only


def test_boundary_lengths():
    with pytest.raises(ValueError):
        ShuffledWindowSampler(numbered(8), seq_len=8, batch_size=1, seed=0)  # 7 targets < one window
    s = ShuffledWindowSampler(numbered(9), seq_len=8, batch_size=1, seed=0)
    assert s.window_count == 1 and s.dropped_targets == 0


def test_shuffle_is_deterministic_and_seed_dependent():
    a = ShuffledWindowSampler(numbered(4000), 8, 4, seed=3)
    b = ShuffledWindowSampler(numbered(4000), 8, 4, seed=3)
    c = ShuffledWindowSampler(numbered(4000), 8, 4, seed=4)
    assert a.order_fingerprint() == b.order_fingerprint() != c.order_fingerprint()
    assert all(torch.equal(p[0], q[0]) for p, q in zip(drain(a), drain(b)))
    assert sorted(a.order.tolist()) == list(range(a.window_count))  # a permutation: without replacement


def test_exhaustion_raises_without_wraparound():
    s = ShuffledWindowSampler(numbered(8 * 10 + 1), 8, 4, seed=0)  # 10 windows
    s.next_batch(), s.next_batch()
    assert s.remaining_windows == 2
    with pytest.raises(SamplerExhausted):
        s.next_batch()
    assert s.cursor == 8  # no partial consumption on failure


def test_sampler_resume_continues_exactly():
    tokens = numbered(8 * 100 + 5)
    full = ShuffledWindowSampler(tokens, 8, 3, seed=9)
    expected = [full.next_batch() for _ in range(20)]
    first = ShuffledWindowSampler(tokens, 8, 3, seed=9)
    got = [first.next_batch() for _ in range(7)]
    state = first.state_dict()
    resumed = ShuffledWindowSampler(tokens, 8, 3, seed=9)
    resumed.load_state_dict(state)
    got += [resumed.next_batch() for _ in range(13)]
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(got, expected))
    assert resumed.consumed_fingerprint() == full.consumed_fingerprint()


@pytest.mark.parametrize("change", [{"seed": 10}, {"seq_len": 4}, {"batch_size": 2}])
def test_sampler_restore_rejects_changed_immutables(change):
    tokens = numbered(8 * 100 + 5)
    state = ShuffledWindowSampler(tokens, 8, 3, seed=9).state_dict()
    args = {"seq_len": 8, "batch_size": 3, "seed": 9, **change}
    with pytest.raises(ValueError, match="differs"):
        ShuffledWindowSampler(tokens, **args).load_state_dict(state)
    with pytest.raises(ValueError, match="window_count"):
        ShuffledWindowSampler(numbered(8 * 101 + 5), 8, 3, seed=9).load_state_dict(state)


# Token-file hash verification ----------------------------------------------------------------

def test_streaming_hash_and_corruption_detected(tmp_path):
    import hashlib

    arr = np.arange(100_000, dtype=np.uint16) % 24_000
    for split in ("train", "val"):
        arr.tofile(tmp_path / f"{split}.bin")
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    assert file_sha256_streaming(tmp_path / "train.bin", chunk_bytes=4096) == digest
    meta = {"splits": {"train": {"sha256": digest}, "val": {"sha256": digest}}}
    verify_token_files(tmp_path, meta)
    with (tmp_path / "val.bin").open("r+b") as fh:  # flip one byte in the middle
        fh.seek(12_345)
        byte = fh.read(1)
        fh.seek(12_345)
        fh.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ValueError, match="val.bin: sha256"):
        verify_token_files(tmp_path, meta)


def test_validation_batches_fixed():
    tokens = corpus()
    a = fixed_validation_batches(tokens, 16, 2, 3)
    b = fixed_validation_batches(tokens, 16, 2, 3)
    assert len(a) == 3
    assert all(torch.equal(x1, x2) and torch.equal(y1, y2) for (x1, y1), (x2, y2) in zip(a, b))
    assert all(torch.equal(x[:, 1:], y[:, :-1]) for x, y in a)
    with pytest.raises(ValueError, match="too small"):
        fixed_validation_batches(corpus(50), 16, 2, 3)
