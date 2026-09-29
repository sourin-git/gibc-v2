import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gibc.data import (
    Batcher,
    fixed_validation_batches,
    open_token_file,
    prepare_token_splits,
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


def test_validation_batches_fixed():
    tokens = corpus()
    a = fixed_validation_batches(tokens, 16, 2, 3)
    b = fixed_validation_batches(tokens, 16, 2, 3)
    assert len(a) == 3
    assert all(torch.equal(x1, x2) and torch.equal(y1, y2) for (x1, y1), (x2, y2) in zip(a, b))
    assert all(torch.equal(x[:, 1:], y[:, :-1]) for x, y in a)
    with pytest.raises(ValueError, match="too small"):
        fixed_validation_batches(corpus(50), 16, 2, 3)
