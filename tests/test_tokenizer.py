"""Tests for the committed tokenizer artifact (results/tokenizer/) and the tokenizer helpers."""

import json
from pathlib import Path

import pytest
from tokenizers import pre_tokenizers

from gibc.config import ModelConfig
from gibc.tokenizer import (
    EOT_TOKEN,
    decode,
    encode_documents,
    encode_text,
    eot_id,
    file_sha256,
    load_tokenizer,
    train_tokenizer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = REPO_ROOT / "results" / "tokenizer"
TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"
META = json.loads((TOKENIZER_DIR / "tokenizer_meta.json").read_text(encoding="utf-8"))
MODEL_VOCAB = ModelConfig.from_json(REPO_ROOT / "configs" / "model" / "gibc_43m.json").vocab_size

ROUNDTRIP_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    " leading space",
    "   three leading spaces",
    "trailing spaces   ",
    "repeated     spaces    inside",
    "\ttab\tseparated\tvalues\t",
    "line one\nline two\n\nline four after a blank line\n",
    "windows\r\nline\r\nendings\r\n",
    "\n\n\n",
    "naïve café résumé façade Ångström",
    "中文字符 日本語のテキスト 한국어 텍스트",
    "emoji 👍🏽 🧪 👨‍👩‍👧 flags 🇮🇳",
    "combining é vs é, zero​width, nbsp here",
    "RTL: שלום עולם مرحبا بالعالم",
    "math ∑ ∫ √ ≤ ≥ ≠ π ∞ and © ® ™ § ¶",
    "punctuation: !?.,;:'\"()[]{}<>@#$%^&*-_=+|\\/~` … — – « » ‘ ’ “ ”",
    'def f(x: int) -> int:\n    """Doc."""\n    return {"a": [x, x**2]}  # comment\n',
    "for (int i = 0; i < n; ++i) { total += a[i]; }\n",
    "<html><body class=\"x\">&amp; &lt;tag&gt;</body></html>",
    "control \x00\x01\x7f chars",
    "",
    "a",
    " ",
    "\n",
    ".",
]


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(TOKENIZER_PATH)


def test_vocab_size_is_exact_and_matches_model(tok):
    assert tok.get_vocab_size(with_added_tokens=True) == 24_000
    assert META["vocab_size"] == MODEL_VOCAB == 24_000


def test_artifact_hash_matches_metadata():
    assert file_sha256(TOKENIZER_PATH) == META["sha256"]


def test_special_tokens_match_metadata(tok):
    added = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))["added_tokens"]
    assert [(t["content"], t["id"], t["special"]) for t in added] == [(EOT_TOKEN, META["eot_id"], True)]
    assert META["special_tokens"] == {EOT_TOKEN: eot_id(tok)}
    assert META["bos_token"] is None and META["pad_token"] is None and META["unk_token"] is None


def test_full_byte_alphabet_present(tok):
    vocab = tok.get_vocab(with_added_tokens=False)
    assert all(symbol in vocab for symbol in pre_tokenizers.ByteLevel.alphabet())
    assert len(pre_tokenizers.ByteLevel.alphabet()) == 256


def test_no_tokens_are_added_automatically(tok):
    assert encode_text(tok, "") == []
    ids = encode_text(tok, "hello")
    assert eot_id(tok) not in ids


@pytest.mark.parametrize("text", ROUNDTRIP_TEXTS)
def test_roundtrip(tok, text):
    assert decode(tok, encode_text(tok, text)) == text


def test_literal_eot_text_is_not_the_special_id(tok):
    eot = eot_id(tok)
    for text in [EOT_TOKEN, f"before{EOT_TOKEN}after", f" {EOT_TOKEN} ", f"{EOT_TOKEN}{EOT_TOKEN}"]:
        ids = encode_text(tok, text)
        assert eot not in ids
        assert decode(tok, ids) == text


def test_one_boundary_per_document(tok):
    docs = ["first doc", "", f"has literal {EOT_TOKEN} inside", "last\n"]
    encoded = encode_documents(tok, docs)
    eot = eot_id(tok)
    assert len(encoded) == len(docs)
    for doc, ids in zip(docs, encoded):
        assert ids[-1] == eot
        assert ids.count(eot) == 1
        assert decode(tok, ids[:-1]) == doc
    assert sum(ids.count(eot) for ids in encoded) == len(docs)


def test_save_reload_gives_identical_ids(tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    reloaded = load_tokenizer(path)
    for text in ROUNDTRIP_TEXTS + [f"x{EOT_TOKEN}y"]:
        assert encode_text(reloaded, text) == encode_text(tok, text)
    assert eot_id(reloaded) == eot_id(tok)
    assert reloaded.get_vocab_size(with_added_tokens=True) == 24_000


SMALL_CORPUS = [
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "The mitochondria produce adenosine triphosphate through cellular respiration pathways.",
    "Tectonic plates drift slowly, forming mountains, earthquakes and volcanic island arcs.",
    "def area(radius):\n    return 3.14159 * radius ** 2\n",
] * 20


def test_train_small_tokenizer_has_contract_properties():
    small = train_tokenizer(SMALL_CORPUS, vocab_size=300, min_frequency=2)
    assert small.get_vocab_size(with_added_tokens=True) == 300
    assert eot_id(small) == 0
    assert encode_text(small, EOT_TOKEN).count(0) == 0
    assert decode(small, encode_text(small, "unseen: 中文 👍🏽\t\n")) == "unseen: 中文 👍🏽\t\n"


def test_train_fails_loudly_if_vocab_not_reached():
    with pytest.raises(ValueError, match="required exactly"):
        train_tokenizer(["tiny corpus"] * 5, vocab_size=5_000)
