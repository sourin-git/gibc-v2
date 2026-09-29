import io
import json

import pytest

from gibc.acquire import AcquisitionConfig, filter_and_write, is_validation_doc

REVISION = "0" * 40


def make_cfg(**overrides) -> AcquisitionConfig:
    base = dict(
        dataset="d", config_name="c", data_dir="x", revision=REVISION,
        max_text_bytes=10**9, max_documents=None, val_fraction=0.0, val_salt="s",
    )
    return AcquisitionConfig(**{**base, **overrides})


def run(rows, cfg):
    writers = {"train": io.StringIO(), "val": io.StringIO()}
    stats = filter_and_write(rows, cfg, writers)
    docs = {split: [json.loads(line) for line in w.getvalue().splitlines()] for split, w in writers.items()}
    return stats, docs


def row(i, url="https://example.com/a", text="hello world"):
    return {"id": f"doc-{i}", "url": url, "text": text}


def test_exclusion_and_counts():
    rows = [
        row(0),
        row(1, url="https://en.wikipedia.org/wiki/X"),
        row(2, url="https://notwikipedia.org/x"),
        row(3, url=None),
        row(4, url="not a url"),
        row(5, text="   \n "),
        row(6, url="https://de.m.wikipedia.org/wiki/Y", text="wiki text"),
    ]
    stats, docs = run(rows, make_cfg())
    assert stats.documents_read == 7
    assert stats.rejected_wikipedia_source == 2
    assert stats.rejected_wikipedia_source_text_bytes == len("hello world") + len("wiki text")
    assert stats.rejected_empty_text == 1
    assert stats.retained_without_usable_url == 2
    assert [d["id"] for d in docs["train"]] == ["doc-0", "doc-2", "doc-3", "doc-4"]
    assert stats.stop_reason == "source_exhausted"


def test_unicode_text_written_losslessly():
    text = "naïve café — 中文 👍🏽\n\tcode()  "
    stats, docs = run([row(0, text=text)], make_cfg())
    assert docs["train"][0]["text"] == text
    assert stats.accepted["train"].text_bytes == len(text.encode("utf-8"))
    assert stats.accepted["train"].characters == len(text)


def test_stops_at_text_byte_bound_deterministically():
    rows = [row(i, text="x" * 10) for i in range(100)]
    stats, docs = run(rows, make_cfg(max_text_bytes=35))
    assert stats.stop_reason == "max_text_bytes"
    assert stats.accepted_documents == 4  # the document that reaches the bound is included
    assert stats.documents_read == 4


def test_stops_at_document_bound():
    stats, _ = run([row(i) for i in range(100)], make_cfg(max_documents=7))
    assert stats.stop_reason == "max_documents"
    assert stats.accepted_documents == 7


def test_validation_split_depends_only_on_id():
    ids = [f"<urn:uuid:{i}>" for i in range(20_000)]
    val = [i for i in ids if is_validation_doc(i, 0.01, "salt")]
    assert 120 < len(val) < 280  # ~1% (expected 200, sd ~14)
    assert val != [i for i in ids if is_validation_doc(i, 0.01, "other-salt")]
    assert not any(is_validation_doc(i, 0.0, "salt") for i in ids)

    stats, docs = run([row(i) for i in range(2000)], make_cfg(val_fraction=0.1))
    assert stats.accepted["val"].documents == len(docs["val"]) > 0
    assert all(is_validation_doc(d["id"], 0.1, "s") for d in docs["val"])
    assert not any(is_validation_doc(d["id"], 0.1, "s") for d in docs["train"])


@pytest.mark.parametrize(
    "overrides",
    [{"revision": "main"}, {"max_text_bytes": 0}, {"max_documents": 0}, {"val_fraction": 1.0}],
)
def test_config_validation(overrides):
    with pytest.raises(ValueError):
        make_cfg(**overrides)
