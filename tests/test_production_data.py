import json
from pathlib import Path

import pytest

from gibc.acquire import AcquisitionConfig, is_validation_doc
from gibc.production_data import acquire_chunk, completed_chunks, pilot_end_cursor, round_robin_order
from gibc.tokenizer import load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
ACQ = AcquisitionConfig(dataset="d", config_name="c", data_dir="x", revision="0" * 40, max_text_bytes=10**9,
                        max_documents=None, val_fraction=0.1, val_salt="s")


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(REPO_ROOT / "results" / "tokenizer" / "tokenizer.json")


def test_round_robin_order_matches_pilot_policy():
    # 3 files with 2, 3, 1 row groups: all group-0s first, then group-1s, ...
    assert round_robin_order([2, 3, 1]) == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 2)]


def _manifest(files, entries, read):
    return {"source_files": files, "counts": {"documents_read": read},
            "consumed_row_groups": [{"file": files[f], "row_group": g, "num_rows": n, "rows_consumed": c}
                                    for f, g, n, c in entries]}


def test_pilot_cursor_mid_row_group_and_verification():
    files = ["a", "b"]
    rows = {(0, 0): 10, (1, 0): 10, (0, 1): 10, (1, 1): 10}
    m = _manifest(files, [(0, 0, 10, 10), (1, 0, 10, 10), (0, 1, 10, 4)], 24)
    assert pilot_end_cursor(m, files, [2, 2], rows) == {"position": 2, "row": 4}  # next unread row is #4

    full = _manifest(files, [(0, 0, 10, 10), (1, 0, 10, 10)], 20)
    assert pilot_end_cursor(full, files, [2, 2], rows) == {"position": 2, "row": 0}

    with pytest.raises(ValueError, match="expected"):  # out of round-robin order
        pilot_end_cursor(_manifest(files, [(1, 0, 10, 10)], 10), files, [2, 2], rows)
    with pytest.raises(ValueError, match="not fully consumed"):
        pilot_end_cursor(_manifest(files, [(0, 0, 10, 5), (1, 0, 10, 3)], 8), files, [2, 2], rows)
    with pytest.raises(ValueError, match="documents_read"):
        pilot_end_cursor(_manifest(files, [(0, 0, 10, 10), (1, 0, 10, 2)], 13), files, [2, 2], rows)
    with pytest.raises(ValueError, match="file list"):
        pilot_end_cursor(m, ["a", "c"], [2, 2], rows)


def test_real_pilot_manifest_cursor():
    m = json.loads((REPO_ROOT / "results" / "data" / "pilot_manifest.json").read_text(encoding="utf-8"))
    files = m["source_files"]
    rows = {(f, g): 1000 for f in range(14) for g in range(2)}
    cursor = pilot_end_cursor(m, files, [726] * 14, rows)
    assert cursor == {"position": 21, "row": 148}  # 007_00000.parquet, row group 1, next zero-based row 148
    assert round_robin_order([726] * 14)[21] == (7, 1)


class FakeReader:
    """Row groups of synthetic documents, in the same round-robin structure as SourceReader."""

    def __init__(self, row_groups_per_file, rows_per_group=5):
        self.files = [f"f{i}" for i in range(len(row_groups_per_file))]
        self.order = round_robin_order(row_groups_per_file)
        self.rows_per_group = rows_per_group

    def read(self, position):
        f, g = self.order[position]
        docs = []
        for r in range(self.rows_per_group):
            doc_id = f"{f}-{g}-{r}"
            url = "https://en.wikipedia.org/wiki/X" if r == 1 else ("https://example.com/" + doc_id)
            text = "" if r == 2 else f"Document {doc_id}: some ordinary text about topic {r}."
            docs.append({"id": doc_id, "url": url, "text": text})
        return docs


def test_chunk_starts_at_cursor_and_is_recorded(tok, tmp_path):
    reader = FakeReader([3, 3])
    rec = acquire_chunk(reader, ACQ, tok, {"position": 1, "row": 3}, max_positions=2, tokens_before=0,
                        target=10**9, chunk_dir=tmp_path / "chunk_00000")
    ids = [json.loads(l)["id"] for s in ("train", "val")
           for l in (tmp_path / "chunk_00000" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "1-0-3" in ids and "1-0-2" not in ids and "1-0-0" not in ids  # rows before the cursor never re-read
    assert rec["start"] == {"position": 1, "row": 3} and rec["end"] == {"position": 3, "row": 0}
    assert [(e["first_row"], e["rows_consumed"]) for e in rec["consumed_row_groups"]] == [(3, 2), (0, 5)]
    assert rec["counts"]["documents_read"] == 7
    assert rec["counts"]["rejected_wikipedia_source"] == 1 and rec["counts"]["rejected_empty_text"] == 1
    assert rec["counts"]["accepted_train"] + rec["counts"]["accepted_val"] == 5
    for s in ("train", "val"):
        for line in (tmp_path / "chunk_00000" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
            assert is_validation_doc(json.loads(line)["id"], 0.1, "s") == (s == "val")
    assert not (tmp_path / "chunk_00000.partial").exists()


def test_chunks_continue_exactly_and_stop_at_target(tok, tmp_path):
    reader = FakeReader([4, 4])
    chunks = tmp_path / "chunks"
    first = acquire_chunk(reader, ACQ, tok, {"position": 0, "row": 0}, 3, 0, 10**9, chunks / "chunk_00000")
    per_doc = first["tokens"]["train_with_eot"] / first["counts"]["accepted_train"]
    target = first["tokens"]["train_with_eot"] + int(per_doc * 2)  # reached a few docs into the next chunk
    second = acquire_chunk(reader, ACQ, tok, first["end"], 3, first["tokens"]["train_with_eot"], target,
                           chunks / "chunk_00001")
    assert second["start"] == first["end"] and second["target_reached"]
    assert second["train_tokens_cumulative"] >= target
    assert second["train_tokens_cumulative"] - second["tokens"]["train_with_eot"] < target
    records = completed_chunks(chunks)
    assert [r["chunk"] for r in records] == ["chunk_00000", "chunk_00001"]
    # Every document read exactly once across chunks.
    all_ids = [json.loads(l)["id"] for c in ("chunk_00000", "chunk_00001") for s in ("train", "val")
               for l in (chunks / c / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(all_ids) == len(set(all_ids))


def test_completed_chunks_drop_partial_and_detect_corruption(tok, tmp_path):
    reader = FakeReader([2])
    chunks = tmp_path / "chunks"
    acquire_chunk(reader, ACQ, tok, {"position": 0, "row": 0}, 1, 0, 10**9, chunks / "chunk_00000")
    (chunks / "chunk_00001.partial").mkdir()
    (chunks / "chunk_00001.partial" / "train.jsonl").write_text("half written", encoding="utf-8")
    assert [r["chunk"] for r in completed_chunks(chunks)] == ["chunk_00000"]
    assert not (chunks / "chunk_00001.partial").exists()
    with (chunks / "chunk_00000" / "train.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("corruption\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        completed_chunks(chunks)
