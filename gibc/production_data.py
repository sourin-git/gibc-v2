"""Production corpus: exact continuation of the Stage 1 pilot in resumable chunks, then an
exact-text duplicate audit and immutable uint16 token files.

Continuation contract. The pilot visited row groups of the pinned FineWeb-Edu `sample-10BT`
files round-robin by row-group index over the sorted file list, and stopped part-way through
one row group. That position (the "cursor") is derived from the pilot manifest and verified
against the real file layout: every consumed row group must match the expected round-robin
sequence, all but the last must be complete, and the rows must sum to `documents_read`.
Acquisition resumes at exactly that row with the same revision, file order, filters
(Wikipedia-source exclusion, empty-text rejection), validation salt and fraction. The pilot is
therefore included exactly once.

Chunks. Each chunk reads at most `chunk_row_groups` round-robin positions, and it stops early
at the document where the cumulative TRAIN tokens (pilot + chunks, under our tokenizer, with
one EOT per document) first reach the target. A chunk is written to `<chunk>.partial/` and
renamed to `<chunk>/` only after its `chunk.json` (cursors, counts, file hashes) is complete.
On restart, completed chunks are re-verified (file hashes, cursor continuity) and reused;
partial ones are deleted and redone.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from gibc.acquire import COLUMNS, AcquisitionConfig, is_validation_doc, read_jsonl_documents, resolve_parquet_files
from gibc.source_filter import is_wikipedia_source, normalized_hostname

CHUNK_FORMAT = "gibc.production_chunk.v1"


@dataclass(frozen=True)
class ProductionConfig:
    pilot_config: str  # repo-relative path of the pilot AcquisitionConfig (source of revision/salt/fraction)
    pilot_name: str  # $GIBC_WORK_DIR/data/<pilot_name>
    name: str  # $GIBC_WORK_DIR/data/<name>
    target_train_tokens: int  # stored train tokens incl. EOT, pilot + continuation
    chunk_row_groups: int
    min_free_gib: float
    bytes_per_token_estimate: float  # only for the disk-space estimate
    read_retries: int

    @classmethod
    def from_json(cls, path: str | Path) -> ProductionConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown ProductionConfig keys: {sorted(unknown)}")
        return cls(**data)


def round_robin_order(row_groups_per_file: list[int]) -> list[tuple[int, int]]:
    """The pilot's visiting order: (file_index, row_group) by row-group index, then sorted file order."""
    return [(f, g) for g in range(max(row_groups_per_file)) for f, n in enumerate(row_groups_per_file) if g < n]


def pilot_end_cursor(manifest: dict[str, Any], files: list[str], row_groups_per_file: list[int],
                     rows_per_group: dict[tuple[int, int], int]) -> dict[str, int]:
    """Derive and verify where the pilot stopped. Returns {"position": i, "row": r}: the next unread
    row is zero-based row r of round-robin position i."""
    if manifest["source_files"] != files:
        raise ValueError("pilot source file list differs from the current sorted file list")
    order = round_robin_order(row_groups_per_file)
    consumed = manifest["consumed_row_groups"]
    for i, entry in enumerate(consumed):
        f, g = order[i]
        if entry["file"] != files[f] or entry["row_group"] != g:
            raise ValueError(f"pilot entry {i} is {entry['file']}#{entry['row_group']}, expected {files[f]}#{g}")
        if entry["num_rows"] != rows_per_group[(f, g)]:
            raise ValueError(f"pilot entry {i}: num_rows {entry['num_rows']} != file layout {rows_per_group[(f, g)]}")
        if i < len(consumed) - 1 and entry["rows_consumed"] != entry["num_rows"]:
            raise ValueError(f"pilot entry {i} was not fully consumed but is not the last one")
    if sum(e["rows_consumed"] for e in consumed) != manifest["counts"]["documents_read"]:
        raise ValueError("pilot consumed rows do not sum to documents_read")
    last = consumed[-1]
    if last["rows_consumed"] == last["num_rows"]:
        return {"position": len(consumed), "row": 0}
    return {"position": len(consumed) - 1, "row": last["rows_consumed"]}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_chunks(chunks_dir: Path) -> list[dict[str, Any]]:
    """Verified completed chunks in order; deletes leftover partial chunks."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for partial in chunks_dir.glob("*.partial"):
        shutil.rmtree(partial)
    records = []
    for d in sorted(p for p in chunks_dir.iterdir() if p.is_dir()):
        record = json.loads((d / "chunk.json").read_text(encoding="utf-8"))
        for name, info in record["outputs"].items():
            if _sha256_file(d / name) != info["sha256"]:
                raise ValueError(f"{d / name}: hash mismatch with chunk.json; delete the chunk to redo it")
        if records and record["start"] != records[-1]["end"]:
            raise ValueError(f"{d.name}: start cursor {record['start']} != previous end {records[-1]['end']}")
        records.append(record)
    return records


class SourceReader:
    """Open handles to the pinned parquet files; reads row groups with retries."""

    def __init__(self, acq: AcquisitionConfig, files: list[str], retries: int) -> None:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        self.files = files
        self.retries = retries
        fs = HfFileSystem()
        self.handles = [fs.open(f"datasets/{acq.dataset}@{acq.revision}/{path}", "rb") for path in files]
        self.parquet = [pq.ParquetFile(h) for h in self.handles]
        self.row_groups_per_file = [p.metadata.num_row_groups for p in self.parquet]
        self.rows_per_group = {(f, g): p.metadata.row_group(g).num_rows
                               for f, p in enumerate(self.parquet) for g in range(p.metadata.num_row_groups)}
        self.order = round_robin_order(self.row_groups_per_file)

    def read(self, position: int) -> list[dict[str, Any]]:
        f, g = self.order[position]
        for attempt in range(self.retries + 1):
            try:
                return self.parquet[f].read_row_group(g, columns=list(COLUMNS)).to_pylist()
            except Exception:  # network/transient errors from fsspec/pyarrow
                if attempt == self.retries:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")

    def close(self) -> None:
        for h in self.handles:
            h.close()


def acquire_chunk(reader: SourceReader, acq: AcquisitionConfig, tok, start: dict[str, int], max_positions: int,
                  tokens_before: int, target: int, chunk_dir: Path) -> dict[str, Any]:
    """Read one chunk starting at `start`; returns its record (also written as chunk.json)."""
    partial = chunk_dir.with_name(chunk_dir.name + ".partial")
    partial.mkdir(parents=True)
    counts = {"documents_read": 0, "accepted_train": 0, "accepted_val": 0, "rejected_wikipedia_source": 0,
              "rejected_empty_text": 0, "retained_without_usable_url": 0}
    text_bytes = {"read": 0, "accepted_train": 0, "accepted_val": 0, "rejected_wikipedia_source": 0}
    tokens = {"train_with_eot": 0, "val_with_eot": 0}
    consumed = []
    position, row = start["position"], start["row"]
    reached = False
    with (partial / "train.jsonl").open("w", encoding="utf-8", newline="\n") as train_fh, \
         (partial / "val.jsonl").open("w", encoding="utf-8", newline="\n") as val_fh:
        while position < len(reader.order) and len(consumed) < max_positions and not reached:
            rows = reader.read(position)
            f, g = reader.order[position]
            decisions = []
            for r in rows[row:]:
                text, url = r["text"] or "", r["url"]
                if is_wikipedia_source(url):
                    decisions.append(("wiki", r))
                elif not text.strip():
                    decisions.append(("empty", r))
                else:
                    split = "val" if is_validation_doc(r["id"], acq.val_fraction, acq.val_salt) else "train"
                    decisions.append((split, r))
            # Token count per accepted document under our tokenizer, + 1 EOT, exactly as the token
            # files will store it (same tokenizer object, same encode_special_tokens setting).
            accepted = [i for i, (kind, _) in enumerate(decisions) if kind in ("train", "val")]
            n_tokens = [0] * len(decisions)
            for i, enc in zip(accepted, tok.encode_batch([decisions[i][1]["text"] for i in accepted])):
                n_tokens[i] = len(enc.ids) + 1
            taken = 0
            for (kind, r), n_tok in zip(decisions, n_tokens):
                text = r["text"] or ""
                n_bytes = len(text.encode("utf-8"))
                counts["documents_read"] += 1
                text_bytes["read"] += n_bytes
                taken += 1
                if kind == "wiki":
                    counts["rejected_wikipedia_source"] += 1
                    text_bytes["rejected_wikipedia_source"] += n_bytes
                elif kind == "empty":
                    counts["rejected_empty_text"] += 1
                else:
                    if normalized_hostname(r["url"]) is None:
                        counts["retained_without_usable_url"] += 1
                    fh = train_fh if kind == "train" else val_fh
                    fh.write(json.dumps({"id": r["id"], "url": r["url"], "text": text}, ensure_ascii=False) + "\n")
                    counts[f"accepted_{kind}"] += 1
                    text_bytes[f"accepted_{kind}"] += n_bytes
                    tokens[f"{kind}_with_eot"] += n_tok
                    if kind == "train" and tokens_before + tokens["train_with_eot"] >= target:
                        reached = True
                        break
            consumed.append({"file": reader.files[f], "row_group": g, "num_rows": len(rows),
                             "first_row": row, "rows_consumed": taken})
            if reached and row + taken < len(rows):
                row += taken  # stopped inside this row group
            else:
                position, row = position + 1, 0
    record = {
        "format": CHUNK_FORMAT, "chunk": chunk_dir.name, "start": start, "end": {"position": position, "row": row},
        "consumed_row_groups": consumed, "counts": counts, "text_bytes": text_bytes, "tokens": tokens,
        "train_tokens_cumulative": tokens_before + tokens["train_with_eot"], "target_reached": reached,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outputs": {name: {"bytes": (partial / name).stat().st_size, "sha256": _sha256_file(partial / name)}
                    for name in ("train.jsonl", "val.jsonl")},
    }
    (partial / "chunk.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    partial.rename(chunk_dir)
    return record


def iter_corpus_documents(pilot_dir: Path, chunks_dir: Path, chunk_names: list[str], split: str) -> Iterator[dict]:
    """Pilot documents first, then each chunk in order: the single canonical corpus order."""
    yield from read_jsonl_documents(pilot_dir / f"{split}.jsonl")
    for name in chunk_names:
        yield from read_jsonl_documents(chunks_dir / name / f"{split}.jsonl")


def text_digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()
