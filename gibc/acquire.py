"""Bounded acquisition of FineWeb-Edu documents, with Wikipedia-source exclusion.

Only the `id`, `url` and `text` columns of the dataset's parquet files are read, at a pinned
revision, one row group (1,000 rows) at a time over HTTP range requests. Nothing close to the
full ~28 GB sample is downloaded. Row groups are visited round-robin across the files (row
group 0 of every file, then row group 1, ...) so that a small pilot is not drawn from a single
file. Acquisition stops at a deterministic bound on accepted text bytes or accepted documents.

Hugging Face libraries are imported inside functions so that callers can check HF_HOME first
(`gibc.paths.require_hf_home`).
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from gibc.source_filter import EXCLUDED_DOMAIN, is_wikipedia_source, normalized_hostname

COLUMNS = ("id", "url", "text")
SPLITS = ("train", "val")


@dataclass(frozen=True)
class AcquisitionConfig:
    dataset: str
    config_name: str
    data_dir: str
    revision: str
    max_text_bytes: int
    max_documents: int | None
    val_fraction: float
    val_salt: str

    def __post_init__(self) -> None:
        if len(self.revision) != 40:
            raise ValueError("revision must be a full 40-character commit SHA (pinned)")
        if self.max_text_bytes <= 0:
            raise ValueError("max_text_bytes must be positive")
        if self.max_documents is not None and self.max_documents <= 0:
            raise ValueError("max_documents must be positive or null")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")

    @classmethod
    def from_json(cls, path: str | Path) -> AcquisitionConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown AcquisitionConfig keys: {sorted(unknown)}")
        return cls(**data)


def is_validation_doc(doc_id: str, fraction: float, salt: str) -> bool:
    """Order-independent split: a document's split depends only on its id, never on read order."""
    digest = hashlib.sha256(f"{salt}\x00{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") < fraction * 2**64


@dataclass
class SplitStats:
    documents: int = 0
    text_bytes: int = 0
    characters: int = 0


@dataclass
class AcquisitionStats:
    documents_read: int = 0
    text_bytes_read: int = 0
    rejected_wikipedia_source: int = 0
    rejected_wikipedia_source_text_bytes: int = 0
    rejected_empty_text: int = 0
    retained_without_usable_url: int = 0
    accepted: dict[str, SplitStats] = field(default_factory=lambda: {s: SplitStats() for s in SPLITS})
    stop_reason: str = "source_exhausted"

    @property
    def accepted_documents(self) -> int:
        return sum(s.documents for s in self.accepted.values())

    @property
    def accepted_text_bytes(self) -> int:
        return sum(s.text_bytes for s in self.accepted.values())


def filter_and_write(
    rows: Iterable[dict[str, Any]], cfg: AcquisitionConfig, writers: dict[str, TextIO]
) -> AcquisitionStats:
    """Apply Wikipedia-source exclusion, split, and write accepted documents as JSONL until a bound is hit.

    Text bytes are UTF-8 bytes of the `text` field. The bound is checked after every document,
    so the document that reaches it is included and the result is deterministic.
    """
    stats = AcquisitionStats()
    for row in rows:
        text = row["text"] or ""
        url = row["url"]
        n_bytes = len(text.encode("utf-8"))
        stats.documents_read += 1
        stats.text_bytes_read += n_bytes
        if is_wikipedia_source(url):
            stats.rejected_wikipedia_source += 1
            stats.rejected_wikipedia_source_text_bytes += n_bytes
        elif not text.strip():
            stats.rejected_empty_text += 1
        else:
            if normalized_hostname(url) is None:
                stats.retained_without_usable_url += 1
            split = "val" if is_validation_doc(row["id"], cfg.val_fraction, cfg.val_salt) else "train"
            writers[split].write(json.dumps({"id": row["id"], "url": url, "text": text}, ensure_ascii=False) + "\n")
            split_stats = stats.accepted[split]
            split_stats.documents += 1
            split_stats.text_bytes += n_bytes
            split_stats.characters += len(text)

        if stats.accepted_text_bytes >= cfg.max_text_bytes:
            stats.stop_reason = "max_text_bytes"
            break
        if cfg.max_documents is not None and stats.accepted_documents >= cfg.max_documents:
            stats.stop_reason = "max_documents"
            break
    return stats


def read_jsonl_documents(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def resolve_parquet_files(cfg: AcquisitionConfig) -> list[str]:
    """List the config's parquet files at the pinned revision, after checking the dataset card mapping."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(cfg.dataset, revision=cfg.revision)
    if info.sha != cfg.revision:
        raise RuntimeError(f"requested revision {cfg.revision} resolved to {info.sha}")
    card_configs = (info.card_data.to_dict() if info.card_data else {}).get("configs", [])
    data_files = next((c.get("data_files") for c in card_configs if c.get("config_name") == cfg.config_name), None)
    expected = [{"split": "train", "path": f"{cfg.data_dir}/*"}]
    if data_files != expected:
        raise RuntimeError(f"dataset card maps {cfg.config_name!r} to {data_files}, expected {expected}")
    entries = api.list_repo_tree(cfg.dataset, repo_type="dataset", revision=cfg.revision, path_in_repo=cfg.data_dir)
    files = sorted(e.path for e in entries if e.path.endswith(".parquet"))
    if not files:
        raise RuntimeError(f"no parquet files under {cfg.data_dir} at {cfg.revision}")
    return files


def iter_rows_round_robin(cfg: AcquisitionConfig, files: list[str], progress: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield rows by row-group index across files. `progress` records every row group touched and
    how many of its rows were yielded (the consumer processes each row before pulling the next)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    handles = [fs.open(f"datasets/{cfg.dataset}@{cfg.revision}/{path}", "rb") for path in files]
    try:
        parquet_files = [pq.ParquetFile(h) for h in handles]
        max_groups = max(p.metadata.num_row_groups for p in parquet_files)
        for group in range(max_groups):
            for path, pf in zip(files, parquet_files):
                if group >= pf.metadata.num_row_groups:
                    continue
                table = pf.read_row_group(group, columns=list(COLUMNS))
                entry = {"file": path, "row_group": group, "num_rows": table.num_rows, "rows_consumed": 0}
                progress.append(entry)
                for row in table.to_pylist():
                    entry["rows_consumed"] += 1
                    yield row
    finally:
        for h in handles:
            h.close()


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def acquire(cfg: AcquisitionConfig, out_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    """Acquire a bounded sample into out_dir/{train,val}.jsonl plus manifest.json. Returns the manifest."""
    import huggingface_hub
    import pyarrow

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass overwrite=True to replace it")
        shutil.rmtree(out_dir)
    partial_dir = out_dir.with_name(out_dir.name + ".partial")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    files = resolve_parquet_files(cfg)
    progress: list[dict[str, Any]] = []
    # newline="\n" keeps the JSONL byte-identical on Windows (no CRLF translation).
    with (
        (partial_dir / "train.jsonl").open("w", encoding="utf-8", newline="\n") as train_fh,
        (partial_dir / "val.jsonl").open("w", encoding="utf-8", newline="\n") as val_fh,
    ):
        rows = iter_rows_round_robin(cfg, files, progress)
        try:
            stats = filter_and_write(rows, cfg, {"train": train_fh, "val": val_fh})
        finally:
            rows.close()

    manifest = {
        "schema": "gibc.acquisition.v1",
        "dataset": cfg.dataset,
        "config_name": cfg.config_name,
        "data_dir": cfg.data_dir,
        "revision": cfg.revision,
        "source_split": "train",
        "access_method": (
            "huggingface_hub.HfFileSystem range reads of parquet row groups, columns "
            f"{list(COLUMNS)}; row groups visited round-robin across files by row-group index"
        ),
        "settings": asdict(cfg),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "stop_reason": stats.stop_reason,
        "counts": {
            "documents_read": stats.documents_read,
            "accepted_documents": stats.accepted_documents,
            "accepted_train_documents": stats.accepted["train"].documents,
            "accepted_val_documents": stats.accepted["val"].documents,
            "rejected_wikipedia_source": stats.rejected_wikipedia_source,
            "rejected_empty_text": stats.rejected_empty_text,
            "rejected_total": stats.rejected_wikipedia_source + stats.rejected_empty_text,
            "retained_without_usable_url": stats.retained_without_usable_url,
        },
        "text_bytes": {
            "note": "UTF-8 bytes of the `text` field; network transfer volume is not measured",
            "read": stats.text_bytes_read,
            "accepted": stats.accepted_text_bytes,
            "accepted_train": stats.accepted["train"].text_bytes,
            "accepted_val": stats.accepted["val"].text_bytes,
            "rejected_wikipedia_source": stats.rejected_wikipedia_source_text_bytes,
        },
        "characters": {split: stats.accepted[split].characters for split in SPLITS},
        "wikipedia_source_exclusion": {
            "rule": (
                f"reject if the normalized URL hostname is {EXCLUDED_DOMAIN} or ends with .{EXCLUDED_DOMAIN}; "
                "documents without a usable URL are retained and counted"
            ),
            "scope": (
                "Reduces direct Wikipedia-source overlap only. Does not guarantee absence of mirrored or "
                "quoted Wikipedia, WikiText or benchmark content; no contamination-free claim is made."
            ),
        },
        "validation_split": {
            "rule": "sha256(val_salt + NUL + id)[:8] as big-endian uint64 < val_fraction * 2^64",
            "val_fraction": cfg.val_fraction,
            "val_salt": cfg.val_salt,
        },
        "source_files": files,
        "consumed_row_groups": progress,
        "outputs": {name: _file_record(partial_dir / name) for name in ("train.jsonl", "val.jsonl")},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pyarrow": pyarrow.__version__,
            "huggingface_hub": huggingface_hub.__version__,
        },
    }
    (partial_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    partial_dir.rename(out_dir)
    return manifest
