"""Continue the Stage 1 pilot acquisition, in resumable chunks, until the target train-token count.

Usage: python scripts/acquire_production.py --config configs/data/production.json
Safe to re-run after an interruption: completed chunks are verified and reused, partial ones redone.
Writes $GIBC_WORK_DIR/data/<name>/chunks/chunk_NNNNN/ and acquisition_manifest.json (copied to
results/data/<name>_acquisition_manifest.json).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone

from gibc.paths import REPO_ROOT, require_hf_home, work_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunked production acquisition (pilot continuation).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="verify cursor/disk/state, acquire nothing")
    args = parser.parse_args()

    work = work_dir()
    print(f"HF_HOME = {require_hf_home()}")
    from gibc.acquire import AcquisitionConfig, resolve_parquet_files
    from gibc.production_data import (
        ProductionConfig, SourceReader, _sha256_file, acquire_chunk, completed_chunks, pilot_end_cursor,
    )
    from gibc.tokenizer import file_sha256, load_tokenizer

    pcfg = ProductionConfig.from_json(args.config)
    acq = AcquisitionConfig.from_json(REPO_ROOT / pcfg.pilot_config)
    pilot_dir = work / "data" / pcfg.pilot_name
    out_dir = work / "data" / pcfg.name
    chunks_dir = out_dir / "chunks"

    # Pilot: settings and files must be exactly what Stage 1 recorded.
    manifest_path = pilot_dir / "manifest.json"
    pilot_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if pilot_manifest["settings"] != asdict(acq):
        raise ValueError("pilot manifest settings differ from the pilot config")
    if file_sha256(manifest_path) != file_sha256(REPO_ROOT / "results" / "data" / f"{pcfg.pilot_name}_manifest.json"):
        raise ValueError("work-dir pilot manifest differs from the committed copy in results/data/")
    for name, info in pilot_manifest["outputs"].items():
        if _sha256_file(pilot_dir / name) != info["sha256"]:
            raise ValueError(f"pilot {name} does not match its manifest hash")

    # Tokenizer: the frozen Stage 1 artifact.
    tokenizer_path = REPO_ROOT / "results" / "tokenizer" / "tokenizer.json"
    tokenizer_meta = json.loads((REPO_ROOT / "results" / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    tokenizer_sha = file_sha256(tokenizer_path)
    if tokenizer_sha != tokenizer_meta["sha256"]:
        raise ValueError("tokenizer.json does not match tokenizer_meta.json")
    pilot_tokens_meta = json.loads((REPO_ROOT / "results" / "data" / f"{pcfg.pilot_name}_tokens_meta.json").read_text(encoding="utf-8"))
    if pilot_tokens_meta["tokenizer_sha256"] != tokenizer_sha or pilot_tokens_meta["source_manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("pilot token counts were not produced from this pilot with this tokenizer")
    pilot_train_tokens = pilot_tokens_meta["splits"]["train"]["tokens"]

    files = resolve_parquet_files(acq)
    reader = SourceReader(acq, files, pcfg.read_retries)
    try:
        cursor = pilot_end_cursor(pilot_manifest, files, reader.row_groups_per_file, reader.rows_per_group)
        f, g = reader.order[cursor["position"]]
        print(f"pilot cursor (derived + verified): round-robin position {cursor['position']} = {files[f]} "
              f"row group {g}, next zero-based row {cursor['row']}; pilot train tokens {pilot_train_tokens:,}")

        chunks = completed_chunks(chunks_dir)
        if chunks and chunks[0]["start"] != cursor:
            raise ValueError(f"first chunk starts at {chunks[0]['start']}, pilot cursor is {cursor}")
        cumulative = pilot_train_tokens + sum(c["tokens"]["train_with_eot"] for c in chunks)
        done = bool(chunks) and chunks[-1]["target_reached"]
        print(f"{len(chunks)} completed chunk(s) reused; cumulative train tokens {cumulative:,} / "
              f"{pcfg.target_train_tokens:,}")

        if not done:
            remaining = pcfg.target_train_tokens - cumulative
            needed = remaining * pcfg.bytes_per_token_estimate * 1.05 + pcfg.target_train_tokens * 2 * 1.02
            free = shutil.disk_usage(work).free
            print(f"disk: free {free / 2**30:.1f} GiB, estimated need {needed / 2**30:.1f} GiB "
                  f"(JSONL for {remaining:,} tokens + final uint16 files), floor {pcfg.min_free_gib} GiB")
            if free - needed < pcfg.min_free_gib * 2**30:
                raise RuntimeError("refusing to acquire: would leave less than the minimum free disk space")
        if args.dry_run:
            print("dry run: nothing acquired")
            return 0

        start = chunks[-1]["end"] if chunks else cursor
        tok = load_tokenizer(tokenizer_path)
        while not done:
            name = f"chunk_{len(chunks):05d}"
            record = acquire_chunk(reader, acq, tok, start, pcfg.chunk_row_groups, cumulative,
                                   pcfg.target_train_tokens, chunks_dir / name)
            chunks.append(record)
            cumulative = record["train_tokens_cumulative"]
            done = record["target_reached"]
            start = record["end"]
            c = record["counts"]
            print(f"{name}: read {c['documents_read']:,} docs (train {c['accepted_train']:,}, val {c['accepted_val']:,}, "
                  f"wiki-rejected {c['rejected_wikipedia_source']}, empty {c['rejected_empty_text']}), "
                  f"train tokens +{record['tokens']['train_with_eot']:,} -> {cumulative:,}", flush=True)
            if record["end"]["position"] >= len(reader.order) and not done:
                raise RuntimeError("source exhausted before reaching the target")
    finally:
        reader.close()

    totals: dict[str, int] = {}
    for c in chunks:
        for section in ("counts", "text_bytes", "tokens"):
            for k, v in c[section].items():
                totals[f"{section}.{k}"] = totals.get(f"{section}.{k}", 0) + v
    manifest = {
        "format": "gibc.production_acquisition.v1",
        "complete": True,
        "dataset": acq.dataset, "config_name": acq.config_name, "revision": acq.revision,
        "settings": asdict(acq), "production_config": asdict(pcfg),
        "tokenizer_sha256": tokenizer_sha,
        "pilot": {"name": pcfg.pilot_name, "manifest_sha256": file_sha256(manifest_path),
                  "train_tokens_with_eot": pilot_train_tokens, "end_cursor": cursor},
        "continuation_totals": totals,
        "train_tokens_with_eot_total": cumulative,
        "chunks": chunks,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "acquisition_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    copy = REPO_ROOT / "results" / "data" / f"{pcfg.name}_acquisition_manifest.json"
    shutil.copyfile(out_dir / "acquisition_manifest.json", copy)
    print(f"complete: {len(chunks)} chunks, train tokens (pilot + continuation) {cumulative:,}; manifest -> {copy.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
