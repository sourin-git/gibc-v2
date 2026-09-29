"""Build the immutable production token files from the pilot + completed acquisition chunks.

Usage: python scripts/build_production_tokens.py --config configs/data/production.json
1. Verifies the acquisition is complete and every chunk / pilot file matches its recorded hash.
2. Exact-text duplicate audit (SHA-256 of the UTF-8 text; NOT near-duplicate detection): counts
   duplicates, and deterministically removes every validation document whose exact text also
   occurs in train (the validation copy is removed; train is untouched).
3. Tokenizes with the frozen Stage 1 tokenizer (one EOT per document) into
   $GIBC_WORK_DIR/tokens/<name>/{train,val}.bin + meta.json, marks them read-only, and copies meta
   and the audit to results/data/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
from datetime import datetime, timezone

from gibc.paths import REPO_ROOT, work_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build production token files.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    from gibc.config import ModelConfig
    from gibc.data import open_token_file, write_token_file
    from gibc.production_data import ProductionConfig, _sha256_file, completed_chunks, iter_corpus_documents, text_digest
    from gibc.tokenizer import eot_id, file_sha256, load_tokenizer

    pcfg = ProductionConfig.from_json(args.config)
    work = work_dir()
    pilot_dir = work / "data" / pcfg.pilot_name
    data_dir = work / "data" / pcfg.name
    chunks_dir = data_dir / "chunks"
    out_dir = work / "tokens" / pcfg.name
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; production token files are immutable (delete it deliberately to rebuild)")

    manifest = json.loads((data_dir / "acquisition_manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"]:
        raise ValueError("acquisition is not complete")
    chunks = completed_chunks(chunks_dir)  # re-verifies every chunk file hash and cursor continuity
    if [c["chunk"] for c in chunks] != [c["chunk"] for c in manifest["chunks"]]:
        raise ValueError("chunk directory does not match the acquisition manifest")
    pilot_manifest = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, info in pilot_manifest["outputs"].items():
        if _sha256_file(pilot_dir / name) != info["sha256"]:
            raise ValueError(f"pilot {name} does not match its manifest hash")
    tokenizer_path = REPO_ROOT / "results" / "tokenizer" / "tokenizer.json"
    tokenizer_sha = file_sha256(tokenizer_path)
    if tokenizer_sha != manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer differs from the one used to count acquisition tokens")
    vocab_size = ModelConfig.from_json(REPO_ROOT / "configs" / "model" / "gibc_43m.json").vocab_size

    free = shutil.disk_usage(work).free
    needed = pcfg.target_train_tokens * 2 * 1.03
    if free - needed < pcfg.min_free_gib * 2**30:
        raise RuntimeError(f"refusing: {free / 2**30:.1f} GiB free, need {needed / 2**30:.1f} GiB + {pcfg.min_free_gib} GiB floor")

    names = [c["chunk"] for c in chunks]
    t0 = time.monotonic()
    # Exact-text duplicate audit.
    train_counts: dict[bytes, int] = {}
    n_train = 0
    for doc in iter_corpus_documents(pilot_dir, chunks_dir, names, "train"):
        d = text_digest(doc["text"])
        train_counts[d] = train_counts.get(d, 0) + 1
        n_train += 1
    val_seen: dict[bytes, int] = {}
    removed = []
    n_val = 0
    for index, doc in enumerate(iter_corpus_documents(pilot_dir, chunks_dir, names, "val")):
        d = text_digest(doc["text"])
        val_seen[d] = val_seen.get(d, 0) + 1
        n_val += 1
        if d in train_counts:
            removed.append({"val_index": index, "id": doc["id"], "url": doc["url"], "text_sha256": d.hex()})
    removed_index = {r["val_index"] for r in removed}
    audit = {
        "method": "exact SHA-256 of UTF-8 text; no near-duplicate or semantic matching; not a decontamination claim",
        "train_documents": n_train,
        "train_unique_texts": len(train_counts),
        "train_duplicate_documents": n_train - len(train_counts),
        "train_texts_with_copies": sum(1 for c in train_counts.values() if c > 1),
        "val_documents": n_val,
        "val_unique_texts": len(val_seen),
        "val_duplicate_documents": n_val - len(val_seen),
        "val_documents_removed_as_train_overlap": len(removed),
        "rule": "a validation document whose exact text also occurs in train is removed from validation; "
                "train and other duplicates are kept",
        "removed": removed,
    }
    audit_seconds = time.monotonic() - t0
    print(f"audit ({audit_seconds:.0f} s): train {n_train:,} docs, {audit['train_duplicate_documents']:,} exact "
          f"duplicates; val {n_val:,} docs, {len(removed)} removed as train overlap", flush=True)
    del train_counts, val_seen

    tok = load_tokenizer(tokenizer_path)
    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    t1 = time.monotonic()
    train_info = write_token_file(tok, (d["text"] for d in iter_corpus_documents(pilot_dir, chunks_dir, names, "train")),
                                  partial / "train.bin", vocab_size)
    val_texts = (d["text"] for i, d in enumerate(iter_corpus_documents(pilot_dir, chunks_dir, names, "val"))
                 if i not in removed_index)
    val_info = write_token_file(tok, val_texts, partial / "val.bin", vocab_size)
    for split, info in (("train", train_info), ("val", val_info)):
        open_token_file(partial / f"{split}.bin", vocab_size, info["tokens"])  # re-check range + length from disk
    if train_info["tokens"] < pcfg.target_train_tokens:
        raise ValueError(f"train tokens {train_info['tokens']:,} below target {pcfg.target_train_tokens:,}")
    expected_train = manifest["train_tokens_with_eot_total"]
    if train_info["tokens"] != expected_train:
        raise ValueError(f"train tokens {train_info['tokens']:,} != acquisition count {expected_train:,}")

    meta = {
        "format": "gibc.tokens.v1",
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "eot_id": eot_id(tok),
        "eot_policy": "exactly one EOT appended after every document; documents concatenated; no BOS, no padding",
        "tokenizer_sha256": tokenizer_sha,
        "dataset": {"id": manifest["dataset"], "config_name": manifest["config_name"], "revision": manifest["revision"]},
        "sources": {
            "order": "pilot documents, then chunks in order",
            "pilot_manifest_sha256": manifest["pilot"]["manifest_sha256"],
            "acquisition_manifest_sha256": file_sha256(data_dir / "acquisition_manifest.json"),
            "chunks": [{"chunk": c["chunk"], "start": c["start"], "end": c["end"], "outputs": c["outputs"]} for c in chunks],
        },
        "exclusions": {
            "pilot": pilot_manifest["counts"],
            "continuation": {k: v for k, v in manifest["continuation_totals"].items() if k.startswith("counts.")},
        },
        "duplicate_audit": {k: v for k, v in audit.items() if k != "removed"},
        "splits": {"train": train_info, "val": val_info},
        "target_train_tokens": pcfg.target_train_tokens,
        "command": f"python scripts/build_production_tokens.py --config {args.config}",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenize_seconds": round(time.monotonic() - t1, 1),
    }
    (partial / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    partial.rename(out_dir)
    for name in ("train.bin", "val.bin", "meta.json"):
        os.chmod(out_dir / name, stat.S_IREAD)  # immutable: read-only
    results = REPO_ROOT / "results" / "data"
    shutil.copyfile(out_dir / "meta.json", results / f"{pcfg.name}_tokens_meta.json")
    (results / f"{pcfg.name}_duplicate_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8",
                                                               newline="\n")
    for split, info in meta["splits"].items():
        print(f"{split}: {info['documents']:,} docs, {info['tokens']:,} tokens, {info['bytes']:,} bytes, sha256 {info['sha256']}")
    print(f"wrote {out_dir} (read-only) in {time.monotonic() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
