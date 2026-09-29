"""Train our byte-level BPE tokenizer on an acquired sample's TRAIN split, verify it, and report stats.

Usage: python scripts/train_tokenizer.py --data-name pilot
Reads $GIBC_WORK_DIR/data/<name>/{train,val}.jsonl and writes to results/tokenizer/:
tokenizer.json, tokenizer_meta.json and <name>_token_stats.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import tokenizers

from gibc.acquire import read_jsonl_documents
from gibc.config import ModelConfig
from gibc.paths import REPO_ROOT, work_dir
from gibc.tokenizer import (
    EOT_TOKEN,
    SPECIAL_TOKENS,
    corpus_token_stats,
    decode,
    encode_text,
    eot_id,
    file_sha256,
    load_tokenizer,
    train_tokenizer,
)

RELOAD_CHECK_DOCS = 2000


def texts(path: Path):
    return (doc["text"] for doc in read_jsonl_documents(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and verify the byte-level BPE tokenizer.")
    parser.add_argument("--data-name", required=True, help="acquired sample under $GIBC_WORK_DIR/data/")
    parser.add_argument("--model-config", default=str(REPO_ROOT / "configs" / "model" / "gibc_43m.json"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "tokenizer"))
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    data_dir = work_dir() / "data" / args.data_name
    out_dir = Path(args.out_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vocab_size = ModelConfig.from_json(args.model_config).vocab_size
    train_path, val_path = data_dir / "train.jsonl", data_dir / "val.jsonl"

    print(f"training on {train_path} ({manifest['counts']['accepted_train_documents']:,} documents), "
          f"vocab_size={vocab_size}")
    tok = train_tokenizer(
        texts(train_path),
        vocab_size=vocab_size,
        min_frequency=args.min_frequency,
        length=manifest["counts"]["accepted_train_documents"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out_dir / "tokenizer.json"
    tok.save(str(tokenizer_path))

    # Save/reload must be lossless: identical ids on real documents (val + a train prefix).
    reloaded = load_tokenizer(tokenizer_path)
    check_docs = list(texts(val_path)) + list(itertools.islice(texts(train_path), RELOAD_CHECK_DOCS))
    mismatches = sum(encode_text(tok, t) != encode_text(reloaded, t) for t in check_docs)
    roundtrip_failures = sum(decode(reloaded, encode_text(reloaded, t)) != t for t in check_docs)
    if mismatches or roundtrip_failures:
        raise RuntimeError(f"reload id mismatches: {mismatches}, round-trip failures: {roundtrip_failures}")

    special = {token: reloaded.token_to_id(token) for token in SPECIAL_TOKENS}
    meta = {
        "tokenizer_file": tokenizer_path.name,
        "sha256": file_sha256(tokenizer_path),
        "vocab_size": reloaded.get_vocab_size(with_added_tokens=True),
        "model": "BPE, byte-level (full 256-symbol initial alphabet), trained from scratch",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False, use_regex=True)",
        "normalizer": None,
        "decoder": "ByteLevel",
        "post_processor": None,
        "special_tokens": special,
        "eot_token": EOT_TOKEN,
        "eot_id": eot_id(reloaded),
        "bos_token": None,
        "pad_token": None,
        "unk_token": None,
        "policies": {
            "document_boundary": "EOT id appended exactly once after every document; no BOS",
            "literal_special_text": (
                "raw text containing the EOT spelling is encoded as ordinary bytes "
                "(tokenizers encode_special_tokens=True, set by gibc.tokenizer.load_tokenizer; "
                "not persisted in tokenizer.json)"
            ),
            "pad": (
                "no PAD token; if batched evaluation needs a pad id, reuse the EOT id with padded "
                "positions masked; add a dedicated PAD (inside the 24,000) only if that proves insufficient"
            ),
        },
        "trainer": {"algorithm": "tokenizers.trainers.BpeTrainer", "min_frequency": args.min_frequency},
        "training_corpus": {
            "data_name": args.data_name,
            "split": "train",
            "documents": manifest["counts"]["accepted_train_documents"],
            "text_bytes": manifest["text_bytes"]["accepted_train"],
            "source_manifest_sha256": file_sha256(manifest_path),
            "train_jsonl_sha256": manifest["outputs"]["train.jsonl"]["sha256"],
        },
        "reload_check": {"documents": len(check_docs), "id_mismatches": 0, "roundtrip_failures": 0},
        "tokenizers_version": tokenizers.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")

    stats = {split: corpus_token_stats(reloaded, texts(data_dir / f"{split}.jsonl")) for split in ("train", "val")}
    stats_path = out_dir / f"{args.data_name}_token_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"vocab_size: {meta['vocab_size']}  special tokens: {special}  sha256: {meta['sha256']}")
    print(f"save/reload: identical ids and exact round trip on {len(check_docs):,} documents")
    for split, s in stats.items():
        print(
            f"{split}: docs {s['documents']:,}  chars {s['characters']:,}  bytes {s['utf8_bytes']:,}  "
            f"tokens {s['tokens']:,} (+EOT {s['tokens_with_eot']:,})  "
            f"avg tok/doc {s['avg_tokens_per_document']:.1f}  bytes/tok {s['bytes_per_token']:.3f}  "
            f"chars/tok {s['characters_per_token']:.3f}"
        )
    print(f"wrote {tokenizer_path}, tokenizer_meta.json, {stats_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
