"""Tokenize an acquired sample into flat uint16 token files under $GIBC_WORK_DIR/tokens/<name>/.

Usage: python scripts/prepare_tokens.py --data-name pilot
Reads $GIBC_WORK_DIR/data/<name>/{train,val}.jsonl and writes train.bin, val.bin and meta.json;
copies meta.json to results/data/<name>_tokens_meta.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

from gibc.config import ModelConfig
from gibc.data import prepare_token_splits
from gibc.paths import REPO_ROOT, work_dir
from gibc.tokenizer import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize an acquired sample into uint16 token files.")
    parser.add_argument("--data-name", required=True, help="acquired sample under $GIBC_WORK_DIR/data/")
    parser.add_argument("--model-config", default=str(REPO_ROOT / "configs" / "model" / "gibc_43m.json"))
    parser.add_argument("--tokenizer", default=str(REPO_ROOT / "results" / "tokenizer" / "tokenizer.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = work_dir() / "data" / args.data_name
    out_dir = work_dir() / "tokens" / args.data_name
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to rebuild")
        shutil.rmtree(out_dir)

    tokenizer_meta = json.loads((REPO_ROOT / "results" / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    if file_sha256(args.tokenizer) != tokenizer_meta["sha256"]:
        raise ValueError("tokenizer.json does not match the sha256 in tokenizer_meta.json")

    t0 = time.monotonic()
    meta = prepare_token_splits(
        data_dir, out_dir, REPO_ROOT / "results" / "tokenizer" / "tokenizer.json",
        ModelConfig.from_json(args.model_config).vocab_size,
        command=f"python scripts/prepare_tokens.py --data-name {args.data_name}",
    )
    copy = REPO_ROOT / "results" / "data" / f"{args.data_name}_tokens_meta.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_dir / "meta.json", copy)

    print(f"wrote {out_dir} in {time.monotonic() - t0:.1f} s (dtype {meta['dtype']}, eot id {meta['eot_id']})")
    for split, s in meta["splits"].items():
        print(f"  {split}: {s['documents']:,} docs  {s['tokens']:,} tokens (incl. {s['eot_tokens']:,} EOT)  "
              f"{s['bytes']:,} bytes  sha256 {s['sha256'][:16]}...")
    print(f"meta copied to {copy.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
