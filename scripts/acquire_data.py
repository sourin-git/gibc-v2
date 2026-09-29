"""Acquire a bounded FineWeb-Edu sample into $GIBC_WORK_DIR/data/<name>/ (Wikipedia-source exclusion applied).

Usage: python scripts/acquire_data.py --config configs/data/pilot.json --name pilot
Writes train.jsonl, val.jsonl and manifest.json, and copies the manifest to results/data/<name>_manifest.json.
"""

from __future__ import annotations

import argparse
import shutil
import sys

from gibc.paths import REPO_ROOT, require_hf_home, work_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded FineWeb-Edu acquisition.")
    parser.add_argument("--config", required=True, help="acquisition config JSON")
    parser.add_argument("--name", required=True, help="output name under $GIBC_WORK_DIR/data/")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()

    out_dir = work_dir() / "data" / args.name
    print(f"GIBC_WORK_DIR ok; HF_HOME = {require_hf_home()}")

    from gibc.acquire import AcquisitionConfig, acquire  # after the HF_HOME check

    cfg = AcquisitionConfig.from_json(args.config)
    manifest = acquire(cfg, out_dir, overwrite=args.overwrite)

    results_copy = REPO_ROOT / "results" / "data" / f"{args.name}_manifest.json"
    results_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_dir / "manifest.json", results_copy)

    counts, text_bytes = manifest["counts"], manifest["text_bytes"]
    print(f"output: {out_dir}")
    print(f"stop reason: {manifest['stop_reason']}  elapsed: {manifest['elapsed_seconds']} s")
    for key, value in counts.items():
        print(f"  {key:<30} {value:>12,}")
    for key, value in text_bytes.items():
        if key != "note":
            print(f"  text_bytes.{key:<19} {value:>12,}")
    print(f"manifest copied to {results_copy.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
