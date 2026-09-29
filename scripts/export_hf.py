"""Export any compatible project checkpoint to a local Hugging Face Llama directory.

Usage: python scripts/export_hf.py --checkpoint <path.pt> [--out <dir>]
Default output: $GIBC_WORK_DIR/exports/<run_name>_step_<step>/. Nothing is downloaded. Verify the
export in a FRESH process afterwards with scripts/verify_hf_export.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gibc.checkpoint import load_model_from_checkpoint
from gibc.hf_export import export
from gibc.paths import REPO_ROOT, work_dir
from gibc.tokenizer import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a project checkpoint to local HF Llama format.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    model, ckpt = load_model_from_checkpoint(ckpt_path)
    run_name, step = ckpt["train_config"]["run_name"], ckpt["state"]["step"]
    out_dir = Path(args.out) if args.out else work_dir() / "exports" / f"{run_name}_step_{step:07d}"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    tokenizer_json = REPO_ROOT / "results" / "tokenizer" / "tokenizer.json"
    tokenizer_sha = file_sha256(tokenizer_json)
    if tokenizer_sha != ckpt["data"]["tokenizer_sha256"]:
        raise ValueError("tokenizer.json differs from the tokenizer the checkpoint was trained with")

    manifest = export(model, tokenizer_json, out_dir, meta={
        "source_checkpoint": str(ckpt_path), "source_checkpoint_sha256": file_sha256(ckpt_path),
        "run_name": run_name, "step": step, "tokens_trained": ckpt["state"]["tokens"],
        "model_config": ckpt["model_config"], "tokenizer_sha256": tokenizer_sha,
    })
    print(json.dumps({k: manifest[k] for k in ("run_name", "step", "unique_trainable_parameters", "weights_tied",
                                              "tokenizer")}, indent=2))
    print(f"weight mapping: {manifest['weight_mapping']['names_compared']} names compared, "
          f"all equal = {manifest['weight_mapping']['all_equal']}")
    print(f"exported to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
