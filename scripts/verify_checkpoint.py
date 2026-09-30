"""Verify a training checkpoint: SHA-256, strict load, unique trainable parameters, weight tying, run state.

Usage: python scripts/verify_checkpoint.py --checkpoint <path.pt> [--out <json>]
Exits 1 unless the checkpoint loads strictly, has 42,968,576 unique trainable parameters and a tied head.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gibc.checkpoint import load_model_from_checkpoint
from gibc.data import file_sha256_streaming
from gibc.model import parameter_report, weights_are_tied

EXPECTED_PARAMS = 42_968_576


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a training checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    path = Path(args.checkpoint).resolve()
    model, ckpt = load_model_from_checkpoint(path)  # strict load_state_dict
    report = parameter_report(model)
    result = {
        "checkpoint": str(path), "sha256": file_sha256_streaming(path), "bytes": path.stat().st_size,
        "format": ckpt["format"], "state": ckpt["state"], "created_utc": ckpt["created_utc"],
        "source": ckpt["source"], "data": ckpt["data"], "sampler_info": ckpt.get("sampler_info"),
        "model_config": ckpt["model_config"], "run_name": ckpt["train_config"]["run_name"],
        "parameter_report": report, "unique_trainable_parameters": report["total unique trainable"],
        "weights_tied": weights_are_tied(model),
    }
    ok = result["unique_trainable_parameters"] == EXPECTED_PARAMS and result["weights_tied"]
    result["passed"] = ok
    text = json.dumps(result, indent=2, default=str) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    s = ckpt["state"]
    print(f"checkpoint {path.name}: sha256 {result['sha256']}")
    print(f"  step {s['step']:,}  tokens {s['tokens']:,}  microsteps {s['microsteps']:,}  skipped {s['skipped_steps']}")
    print(f"  unique trainable parameters {result['unique_trainable_parameters']:,}  tied {result['weights_tied']}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
