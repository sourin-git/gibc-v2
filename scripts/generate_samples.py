"""Generate a FIXED set of samples from a checkpoint and record prompts, settings and outputs.

Usage: python scripts/generate_samples.py --checkpoint <path.pt> --out <json>
Every prompt is run exactly once greedy and once sampled with a fixed seed; nothing is regenerated
or selected. Uses the same path as scripts/generate.py (gibc.generate.generate, our tokenizer).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from gibc.checkpoint import load_model_from_checkpoint
from gibc.data import file_sha256_streaming
from gibc.generate import generate
from gibc.paths import REPO_ROOT
from gibc.tokenizer import decode, encode_text, eot_id, load_tokenizer

PROMPTS = {
    "factual": "The capital of France is",
    "encyclopedic": "Photosynthesis is the process by which",
    "scientific_explanation": "Water boils at a lower temperature at high altitude because",
    "simple_reasoning": "If a train travels 60 miles in one hour, then in three hours it travels",
    "open_ended": "Once upon a time, in a small village by the sea,",
}
SETTINGS = {"greedy": {"temperature": 0.0, "top_k": None},
            "sampled": {"temperature": 0.8, "top_k": 50, "seed": 1234}}
MAX_NEW_TOKENS = 60


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed-prompt generation samples.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    path = Path(args.checkpoint).resolve()
    model, ckpt = load_model_from_checkpoint(path, "cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(REPO_ROOT / "results" / "tokenizer" / "tokenizer.json")
    rows = []
    for name, prompt in PROMPTS.items():
        ids = encode_text(tok, prompt)
        row = {"name": name, "prompt": prompt}
        for label, s in SETTINGS.items():
            gen = torch.Generator().manual_seed(s["seed"]) if "seed" in s else None
            out = generate(model, ids, MAX_NEW_TOKENS, temperature=s["temperature"], top_k=s["top_k"],
                           stop_id=eot_id(tok), generator=gen)
            row[label] = decode(tok, out[len(ids):])
        rows.append(row)
        print(f"[{name}] {prompt!r}\n  greedy : {row['greedy']!r}\n  sampled: {row['sampled']!r}")
    result = {"checkpoint": str(path), "checkpoint_sha256": file_sha256_streaming(path),
              "step": ckpt["state"]["step"], "max_new_tokens": MAX_NEW_TOKENS, "settings": SETTINGS,
              "stop": "EOT id 0", "note": "fixed prompts, one output per setting, no retries or selection",
              "samples": rows}
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
