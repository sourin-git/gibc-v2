"""Generate text from a training checkpoint.

Usage: python scripts/generate.py --checkpoint <path> --prompt "Some text" [--max-new-tokens 40]
       [--temperature 0.8] [--top-k 50] [--seed 0]    (temperature 0 = greedy)
"""

from __future__ import annotations

import argparse
import sys

import torch

from gibc.checkpoint import load_checkpoint
from gibc.config import ModelConfig
from gibc.generate import generate
from gibc.model import CausalLM
from gibc.paths import REPO_ROOT
from gibc.tokenizer import decode, encode_text, eot_id, file_sha256, load_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample from a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer", default=str(REPO_ROOT / "results" / "tokenizer" / "tokenizer.json"))
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    if file_sha256(args.tokenizer) != ckpt["data"]["tokenizer_sha256"]:
        raise ValueError("tokenizer does not match the one the checkpoint's training data was built with")
    model = CausalLM(ModelConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    tok = load_tokenizer(args.tokenizer)

    prompt_ids = encode_text(tok, args.prompt)
    out = generate(model, prompt_ids, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                   stop_id=eot_id(tok), generator=torch.Generator().manual_seed(args.seed))
    print(f"checkpoint step {ckpt['state']['step']}, tokens {ckpt['state']['tokens']:,}")
    print(args.prompt + decode(tok, out[len(prompt_ids):]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
