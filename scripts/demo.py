"""Terminal demo: local text generation from the FINAL GIBC-43M checkpoint (our own weights, no external model).

Usage:
  python scripts/demo.py --prompt "Photosynthesis is the process by which"
  python scripts/demo.py --fixed            # the fixed prompt set used in results (greedy + seeded sampling)
  python scripts/demo.py                    # interactive: type prompts, empty line to quit
Options: --checkpoint <path.pt> (default: the final production checkpoint), --max-new-tokens 60,
         --temperature 0.8 (0 = greedy), --top-k 50, --seed 1234, --skip-hash (skip the SHA-256 check)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from gibc.checkpoint import load_model_from_checkpoint
from gibc.data import file_sha256_streaming
from gibc.generate import generate
from gibc.model import parameter_report, weights_are_tied
from gibc.paths import REPO_ROOT, work_dir
from gibc.tokenizer import decode, encode_text, eot_id, load_tokenizer

FINAL_RESULTS = REPO_ROOT / "results" / "FINAL_EVALUATION_production_step_0061036" / "final_results.json"
FIXED_PROMPTS = [
    "Photosynthesis is the process by which",
    "Water boils at a lower temperature at high altitude because",
    "Once upon a time, in a small village by the sea,",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="GIBC-43M local generation demo.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    final = json.loads(FINAL_RESULTS.read_text(encoding="utf-8"))
    path = Path(args.checkpoint) if args.checkpoint else work_dir() / "runs" / "production" / "checkpoints" / "final_step_0061036.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("GIBC-43M  |  local inference from OUR checkpoint (trained from scratch; no external model/API)")
    print("=" * 78)
    t0 = time.perf_counter()
    if not args.skip_hash:
        sha = file_sha256_streaming(path)
        expected = final["checkpoint"]["sha256"]
        print(f"checkpoint : {path}")
        print(f"sha256     : {sha}  [{'matches final_results.json' if sha == expected else 'DOES NOT MATCH final_results.json'}]")
    model, ckpt = load_model_from_checkpoint(path, device)
    tok = load_tokenizer(REPO_ROOT / "results" / "tokenizer" / "tokenizer.json")
    report = parameter_report(model)
    cfg = model.config
    print(f"trained    : step {ckpt['state']['step']:,}, {ckpt['state']['tokens']:,} target tokens")
    print(f"parameters : {report['total unique trainable']:,} unique trainable  (limit 50,000,000; "
          f"embedding/head tied: {weights_are_tied(model)})")
    print(f"model      : {cfg.n_layers} layers, d_model {cfg.d_model}, {cfg.n_heads} heads, d_ff {cfg.d_ff}, "
          f"vocab {cfg.vocab_size:,}, context {cfg.context_length}")
    print(f"device     : {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}  (loaded in {time.perf_counter() - t0:.1f} s)")
    mode = "greedy" if args.temperature == 0 else f"temperature {args.temperature}, top-k {args.top_k}, seed {args.seed}"
    print(f"decoding   : {mode}, up to {args.max_new_tokens} new tokens, stop at <|endoftext|>")
    print("note       : small 43M-parameter model; output is often repetitive or factually wrong.")
    print("-" * 78)

    def run(prompt: str, temperature: float) -> None:
        ids = encode_text(tok, prompt)
        gen = torch.Generator().manual_seed(args.seed) if temperature > 0 else None
        t = time.perf_counter()
        out = generate(model, ids, args.max_new_tokens, temperature=temperature, top_k=args.top_k,
                       stop_id=eot_id(tok), generator=gen)
        new = out[len(ids):]
        label = "greedy" if temperature == 0 else f"sampled T={temperature}"
        print(f"\nPROMPT : {prompt}\nOUTPUT ({label}, {len(new)} tokens, {time.perf_counter() - t:.2f} s):")
        bold, reset = ("\033[1m", "\033[0m") if sys.stdout.isatty() else ("", "")  # highlight the model's text
        print(f"{prompt}{bold}{decode(tok, new)}{reset}")

    if args.fixed:
        for p in FIXED_PROMPTS:
            run(p, 0.0)
            run(p, args.temperature if args.temperature > 0 else 0.8)
    elif args.prompt:
        run(args.prompt, args.temperature)
    else:
        while True:
            try:
                prompt = input("\nprompt> ").strip()
            except EOFError:
                break
            if not prompt:
                break
            run(prompt, args.temperature)
    return 0


if __name__ == "__main__":
    sys.exit(main())
