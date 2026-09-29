"""Stage 2 smoke test: tokenizer <-> model integration, a small CUDA forward/backward, and generation.

Usage: python scripts/smoke_model.py
The model has RANDOM weights; generated text is meaningless. The batch/sequence here is
deliberately small: this is not a memory or throughput benchmark. Writes results/smoke/model_smoke.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone

import torch

from gibc.config import ModelConfig
from gibc.generate import generate
from gibc.model import CausalLM, weights_are_tied
from gibc.paths import REPO_ROOT
from gibc.tokenizer import decode, encode_text, eot_id, load_tokenizer

PROMPT = "The water cycle describes how water"


def main() -> int:
    parser = argparse.ArgumentParser(description="Model smoke test (random weights).")
    parser.add_argument("--model-config", default=str(REPO_ROOT / "configs" / "model" / "gibc_43m.json"))
    parser.add_argument("--tokenizer", default=str(REPO_ROOT / "results" / "tokenizer" / "tokenizer.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = ModelConfig.from_json(args.model_config)
    tok = load_tokenizer(args.tokenizer)
    result: dict = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "note": "random-initialized weights; outputs are not meaningful"}

    # Tokenizer <-> model integration.
    tok_vocab = tok.get_vocab_size(with_added_tokens=True)
    prompt_ids = encode_text(tok, PROMPT)
    torch.manual_seed(args.seed)
    model = CausalLM(cfg)
    with torch.no_grad():
        logits, _ = model.eval()(torch.tensor([prompt_ids]))
    integration = {
        "tokenizer_vocab": tok_vocab,
        "model_vocab": cfg.vocab_size,
        "eot_id": eot_id(tok),
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "logits_shape": list(logits.shape),
    }
    assert tok_vocab == cfg.vocab_size == 24_000 and integration["eot_id"] == 0
    assert list(logits.shape) == [1, len(prompt_ids), cfg.vocab_size]
    result["integration"] = integration
    print(f"integration: tokenizer vocab {tok_vocab} == model vocab {cfg.vocab_size}; eot id "
          f"{integration['eot_id']}; prompt -> {len(prompt_ids)} ids -> logits {tuple(logits.shape)}")

    # CUDA forward/backward.
    if not torch.cuda.is_available():
        print("CUDA not available: skipping CUDA smoke")
        result["cuda"] = None
    else:
        device = torch.device("cuda")
        model = model.to(device).train()
        torch.cuda.reset_peak_memory_stats(device)
        gen = torch.Generator().manual_seed(args.seed)
        ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len + 1), generator=gen).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(ids[:, :-1], ids[:, 1:])
        loss.backward()
        torch.cuda.synchronize(device)
        grads_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters())
        cuda = {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "precision": "bf16 autocast, fp32 parameters and gradients",
            "loss": loss.item(),
            "ln_vocab": math.log(cfg.vocab_size),
            "grads_finite": grads_finite,
            "weights_tied_on_cuda": weights_are_tied(model),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
        assert math.isfinite(cuda["loss"]) and grads_finite and cuda["weights_tied_on_cuda"]
        result["cuda"] = cuda
        print(f"cuda: {cuda['gpu']}  B={args.batch_size} T={args.seq_len}  {cuda['precision']}")
        print(f"      loss {cuda['loss']:.4f} (ln V = {cuda['ln_vocab']:.4f})  grads finite {grads_finite}")
        print(f"      peak allocated {cuda['peak_allocated_mib']:.1f} MiB  peak reserved {cuda['peak_reserved_mib']:.1f} MiB")
        model.zero_grad(set_to_none=True)

    # Generation through our tokenizer (random weights).
    greedy = generate(model, prompt_ids, max_new_tokens=20, temperature=0, stop_id=eot_id(tok))
    sampled = generate(model, prompt_ids, max_new_tokens=20, temperature=1.0, top_k=50, stop_id=eot_id(tok),
                       generator=torch.Generator().manual_seed(args.seed))
    result["generation"] = {
        "device": str(next(model.parameters()).device),
        "greedy_ids": greedy[len(prompt_ids):],
        "greedy_text": decode(tok, greedy[len(prompt_ids):]),
        "sampled_ids": sampled[len(prompt_ids):],
        "sampled_text": decode(tok, sampled[len(prompt_ids):]),
        "settings": {"max_new_tokens": 20, "sampled": {"temperature": 1.0, "top_k": 50, "seed": args.seed}},
    }
    print("generation (RANDOM WEIGHTS - not meaningful):")
    print(f"  greedy : {PROMPT!r} + {result['generation']['greedy_text']!r}")
    print(f"  sampled: {PROMPT!r} + {result['generation']['sampled_text']!r}")

    out = REPO_ROOT / "results" / "smoke" / "model_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
