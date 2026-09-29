"""WikiText-103 test token perplexity under the documented project methodology (see gibc/wikitext.py).

Usage: python scripts/eval_wikitext103.py --export <dir> [--max-targets N] [--revision SHA]
With --max-targets the run is a bounded SMOKE TEST (first N target tokens only): NOT OFFICIAL.
The HF export (the same artifact lm-eval uses) is scored on CUDA in fp32. For verification the run
repeats the scoring to check determinism, and compares window NLLs with our source model on
sampled windows.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

SMOKE_WARNING = "SMOKE TEST ONLY - bounded subset of WikiText-103 test; NOT an official or reportable perplexity."


def main() -> int:
    parser = argparse.ArgumentParser(description="WikiText-103 token perplexity (project methodology).")
    parser.add_argument("--export", required=True)
    parser.add_argument("--max-targets", type=int, help="smoke test: score only the first N target tokens")
    parser.add_argument("--revision", help="dataset commit SHA (default: resolve the current one and record it)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out")
    args = parser.parse_args()

    import torch
    from huggingface_hub import HfApi

    from gibc import wikitext
    from gibc.checkpoint import load_model_from_checkpoint
    from gibc.hf_export import load_export
    from gibc.paths import REPO_ROOT
    from gibc.scoring import hf_logits_fn, ours_logits_fn
    from gibc.tokenizer import encode_text, eot_id, file_sha256, load_tokenizer

    torch.set_float32_matmul_precision("highest")
    revision = args.revision or HfApi().dataset_info(wikitext.DATASET).sha
    export_dir = Path(args.export).resolve()
    manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
    tok_path = REPO_ROOT / "results" / "tokenizer" / "tokenizer.json"
    tok = load_tokenizer(tok_path)

    text, text_info = wikitext.load_text(revision)
    t_tok = time.perf_counter()
    ids = encode_text(tok, text)
    tokenize_seconds = time.perf_counter() - t_tok
    n_full = len(ids)
    if 0 in ids:
        raise ValueError("EOT id appears inside the tokenized WikiText text")
    n = min(n_full, args.max_targets) if args.max_targets else n_full
    stream = [eot_id(tok)] + ids[:n]

    hf, _ = load_export(export_dir, "cuda")
    run1 = wikitext.score_stream(hf_logits_fn(hf), stream, batch_size=args.batch_size, device="cuda")
    run2 = wikitext.score_stream(hf_logits_fn(hf), stream, batch_size=args.batch_size, device="cuda")

    # Source-vs-HF parity on sampled windows (NLL of the scored targets of each window).
    ours, _ = load_model_from_checkpoint(Path(manifest["source_checkpoint"]), "cuda")
    windows = wikitext.rolling_windows(n)
    picks = sorted({0, len(windows) // 2, len(windows) - 1})
    parity = []
    for k in picks:
        s, e, first = windows[k]
        sub = stream[s:e + 1]  # a stream whose first token is context; score the same targets
        nll = {}
        for label, fn in (("hf", hf_logits_fn(hf)), ("ours", ours_logits_fn(ours))):
            inp = torch.tensor([sub[:-1]], device="cuda")
            logp = torch.log_softmax(fn(inp)[0].float(), -1).gather(1, torch.tensor(sub[1:], device="cuda")[:, None])
            nll[label] = -logp[first - 1 - s:].double().sum().item()
        parity.append({"window": k, "start": s, "end": e, "scored": e - first + 1, **nll,
                       "abs_diff": abs(nll["hf"] - nll["ours"])})

    per_window = run1["seconds"] / run1["windows"]
    full_windows = len(wikitext.rolling_windows(n_full))
    result = {
        "label": wikitext.METHOD_LABEL,
        "warning": SMOKE_WARNING if n < n_full else None,
        "methodology": inspect.getdoc(wikitext),
        "dataset": {"id": wikitext.DATASET, "config": wikitext.CONFIG, "split": wikitext.SPLIT, "revision": revision,
                    **text_info},
        "tokenizer_sha256": file_sha256(tok_path),
        "export": str(export_dir), "source_checkpoint": manifest["source_checkpoint"],
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "window": {"max_len": wikitext.MAX_LEN, "stride": wikitext.STRIDE, "prefix": "one EOT (id 0), context only"},
        "full_test_tokens": n_full, "scored_tokens": run1["scored_tokens"], "windows": run1["windows"],
        "total_nll": run1["total_nll"], "token_perplexity": run1["token_perplexity"],
        "deterministic_repeat": run1["total_nll"] == run2["total_nll"],
        "repeat_total_nll": run2["total_nll"], "seconds": run1["seconds"], "tokenize_seconds": tokenize_seconds,
        "source_vs_hf_window_parity": parity,
        "ESTIMATED_full_seconds": per_window * full_windows, "full_windows": full_windows,
    }
    name = f"SMOKE_NOT_OFFICIAL_wikitext103_{n}targets_{export_dir.name}" if n < n_full else f"wikitext103_{export_dir.name}"
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "eval_preflight" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{'SMOKE (NOT OFFICIAL): ' if n < n_full else ''}scored {run1['scored_tokens']:,} of {n_full:,} test tokens in "
          f"{run1['windows']} windows, {run1['seconds']:.1f} s; token ppl {run1['token_perplexity']:.3f}; "
          f"deterministic {result['deterministic_repeat']}; max window parity |diff| "
          f"{max(p['abs_diff'] for p in parity):.2e} nats; ESTIMATED full {result['ESTIMATED_full_seconds']:.0f} s "
          f"({full_windows} windows); revision {revision}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
