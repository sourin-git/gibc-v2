"""SMOKE TEST of the four required lm-eval tasks on a LOCAL HF export. NOT OFFICIAL RESULTS.

Usage: python scripts/eval_lm_eval_smoke.py --export <dir> [--limit 20] [--batch-size 8]
Runs hellaswag, arc_easy, piqa, winogrande with lm-eval's standard HF backend, 0-shot (our documented
methodology; no organizer few-shot protocol is available), float32, max_length 512. It only exercises
dataset loading, prompt formatting, tokenization, likelihood scoring, batching, serialization and
CUDA inference. The limited-example scores must never be reported, tuned on, or treated as model quality.
Each logged request is also re-scored with OUR model + tokenizer to cross-check the harness path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande"]
WARNING = ("SMOKE TEST ONLY - NOT OFFICIAL RESULTS. Limited examples on a 1,024-step LR-sanity checkpoint; "
           "scores are not model quality and must not be reported, tuned on, or used for decisions.")


def request_pairs(sample: dict) -> list[tuple[str, str]]:
    args = sample["arguments"]
    if isinstance(args, dict):  # {"gen_args_0": {"arg_0": ctx, "arg_1": cont}, ...}
        return [(v["arg_0"], v["arg_1"]) for _, v in sorted(args.items(), key=lambda kv: int(kv[0].split("_")[-1]))]
    return [tuple(a) for a in args]


def main() -> int:
    parser = argparse.ArgumentParser(description="lm-eval smoke test (NOT OFFICIAL).")
    parser.add_argument("--export", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out")
    args = parser.parse_args()

    import lm_eval
    import torch
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable

    from gibc.checkpoint import load_model_from_checkpoint
    from gibc.paths import REPO_ROOT
    from gibc.scoring import continuation_logprobs, encode_pair, ours_logits_fn
    from gibc.tokenizer import encode_text, load_tokenizer

    torch.set_float32_matmul_precision("highest")
    export_dir = Path(args.export).resolve()
    manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else (REPO_ROOT / "results" / "eval_preflight" /
                                              f"SMOKE_NOT_OFFICIAL_lm_eval_limit{args.limit}_{export_dir.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README_NOT_OFFICIAL.txt").write_text(WARNING + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    lm = HFLM(pretrained=str(export_dir), tokenizer=str(export_dir), device="cuda", dtype="float32",
              max_length=512, batch_size=args.batch_size)
    load_seconds = time.perf_counter() - t0
    ours, _ = load_model_from_checkpoint(Path(manifest["source_checkpoint"]), "cuda")
    our_tok = load_tokenizer(REPO_ROOT / "results" / "tokenizer" / "tokenizer.json")

    summary = {"warning": WARNING, "export": str(export_dir), "source_checkpoint": manifest["source_checkpoint"],
               "source_checkpoint_sha256": manifest["source_checkpoint_sha256"], "limit": args.limit,
               "num_fewshot": 0, "dtype": "float32", "max_length": lm.max_length, "batch_size": args.batch_size,
               "model_load_seconds": round(load_seconds, 1), "tasks": {}}
    for task in TASKS:
        t = time.perf_counter()
        res = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=0, limit=args.limit, log_samples=True,
                                      bootstrap_iters=0)
        seconds = time.perf_counter() - t
        samples = res["samples"][task]
        # Cross-check: harness log-likelihoods vs our own model/tokenizer on every logged request.
        worst = 0.0
        n_requests = 0
        for s in samples:
            harness = [r[0] if isinstance(r, (list, tuple)) else r for r in s["filtered_resps"]]
            harness = [h[0] if isinstance(h, (list, tuple)) else h for h in harness]
            for (context, cont), h in zip(request_pairs(s), harness):
                ctx_ids, cont_ids = encode_pair(lambda x: encode_text(our_tok, x), context, cont)
                ll = continuation_logprobs(ours_logits_fn(ours), ctx_ids, cont_ids, 512, device="cuda").sum().item()
                worst = max(worst, abs(ll - float(h)))
                n_requests += 1
        record = {"warning": WARNING, "task": task, "seconds": seconds, "examples": len(samples),
                  "requests_crosschecked": n_requests, "max_abs_loglik_diff_vs_source_model": worst,
                  "results_NOT_OFFICIAL": res["results"][task], "n_samples": res["n-samples"].get(task),
                  "versions": res["versions"].get(task), "config": res["configs"].get(task),
                  "lm_eval_config": res.get("config"), "samples": samples}
        (out_dir / f"{task}.json").write_text(json.dumps(record, indent=2, default=handle_non_serializable) + "\n",
                                             encoding="utf-8", newline="\n")
        summary["tasks"][task] = {"examples": len(samples), "seconds": round(seconds, 1), "requests": n_requests,
                                  "max_abs_loglik_diff_vs_source_model": worst,
                                  "version": res["versions"].get(task),
                                  "metrics_NOT_OFFICIAL": {k: v for k, v in res["results"][task].items()
                                                           if k.startswith("acc")}}
        print(f"{task}: {len(samples)} examples, {n_requests} requests, {seconds:.1f} s, "
              f"harness-vs-source max |diff| {worst:.2e}  [NOT OFFICIAL]", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=handle_non_serializable) + "\n",
                                          encoding="utf-8", newline="\n")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
