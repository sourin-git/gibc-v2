"""Context-length / truncation audit of the four required tasks + runtime ESTIMATES. Computes no scores.

Usage: python scripts/audit_eval_requests.py --export <dir> [--timing-sample 400]
Builds ALL requests of hellaswag, arc_easy, piqa and winogrande (0-shot) through lm-eval 0.4.13, then
tokenizes each (context, continuation) pair with the harness's own `_encode_pair`, i.e. with our
exported tokenizer. It reports lengths against the harness rule: input = (ctx + cont)[-(max_length + 1):][:-1],
a left-truncation of the context, with an assert that len(cont) <= max_length. It then times
`loglikelihood` on a random request sample (log-probs discarded; no accuracy is computed) and
extrapolates the full runtime. These are ESTIMATES.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluation request audit (no scores).")
    parser.add_argument("--export", required=True)
    parser.add_argument("--timing-sample", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.utils import handle_non_serializable

    from gibc.paths import REPO_ROOT

    torch.set_float32_matmul_precision("highest")
    export_dir = Path(args.export).resolve()
    lm = HFLM(pretrained=str(export_dir), tokenizer=str(export_dir), device="cuda", dtype="float32",
              max_length=512, batch_size=args.batch_size)
    tasks = get_task_dict(TASKS, TaskManager())
    report = {"note": "request audit + runtime ESTIMATES; no accuracy computed", "max_length": lm.max_length,
              "truncation_rule": "input = (context_enc + continuation_enc)[-(max_length+1):][:-1] (left-truncate context); "
                                 "assert len(continuation_enc) <= max_length",
              "batch_size": args.batch_size, "dtype": "float32", "tasks": {}}
    rng = random.Random(0)
    for name in TASKS:
        task = tasks[name]
        task.build_all_requests(limit=None, rank=0, world_size=1)
        instances = task.instances
        lengths, cont_lengths, empty_ctx = [], [], 0
        for inst in instances:
            context, cont = inst.args
            if context == "":
                empty_ctx += 1
                ctx_enc, cont_enc = [lm.prefix_token_id], lm.tok_encode(cont, add_special_tokens=False)
            else:
                ctx_enc, cont_enc = lm._encode_pair(context, cont)
            lengths.append(len(ctx_enc) + len(cont_enc))
            cont_lengths.append(len(cont_enc))
        lengths, cont_lengths = np.array(lengths), np.array(cont_lengths)
        needs_trunc = int((lengths > lm.max_length + 1).sum())

        sample = rng.sample(instances, min(args.timing_sample, len(instances)))
        lm.loglikelihood(sample[:16], disable_tqdm=True)  # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lm.loglikelihood(sample, disable_tqdm=True)
        torch.cuda.synchronize()
        per_request = (time.perf_counter() - t0) / len(sample)
        cfg = task.config
        report["tasks"][name] = {
            "dataset": f"{cfg.dataset_path}/{cfg.dataset_name}", "split_scored": cfg.test_split or cfg.validation_split,
            "version": (cfg.metadata or {}).get("version"), "documents": len(task.eval_docs),
            "requests": len(instances), "empty_context_requests": empty_ctx,
            "tokens_ctx_plus_cont": {"max": int(lengths.max()), "p50": float(np.percentile(lengths, 50)),
                                     "p99": float(np.percentile(lengths, 99)), "mean": float(lengths.mean())},
            "continuation_tokens_max": int(cont_lengths.max()),
            "requests_needing_truncation": needs_trunc,
            "fraction_needing_truncation": needs_trunc / len(instances),
            "continuation_exceeds_max_length": int((cont_lengths > lm.max_length).sum()),
            "timing_sample_requests": len(sample), "seconds_per_request_measured": per_request,
            "ESTIMATED_full_scoring_seconds": per_request * len(instances),
            "task_config": cfg.to_dict() if hasattr(cfg, "to_dict") else str(cfg),
        }
        r = report["tasks"][name]
        print(f"{name}: {r['documents']:,} docs, {r['requests']:,} requests, max tokens {r['tokens_ctx_plus_cont']['max']}, "
              f"truncation needed {needs_trunc}, max continuation {r['continuation_tokens_max']}, "
              f"~{per_request * 1000:.1f} ms/request -> ESTIMATE {r['ESTIMATED_full_scoring_seconds'] / 60:.1f} min", flush=True)
    out = REPO_ROOT / "results" / "eval_preflight" / f"eval_request_audit_{export_dir.name}.json"
    out.write_text(json.dumps(report, indent=2, default=handle_non_serializable) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
