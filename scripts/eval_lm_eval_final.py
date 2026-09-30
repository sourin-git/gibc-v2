"""FINAL full evaluation of the four required lm-eval tasks on a verified local HF export (no --limit).

Usage: python scripts/eval_lm_eval_final.py --export <dir> --out <final results dir>
Same settings as the Stage 6 smoke test: lm-eval 0.4.13 standard HF backend, local export only, CUDA,
float32, max_length 512, batch size 8, 0-shot (our documented methodology), and unchanged task
definitions. Per-sample logs (large) go to $GIBC_WORK_DIR/eval_samples/, recorded by sha256.
A fixed random subset of logged requests is re-scored with the SOURCE checkpoint to cross-check the path.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande"]
CROSSCHECK_REQUESTS = 200


def main() -> int:
    parser = argparse.ArgumentParser(description="FINAL lm-eval evaluation (full, no limit).")
    parser.add_argument("--export", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import lm_eval
    import torch
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable

    from gibc.checkpoint import load_model_from_checkpoint
    from gibc.data import file_sha256_streaming
    from gibc.paths import work_dir
    from gibc.scoring import continuation_logprobs, encode_pair, ours_logits_fn
    from gibc.tokenizer import encode_text, load_tokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from eval_lm_eval_smoke import request_pairs

    torch.set_float32_matmul_precision("highest")
    export_dir = Path(args.export).resolve()
    manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = work_dir() / "eval_samples" / export_dir.name
    samples_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    lm = HFLM(pretrained=str(export_dir), tokenizer=str(export_dir), device="cuda", dtype="float32",
              max_length=512, batch_size=args.batch_size)
    load_seconds = time.perf_counter() - t0
    ours, _ = load_model_from_checkpoint(Path(manifest["source_checkpoint"]), "cuda")
    our_tok = load_tokenizer(Path(__file__).resolve().parents[1] / "results" / "tokenizer" / "tokenizer.json")
    hub = Path(__import__("os").environ["HF_HOME"]) / "hub"

    summary = {"status": "FINAL", "export": str(export_dir), "source_checkpoint": manifest["source_checkpoint"],
               "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
               "settings": {"backend": "lm-eval HFLM (hf)", "device": "cuda", "dtype": "float32", "max_length": lm.max_length,
                            "batch_size": args.batch_size, "num_fewshot": 0, "limit": None,
                            "few_shot_note": "0-shot is the project's documented methodology (no organizer protocol given)"},
               "command": " ".join(["python", *sys.argv]), "model_load_seconds": round(load_seconds, 1), "tasks": {}}
    rng = random.Random(0)
    for task in TASKS:
        t = time.perf_counter()
        res = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=0, log_samples=True)
        seconds = time.perf_counter() - t
        samples = res["samples"][task]
        samples_path = samples_dir / f"{task}_samples.json"
        samples_path.write_text(json.dumps(samples, default=handle_non_serializable), encoding="utf-8")

        pairs = [(pair, h) for s in samples for pair, h in zip(
            request_pairs(s), [r[0] if isinstance(r, (list, tuple)) else r for r in s["filtered_resps"]])]
        worst = 0.0
        for (context, cont), h in rng.sample(pairs, min(CROSSCHECK_REQUESTS, len(pairs))):
            h = h[0] if isinstance(h, (list, tuple)) else h
            ctx_ids, cont_ids = encode_pair(lambda x: encode_text(our_tok, x), context, cont)
            ll = continuation_logprobs(ours_logits_fn(ours), ctx_ids, cont_ids, 512, device="cuda").sum().item()
            worst = max(worst, abs(ll - float(h)))

        cfg = res["configs"][task]
        repo = cfg["dataset_path"].replace("/", "--")
        ref = hub / f"datasets--{repo}" / "refs" / "main"
        record = {"status": "FINAL", "task": task, "seconds": seconds, "examples": len(samples),
                  "requests": len(pairs), "n_samples": res["n-samples"].get(task),
                  "results": res["results"][task], "version": res["versions"].get(task), "config": cfg,
                  "dataset_revision": ref.read_text(encoding="utf-8").strip() if ref.exists() else None,
                  "samples_file": str(samples_path), "samples_sha256": file_sha256_streaming(samples_path),
                  "crosscheck_requests": min(CROSSCHECK_REQUESTS, len(pairs)),
                  "crosscheck_max_abs_loglik_diff_vs_source": worst, "lm_eval_config": res.get("config")}
        (out_dir / f"lm_eval_{task}.json").write_text(json.dumps(record, indent=2, default=handle_non_serializable) + "\n",
                                                      encoding="utf-8", newline="\n")
        metrics = {k: v for k, v in res["results"][task].items() if k.startswith("acc")}
        summary["tasks"][task] = {"examples": len(samples), "requests": len(pairs), "seconds": round(seconds, 1),
                                  "version": record["version"], "split": cfg.get("test_split") or cfg.get("validation_split"),
                                  "dataset": f"{cfg['dataset_path']}/{cfg.get('dataset_name')}",
                                  "dataset_revision": record["dataset_revision"], "metrics": metrics,
                                  "crosscheck_max_abs_loglik_diff_vs_source": worst}
        print(f"{task}: {len(samples)} examples, {len(pairs)} requests, {seconds:.1f} s, {metrics}, "
              f"cross-check max |diff| {worst:.2e}", flush=True)
    (out_dir / "lm_eval_summary.json").write_text(json.dumps(summary, indent=2, default=handle_non_serializable) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
