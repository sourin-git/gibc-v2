"""Benchmark one training configuration at seq_len 512 on pilot tokens; append the result as a JSON line.

Usage: python scripts/benchmark.py --precision bf16 --micro-batch 4 --grad-accum 8
       [--warmup 3] [--measured 15] [--sustain-seconds 0] [--out results/benchmark/stage4_runs.jsonl]
Run each configuration in its own process so that an OOM cannot affect the next measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from gibc.benchmark import run_benchmark
from gibc.config import ModelConfig
from gibc.data import load_token_meta, open_token_file
from gibc.paths import REPO_ROOT, work_dir
from gibc.train import TrainConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-configuration training benchmark.")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=15)
    parser.add_argument("--sustain-seconds", type=float, default=0.0)
    parser.add_argument("--tokens", default="pilot")
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "benchmark" / "stage4_runs.jsonl"))
    args = parser.parse_args()

    model_cfg = ModelConfig.from_json(REPO_ROOT / "configs" / "model" / "gibc_43m.json")
    base_cfg = TrainConfig.from_json(REPO_ROOT / "configs" / "train" / "smoke.json")  # optimizer settings only
    tokens_dir = work_dir() / "tokens" / args.tokens
    meta = load_token_meta(tokens_dir)
    tokens = open_token_file(tokens_dir / "train.bin", model_cfg.vocab_size, meta["splits"]["train"]["tokens"])

    result = run_benchmark(tokens, model_cfg, base_cfg, args.precision, args.micro_batch, args.grad_accum,
                           args.warmup, args.measured, args.sustain_seconds)
    result["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["command"] = " ".join(["python", "scripts/benchmark.py", *sys.argv[1:]])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(result) + "\n")

    if result["status"] == "ok":
        print(f"{args.precision} B={args.micro_batch} accum={args.grad_accum} ({result['tokens_per_update']:,} tok/update)")
        print(f"  whole interval: {result['tokens_per_s_whole_interval']:,.0f} tok/s over "
              f"{result['whole_interval_updates']} updates / {result['whole_interval_seconds']:.1f} s")
        print(f"  per update (synced both sides): mean {result['tokens_per_s_per_update_mean']:,.0f} tok/s, "
              f"median {result['tokens_per_s_per_update_median']:,.0f}; step {result['step_time_s_mean']:.3f} s "
              f"(min {result['step_time_s_min']:.3f}, max {result['step_time_s_max']:.3f}) over {result['per_update_count']}")
        print(f"  peak alloc {result['peak_allocated_mib']:,.0f} MiB, reserved {result['peak_reserved_mib']:,.0f} MiB, "
              f"skips {result['scaler_skips_measured']}, GPU {result['gpu_samples']}")
    else:
        print(f"{args.precision} B={args.micro_batch}: {result['status']} ({result.get('error', '')}); "
              f"allocated after cleanup {result['allocated_after_cleanup_mib']:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
