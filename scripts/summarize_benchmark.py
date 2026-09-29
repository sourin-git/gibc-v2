"""Summarize Stage 4 benchmark results and derive time ESTIMATES from the measured sustained throughput.

Usage: python scripts/summarize_benchmark.py
Reads results/benchmark/stage4_runs.jsonl and stage4_sustained.jsonl; writes results/benchmark/stage4_summary.json.
Estimates are arithmetic on measured throughput, not measured training times.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gibc.paths import REPO_ROOT

# Planning allowance on top of raw throughput: validation (~1%), checkpoints (<0.1%), startup,
# occasional interruptions/resumes (~10%), and uncertainty from only a 2-minute thermal check.
PLANNING_OVERHEAD = 1.20
TOKEN_TARGETS = [25_000_000, 50_000_000, 75_000_000, 100_000_000, 250_000_000, 500_000_000,
                 750_000_000, 1_000_000_000]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    bench_dir = REPO_ROOT / "results" / "benchmark"
    runs = read_jsonl(bench_dir / "stage4_runs.jsonl")
    sustained = read_jsonl(bench_dir / "stage4_sustained.jsonl")[-1]
    keys = ["precision", "micro_batch", "seq_len", "activation_checkpointing", "grad_accum", "tokens_per_update",
            "tokens_per_s_mean", "tokens_per_s_median", "step_time_s_mean", "peak_allocated_mib",
            "peak_reserved_mib", "scaler_skips_measured", "updates_timed", "status"]
    table = [{k: r.get(k) for k in keys} for r in runs + [sustained]]

    rate = sustained["tokens_per_s_mean"]
    estimates = []
    for tokens in TOKEN_TARGETS:
        raw_h = tokens / rate / 3600
        estimates.append({"tokens": tokens, "raw_hours": round(raw_h, 2),
                          "planning_hours": round(raw_h * PLANNING_OVERHEAD, 2)})
    summary = {
        "note": "Throughput rows are measurements; time estimates are arithmetic ESTIMATES from the sustained rate.",
        "table": table,
        "sustained": {k: sustained[k] for k in ("precision", "micro_batch", "grad_accum", "tokens_per_update",
                                               "tokens_per_s_mean", "tokens_per_s_median", "updates_timed",
                                               "peak_allocated_mib", "peak_reserved_mib", "gpu_samples")},
        "estimate_basis_tokens_per_s": rate,
        "planning_overhead_factor": PLANNING_OVERHEAD,
        "estimates": estimates,
    }
    out = bench_dir / "stage4_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")

    for row in table:
        mean = row["tokens_per_s_mean"]
        print(f"{row['precision']:>5} B={row['micro_batch']:<2} accum={row['grad_accum']:<2} "
              f"{row['tokens_per_update']:>6,} tok/upd  "
              + (f"{mean:>7,.0f} tok/s  step {row['step_time_s_mean']:.3f} s  "
                 f"alloc {row['peak_allocated_mib']:>5,.0f}  resv {row['peak_reserved_mib']:>5,.0f} MiB  "
                 f"skips {row['scaler_skips_measured']}" if mean else row["status"]))
    print(f"estimate basis: {rate:,.0f} tok/s (sustained), overhead x{PLANNING_OVERHEAD}")
    for e in estimates:
        print(f"  {e['tokens']/1e6:>6,.0f}M tokens: raw {e['raw_hours']:>6.2f} h   planning {e['planning_hours']:>6.2f} h")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
