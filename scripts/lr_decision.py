"""Apply the PREDECLARED Stage 5 LR decision rule to the two sanity runs, from their metrics logs.

Usage: python scripts/lr_decision.py
Reads $GIBC_WORK_DIR/runs/lr_6e-4 and lr_1e-3 metrics.jsonl; copies them to results/lr/ and writes
results/lr/lr_decision.json.

Rule (fixed before the runs):
- reject a candidate if its loss or gradients became non-finite, or if BOTH late validation losses
  (steps 768 and 1024) exceed its step-512 loss by > 0.10 nats/token;
- also require late_mean = mean(val@768, val@1024) < val@0;
- choose 1e-3 only if it is stable, its late_mean beats stable 6e-4 by >= 0.02, and its val@1024 is
  not worse than 6e-4's; otherwise choose stable 6e-4; if only one passes choose it; if neither
  passes, select nothing (stop and decide on the single permitted 3e-4 fallback).
"""

from __future__ import annotations

import json
import math
import shutil
import sys

from gibc.paths import REPO_ROOT, work_dir

RUNS = {"6e-4": "lr_6e-4", "1e-3": "lr_1e-3"}
VAL_STEPS = (0, 256, 512, 768, 1024)


def load(run: str) -> list[dict]:
    path = work_dir() / "runs" / run / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assess(events: list[dict]) -> dict:
    val = {e["step"]: e for e in events if e["event"] == "val"}
    missing = [s for s in VAL_STEPS if s not in val]
    train = [e for e in events if e["event"] == "train"]
    non_finite_events = [e for e in events if e["event"] in ("non_finite", "scaler_skip")]
    finite = (not non_finite_events and all(math.isfinite(e["train_loss"]) and math.isfinite(e["grad_norm"]) for e in train)
              and all(math.isfinite(v["val_loss"]) for v in val.values()))
    out = {"val_loss": {s: val[s]["val_loss"] for s in VAL_STEPS if s in val},
           "val_ppl": {s: val[s]["val_ppl"] for s in VAL_STEPS if s in val},
           "val_tokens": sorted({v["val_tokens"] for v in val.values()}),
           "missing_val_steps": missing, "finite": finite, "final_step": max((e["step"] for e in train), default=0)}
    if missing or not finite:
        out.update({"stable": False, "reason": "missing validation points" if missing else "non-finite values"})
        return out
    v = out["val_loss"]
    late_mean = (v[768] + v[1024]) / 2
    both_late_up = (v[768] - v[512] > 0.10) and (v[1024] - v[512] > 0.10)
    improves = late_mean < v[0]
    out.update({"late_mean": late_mean, "both_late_exceed_512_by_0.10": both_late_up, "late_mean_below_step0": improves,
                "stable": (not both_late_up) and improves})
    out["reason"] = "passes" if out["stable"] else ("late divergence" if both_late_up else "no improvement over step 0")
    return out


def decide(a: dict, b: dict) -> tuple[str | None, str]:
    """a = 6e-4, b = 1e-3."""
    if a["stable"] and b["stable"]:
        margin = a["late_mean"] - b["late_mean"]
        final_ok = b["val_loss"][1024] <= a["val_loss"][1024]
        if margin >= 0.02 and final_ok:
            return "1e-3", f"both stable; 1e-3 late_mean better by {margin:.4f} >= 0.02 and its val@1024 is not worse"
        return "6e-4", (f"both stable; 1e-3 late_mean margin {margin:.4f} (needs >= 0.02) and val@1024 "
                        f"{'not worse' if final_ok else 'worse'} -> default to 6e-4")
    if a["stable"]:
        return "6e-4", "only 6e-4 passes"
    if b["stable"]:
        return "1e-3", "only 1e-3 passes"
    return None, "neither candidate passes; STOP (decide on the permitted 3e-4 fallback)"


def windows(events: list[dict]) -> list[dict]:
    rows = []
    for e in events:
        if e["event"] == "train":
            rows.append({k: e.get(k) for k in ("step", "interval_steps", "train_loss", "lr", "grad_norm", "clip_fraction",
                                               "tokens_per_s", "cuda_peak_allocated_mib", "cuda_peak_reserved_mib",
                                               "elapsed_s", "temp_c", "sampler_cursor")})
    return rows


def main() -> int:
    out_dir = REPO_ROOT / "results" / "lr"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for label, run in RUNS.items():
        events = load(run)
        shutil.copyfile(work_dir() / "runs" / run / "metrics.jsonl", out_dir / f"{run}_metrics.jsonl")
        end = [e for e in events if e["event"] == "end"][-1]
        runs[label] = {"assessment": assess(events), "train_log": windows(events),
                       "resumes": [e for e in events if e["event"] == "resume"],
                       "end": {k: end.get(k) for k in ("step", "microsteps", "tokens", "cursor", "consumed_sha256",
                                                       "train_seconds", "wall_seconds", "cuda_peak_allocated_mib",
                                                       "cuda_peak_reserved_mib")}}
    selected, reason = decide(runs["6e-4"]["assessment"], runs["1e-3"]["assessment"])
    same_windows = runs["6e-4"]["end"]["consumed_sha256"] == runs["1e-3"]["end"]["consumed_sha256"]
    decision = {"rule": __doc__.split("Rule (fixed before the runs):")[1].strip(), "selected_lr": selected,
                "reason": reason, "identical_window_sequence_both_runs": same_windows, "runs": runs}
    (out_dir / "lr_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8", newline="\n")

    for label, r in runs.items():
        a = r["assessment"]
        print(f"lr {label}: " + "  ".join(f"val@{s} {a['val_loss'][s]:.4f}" for s in a["val_loss"])
              + f"  late_mean {a.get('late_mean', float('nan')):.4f}  stable {a['stable']} ({a['reason']})")
    print(f"identical window sequence in both runs: {same_windows}")
    print(f"SELECTED: {selected}  -- {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
