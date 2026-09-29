"""Preflight data checks for a training config (read-only; trains nothing).

Usage: python scripts/verify_production_data.py --config configs/train/production.json
Checks: token files re-hashed (streaming) against meta.json; tokenizer hash; train tokens >= the
acquisition target; token ids in range; enough shuffled full windows for every update; the
validation split supports the configured validation subset; free disk >= the floor. Exits 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from gibc.paths import REPO_ROOT, work_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify production data before training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", default=str(REPO_ROOT / "configs" / "data" / "production.json"))
    args = parser.parse_args()

    from gibc.config import ModelConfig
    from gibc.data import ShuffledWindowSampler, fixed_validation_batches, load_token_meta, open_token_file, verify_token_files
    from gibc.production_data import ProductionConfig
    from gibc.tokenizer import file_sha256
    from gibc.train import TrainConfig

    cfg = TrainConfig.from_json(args.config)
    pcfg = ProductionConfig.from_json(args.data_config)
    model_cfg = ModelConfig.from_json(REPO_ROOT / cfg.model_config)
    tokens_dir = work_dir() / "tokens" / cfg.tokens
    meta = load_token_meta(tokens_dir)
    checks: list[tuple[str, bool, str]] = []

    verify_token_files(tokens_dir, meta)  # raises on mismatch
    checks.append(("token file sha256 (streaming re-hash)", True,
                   f"train {meta['splits']['train']['sha256'][:16]}..., val {meta['splits']['val']['sha256'][:16]}..."))
    tok_sha = file_sha256(REPO_ROOT / "results" / "tokenizer" / "tokenizer.json")
    checks.append(("tokenizer sha256 matches token meta", tok_sha == meta["tokenizer_sha256"], tok_sha[:16] + "..."))

    train = open_token_file(tokens_dir / "train.bin", model_cfg.vocab_size, meta["splits"]["train"]["tokens"])
    val = open_token_file(tokens_dir / "val.bin", model_cfg.vocab_size, meta["splits"]["val"]["tokens"])
    checks.append(("token ids in [0, vocab)", True, "checked at open"))
    checks.append(("train tokens >= target", train.size >= pcfg.target_train_tokens,
                   f"{train.size:,} >= {pcfg.target_train_tokens:,}"))

    sampler = ShuffledWindowSampler(train, cfg.seq_len, cfg.micro_batch_size, cfg.seed)
    needed = cfg.max_steps * cfg.grad_accum_steps * cfg.micro_batch_size
    checks.append(("enough shuffled full windows", needed <= sampler.window_count,
                   f"need {needed:,} of W={sampler.window_count:,} (margin {sampler.window_count - needed:,}; "
                   f"dropped tail {sampler.dropped_targets} targets)"))
    try:
        batches = fixed_validation_batches(val, cfg.seq_len, cfg.micro_batch_size, cfg.eval_batches)
        targets = sum(y.numel() for _, y in batches)
        checks.append(("validation subset supported", True, f"{targets:,} targets from {val.size:,} val tokens"))
    except ValueError as exc:
        checks.append(("validation subset supported", False, str(exc)))
    free = shutil.disk_usage(work_dir()).free / 2**30
    checks.append(("free disk >= floor", free >= pcfg.min_free_gib, f"{free:.2f} GiB free (floor {pcfg.min_free_gib})"))

    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    ok = all(c[1] for c in checks)
    result = {"config": args.config, "ok": ok, "window_count": sampler.window_count,
              "windows_needed": needed, "dropped_targets": sampler.dropped_targets,
              "order_sha256": sampler.order_fingerprint(), "checks": [list(c) for c in checks]}
    (REPO_ROOT / "results" / "data" / "production_preflight_data.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
