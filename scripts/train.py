"""Train (or resume) a run under $GIBC_WORK_DIR/runs/<run_name>/.

Fresh:  python scripts/train.py --config configs/train/smoke.json [--overwrite] [--stop-at-step N]
Resume: python scripts/train.py --resume <run_dir>/checkpoints/step_XXXXXXX.pt [--stop-at-step N]

On resume, the train/model configs come from the checkpoint (they cannot drift). Configs marked
"placeholder": true are refused: they are unapproved (see docs/PLAN.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gibc.checkpoint import load_checkpoint
from gibc.config import ModelConfig
from gibc.paths import REPO_ROOT, work_dir
from datetime import datetime, timezone

import numpy as np

from gibc.data import SHUFFLED_SAMPLER_VERSION, ShuffledWindowSampler, load_token_meta
from gibc.hf_export import unique_trainable_parameters
from gibc.model import CausalLM
from gibc.tokenizer import file_sha256
from gibc.train import TrainConfig, environment_info, git_state, prepare_run_dir, train


def data_provenance(cfg: TrainConfig) -> dict:
    """Corpus identity as recorded in meta.json (train() re-verifies the actual file hashes) + sampler identity."""
    tokens_dir = work_dir() / "tokens" / cfg.tokens
    meta = load_token_meta(tokens_dir)
    info = {"tokens_dir": str(tokens_dir), "tokenizer_sha256": meta["tokenizer_sha256"], "dataset": meta.get("dataset"),
            **{split: {k: meta["splits"][split][k] for k in ("tokens", "documents", "sha256")} for split in ("train", "val")}}
    if cfg.sampler == SHUFFLED_SAMPLER_VERSION:
        tokens = np.memmap(tokens_dir / "train.bin", dtype=np.uint16, mode="r")
        sampler = ShuffledWindowSampler(tokens, cfg.seq_len, cfg.micro_batch_size, cfg.seed)
        info["sampler"] = {**{k: v for k, v in sampler.state_dict().items() if k != "cursor"},
                           "dropped_targets": sampler.dropped_targets}
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-GPU training.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="train config JSON (fresh run)")
    source.add_argument("--resume", help="checkpoint to resume from")
    parser.add_argument("--stop-at-step", type=int, help="checkpoint and exit after this optimizer step")
    parser.add_argument("--overwrite", action="store_true", help="fresh run: delete an existing run dir")
    args = parser.parse_args()

    if args.resume:
        resume = Path(args.resume).resolve()
        ckpt = load_checkpoint(resume)
        cfg = TrainConfig.from_dict(ckpt["train_config"])
        model_cfg = ModelConfig.from_dict(ckpt["model_config"])
        run_dir = resume.parent.parent
        del ckpt
    else:
        resume = None
        cfg = TrainConfig.from_json(args.config)
        model_cfg = ModelConfig.from_json(REPO_ROOT / cfg.model_config)
        run_dir = None
    if cfg.placeholder:
        print(f"refusing to run {cfg.run_name!r}: config is an unapproved placeholder ({cfg.notes})")
        return 2
    if run_dir is None:
        run_dir = work_dir() / "runs" / cfg.run_name

    if resume is None:
        source = git_state(REPO_ROOT)  # before anything is written, so the recorded tree state is the launch state
        prepare_run_dir(run_dir, args.overwrite)
        freeze = REPO_ROOT / "results" / "env" / "pip-freeze.txt"
        run_info = {"command": " ".join([Path(sys.executable).name, *sys.argv]),
                    "start_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "start_local": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "train_config": cfg.to_dict(), "model_config": model_cfg.to_dict(),
                    "unique_trainable_parameters": unique_trainable_parameters(CausalLM(model_cfg)),
                    "tokens_per_optimizer_step": cfg.tokens_per_step,
                    "planned_targets": cfg.tokens_per_step * cfg.max_steps,
                    "source": source, "environment": environment_info(),
                    "data": data_provenance(cfg), "pip_freeze_sha256": file_sha256(freeze) if freeze.exists() else None}
        (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8", newline="\n")

    summary = train(cfg, model_cfg, work_dir() / "tokens" / cfg.tokens, run_dir, REPO_ROOT,
                    resume=resume, stop_at_step=args.stop_at_step, print_fn=lambda s: print(s, flush=True))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
