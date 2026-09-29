"""Single-GPU training loop: AdamW, warmup + cosine LR, fp32/bf16/fp16, gradient accumulation,
fixed-batch validation, atomic checkpoints and exact logical resume.

Step accounting: one optimizer ("global") step = `grad_accum_steps` microsteps, each consuming
micro_batch_size x seq_len input tokens. The LR schedule is a pure function of the global step,
so restoring the step restores the schedule.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from gibc.checkpoint import CHECKPOINT_FORMAT, capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from gibc.config import ModelConfig
from gibc.data import (
    SHUFFLED_SAMPLER_VERSION,
    Batcher,
    ShuffledWindowSampler,
    fixed_validation_batches,
    load_token_meta,
    open_token_file,
    verify_token_files,
)
from gibc.gpu import gpu_sample
from gibc.model import CausalLM, parameter_report

PRECISIONS = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
SAMPLERS = ("random_v1", SHUFFLED_SAMPLER_VERSION)
# fp16: GradScaler halves its scale on every overflow skip; this many in a row means real divergence.
MAX_CONSECUTIVE_SCALER_SKIPS = 20


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    placeholder: bool  # true = unapproved placeholder; scripts/train.py refuses to run it
    notes: str
    model_config: str  # repo-relative path
    tokens: str  # name under $GIBC_WORK_DIR/tokens/
    seq_len: int
    micro_batch_size: int
    grad_accum_steps: int
    max_steps: int  # optimizer steps; also the length of the LR schedule
    precision: str
    lr: float  # peak
    min_lr_ratio: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    grad_clip: float  # 0 disables clipping (the norm is still measured)
    seed: int
    log_every: int
    eval_every: int
    eval_batches: int
    checkpoint_every_steps: int
    checkpoint_every_minutes: float  # 0 disables time-based checkpoints
    keep_checkpoints: int
    allow_tf32: bool
    fused_adamw: bool
    device: str
    status: dict[str, Any] = field(default_factory=dict)  # free-form approval markers; not used by training
    sampler: str = "random_v1"  # "random_v1" (with replacement) | SHUFFLED_SAMPLER_VERSION (one pass, no replacement)
    run_until_step: int | None = None  # stop (with checkpoint) here; the LR schedule still spans max_steps
    milestone_fractions: list[float] = field(default_factory=list)  # protected checkpoints at these fractions

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}")
        if self.run_until_step is not None and not 0 < self.run_until_step <= self.max_steps:
            raise ValueError("run_until_step must be in (0, max_steps]")
        if any(not 0 < f <= 1 for f in self.milestone_fractions):
            raise ValueError("milestone_fractions must be in (0, 1]")
        for name in ("seq_len", "micro_batch_size", "grad_accum_steps", "max_steps", "log_every",
                     "eval_every", "eval_batches", "checkpoint_every_steps", "keep_checkpoints"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps]")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.lr <= 0 or self.grad_clip < 0 or self.checkpoint_every_minutes < 0:
            raise ValueError("lr must be > 0; grad_clip and checkpoint_every_minutes must be >= 0")

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.seq_len * self.grad_accum_steps

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown TrainConfig keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NonFiniteError(RuntimeError):
    pass


def lr_at(step: int, cfg: TrainConfig) -> float:
    """LR for the optimizer step with 0-based index `step`: linear warmup to cfg.lr over
    warmup_steps (step 0 -> lr/warmup), then cosine decay to lr*min_lr_ratio at step max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    min_lr = cfg.lr * cfg.min_lr_ratio
    decay_steps = cfg.max_steps - cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / decay_steps) if decay_steps > 0 else 1.0
    return min_lr + (cfg.lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: CausalLM, cfg: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    """Two groups: >=2-D weights (incl. the tied embedding, once) get weight decay; 1-D RMSNorm scales do not."""
    params = dict(model.named_parameters())  # deduplicated: the tied head appears only as the embedding
    decay = [p for p in params.values() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in params.values() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay, "name": "decay"},
         {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"}],
        lr=lr_at(0, cfg), betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
        fused=cfg.fused_adamw and device.type == "cuda",
    )
    check_optimizer_coverage(model, optimizer)
    return optimizer


def check_optimizer_coverage(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Every trainable Parameter object in exactly one group, nothing else."""
    in_groups = [id(p) for group in optimizer.param_groups for p in group["params"]]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if len(in_groups) != len(set(in_groups)):
        raise ValueError("a parameter appears more than once in the optimizer groups")
    if set(in_groups) != trainable:
        raise ValueError(f"optimizer groups cover {len(set(in_groups))} params, model has {len(trainable)} trainable")


def autocast_context(precision: str, device: torch.device):
    dtype = PRECISIONS[precision]
    return nullcontext() if dtype is None else torch.autocast(device.type, dtype=dtype)


@torch.no_grad()
def evaluate(model: CausalLM, batches: list[tuple[torch.Tensor, torch.Tensor]], precision: str,
             device: torch.device) -> dict[str, float]:
    """Token-weighted mean loss over fixed batches; restores the previous train/eval mode."""
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = 0
    try:
        for x, y in batches:
            with autocast_context(precision, device):
                _, loss = model(x.to(device), y.to(device), validate_targets=False)
            total_loss += loss.double() * y.numel()
            total_tokens += y.numel()
    finally:
        model.train(was_training)
    mean = (total_loss / total_tokens).item()
    return {"val_loss": mean, "val_ppl": math.exp(mean) if mean < 700 else math.inf, "val_tokens": total_tokens}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def cuda_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    mib = 2**20
    return {"cuda_allocated_mib": round(torch.cuda.memory_allocated(device) / mib, 1),
            "cuda_reserved_mib": round(torch.cuda.memory_reserved(device) / mib, 1),
            "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / mib, 1),
            "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / mib, 1)}


@dataclass
class TrainState:
    step: int = 0  # optimizer steps completed
    microsteps: int = 0  # forward/backward passes completed
    tokens: int = 0  # input tokens processed
    train_seconds: float = 0.0  # time inside training steps (excludes eval/checkpoint)
    wall_seconds: float = 0.0  # total wall time across all processes of this run
    skipped_steps: int = 0  # fp16 GradScaler overflow skips


class MetricsLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, record: dict[str, Any]) -> None:
        record = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record) + "\n")


def train(
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    tokens_dir: Path,
    run_dir: Path,
    repo_root: Path,
    resume: Path | None = None,
    stop_at_step: int | None = None,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train until cfg.max_steps (or stop_at_step, which checkpoints and returns). Returns a summary."""
    process_start = time.perf_counter()
    device = torch.device(cfg.device)
    if cfg.seq_len > model_cfg.context_length:
        raise ValueError(f"seq_len {cfg.seq_len} exceeds context_length {model_cfg.context_length}")
    if cfg.allow_tf32:
        torch.set_float32_matmul_precision("high")

    meta = load_token_meta(tokens_dir)
    if meta["vocab_size"] != model_cfg.vocab_size:
        raise ValueError(f"token files built for vocab {meta['vocab_size']}, model has {model_cfg.vocab_size}")
    # Once per process (fresh start or resume), before anything trusts the corpus: recompute the
    # actual file hashes (streaming) instead of trusting the hashes recorded in meta.json.
    t_verify = time.perf_counter()
    verify_token_files(tokens_dir, meta)
    verify_seconds = time.perf_counter() - t_verify
    split_tokens = {s: open_token_file(tokens_dir / f"{s}.bin", model_cfg.vocab_size, meta["splits"][s]["tokens"])
                    for s in ("train", "val")}
    data_identity = {"tokens_dir": tokens_dir.name, "tokenizer_sha256": meta["tokenizer_sha256"],
                     "train_sha256": meta["splits"]["train"]["sha256"], "val_sha256": meta["splits"]["val"]["sha256"]}

    set_seeds(cfg.seed)
    model = CausalLM(model_cfg).to(device)
    optimizer = build_optimizer(model, cfg, device)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.precision == "fp16")
    if cfg.sampler == SHUFFLED_SAMPLER_VERSION:
        batcher = ShuffledWindowSampler(split_tokens["train"], cfg.seq_len, cfg.micro_batch_size, cfg.seed)
    else:
        batcher = Batcher(split_tokens["train"], cfg.seq_len, cfg.micro_batch_size, cfg.seed)
    val_batches = fixed_validation_batches(split_tokens["val"], cfg.seq_len, cfg.micro_batch_size, cfg.eval_batches)
    state = TrainState()
    stop_targets = [s for s in (stop_at_step, cfg.run_until_step, cfg.max_steps) if s is not None]
    stop_step = min(stop_targets)
    milestones = {max(1, round(f * cfg.max_steps)) for f in cfg.milestone_fractions}

    def sampler_info() -> dict[str, Any]:
        if not isinstance(batcher, ShuffledWindowSampler):
            return {"sampler": cfg.sampler}
        return {"sampler": cfg.sampler, "window_count": batcher.window_count, "cursor": batcher.cursor,
                "dropped_targets": batcher.dropped_targets, "order_sha256": batcher.order_fingerprint(),
                "consumed_sha256": batcher.consumed_fingerprint()}

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    metrics = MetricsLog(run_dir / "metrics.jsonl")
    source = git_state(repo_root)

    if resume is not None:
        ckpt = load_checkpoint(resume)
        if (TrainConfig.from_dict(ckpt["train_config"]) != cfg
                or ModelConfig.from_dict(ckpt["model_config"]) != model_cfg):
            raise ValueError("resume config differs from the checkpoint's config")
        if ckpt["data"] != data_identity:
            raise ValueError(f"token data differs from the checkpoint's: {ckpt['data']} vs {data_identity}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        batcher.load_state_dict(ckpt["batcher"])
        restore_rng_state(ckpt["rng"])
        state = TrainState(**ckpt["state"])
        check_optimizer_coverage(model, optimizer)
        metrics.write({"event": "resume", "checkpoint": str(resume), "step": state.step, "tokens": state.tokens,
                       "microsteps": state.microsteps, "source": source, "data_verified_seconds": round(verify_seconds, 2),
                       **sampler_info()})
        print_fn(f"resumed from {resume}: step {state.step}, tokens {state.tokens:,}")
    else:
        report = parameter_report(model)
        metrics.write({"event": "start", "params_unique_trainable": report["total unique trainable"],
                       "train_config": cfg.to_dict(), "model_config": model_cfg.to_dict(), "data": data_identity,
                       "data_verified_seconds": round(verify_seconds, 2), "stop_step": stop_step,
                       "source": source, "torch": str(torch.__version__), **sampler_info(),
                       "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
        print_fn(f"fresh run: {report['total unique trainable']:,} unique trainable params, "
                 f"{cfg.tokens_per_step:,} tokens/optimizer step, {cfg.precision}")

    if isinstance(batcher, ShuffledWindowSampler):
        needed = (stop_step - state.step) * cfg.grad_accum_steps * cfg.micro_batch_size
        if needed > batcher.remaining_windows:
            raise ValueError(f"training to step {stop_step} needs {needed:,} more windows but only "
                             f"{batcher.remaining_windows:,} of {batcher.window_count:,} remain (no wraparound)")

    wall_base = state.wall_seconds
    last_ckpt_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def checkpoint(reason: str) -> Path:
        nonlocal last_ckpt_time
        state.wall_seconds = wall_base + (time.perf_counter() - process_start)
        # Only "step_*" files rotate; "milestone_*" and "final_*" are never deleted automatically.
        prefix = "final" if state.step == cfg.max_steps else "milestone" if state.step in milestones else None
        path = ckpt_dir / (f"{prefix}_step_{state.step:07d}.pt" if prefix else f"step_{state.step:07d}.pt")
        t0 = time.perf_counter()
        save_checkpoint({
            "format": CHECKPOINT_FORMAT,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": {"type": "linear_warmup_cosine", "next_step": state.step, "lr_next": lr_at(state.step, cfg)},
            "state": asdict(state),
            "accumulation": {"microsteps_into_current_step": 0},  # checkpoints only at step boundaries
            "batcher": batcher.state_dict(),
            "rng": capture_rng_state(),
            "train_config": cfg.to_dict(),
            "model_config": model_cfg.to_dict(),
            "data": data_identity,
            "source": source,
            "torch": str(torch.__version__),
            "sampler_info": sampler_info(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, path)
        for old in sorted(ckpt_dir.glob("step_*.pt"))[:-cfg.keep_checkpoints]:
            old.unlink()
        last_ckpt_time = time.perf_counter()
        metrics.write({"event": "checkpoint", "reason": reason, "step": state.step, "tokens": state.tokens,
                       "path": str(path), "bytes": path.stat().st_size, "seconds": round(time.perf_counter() - t0, 2),
                       **sampler_info()})
        print_fn(f"checkpoint ({reason}): {path} ({path.stat().st_size / 2**20:.1f} MiB)")
        return path

    def validate() -> dict[str, float]:
        result = evaluate(model, val_batches, cfg.precision, device)
        metrics.write({"event": "val", "step": state.step, "tokens": state.tokens, **result})
        print_fn(f"val  step {state.step:>6}  loss {result['val_loss']:.4f}  ppl {result['val_ppl']:.2f}  "
                 f"({result['val_tokens']:,} tokens)")
        return result

    if state.step == 0:
        validate()

    model.train()
    interval = {"tokens": 0, "seconds": 0.0, "loss_sum": 0.0, "steps": 0, "clipped": 0}
    last_val: dict[str, float] | None = None
    last_ckpt: Path | None = resume
    first_loss: float | None = None
    step_loss = float("nan")
    consecutive_skips = 0
    use_pin = device.type == "cuda"

    while state.step < stop_step:
        t_step = time.perf_counter()
        lr = lr_at(state.step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_sum = torch.zeros((), device=device)
        for _ in range(cfg.grad_accum_steps):
            x, y = batcher.next_batch()
            if use_pin:
                x, y = x.pin_memory(), y.pin_memory()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_context(cfg.precision, device):
                _, loss = model(x, y, validate_targets=False)  # token files are range-checked at open
            scaler.scale(loss / cfg.grad_accum_steps).backward()
            loss_sum += loss.detach()
            state.microsteps += 1
            state.tokens += x.numel()

        scaler.unscale_(optimizer)  # no-op unless fp16; clipping must see unscaled gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip or float("inf"))
        step_loss, norm = torch.stack([loss_sum / cfg.grad_accum_steps, grad_norm.float()]).tolist()  # one sync
        if not math.isfinite(step_loss) or (not math.isfinite(norm) and not scaler.is_enabled()):
            record = {"event": "non_finite", "step": state.step, "microsteps": state.microsteps,
                      "loss": step_loss, "grad_norm": norm, "lr": lr,
                      "latest_checkpoint": str(last_ckpt) if last_ckpt else None}
            metrics.write(record)
            raise NonFiniteError(f"non-finite loss/grad at step {state.step}: {record}; no checkpoint written")

        scale_before = scaler.get_scale() if scaler.is_enabled() else None
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        seconds = time.perf_counter() - t_step
        state.train_seconds += seconds
        interval["tokens"] += cfg.tokens_per_step
        interval["seconds"] += seconds

        # GradScaler lowers its scale in update() exactly when step() found inf/NaN gradients and
        # skipped optimizer.step(). A skipped update is not an optimizer step: the global step, and
        # with it the LR schedule, logging, validation and checkpoint triggers, stay where they are.
        # Microsteps and (attempted) tokens were already counted above.
        if scaler.is_enabled() and scaler.get_scale() < scale_before:
            state.skipped_steps += 1
            consecutive_skips += 1
            metrics.write({"event": "scaler_skip", "step": state.step, "microsteps": state.microsteps,
                           "tokens": state.tokens, "scale_before": scale_before, "scale_after": scaler.get_scale(),
                           "grad_norm": norm, "skipped_steps": state.skipped_steps})
            if consecutive_skips >= MAX_CONSECUTIVE_SCALER_SKIPS:
                raise NonFiniteError(f"{consecutive_skips} consecutive GradScaler overflow skips at step "
                                     f"{state.step}; no checkpoint written")
            continue
        consecutive_skips = 0
        state.step += 1

        first_loss = step_loss if first_loss is None else first_loss
        interval["loss_sum"] += step_loss
        interval["steps"] += 1
        interval["clipped"] += int(cfg.grad_clip > 0 and norm > cfg.grad_clip)
        stopping = state.step >= stop_step

        if state.step == 1 or state.step % cfg.log_every == 0 or stopping:
            record = {"event": "train", "step": state.step, "microsteps": state.microsteps, "tokens": state.tokens,
                      "train_loss": interval["loss_sum"] / interval["steps"], "last_step_loss": step_loss,
                      "lr": lr, "grad_norm": norm, "tokens_per_s": interval["tokens"] / interval["seconds"],
                      "elapsed_s": round(wall_base + time.perf_counter() - process_start, 1),
                      "train_s": round(state.train_seconds, 2), "skipped_steps": state.skipped_steps,
                      "clip_fraction": interval["clipped"] / interval["steps"], "interval_steps": interval["steps"],
                      "sampler_cursor": getattr(batcher, "cursor", None),
                      **cuda_memory(device), **(gpu_sample() if device.type == "cuda" else {})}
            metrics.write(record)
            mem = f"  mem {record.get('cuda_allocated_mib', 0):.0f}/{record.get('cuda_reserved_mib', 0):.0f} MiB " \
                  f"(peak {record.get('cuda_peak_allocated_mib', 0):.0f})" if device.type == "cuda" else ""
            print_fn(f"step {state.step:>6}/{cfg.max_steps}  micro {state.microsteps:>7}  tok {state.tokens:>11,}  "
                     f"loss {record['train_loss']:.4f}  lr {lr:.2e}  gnorm {norm:.3f}  "
                     f"{record['tokens_per_s']:,.0f} tok/s{mem}")
            interval = {"tokens": 0, "seconds": 0.0, "loss_sum": 0.0, "steps": 0, "clipped": 0}

        if state.step % cfg.eval_every == 0 or state.step in (cfg.max_steps, cfg.run_until_step):
            last_val = validate()
        minutes = (time.perf_counter() - last_ckpt_time) / 60
        if (state.step % cfg.checkpoint_every_steps == 0 or stopping or state.step in milestones
                or (cfg.checkpoint_every_minutes and minutes >= cfg.checkpoint_every_minutes)):
            last_ckpt = checkpoint("stop" if stopping and state.step < cfg.max_steps else
                                   "final" if state.step == cfg.max_steps else
                                   "milestone" if state.step in milestones else "scheduled")

    state.wall_seconds = wall_base + (time.perf_counter() - process_start)
    summary = {"step": state.step, "microsteps": state.microsteps, "tokens": state.tokens,
               "first_step_loss_this_process": first_loss, "last_step_loss": step_loss,
               "last_val": last_val, "latest_checkpoint": str(last_ckpt) if last_ckpt else None,
               "train_seconds": state.train_seconds, "wall_seconds": state.wall_seconds,
               "skipped_steps": state.skipped_steps, "finished": state.step >= cfg.max_steps,
               "stop_step": stop_step, **sampler_info(), **cuda_memory(device)}
    metrics.write({"event": "end", **summary})
    return summary


def prepare_run_dir(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} exists; use --resume <checkpoint> or --overwrite")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)


def environment_info() -> dict[str, Any]:
    return {"python": sys.version.split()[0], "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__}
