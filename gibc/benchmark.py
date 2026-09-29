"""Hardware benchmark: real training updates on pilot tokens, timed after warmup.

One update mirrors `gibc.train.train` exactly: `grad_accum` microsteps of pinned/non-blocking
batches, each running forward + loss under autocast and backward on the scaled loss/accum. Then
unscale, clip, one sync to read loss and grad norm, GradScaler step/update, and
zero_grad(set_to_none=True). The optimizer is the real fused AdamW from `build_optimizer`, and
its state exists before measurement because it is created during warmup. The LR is held
constant; it does not affect speed.

Timing (Stage 5 correction): the in-update sync (reading loss and grad norm) happens BEFORE the
optimizer step, so the optimizer kernels of the last update could still be running when a
Stage 4 timer stopped. Now:
- `whole_interval`: torch.cuda.synchronize() before the first and after the last measured
  update; the planning figure. No extra syncs inside, exactly like the training loop.
- `per_update`: separate updates, each bracketed by torch.cuda.synchronize() on both sides, so
  each time includes forward, backward, accumulation, clipping, optimizer step, zero_grad and
  completion of all CUDA work.
"""

from __future__ import annotations

import dataclasses
import gc
import statistics
import time
from typing import Any

import numpy as np
import torch

from gibc.config import ModelConfig
from gibc.data import Batcher
from gibc.gpu import gpu_sample
from gibc.model import CausalLM
from gibc.train import TrainConfig, autocast_context, build_optimizer

MIB = 2**20


def run_benchmark(tokens: np.ndarray, model_cfg: ModelConfig, base_cfg: TrainConfig, precision: str,
                  micro_batch: int, grad_accum: int, warmup_updates: int, measured_updates: int,
                  sustain_seconds: float = 0.0, seed: int = 0) -> dict[str, Any]:
    device = torch.device("cuda")
    seq_len = model_cfg.context_length
    cfg = dataclasses.replace(base_cfg, seq_len=seq_len, micro_batch_size=micro_batch, grad_accum_steps=grad_accum,
                              precision=precision, device="cuda", fused_adamw=True, warmup_steps=0)
    result: dict[str, Any] = {
        "timing_method": "v2: synchronize() at measurement boundaries (after optimizer step + zero_grad)",
        "precision": precision, "micro_batch": micro_batch, "seq_len": seq_len, "activation_checkpointing": False,
        "grad_accum": grad_accum, "tokens_per_microstep": micro_batch * seq_len,
        "tokens_per_update": cfg.tokens_per_step, "warmup_updates": warmup_updates,
        "measured_updates": measured_updates, "sustain_seconds": sustain_seconds,
        "gpu": torch.cuda.get_device_name(device), "torch": str(torch.__version__),
    }
    if cfg.allow_tf32:
        torch.set_float32_matmul_precision("high")
    result["allocated_before_mib"] = torch.cuda.memory_allocated(device) / MIB
    torch.manual_seed(seed)
    model = optimizer = scaler = None
    try:
        model = CausalLM(model_cfg).to(device).train()
        optimizer = build_optimizer(model, cfg, device)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        batcher = Batcher(tokens, seq_len, micro_batch, seed)
        skips = 0

        def update() -> float:
            nonlocal skips
            loss_sum = torch.zeros((), device=device)
            for _ in range(grad_accum):
                x, y = batcher.next_batch()
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
                with autocast_context(precision, device):
                    _, loss = model(x, y, validate_targets=False)
                scaler.scale(loss / grad_accum).backward()
                loss_sum += loss.detach()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            loss_value, _ = torch.stack([loss_sum / grad_accum, grad_norm.float()]).tolist()
            scale_before = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                skips += 1
            optimizer.zero_grad(set_to_none=True)
            return loss_value

        for _ in range(warmup_updates):
            update()
        torch.cuda.synchronize(device)
        warmup_skips, skips = skips, 0
        torch.cuda.reset_peak_memory_stats(device)
        gpu = [{"t_s": 0.0, **gpu_sample()}]

        # 1) Whole interval: sync, N back-to-back updates, sync.
        losses = []
        t_start = time.perf_counter()
        n = 0
        while n < measured_updates or (sustain_seconds and time.perf_counter() - t_start < sustain_seconds):
            losses.append(update())
            n += 1
        torch.cuda.synchronize(device)
        interval_s = time.perf_counter() - t_start
        gpu.append({"t_s": round(interval_s, 1), **gpu_sample()})

        # 2) Per update: every update bracketed by synchronize() on both sides.
        step_times = []
        for _ in range(measured_updates):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            losses.append(update())
            torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - t0)
        rates = [cfg.tokens_per_step / t for t in step_times]
        result.update({
            "status": "ok",
            "whole_interval_updates": n,
            "whole_interval_seconds": interval_s,
            "tokens_per_s_whole_interval": cfg.tokens_per_step * n / interval_s,
            "per_update_count": len(step_times),
            "tokens_per_s_per_update_mean": cfg.tokens_per_step * len(step_times) / sum(step_times),
            "tokens_per_s_per_update_median": statistics.median(rates),
            "step_time_s_mean": sum(step_times) / len(step_times),
            "step_time_s_median": statistics.median(step_times),
            "step_time_s_min": min(step_times), "step_time_s_max": max(step_times),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
            "device_total_mib": torch.cuda.mem_get_info(device)[1] / MIB,
            "scaler_skips_measured": skips,
            "scaler_skips_warmup": warmup_skips,
            "loss_first": losses[0], "loss_last": losses[-1],
            "all_losses_finite": bool(np.all(np.isfinite(losses))),
            "gpu_samples": gpu,
        })
    except torch.OutOfMemoryError as exc:
        result.update({"status": "OOM", "error": str(exc).splitlines()[0]})
    finally:
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
        result["allocated_after_cleanup_mib"] = torch.cuda.memory_allocated(device) / MIB
    return result
