"""Stage 4 hardware benchmark: real training updates on pilot tokens, timed after warmup.

One update mirrors `gibc.train.train` exactly: `grad_accum` microsteps of pinned/non-blocking
batches, each running forward + loss under autocast and backward on the scaled loss/accum. Then
unscale, clip, one sync to read loss and grad norm, GradScaler step/update, and
zero_grad(set_to_none=True). The optimizer is the real fused AdamW from `build_optimizer`, and
its state exists before measurement because it is created during warmup. The LR is held
constant; it does not affect speed.
"""

from __future__ import annotations

import dataclasses
import gc
import statistics
import subprocess
import time
from typing import Any

import numpy as np
import torch

from gibc.config import ModelConfig
from gibc.data import Batcher
from gibc.model import CausalLM
from gibc.train import TrainConfig, autocast_context, build_optimizer

MIB = 2**20


def gpu_sample() -> dict[str, Any]:
    """Temperature, SM clock and power from nvidia-smi (read-only query)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip().split(", ")
        return {"temp_c": float(out[0]), "sm_clock_mhz": float(out[1]), "power_w": float(out[2])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def run_benchmark(tokens: np.ndarray, model_cfg: ModelConfig, base_cfg: TrainConfig, precision: str,
                  micro_batch: int, grad_accum: int, warmup_updates: int, measured_updates: int,
                  sustain_seconds: float = 0.0, seed: int = 0) -> dict[str, Any]:
    device = torch.device("cuda")
    seq_len = model_cfg.context_length
    cfg = dataclasses.replace(base_cfg, seq_len=seq_len, micro_batch_size=micro_batch, grad_accum_steps=grad_accum,
                              precision=precision, device="cuda", fused_adamw=True, warmup_steps=0)
    result: dict[str, Any] = {
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

        def update() -> tuple[float, float]:
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
            loss_value, norm = torch.stack([loss_sum / grad_accum, grad_norm.float()]).tolist()
            scale_before = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                skips += 1
            optimizer.zero_grad(set_to_none=True)
            return loss_value, norm

        for _ in range(warmup_updates):
            update()
        torch.cuda.synchronize(device)
        warmup_skips, skips = skips, 0
        torch.cuda.reset_peak_memory_stats(device)

        step_times, losses, timeline, gpu = [], [], [], [gpu_sample()]
        start = time.perf_counter()
        next_gpu_sample = start + 15.0
        n = 0
        while n < measured_updates or (sustain_seconds and time.perf_counter() - start < sustain_seconds):
            t0 = time.perf_counter()
            loss_value, norm = update()  # ends with a GPU sync (.tolist()), so the timing is real
            dt = time.perf_counter() - t0
            step_times.append(dt)
            losses.append(loss_value)
            n += 1
            timeline.append({"t_s": round(time.perf_counter() - start, 2), "tokens_per_s": cfg.tokens_per_step / dt})
            if sustain_seconds and time.perf_counter() >= next_gpu_sample:
                gpu.append({"t_s": round(time.perf_counter() - start, 1), **gpu_sample()})
                next_gpu_sample += 15.0
        total = sum(step_times)
        rates = [cfg.tokens_per_step / t for t in step_times]
        result.update({
            "status": "ok",
            "updates_timed": n,
            "tokens_per_s_mean": cfg.tokens_per_step * n / total,
            "tokens_per_s_median": statistics.median(rates),
            "step_time_s_mean": total / n,
            "step_time_s_median": statistics.median(step_times),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
            "device_total_mib": torch.cuda.mem_get_info(device)[1] / MIB,
            "scaler_skips_measured": skips,
            "scaler_skips_warmup": warmup_skips,
            "loss_first": losses[0], "loss_last": losses[-1],
            "all_losses_finite": all(np.isfinite(losses)),
            "gpu_samples": gpu + [{"t_s": round(time.perf_counter() - start, 1), **gpu_sample()}],
        })
        if sustain_seconds:
            result["timeline"] = timeline
    except torch.OutOfMemoryError as exc:
        result.update({"status": "OOM", "error": str(exc).splitlines()[0]})
    finally:
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
        result["allocated_after_cleanup_mib"] = torch.cuda.memory_allocated(device) / MIB
    return result
