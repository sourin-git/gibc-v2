"""Checkpoint I/O: atomic writes and RNG state capture/restore.

A checkpoint is a plain dict of tensors and Python primitives, so it loads with
`torch.load(weights_only=True)`. It is always loaded to CPU first: RNG state tensors must be CPU
ByteTensors, and model/optimizer state is copied onto the parameters' device by load_state_dict.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_FORMAT = "gibc.checkpoint.v1"


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Write beside the destination (same filesystem), then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unknown checkpoint format {payload.get('format')!r}")
    return payload


def load_model_from_checkpoint(path: Path, device: str = "cpu"):
    """Rebuild the CausalLM from any compatible project checkpoint (config + weights), in eval mode.
    Returns (model, checkpoint_dict)."""
    from gibc.config import ModelConfig
    from gibc.model import CausalLM

    payload = load_checkpoint(path)
    model = CausalLM(ModelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


def capture_rng_state() -> dict[str, Any]:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": name, "keys": keys.tolist(), "pos": pos, "has_gauss": has_gauss, "cached": cached},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.array(n["keys"], dtype=np.uint32), n["pos"], n["has_gauss"], n["cached"]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
