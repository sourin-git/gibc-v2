"""Model configuration and analytic parameter count.

`param_breakdown` is derived from the architecture spec, independently of the model
code. Tests pin it to a hand-derived value and, once the model exists, assert that the
instantiated model has exactly this many trainable parameters.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# GIBC V2 Track 01: maximum total trainable parameters, embeddings and output head included.
PARAM_LIMIT = 50_000_000

_POSITIVE_INT_FIELDS = ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "context_length")
_POSITIVE_FLOAT_FIELDS = ("rope_theta", "norm_eps")


@dataclass(frozen=True)
class ModelConfig:
    """Decoder-only Transformer: pre-norm RMSNorm, RoPE, SwiGLU MLP, no biases anywhere."""

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    context_length: int
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        # bool is a subclass of int, so it is rejected explicitly in the numeric checks.
        for name in _POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in _POSITIVE_FLOAT_FIELDS:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number, got {value!r}")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError(f"tie_embeddings must be a bool, got {self.tie_embeddings!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown ModelConfig keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def param_breakdown(cfg: ModelConfig) -> dict[str, int]:
    """Trainable parameters by component. A tied output head shares the embedding and counts 0."""
    d, v = cfg.d_model, cfg.vocab_size
    attn = 4 * d * d  # q, k, v, o projections; every head has its own k/v (no GQA)
    mlp = 3 * d * cfg.d_ff  # SwiGLU gate, up, down projections
    norms = 2 * d  # RMSNorm weights before attention and before the MLP
    per_layer = attn + mlp + norms
    counts = {
        "embedding": v * d,
        "attn_per_layer": attn,
        "mlp_per_layer": mlp,
        "norms_per_layer": norms,
        "per_layer": per_layer,
        "all_layers": cfg.n_layers * per_layer,
        "final_norm": d,
        "lm_head": 0 if cfg.tie_embeddings else v * d,
    }
    counts["total"] = counts["embedding"] + counts["all_layers"] + counts["final_norm"] + counts["lm_head"]
    return counts
