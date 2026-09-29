"""Decoder-only causal Transformer (Llama-style): pre-norm RMSNorm, RoPE, full MHA, SwiGLU, tied head.

Module and parameter names mirror Hugging Face `LlamaForCausalLM`
(`model.embed_tokens`, `model.layers.N.self_attn.q_proj`, `...mlp.gate_proj`,
`...input_layernorm`, `model.norm`, `lm_head`), so a later export is a direct key match.
RoPE uses the same non-interleaved ("rotate_half") convention as HF Llama.

Target convention (the only one): `forward(input_ids, targets)` expects `targets` to be ALREADY
shifted, i.e. `targets[b, t]` is the label for the logits at position t (the token that follows
input_ids[b, t] in the source text). The model never shifts internally. There is no padding or
ignore-index semantics: every target must be a valid token id.

Initialization (explicit, see `_init_weights`):
- every 2-D weight (token embedding = tied output head, q/k/v/o, gate/up/down) ~ N(0, 0.02²),
  except the residual output projections `self_attn.o_proj.weight` and `mlp.down_proj.weight`
  in every layer, which use std 0.02 / sqrt(2 * n_layers) (GPT-2-style residual scaling);
- every RMSNorm scale = 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gibc.config import ModelConfig

INIT_STD = 0.02
RESIDUAL_PROJECTIONS = ("self_attn.o_proj.weight", "mlp.down_proj.weight")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 for stability under bf16/fp16, cast back, then scale (same order as HF Llama).
        input_dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Unscaled RoPE tables for positions [0, max_positions). Non-persistent buffers, no parameters."""

    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(max_positions, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim], halves duplicated (rotate_half layout)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, head_dim]; cos/sin: [T, head_dim]."""
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d = x.shape
        shape = (batch, seq_len, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        # is_causal=True: the kernel applies the lower-triangular mask itself (no mask tensor built).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(batch, seq_len, d))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class DecoderModel(nn.Module):
    """Embedding, decoder stack and final norm (HF: `LlamaModel`)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.context_length, cfg.rope_theta)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cos, sin = self.rotary(input_ids.shape[1])
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class CausalLM(nn.Module):
    """Language model with the output head tied to the token embedding (HF: `LlamaForCausalLM`)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if not cfg.tie_embeddings:
            raise ValueError("untied embeddings exceed the 50M parameter cap; tie_embeddings must be true")
        self.config = cfg
        self.model = DecoderModel(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight  # one Parameter object, two names
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        residual_std = INIT_STD / math.sqrt(2 * self.config.n_layers)
        # named_parameters() yields the tied tensor once, as model.embed_tokens.weight.
        for name, param in self.named_parameters():
            if name.endswith(RESIDUAL_PROJECTIONS):
                nn.init.normal_(param, mean=0.0, std=residual_std)
            elif param.dim() == 2:
                nn.init.normal_(param, mean=0.0, std=INIT_STD)
            else:
                nn.init.ones_(param)  # RMSNorm scales

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, validate_targets: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (logits [B, T, V], mean cross-entropy or None). `targets` must be pre-shifted (module docstring).

        `validate_targets=True` (default) checks every target id is in [0, vocab_size); on CUDA this
        forces a GPU->CPU sync. The training loop passes False because its token files are
        range-checked when written and again when opened (gibc.data).
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [batch, seq], got shape {tuple(input_ids.shape)}")
        seq_len = input_ids.shape[1]
        if not 1 <= seq_len <= self.config.context_length:
            raise ValueError(f"sequence length {seq_len} outside [1, {self.config.context_length}]")

        logits = self.lm_head(self.model(input_ids))
        if targets is None:
            return logits, None
        if targets.shape != input_ids.shape:
            raise ValueError(f"targets shape {tuple(targets.shape)} != input_ids shape {tuple(input_ids.shape)}")
        # No ignore_index semantics (there is no PAD token): reject anything that is not a real token id,
        # including -100, which F.cross_entropy would otherwise silently skip.
        if validate_targets and bool(((targets < 0) | (targets >= self.config.vocab_size)).any()):
            raise ValueError("targets contain ids outside [0, vocab_size)")
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


def parameter_report(model: CausalLM) -> dict[str, int]:
    """Count UNIQUE trainable Parameter objects of an instantiated model, grouped by component.

    Tied weights are one Parameter; each is counted once (by object identity), under the first
    group it matches. Raises if any parameter falls outside the known groups.
    """
    groups = {
        "embedding (tied with output head)": lambda n: n == "model.embed_tokens.weight",
        "attention projections": lambda n: ".self_attn." in n,
        "mlp projections": lambda n: ".mlp." in n,
        "block rmsnorms": lambda n: n.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")),
        "final rmsnorm": lambda n: n == "model.norm.weight",
    }
    counts = {group: 0 for group in groups}
    seen: set[int] = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        group = next((g for g, match in groups.items() if match(name)), None)
        if group is None:
            raise ValueError(f"unclassified parameter {name}")
        counts[group] += param.numel()
    if not weights_are_tied(model):
        raise ValueError("output head is not tied to the token embedding")
    counts["output head (shares the embedding tensor)"] = 0
    counts["total unique trainable"] = sum(counts.values())
    return counts


def weights_are_tied(model: CausalLM) -> bool:
    """Identity check (same Parameter object and same storage), not value equality."""
    head, emb = model.lm_head.weight, model.model.embed_tokens.weight
    return head is emb and head.data_ptr() == emb.data_ptr()
