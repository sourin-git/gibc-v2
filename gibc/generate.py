"""Minimal sampling loop for smoke tests. No KV cache: every step re-runs the (cropped) context."""

from __future__ import annotations

from typing import Sequence

import torch

from gibc.model import CausalLM


@torch.no_grad()
def generate(
    model: CausalLM,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    stop_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Return prompt_ids + generated ids. temperature == 0 means greedy (argmax).

    The context is cropped to the model's last `context_length` tokens. Generation stops after
    emitting `stop_id` (included in the output) if given. Sampling runs on CPU, so `generator`
    must be a CPU torch.Generator; that also keeps seeded samples device-independent.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty (there is no BOS token to start from)")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    try:
        for _ in range(max_new_tokens):
            context = torch.tensor([ids[-model.config.context_length :]], dtype=torch.long, device=device)
            logits = model(context)[0][0, -1].float()
            if temperature == 0:
                next_id = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs.cpu(), 1, generator=generator))
            ids.append(next_id)
            if stop_id is not None and next_id == stop_id:
                break
    finally:
        model.train(was_training)
    return ids
