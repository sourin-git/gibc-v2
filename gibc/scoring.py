"""Model-agnostic log-likelihood scoring, used to compare our model with its HF export.

`encode_pair` mirrors lm-evaluation-harness 0.4.13 `TemplateLM._encode_pair` for causal models:
trailing whitespace of the context moves to the continuation, the whole string is encoded, and
the continuation is whatever follows the context's own encoding. `continuation_logprobs` mirrors
`HFLM._loglikelihood_tokens`: input = (context + continuation)[-(max_length + 1):][:-1], and the
continuation is scored from the last len(continuation) positions.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F

LogitsFn = Callable[[torch.Tensor], torch.Tensor]  # [B, T] int64 -> [B, T, V] float32


def ours_logits_fn(model) -> LogitsFn:
    @torch.no_grad()
    def fn(ids: torch.Tensor) -> torch.Tensor:
        return model(ids)[0].float()

    return fn


def hf_logits_fn(hf) -> LogitsFn:
    @torch.no_grad()
    def fn(ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return hf(input_ids=ids, attention_mask=attention_mask).logits.float()

    return fn


def encode_pair(encode: Callable[[str], list[int]], context: str, continuation: str) -> tuple[list[int], list[int]]:
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = encode(context + continuation)
    context_enc = encode(context)
    return context_enc, whole[len(context_enc):]


def continuation_logprobs(logits_fn: LogitsFn, context_ids: Sequence[int], cont_ids: Sequence[int],
                          max_length: int, device: str = "cpu") -> torch.Tensor:
    """Per-token log p(continuation token | preceding tokens), float64, with harness truncation."""
    if not context_ids or not cont_ids or len(cont_ids) > max_length:
        raise ValueError("need non-empty context, non-empty continuation no longer than max_length")
    full = list(context_ids) + list(cont_ids)
    inp = torch.tensor([full[-(max_length + 1):][:-1]], dtype=torch.long, device=device)
    logp = F.log_softmax(logits_fn(inp)[0], dim=-1)
    cont = torch.tensor(cont_ids, dtype=torch.long, device=device)
    return logp[-len(cont_ids):].gather(1, cont[:, None]).squeeze(1).double().cpu()


def sequence_logprobs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log p(ids[t] | ids[<t]) for t >= 1 from logits over the same sequence: [T-1], float64."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp[:-1].gather(1, ids[1:, None]).squeeze(1).double()
