"""WikiText-103 test token perplexity under the documented project methodology.

This is the project's own methodology; no organizer evaluator has been provided. It was fixed in
Stage 6, before any final-model result existed:
- Data: `Salesforce/wikitext`, config `wikitext-103-raw-v1`, split `test`, at a pinned revision.
- Text: the rows concatenated in dataset order with NO separator added. Every non-empty row
  already ends in "\n"; empty rows contribute nothing. No other change is made to the text.
- Tokens: our tokenizer (literal special-token text stays ordinary text); no BOS; no EOT between
  articles. A single EOT (id 0) is prepended as context only, so the first real token can be
  predicted; it is never a target. The scored targets are exactly the N tokenizer tokens of the
  text, each scored once.
- Windows: rolling causal windows of at most 512 input tokens with stride 256. The first window
  scores all its targets. Each later window advances the scored frontier by at most 256 targets
  and scores only those new targets, using up to 256 preceding tokens as unscored context. No
  padding.
- Metric: token perplexity = exp(total NLL / N). The NLL is summed in float64 over all N scored
  targets before exponentiation. The window context is not in the denominator.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import torch
import torch.nn.functional as F

DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-103-raw-v1"
SPLIT = "test"
MAX_LEN = 512
STRIDE = 256
METHOD_LABEL = "WikiText-103 test token perplexity under the documented project methodology"


def rolling_windows(n_targets: int, max_len: int = MAX_LEN, stride: int = STRIDE) -> list[tuple[int, int, int]]:
    """Windows over stream = [EOT] + tokens (length n_targets + 1); targets are stream positions 1..n_targets.

    Returns (start, end, first_target): input = stream[start:end] predicts stream[start+1 : end+1];
    only targets first_target..end are scored in that window.
    """
    if not 0 < stride <= max_len:
        raise ValueError("need 0 < stride <= max_len")
    windows, scored = [], 0
    while scored < n_targets:
        end = min(n_targets, scored + (max_len if scored == 0 else stride))
        start = max(0, end - max_len)
        windows.append((start, end, scored + 1))
        scored = end
    return windows


def score_stream(logits_fn: Callable[[torch.Tensor], torch.Tensor], stream: list[int], max_len: int = MAX_LEN,
                 stride: int = STRIDE, batch_size: int = 8, device: str = "cpu") -> dict:
    """Total NLL over targets 1..len(stream)-1 of `stream` (stream[0] is the context-only EOT)."""
    n_targets = len(stream) - 1
    windows = rolling_windows(n_targets, max_len, stride)
    ids = torch.tensor(stream, dtype=torch.long)
    total_nll = 0.0
    scored = 0
    t0 = time.perf_counter()
    i = 0
    while i < len(windows):
        # Batch consecutive windows of equal input length (all but at most one are max_len), so no padding.
        length = windows[i][1] - windows[i][0]
        batch = [windows[i]]
        while len(batch) < batch_size and i + len(batch) < len(windows) and \
                windows[i + len(batch)][1] - windows[i + len(batch)][0] == length:
            batch.append(windows[i + len(batch)])
        inp = torch.stack([ids[s:e] for s, e, _ in batch]).to(device)
        tgt = torch.stack([ids[s + 1:e + 1] for s, e, _ in batch]).to(device)
        logp = F.log_softmax(logits_fn(inp).float(), dim=-1).gather(2, tgt[..., None]).squeeze(-1)
        for row, (s, e, first) in enumerate(batch):
            keep = logp[row, first - 1 - s: e - s]  # positions predicting targets first..e
            total_nll -= keep.double().sum().item()
            scored += keep.numel()
        i += len(batch)
    if scored != n_targets:
        raise AssertionError(f"scored {scored} targets, expected {n_targets}")
    return {"total_nll": total_nll, "scored_tokens": scored, "windows": len(windows),
            "token_perplexity": math.exp(total_nll / scored) if total_nll / scored < 700 else math.inf,
            "seconds": time.perf_counter() - t0}


def load_text(revision: str) -> tuple[str, dict]:
    from datasets import load_dataset

    rows = load_dataset(DATASET, CONFIG, split=SPLIT, revision=revision)["text"]
    bad = [i for i, r in enumerate(rows) if r and not r.endswith("\n")]
    if bad:
        raise ValueError(f"{len(bad)} non-empty rows do not end with a newline (first: {bad[0]})")
    info = {"rows": len(rows), "empty_rows": sum(1 for r in rows if not r)}
    text = "".join(rows)
    info["characters"] = len(text)
    return text, info
