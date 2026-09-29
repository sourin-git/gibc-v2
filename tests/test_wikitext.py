"""Rolling-window WikiText scoring: every target counted exactly once, context excluded, exact totals."""

import math

import pytest
import torch
import torch.nn.functional as F

from gibc.config import ModelConfig
from gibc.model import CausalLM
from gibc.scoring import ours_logits_fn
from gibc.wikitext import rolling_windows, score_stream


@pytest.mark.parametrize("n", [1, 5, 16, 17, 40, 100])
@pytest.mark.parametrize("max_len, stride", [(16, 8), (16, 16), (16, 5), (8, 1)])
def test_windows_score_each_target_once_with_bounded_context(n, max_len, stride):
    scored = []
    for start, end, first in rolling_windows(n, max_len, stride):
        assert 0 <= start < first <= end <= n and end - start <= max_len
        scored += list(range(first, end + 1))
        if first > 1:  # later windows keep >= max_len - stride tokens of unscored context
            assert first - 1 - start >= min(max_len - stride, first - 1)
    assert scored == list(range(1, n + 1))  # targets 1..n, once each, in order; position 0 never scored


def test_uniform_model_gives_exact_totals():
    vocab, n = 50, 37
    stream = [0] + [i % vocab for i in range(1, n + 1)]
    result = score_stream(lambda ids: torch.zeros(*ids.shape, vocab), stream, max_len=16, stride=8, batch_size=3)
    assert result["scored_tokens"] == n
    assert result["total_nll"] == pytest.approx(n * math.log(vocab))
    assert result["token_perplexity"] == pytest.approx(vocab)


def test_matches_brute_force_and_is_batch_invariant():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=2, d_ff=48, context_length=16)
    model = CausalLM(cfg).eval()
    stream = [0] + torch.randint(1, 97, (60,), generator=torch.Generator().manual_seed(1)).tolist()
    fn = ours_logits_fn(model)
    a = score_stream(fn, stream, max_len=16, stride=6, batch_size=1)
    b = score_stream(fn, stream, max_len=16, stride=6, batch_size=8)
    brute = 0.0
    for start, end, first in rolling_windows(60, 16, 6):
        for t in range(first, end + 1):  # score target t with exactly the window's context
            inp = torch.tensor([stream[start:end]])
            logp = F.log_softmax(fn(inp)[0], -1)[t - 1 - start, stream[t]].item()
            brute -= logp
    assert a["total_nll"] == pytest.approx(b["total_nll"], rel=1e-6)
    assert a["total_nll"] == pytest.approx(brute, rel=1e-5)
    assert a["scored_tokens"] == b["scored_tokens"] == 60
