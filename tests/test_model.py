import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gibc.config import PARAM_LIMIT, ModelConfig, param_breakdown
from gibc.generate import generate
from gibc.model import (
    INIT_STD,
    CausalLM,
    RotaryEmbedding,
    apply_rotary,
    parameter_report,
    weights_are_tied,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_CFG = ModelConfig.from_json(REPO_ROOT / "configs" / "model" / "gibc_43m.json")
TINY_CFG = ModelConfig(vocab_size=97, d_model=64, n_layers=2, n_heads=4, d_ff=96, context_length=32)
EXPECTED_TOTAL = 42_968_576


def build(cfg: ModelConfig, seed: int = 0) -> CausalLM:
    torch.manual_seed(seed)
    return CausalLM(cfg)


@pytest.fixture(scope="module")
def main_model() -> CausalLM:
    return build(MAIN_CFG).eval()


@pytest.fixture
def tiny_model() -> CausalLM:
    return build(TINY_CFG)


def random_ids(cfg: ModelConfig, batch: int, seq: int, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (batch, seq), generator=torch.Generator().manual_seed(seed))


# A-D: construction, exact instantiated count, cap, tying ------------------------------------

def test_instantiated_unique_trainable_count_is_exact(main_model):
    unique = {id(p): p for p in main_model.parameters() if p.requires_grad}
    total = sum(p.numel() for p in unique.values())
    assert total == EXPECTED_TOTAL
    assert total == param_breakdown(MAIN_CFG)["total"]
    assert total < PARAM_LIMIT
    assert all(p.requires_grad for p in main_model.parameters())


def test_parameter_report_matches_analytic_components(main_model):
    report = parameter_report(main_model)
    analytic = param_breakdown(MAIN_CFG)
    L = MAIN_CFG.n_layers
    assert report["embedding (tied with output head)"] == analytic["embedding"]
    assert report["attention projections"] == L * analytic["attn_per_layer"]
    assert report["mlp projections"] == L * analytic["mlp_per_layer"]
    assert report["block rmsnorms"] == L * analytic["norms_per_layer"]
    assert report["final rmsnorm"] == analytic["final_norm"]
    assert report["output head (shares the embedding tensor)"] == 0
    assert report["total unique trainable"] == EXPECTED_TOTAL


def test_output_head_is_the_embedding_parameter(main_model):
    assert weights_are_tied(main_model)
    assert main_model.lm_head.weight is main_model.model.embed_tokens.weight
    names = [n for n, _ in main_model.named_parameters(remove_duplicate=False)]
    assert "lm_head.weight" in names and "model.embed_tokens.weight" in names
    assert "lm_head.weight" not in dict(main_model.named_parameters())  # deduplicated
    state = main_model.state_dict()
    assert state["lm_head.weight"].data_ptr() == state["model.embed_tokens.weight"].data_ptr()


def test_tying_survives_an_update(tiny_model):
    with torch.no_grad():
        tiny_model.model.embed_tokens.weight[3, 5] = 123.0
    assert tiny_model.lm_head.weight[3, 5].item() == 123.0


def test_untied_config_rejected():
    with pytest.raises(ValueError, match="tie_embeddings"):
        CausalLM(ModelConfig(**{**TINY_CFG.to_dict(), "tie_embeddings": False}))


def test_parameter_names_and_shapes(main_model):
    d, f, v = MAIN_CFG.d_model, MAIN_CFG.d_ff, MAIN_CFG.vocab_size
    shapes = {n: tuple(p.shape) for n, p in main_model.named_parameters()}
    assert shapes["model.embed_tokens.weight"] == (v, d)
    assert shapes["model.norm.weight"] == (d,)
    for i in range(MAIN_CFG.n_layers):
        p = f"model.layers.{i}."
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert shapes[p + f"self_attn.{proj}.weight"] == (d, d)
        assert shapes[p + "mlp.gate_proj.weight"] == (f, d)
        assert shapes[p + "mlp.up_proj.weight"] == (f, d)
        assert shapes[p + "mlp.down_proj.weight"] == (d, f)
        assert shapes[p + "input_layernorm.weight"] == (d,)
        assert shapes[p + "post_attention_layernorm.weight"] == (d,)
    assert len(shapes) == 2 + 9 * MAIN_CFG.n_layers


# O, P: no RoPE parameters, no biases ---------------------------------------------------------

def test_no_bias_and_no_rope_parameters(main_model):
    assert not any("bias" in n for n, _ in main_model.named_parameters(remove_duplicate=False))
    assert all(m.bias is None for m in main_model.modules() if isinstance(m, nn.Linear))
    assert not any("rotary" in n for n, _ in main_model.named_parameters())
    assert not any("rotary" in k for k in main_model.state_dict())  # non-persistent buffers
    assert list(main_model.model.rotary.parameters()) == []


# Initialization and determinism (N) ----------------------------------------------------------

def test_initialization_statistics(main_model):
    residual_std = INIT_STD / math.sqrt(2 * MAIN_CFG.n_layers)
    for name, param in main_model.named_parameters():
        if param.dim() == 1:
            assert torch.all(param == 1), name
        elif name.endswith(("o_proj.weight", "down_proj.weight")):
            assert param.std().item() == pytest.approx(residual_std, rel=0.05), name
        else:
            assert param.std().item() == pytest.approx(INIT_STD, rel=0.05), name
            assert abs(param.mean().item()) < 1e-3, name


def test_deterministic_initialization():
    a, b, c = build(TINY_CFG, seed=7), build(TINY_CFG, seed=7), build(TINY_CFG, seed=8)
    for (name, pa), (_, pb), (_, pc) in zip(a.named_parameters(), b.named_parameters(), c.named_parameters()):
        assert torch.equal(pa, pb), name
        if pa.dim() == 2:
            assert not torch.equal(pa, pc), name


# E-I: forward, loss, backward, gradients -----------------------------------------------------

def test_forward_shapes_and_initial_loss(main_model):
    ids = random_ids(MAIN_CFG, 2, 64)
    targets = random_ids(MAIN_CFG, 2, 64, seed=2)
    with torch.no_grad():
        logits, loss = main_model(ids, targets)
        logits_only, no_loss = main_model(ids)
    assert logits.shape == (2, 64, MAIN_CFG.vocab_size)
    assert no_loss is None and torch.equal(logits, logits_only)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(math.log(MAIN_CFG.vocab_size), abs=0.5)  # ~uniform at init


def test_target_convention_is_pre_shifted_no_internal_shift(tiny_model):
    ids = random_ids(TINY_CFG, 2, 16)
    targets = random_ids(TINY_CFG, 2, 16, seed=3)
    logits, loss = tiny_model(ids, targets)
    expected = F.cross_entropy(logits.reshape(-1, TINY_CFG.vocab_size), targets.reshape(-1))
    shifted = F.cross_entropy(logits[:, :-1].reshape(-1, TINY_CFG.vocab_size), targets[:, 1:].reshape(-1))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert loss.item() != pytest.approx(shifted.item(), rel=1e-6)


@pytest.mark.parametrize("bad", [-100, -1, TINY_CFG.vocab_size])
def test_invalid_targets_rejected(tiny_model, bad):
    ids = random_ids(TINY_CFG, 1, 8)
    targets = ids.clone()
    targets[0, 3] = bad
    with pytest.raises(ValueError, match="outside"):
        tiny_model(ids, targets)


def test_target_shape_mismatch_rejected(tiny_model):
    ids = random_ids(TINY_CFG, 1, 8)
    with pytest.raises(ValueError, match="targets shape"):
        tiny_model(ids, ids[:, :-1])


def test_backward_populates_gradients(tiny_model):
    ids = random_ids(TINY_CFG, 2, 16)
    _, loss = tiny_model(ids, random_ids(TINY_CFG, 2, 16, seed=4))
    loss.backward()
    representative = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.norm.weight",
    ]
    params = dict(tiny_model.named_parameters())
    for name in representative:
        grad = params[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert all(p.grad is not None for p in tiny_model.parameters())


def test_training_step_reduces_loss_on_fixed_batch(tiny_model):
    ids = random_ids(TINY_CFG, 4, 32)
    targets = torch.roll(ids, -1, dims=1)
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-3)
    first = None
    for _ in range(30):
        _, loss = tiny_model(ids, targets)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.7


# J: causality --------------------------------------------------------------------------------

def test_future_tokens_do_not_affect_earlier_logits(tiny_model):
    tiny_model.eval()
    ids = random_ids(TINY_CFG, 2, 24)
    t = 10
    changed = ids.clone()
    changed[:, t:] = (changed[:, t:] + 1) % TINY_CFG.vocab_size
    with torch.no_grad():
        a, _ = tiny_model(ids)
        b, _ = tiny_model(changed)
    torch.testing.assert_close(a[:, :t], b[:, :t], rtol=0, atol=1e-6)
    assert not torch.allclose(a[:, t:], b[:, t:])


def test_prefix_logits_match_shorter_sequence(tiny_model):
    tiny_model.eval()
    ids = random_ids(TINY_CFG, 1, 20)
    with torch.no_grad():
        full, _ = tiny_model(ids)
        prefix, _ = tiny_model(ids[:, :7])
    torch.testing.assert_close(full[:, :7], prefix, rtol=1e-5, atol=1e-5)


# RoPE semantics ------------------------------------------------------------------------------

def test_rope_is_relative_and_rotate_half_layout():
    rope = RotaryEmbedding(head_dim=64, max_positions=512, theta=10_000.0)
    cos, sin = rope(512)
    q, k = torch.randn(64, generator=torch.Generator().manual_seed(0)), torch.randn(64)

    def rotated(x, pos):
        return apply_rotary(x.view(1, 1, 1, 64), cos[pos : pos + 1], sin[pos : pos + 1]).flatten()

    assert torch.dot(rotated(q, 5), rotated(k, 2)).item() == pytest.approx(
        torch.dot(rotated(q, 305), rotated(k, 302)).item(), abs=1e-3
    )
    torch.testing.assert_close(rotated(q, 0), q)
    # HF Llama layout: frequency i pairs dims (i, i + 32), inv_freq_i = theta^(-2i/64).
    inv_freq = 10_000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    torch.testing.assert_close(cos[7], torch.cat([torch.cos(7 * inv_freq)] * 2))
    torch.testing.assert_close(sin[7], torch.cat([torch.sin(7 * inv_freq)] * 2))


# K-M: sequence length bounds -----------------------------------------------------------------

def test_sequence_length_one(main_model):
    with torch.no_grad():
        logits, loss = main_model(torch.tensor([[5]]), torch.tensor([[6]]))
    assert logits.shape == (1, 1, MAIN_CFG.vocab_size) and torch.isfinite(loss)


def test_sequence_length_512(main_model):
    ids = random_ids(MAIN_CFG, 1, 512)
    with torch.no_grad():
        logits, loss = main_model(ids, ids)
    assert logits.shape == (1, 512, MAIN_CFG.vocab_size) and torch.isfinite(loss)


@pytest.mark.parametrize("shape", [(1, 513), (1, 0), (512,)])
def test_invalid_input_shapes_rejected(main_model, shape):
    with pytest.raises(ValueError):
        main_model(torch.zeros(shape, dtype=torch.long))


# Q: CUDA -------------------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tiny_cuda_forward_backward_bf16(tiny_model):
    model = tiny_model.cuda()
    ids = random_ids(TINY_CFG, 2, 32).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = model(ids, torch.roll(ids, -1, dims=1))
    assert logits.dtype == torch.bfloat16 and loss.dtype == torch.float32
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert weights_are_tied(model)


# Tokenizer integration -----------------------------------------------------------------------

def test_tokenizer_model_integration(main_model):
    from gibc.tokenizer import decode, encode_text, eot_id, load_tokenizer

    tok = load_tokenizer(REPO_ROOT / "results" / "tokenizer" / "tokenizer.json")
    assert tok.get_vocab_size(with_added_tokens=True) == MAIN_CFG.vocab_size == 24_000
    assert eot_id(tok) == 0
    ids = encode_text(tok, "Plants convert sunlight into chemical energy.")
    with torch.no_grad():
        logits, _ = main_model(torch.tensor([ids]))
    assert logits.shape == (1, len(ids), 24_000)
    out = generate(main_model, ids, max_new_tokens=3, temperature=0)
    assert isinstance(decode(tok, out), str) and len(out) == len(ids) + 3


# Generation ----------------------------------------------------------------------------------

def test_generate_greedy_is_deterministic_and_crops_context(tiny_model):
    prompt = list(range(40))  # longer than context_length=32
    a = generate(tiny_model, prompt, max_new_tokens=5, temperature=0)
    b = generate(tiny_model, prompt, max_new_tokens=5, temperature=0)
    assert a == b and len(a) == 45 and a[:40] == prompt
    assert tiny_model.training  # mode restored
    top1 = generate(tiny_model, prompt, max_new_tokens=5, temperature=1.0, top_k=1)
    assert top1 == a


def test_generate_sampling_is_seeded():
    model = build(TINY_CFG)
    run = lambda seed: generate(model, [1, 2, 3], 8, temperature=1.0, top_k=10,
                                generator=torch.Generator().manual_seed(seed))
    assert run(0) == run(0)
    assert all(0 <= t < TINY_CFG.vocab_size for t in run(0))


def test_generate_stops_at_stop_id(tiny_model):
    # Zeroing the final RMSNorm scale makes every logit 0; argmax of a constant vector is index 0.
    with torch.no_grad():
        tiny_model.model.norm.weight.zero_()
    out = generate(tiny_model, [5, 6], max_new_tokens=10, temperature=0, stop_id=0)
    assert out == [5, 6, 0]
