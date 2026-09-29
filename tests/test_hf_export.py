"""HF Llama export: config mapping, weights, parameter count, tying, tokenizer, logits, likelihoods,
padding and harness truncation. Uses a tiny model with the real 24,000-entry tokenizer."""

from pathlib import Path

import pytest
import torch

from gibc.config import ModelConfig
from gibc.hf_export import (
    build_hf_tokenizer,
    compare_weights,
    export,
    hf_weights_tied,
    llama_config,
    load_export,
    unique_trainable_parameters,
)
from gibc.model import CausalLM
from gibc.scoring import continuation_logprobs, encode_pair, hf_logits_fn, ours_logits_fn, sequence_logprobs
from gibc.tokenizer import encode_text, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = REPO_ROOT / "results" / "tokenizer" / "tokenizer.json"
MAIN_CFG = ModelConfig.from_json(REPO_ROOT / "configs" / "model" / "gibc_43m.json")
TINY = ModelConfig(vocab_size=24_000, d_model=32, n_layers=2, n_heads=2, d_ff=48, context_length=64)
TEXTS = ["Hello world", " leading", "a   b\tc\nd\r\n", "naïve 中文 👍🏽", "<|endoftext|>", "x<|endoftext|>y", "",
         'def f(x):\n    return x  # "q"\n', "Question: 2+2?\nAnswer: 4"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    ours = CausalLM(TINY).eval()
    out = tmp_path_factory.mktemp("export")
    manifest = export(ours, TOKENIZER_JSON, out, meta={"source_checkpoint": "test"})
    hf, tok = load_export(out)  # fresh objects, local files only
    return ours, hf, tok, out, manifest


def test_config_mapping():
    c = llama_config(MAIN_CFG, 0)
    assert (c.vocab_size, c.hidden_size, c.intermediate_size, c.num_hidden_layers) == (24_000, 512, 1536, 9)
    assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.max_position_embeddings) == (8, 8, 64, 512)
    assert c.hidden_act == "silu" and c.rms_norm_eps == 1e-5
    assert c.rope_parameters == {"rope_type": "default", "rope_theta": 10_000.0}
    assert not c.attention_bias and not c.mlp_bias and c.tie_word_embeddings
    assert c.bos_token_id is None and c.eos_token_id == 0 and c.pad_token_id is None


def test_hf_main_model_unique_parameter_count_and_tying():
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM(llama_config(MAIN_CFG, 0))
    assert unique_trainable_parameters(hf) == 42_968_576
    assert hf_weights_tied(hf)
    # Counting by name double-counts the tied tensor: exactly the untied ~55.3M figure.
    assert sum(p.numel() for _, p in hf.named_parameters(remove_duplicate=False)) == 55_256_576


def test_weight_mapping_complete_and_exact(exported):
    ours, hf, _, _, manifest = exported
    report = compare_weights(ours, hf)
    assert report["all_equal"] and report["names_compared"] == 2 + 9 * TINY.n_layers + 1
    assert manifest["weight_mapping"]["all_equal"]
    assert unique_trainable_parameters(hf) == unique_trainable_parameters(ours)


def test_tying_survives_save_and_reload(exported):
    from safetensors import safe_open

    _, hf, _, out, _ = exported
    assert hf_weights_tied(hf)
    with safe_open(out / "model.safetensors", "pt") as f:
        assert "lm_head.weight" not in f.keys() and "model.embed_tokens.weight" in f.keys()
    sd = hf.state_dict()
    assert sd["lm_head.weight"].data_ptr() == sd["model.embed_tokens.weight"].data_ptr()


def test_tokenizer_save_reload_parity(exported, tmp_path):
    _, _, tok, _, _ = exported
    ours = load_tokenizer(TOKENIZER_JSON)
    assert len(tok) == 24_000 and tok.eos_token_id == 0 and tok.pad_token_id == 0 and tok.bos_token_id is None
    for s in TEXTS:
        ref = encode_text(ours, s)
        assert tok.encode(s, add_special_tokens=False) == ref
        assert tok.encode(s) == ref and tok(s)["input_ids"] == ref  # nothing inserted (no BOS/EOS)
    tok.save_pretrained(tmp_path)
    from transformers import AutoTokenizer

    again = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert all(again.encode(s) == encode_text(ours, s) for s in TEXTS) and len(again) == 24_000


def test_literal_eot_text_is_not_special(exported):
    _, _, tok, _, _ = exported
    ids = tok.encode("a<|endoftext|>b", add_special_tokens=False)
    assert 0 not in ids and tok.decode(ids) == "a<|endoftext|>b"
    assert build_hf_tokenizer(TOKENIZER_JSON, 512).split_special_tokens


@pytest.mark.parametrize("length", [1, 2, 17, 64])
@pytest.mark.parametrize("batch", [1, 3])
def test_logit_parity(exported, length, batch):
    ours, hf, _, _, _ = exported
    ids = torch.randint(0, 24_000, (batch, length), generator=torch.Generator().manual_seed(length))
    diff = (ours_logits_fn(ours)(ids) - hf_logits_fn(hf)(ids)).abs().max().item()
    assert diff <= 1e-4


@pytest.mark.parametrize("context, cont", [("The capital of France is", " Paris."), ("The answer is ", "yes"),
                                           ("Der Bär", " ist größer — sicher!"), ("x", " 👍🏽 done")])
def test_continuation_likelihood_parity(exported, context, cont):
    ours, hf, tok, _, _ = exported
    our_tok = load_tokenizer(TOKENIZER_JSON)
    ctx_a, cont_a = encode_pair(lambda s: encode_text(our_tok, s), context, cont)
    ctx_b, cont_b = encode_pair(lambda s: tok.encode(s, add_special_tokens=False), context, cont)
    assert (ctx_a, cont_a) == (ctx_b, cont_b)
    if context.endswith(" "):
        assert cont_a == encode_text(our_tok, context + cont)[len(encode_text(our_tok, context.rstrip())):]
    a = continuation_logprobs(ours_logits_fn(ours), ctx_a, cont_a, TINY.context_length)
    b = continuation_logprobs(hf_logits_fn(hf), ctx_b, cont_b, TINY.context_length)
    assert (a - b).abs().max().item() <= 1e-4 and abs(a.sum().item() - b.sum().item()) <= 1e-4 * len(cont_a)


def test_padded_batch_invariance_with_eot_as_pad_and_real_eot_scored(exported):
    _, hf, _, _, _ = exported
    fn = hf_logits_fn(hf)
    seqs = [[5, 6, 7], [11, 12, 0, 13, 14, 15, 16, 17], [1, 2, 3, 4, 5, 0]]  # genuine EOTs are scored targets
    single = [sequence_logprobs(fn(torch.tensor([s]))[0], torch.tensor(s)) for s in seqs]
    width = max(map(len, seqs))
    for side, with_mask in (("right", False), ("right", True), ("left", True)):
        ids = torch.zeros(len(seqs), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            sl = slice(0, len(s)) if side == "right" else slice(width - len(s), width)
            ids[i, sl], mask[i, sl] = torch.tensor(s), 1
        logits = fn(ids, mask if with_mask else None)
        for i, s in enumerate(seqs):
            sl = slice(0, len(s)) if side == "right" else slice(width - len(s), width)
            got = sequence_logprobs(logits[i, sl], torch.tensor(s))
            assert (got - single[i]).abs().max().item() <= 1e-4, (side, with_mask, i)


def test_harness_left_truncates_long_context_like_our_scoring(exported):
    from lm_eval.api.instance import Instance
    from lm_eval.models.huggingface import HFLM

    ours, _, _, out, _ = exported
    our_tok = load_tokenizer(TOKENIZER_JSON)
    context = " ".join(f"word{i}" for i in range(200))  # far beyond the tiny model's 64-token context
    cont = " and the end."
    lm = HFLM(pretrained=str(out), tokenizer=str(out), device="cpu", dtype="float32", batch_size=2)
    assert lm.max_length == TINY.context_length
    (ll, _), = lm.loglikelihood([Instance("loglikelihood", {}, (context, cont), 0)], disable_tqdm=True)
    ctx_ids, cont_ids = encode_pair(lambda s: encode_text(our_tok, s), context, cont)
    assert len(ctx_ids) + len(cont_ids) > TINY.context_length + 1
    ref = continuation_logprobs(ours_logits_fn(ours), ctx_ids, cont_ids, TINY.context_length).sum().item()
    assert abs(ll - ref) <= 1e-4 * len(cont_ids)
    with pytest.raises(ValueError):
        continuation_logprobs(ours_logits_fn(ours), ctx_ids, list(range(1, 66)), TINY.context_length)
