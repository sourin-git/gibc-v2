"""Export a project model to a LOCAL Hugging Face `LlamaForCausalLM` + tokenizer directory.

Nothing is downloaded: the HF model is built from a `LlamaConfig` (random init) and every weight
is then overwritten from our checkpoint. Our module and parameter names already equal HF Llama's
(`model.embed_tokens`, `model.layers.N.self_attn.q_proj`, ..., `model.norm`, `lm_head`), so the
mapping is the identity. It is still verified tensor by tensor (shape, dtype, exact values).

Config choices that differ from transformers 5.17 LlamaConfig defaults:
- rms_norm_eps 1e-5 (default 1e-6)
- tie_word_embeddings True (default False)
- bos_token_id None (default 1)
- eos_token_id 0 (default 2)
- pad_token_id None, because Llama builds nn.Embedding(padding_idx=pad_token_id). Padding is
  handled only at the tokenizer level (pad = EOT id 0, masked or right-padded).
- rope_parameters {"rope_type": "default", "rope_theta": 10000.0}, i.e. unscaled RoPE.

Tokenizer: `TokenizersBackend` built from our exact tokenizer.json. EOS and PAD are the existing
`<|endoftext|>` (id 0); BOS and UNK are None, so nothing is added to the 24,000 entries. No BOS
or EOS is inserted. `split_special_tokens=True` makes literal "<|endoftext|>" text encode as
ordinary bytes, like `gibc.tokenizer.load_tokenizer`; it is saved in tokenizer_config.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gibc.config import ModelConfig
from gibc.model import CausalLM
from gibc.tokenizer import EOT_TOKEN

HF_EXPORT_FORMAT = "gibc.hf_export.v1"


def llama_config(cfg: ModelConfig, eot_id: int):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.d_ff,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_heads,  # full multi-head attention
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.context_length,
        hidden_act="silu",
        rms_norm_eps=cfg.norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": float(cfg.rope_theta)},
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=eot_id,
        pad_token_id=None,
    )


def build_hf_tokenizer(tokenizer_json: Path, context_length: int):
    from transformers import TokenizersBackend

    return TokenizersBackend(
        tokenizer_file=str(tokenizer_json),
        eos_token=EOT_TOKEN,
        pad_token=EOT_TOKEN,
        bos_token=None,
        unk_token=None,
        add_bos_token=False,
        add_eos_token=False,
        split_special_tokens=True,
        model_max_length=context_length,
        clean_up_tokenization_spaces=False,
    )


def unique_trainable_parameters(model: torch.nn.Module) -> int:
    """Each Parameter object counted once, however many names or modules refer to it."""
    return sum(p.numel() for p in {id(p): p for p in model.parameters() if p.requires_grad}.values())


def hf_weights_tied(hf) -> bool:
    head, emb = hf.lm_head.weight, hf.model.embed_tokens.weight
    return head is emb and head.data_ptr() == emb.data_ptr()


def compare_weights(ours: CausalLM, hf) -> dict[str, Any]:
    """Tensor-by-tensor comparison over every name either side exposes (duplicates included)."""
    src = dict(ours.named_parameters(remove_duplicate=False))
    dst = dict(hf.named_parameters(remove_duplicate=False))
    missing = sorted(set(src) - set(dst))
    unexpected = sorted(n for n, p in dst.items() if n not in src and p.requires_grad)
    mismatched = []
    for name, p in src.items():
        q = dst.get(name)
        if q is None:
            continue
        if tuple(p.shape) != tuple(q.shape) or p.dtype != q.dtype or not torch.equal(p.detach().cpu(), q.detach().cpu()):
            mismatched.append({"name": name, "src_shape": list(p.shape), "hf_shape": list(q.shape),
                               "src_dtype": str(p.dtype), "hf_dtype": str(q.dtype)})
    return {"names_compared": len(src), "missing_in_hf": missing, "unexpected_trainable_in_hf": unexpected,
            "mismatched": mismatched, "all_equal": not (missing or unexpected or mismatched)}


def export(model: CausalLM, tokenizer_json: Path, out_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Build the HF model locally, copy our weights, verify, save model + tokenizer + export_manifest.json."""
    import transformers
    from transformers import LlamaForCausalLM

    cfg = model.config
    tok = build_hf_tokenizer(tokenizer_json, cfg.context_length)
    eot = tok.convert_tokens_to_ids(EOT_TOKEN)
    if len(tok) != cfg.vocab_size or eot != 0:
        raise ValueError(f"HF tokenizer has {len(tok)} entries / EOT id {eot}; expected {cfg.vocab_size} / 0")

    hf = LlamaForCausalLM(llama_config(cfg, eot)).float().eval()
    src_state = model.state_dict()
    if set(src_state) != set(hf.state_dict()):
        raise ValueError(f"state-dict key sets differ: ours-only {sorted(set(src_state) - set(hf.state_dict()))}, "
                         f"hf-only {sorted(set(hf.state_dict()) - set(src_state))}")
    hf.load_state_dict(src_state, strict=True)
    weights = compare_weights(model, hf)
    params = unique_trainable_parameters(hf)
    tied = hf_weights_tied(hf)
    if not weights["all_equal"] or not tied or params != unique_trainable_parameters(model):
        raise ValueError(f"export verification failed: tied={tied} params={params:,} weights={weights}")

    out_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    manifest = {
        "format": HF_EXPORT_FORMAT, **meta,
        "transformers_version": transformers.__version__, "torch_version": str(torch.__version__),
        "unique_trainable_parameters": params, "weights_tied": tied, "weight_mapping": weights,
        "tokenizer": {"len": len(tok), "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id,
                      "bos_token_id": tok.bos_token_id, "split_special_tokens": tok.split_special_tokens},
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def load_export(export_dir: Path, device: str = "cpu"):
    """Reload model + tokenizer strictly from the local directory (never the Hub)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = AutoModelForCausalLM.from_pretrained(export_dir, dtype=torch.float32, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)
    return hf.to(device).eval(), tok
