"""Verify a HF export against its source checkpoint, in a FRESH process, loading only local files.

Usage: python scripts/verify_hf_export.py --export <dir> [--checkpoint <path.pt>]
Gates (exit 1 if any fails):
- config mapping; unique trainable params == source; true weight tying after reload;
- every tensor equal to the source;
- tokenizer parity with our loader (edge cases + real FineWeb-Edu docs), len 24000, EOS 0, no BOS,
  literal "<|endoftext|>" text never becomes id 0;
- fp32 logit parity (max abs diff <= 1e-4) across lengths/batch shapes, on CPU and CUDA;
- context/continuation tokenization + per-token/summed log-likelihood parity, including
  lm-eval's own HFLM scoring path and >512-token truncation;
- padded-batch invariance (right padding with/without mask, left padding with mask).
Writes <export>/verification_report.json and results/eval_preflight/hf_export_verification_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

LOGIT_TOL = 1e-4
LOGPROB_TOL = 1e-4

EDGE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.", " leading space", "   three leading", "a     b",
    "\ttab\tseparated\t", "line one\nline two\n\n", "windows\r\nline\r\n", "naïve café 中文 日本語 한국어",
    "emoji 👍🏽 👨‍👩‍👧 🇮🇳", 'def f(x):\n    return {"a": [x, x**2]}  # c\n', "!?.,;:'\"()[]{}<>@#$%^&*-_=+|\\/~`",
    "", "a", " ", "\n", "<|endoftext|>", "before<|endoftext|>after", "Question: What is 2+2?\nAnswer: 4",
]
LOGLIK_EXAMPLES = [
    ("The capital of France is", " Paris."),
    ("Question: Which gas do plants absorb from the air?\nAnswer:", " Carbon dioxide"),
    ("She opened the door and", " walked slowly into the brightly lit kitchen, humming a tune."),
    ("He shouted:", ' "Stop! Don\'t touch that!"'),
    ("The answer is ", "yes"),
    ("Der Bär ist", " größer als der Fuchs — ganz sicher."),
    ("def square(x):\n    return", " x * x\n"),
    ("Emoji test:", " 👍🏽 done"),
    ("A literal marker <|endoftext|> appears", " in this text."),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a local HF export against its source checkpoint.")
    parser.add_argument("--export", required=True)
    parser.add_argument("--checkpoint", help="default: source_checkpoint recorded in export_manifest.json")
    args = parser.parse_args()

    from lm_eval.api.instance import Instance
    from lm_eval.models.huggingface import HFLM
    from safetensors import safe_open

    from gibc.acquire import read_jsonl_documents
    from gibc.checkpoint import load_model_from_checkpoint
    from gibc.data import open_token_file
    from gibc.hf_export import compare_weights, hf_weights_tied, llama_config, load_export, unique_trainable_parameters
    from gibc.paths import REPO_ROOT, work_dir
    from gibc.scoring import continuation_logprobs, encode_pair, hf_logits_fn, ours_logits_fn, sequence_logprobs
    from gibc.tokenizer import encode_text, eot_id, load_tokenizer

    torch.set_float32_matmul_precision("highest")  # no TF32 in the reference comparison
    export_dir = Path(args.export).resolve()
    manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
    ckpt_path = Path(args.checkpoint or manifest["source_checkpoint"])
    ours, ckpt = load_model_from_checkpoint(ckpt_path, "cpu")
    hf, hf_tok = load_export(export_dir, "cpu")
    our_tok = load_tokenizer(REPO_ROOT / "results" / "tokenizer" / "tokenizer.json")
    report: dict = {"export": str(export_dir), "checkpoint": str(ckpt_path), "gates": {}}
    gates = report["gates"]

    # 1. Config mapping.
    expected = llama_config(ours.config, 0).to_dict()
    keys = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "max_position_embeddings", "hidden_act", "rms_norm_eps",
            "rope_parameters", "attention_bias", "mlp_bias", "tie_word_embeddings", "bos_token_id", "eos_token_id",
            "pad_token_id"]
    actual = hf.config.to_dict()
    report["config"] = {k: actual.get(k) for k in keys}
    gates["config_mapping"] = all(actual.get(k) == expected.get(k) for k in keys) and actual["model_type"] == "llama"

    # 2. Parameters, tying, weights, state-dict representation.
    with safe_open(export_dir / "model.safetensors", "pt") as f:
        saved_keys = list(f.keys())
    sd = hf.state_dict()
    report["parameters"] = {
        "source_unique_trainable": unique_trainable_parameters(ours),
        "hf_unique_trainable": unique_trainable_parameters(hf),
        "hf_named_parameters_with_duplicates": sum(p.numel() for _, p in hf.named_parameters(remove_duplicate=False)),
        "hf_weights_tied": hf_weights_tied(hf),
        "state_dict_keys": len(sd),
        "state_dict_lm_head_shares_storage": sd["lm_head.weight"].data_ptr() == sd["model.embed_tokens.weight"].data_ptr(),
        "safetensors_keys": len(saved_keys),
        "safetensors_has_lm_head": "lm_head.weight" in saved_keys,
    }
    p = report["parameters"]
    gates["param_count_42968576"] = p["hf_unique_trainable"] == p["source_unique_trainable"] == 42_968_576
    gates["weights_tied_after_reload"] = p["hf_weights_tied"] and p["state_dict_lm_head_shares_storage"]
    report["weights"] = compare_weights(ours, hf)
    gates["all_weights_equal"] = report["weights"]["all_equal"]

    # 3. Tokenizer parity.
    real_docs = [d["text"] for d in read_jsonl_documents(work_dir() / "data" / "pilot" / "val.jsonl")]
    real_docs += [d["text"] for _, d in zip(range(100), read_jsonl_documents(
        work_dir() / "data" / "production" / "chunks" / "chunk_00000" / "val.jsonl"))]
    mismatches = []
    for s in EDGE_TEXTS + real_docs:
        ref = encode_text(our_tok, s)
        for label, got in (("encode_nospecial", hf_tok.encode(s, add_special_tokens=False)),
                           ("encode_default", hf_tok.encode(s)), ("call", hf_tok(s)["input_ids"])):
            if got != ref:
                mismatches.append({"text": s[:80], "via": label})
    literal = hf_tok.encode("x<|endoftext|>y", add_special_tokens=False)
    report["tokenizer"] = {"texts_compared": len(EDGE_TEXTS) + len(real_docs), "real_fineweb_docs": len(real_docs),
                           "mismatches": mismatches[:20], "len": len(hf_tok), "eos_token_id": hf_tok.eos_token_id,
                           "pad_token_id": hf_tok.pad_token_id, "bos_token_id": hf_tok.bos_token_id,
                           "literal_eot_ids": literal, "class": type(hf_tok).__name__,
                           "split_special_tokens": hf_tok.split_special_tokens}
    gates["tokenizer_parity"] = (not mismatches and len(hf_tok) == 24_000 and hf_tok.eos_token_id == eot_id(our_tok) == 0
                                 and hf_tok.bos_token_id is None and 0 not in literal and hf_tok.encode("") == [])

    # 4. Logit parity (fp32, eval, no autocast).
    val_tokens = open_token_file(work_dir() / "tokens" / "production" / "val.bin", 24_000)
    rng = np.random.default_rng(0)
    cases = []
    for length in (1, 2, 17, 128, 511, 512):
        for batch in (1, 3):
            cases.append((f"random_B{batch}_T{length}", torch.from_numpy(rng.integers(0, 24_000, (batch, length)))))
    for length in (300, 512):
        starts = rng.integers(0, val_tokens.size - length, 2)
        cases.append((f"real_B2_T{length}", torch.from_numpy(np.stack([val_tokens[s:s + length] for s in starts]).astype(np.int64))))
    cases.append(("real_text_encoded", torch.tensor([encode_text(our_tok, real_docs[0])[:512]])))

    def compare(models, device, case_list):
        rows = []
        f_ours, f_hf = ours_logits_fn(models[0]), hf_logits_fn(models[1])
        for name, ids in case_list:
            a, b = f_ours(ids.to(device)), f_hf(ids.to(device))
            d = (a - b).abs()
            rows.append({"case": name, "device": device, "shape": list(ids.shape),
                         "max_abs_diff": d.max().item(), "mean_abs_diff": d.mean().item()})
        return rows

    report["logits"] = compare((ours, hf), "cpu", cases)
    if torch.cuda.is_available():
        ours.cuda(), hf.cuda()
        report["logits"] += compare((ours, hf), "cuda", [c for c in cases if c[0].startswith(("random_B3", "real"))])
        ours.cpu(), hf.cpu()
    gates["logit_parity_max_abs_le_1e-4"] = max(r["max_abs_diff"] for r in report["logits"]) <= LOGIT_TOL

    # 5. Log-likelihood parity: our pipeline vs HF model vs lm-eval's own HFLM path.
    long_ctx = " ".join(real_docs[:6])  # > 512 tokens: exercises harness left-truncation
    examples = LOGLIK_EXAMPLES + [(long_ctx, " and that is the end of the story.")]
    lm = HFLM(pretrained=str(export_dir), tokenizer=str(export_dir), device="cpu", dtype="float32",
              max_length=512, batch_size=4)
    harness = lm.loglikelihood([Instance(request_type="loglikelihood", doc={}, arguments=(c, x), idx=i)
                                for i, (c, x) in enumerate(examples)], disable_tqdm=True)
    rows = []
    for (context, cont), (harness_ll, _) in zip(examples, harness):
        ctx_ids, cont_ids = encode_pair(lambda s: encode_text(our_tok, s), context, cont)
        h_ctx, h_cont = lm._encode_pair(context, cont)
        ours_lp = continuation_logprobs(ours_logits_fn(ours), ctx_ids, cont_ids, 512)
        hf_lp = continuation_logprobs(hf_logits_fn(hf), h_ctx, h_cont, 512)
        rows.append({"context_tokens": len(ctx_ids), "continuation_ids": cont_ids,
                     "boundary_equal": (ctx_ids, cont_ids) == (h_ctx, h_cont),
                     "per_token_max_abs_diff": (ours_lp - hf_lp).abs().max().item(),
                     "sum_ours": ours_lp.sum().item(), "sum_hf": hf_lp.sum().item(), "sum_harness": harness_ll,
                     "sum_abs_diff_hf": abs(ours_lp.sum().item() - hf_lp.sum().item()),
                     "sum_abs_diff_harness": abs(ours_lp.sum().item() - harness_ll),
                     "truncated": len(ctx_ids) + len(cont_ids) > 513})
    report["loglikelihood"] = rows
    gates["loglikelihood_parity"] = all(r["boundary_equal"] and r["per_token_max_abs_diff"] <= LOGPROB_TOL
                                        and r["sum_abs_diff_hf"] <= LOGPROB_TOL * len(r["continuation_ids"])
                                        and r["sum_abs_diff_harness"] <= LOGPROB_TOL * len(r["continuation_ids"])
                                        for r in rows)
    gates["truncation_case_exercised"] = any(r["truncated"] for r in rows)

    # 6. Padded-batch invariance on the HF model (pad id = EOT 0).
    base = encode_text(our_tok, real_docs[1])
    n_cont = 5
    # The last sequence ends "... EOT, 3 tokens", so a genuine EOT (id 0 == the pad id) is a SCORED target.
    seqs = [base[:7], base[:31], base[:64], base[:150], base[:20] + [0] + base[20:23]]
    f_hf = hf_logits_fn(hf)
    single = [sequence_logprobs(f_hf(torch.tensor([s]))[0], torch.tensor(s))[-n_cont:].sum().item() for s in seqs]
    width = max(len(s) for s in seqs)

    def batched(side: str, with_mask: bool) -> list[float]:
        ids = torch.zeros(len(seqs), width, dtype=torch.long)  # pad value 0 == EOT
        mask = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, s in enumerate(seqs):
            sl = slice(0, len(s)) if side == "right" else slice(width - len(s), width)
            ids[i, sl], mask[i, sl] = torch.tensor(s), 1
        logits = f_hf(ids, mask if with_mask else None)
        out = []
        for i, s in enumerate(seqs):
            sl = slice(0, len(s)) if side == "right" else slice(width - len(s), width)
            out.append(sequence_logprobs(logits[i, sl], torch.tensor(s))[-n_cont:].sum().item())
        return out

    padding = {"single": single, "right_no_mask (lm-eval causal path)": batched("right", False),
               "right_with_mask": batched("right", True), "left_with_mask": batched("left", True)}
    diffs = {k: max(abs(a - b) for a, b in zip(v, single)) for k, v in padding.items() if k != "single"}
    eot_scored = 0 in seqs[-1][-n_cont:]
    report["padding"] = {"lengths": [len(s) for s in seqs], "continuation_tokens": n_cont,
                         "eot_in_scored_continuation": eot_scored, "scores": padding, "max_abs_diff": diffs,
                         "eot_example_logprob_finite": bool(np.isfinite(single[-1]))}
    gates["padded_batch_invariance"] = eot_scored and all(v <= LOGPROB_TOL * n_cont for v in diffs.values())

    report["passed"] = all(gates.values())
    text = json.dumps(report, indent=2, default=str) + "\n"
    (export_dir / "verification_report.json").write_text(text, encoding="utf-8", newline="\n")
    out = REPO_ROOT / "results" / "eval_preflight" / f"hf_export_verification_{export_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    for name, ok in gates.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    worst = max(report["logits"], key=lambda r: r["max_abs_diff"])
    print(f"logits: {len(report['logits'])} cases, worst max|diff| {worst['max_abs_diff']:.3e} ({worst['case']}, {worst['device']}); "
          f"mean|diff| max {max(r['mean_abs_diff'] for r in report['logits']):.3e}")
    print(f"loglik: max per-token |diff| {max(r['per_token_max_abs_diff'] for r in rows):.3e}, "
          f"max harness sum |diff| {max(r['sum_abs_diff_harness'] for r in rows):.3e}")
    print(f"padding: {diffs}")
    print(f"report -> {out.relative_to(REPO_ROOT)}   OVERALL {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
