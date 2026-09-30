# GIBC-43M: final results (technical record)

Every value below is copied from a committed artifact. The source of truth is
[`results/FINAL_EVALUATION_production_step_0061036/final_results.json`](../results/FINAL_EVALUATION_production_step_0061036/final_results.json).
Smoke-test outputs under `results/eval_preflight/SMOKE_NOT_OFFICIAL_*` are **not** results.

## 1. Final checkpoint identity

| | |
|---|---|
| File | `$GIBC_WORK_DIR\runs\production\checkpoints\final_step_0061036.pt` (515,750,203 bytes; outside git) |
| SHA-256 | `4f28fc9d3ebfa97a415afe27115aac485496681d4fc324f91889f5b938c3e349` |
| State | step 61,036 · 1,000,013,824 target tokens · 244,144 microsteps · 0 skipped steps |
| Code | git commit `ff26beab34a5f5921297d1f8e61a2b6c18abb27f`, clean tree at launch |
| HF export | `$GIBC_WORK_DIR\exports\production_step_0061036` (local `LlamaForCausalLM` + tokenizer; outside git) |
| Records | `checkpoint_verification.json`, `hf_export_manifest.json`, `results/production/run_info.json` |

## 2. Parameter accounting

| Component | Formula | Count |
|---|---|---|
| Token embedding (shared with the output head) | V·d = 24,000 × 512 | 12,288,000 |
| Attention q, k, v, o per layer | 4·d² | 1,048,576 |
| SwiGLU gate, up, down per layer | 3·d·d_ff = 3 × 512 × 1,536 | 2,359,296 |
| RMSNorm × 2 per layer | 2·d | 1,024 |
| 9 layers | 9 × 3,408,896 | 30,680,064 |
| Final RMSNorm | d | 512 |
| Output head | tied: same tensor as the embedding | 0 |
| **Unique trainable parameters** | | **42,968,576** |

The count is verified four ways:
- by the analytic formula (`scripts/param_budget.py`);
- on the instantiated source model, counting unique Parameter objects (`scripts/verify_params.py`,
  `scripts/verify_checkpoint.py`);
- on the exported HF model after a fresh-process reload;
- in the training log.

The tie is checked by object and storage identity: `lm_head.weight is embed_tokens.weight`, and
`model.safetensors` contains no separate `lm_head` tensor. Summing parameters *by name*
double-counts the tied tensor (55,256,576). The competition figure is the unique count:
**42,968,576 ≤ 50,000,000**.

## 3. Data

| | |
|---|---|
| Source | `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Access | range reads of individual parquet row groups, visited round-robin across the 14 files; the full 28.5 GB sample was never downloaded |
| Pilot | 21,148 documents read (100 MB of accepted text); 196 Wikipedia-source documents excluded |
| Continuation | 27 hash-verified chunks from the exact pilot cursor (`007_00000.parquet`, row group 1, next row 148); 1,043,530 documents read, 9,661 Wikipedia-source documents excluded |
| Wikipedia-source exclusion | URL host `wikipedia.org` or a subdomain; applied before tokenizer training and corpus build; 9,857 documents in total |
| Validation split | 1% by `sha256(salt, document id)` |
| Exact-duplicate audit | 1,044,378 train documents / 1,039,557 unique texts: 4,821 duplicates, counted and **kept**. 87 validation documents whose exact text is in train were **removed** from validation (10,443 → 10,356) |
| Token files | train **1,100,000,632** tokens, sha256 `efb63e4f4902f5e7e84d7b13e1f05e567c1ecb69a1221dd101a397ae6fd8ea7f`; val **10,753,165** tokens, sha256 `f4340757b24634855e78f1daedc22d6a5145b490cccd773bb871376e201ac0e8`; uint16, read-only, one EOT per document |
| Integrity | streaming SHA-256 re-verification at every training start and resume |
| Sampler | `shuffled_windows_v1`, seed 1234: `PCG64(1234).permutation(W)` over W = 2,148,438 non-overlapping 513-token windows (order sha256 `f95bf3ba6b864932cc460f4336067556a1823f6d527ac075cee4ec929ca51228`). The first 1,953,152 windows in that order were used; no window was used twice |

Records: `results/data/pilot_manifest.json`, `production_acquisition_manifest.json`,
`production_duplicate_audit.json`, `production_tokens_meta.json`, `production_preflight_data.json`.

**Contamination caveat:** Wikipedia-source exclusion reduces direct Wikipedia overlap only. Web text
can still contain mirrored or quoted Wikipedia, WikiText or benchmark content. No
contamination-free claim is made.

## 4. Tokenizer

- **Model:** byte-level BPE (HF `tokenizers` 0.23.2), trained from scratch with
  `min_frequency` 2 on the pilot's training split: 20,765 documents, 99,207,133 bytes.
- **Vocabulary:** 24,000 entries, including the only special token `<|endoftext|>` = id 0.
- **Special-token behaviour:** no BOS, PAD or UNK, and no normalizer. Raw text containing
  `<|endoftext|>` is encoded as ordinary bytes (`encode_special_tokens` / `split_special_tokens`).
- **Compression:** 4.52 bytes per token on the pilot training text.
- **sha256** (`results/tokenizer/tokenizer.json`):
  `40f595fbedfff2de6b06c714a4537ef68b592141288c1a327d8b74962c4f7689`.

## 5. Production training

| | |
|---|---|
| Hardware | NVIDIA GeForce RTX 3050 6GB Laptop GPU (55 W cap); Ryzen 5 6600H; 16 GB RAM; Windows 11 |
| Precision / batch | bf16 autocast, fp32 master weights, TF32; micro-batch 8 × 512 × accumulation 4 = 16,384 targets per update |
| Optimizer | fused AdamW β = (0.9, 0.95), eps 1e-8, weight decay 0.1 on 2-D weights (incl. the tied embedding), none on RMSNorm scales; grad-norm clip 1.0 |
| Schedule | linear warmup 256 steps to 1e-3, cosine to 1e-4 at step 61,036 |
| LR selection | two 1,024-step runs on the pilot, with the production schedule horizon; predeclared rule ([`results/lr/lr_decision.json`](../results/lr/lr_decision.json)); val@1024 was 4.9451 (6e-4) vs 4.8853 (1e-3), so **1e-3** was selected (late-mean margin 0.0601 ≥ 0.02) |
| Steps / targets | 61,036 / 1,000,013,824 |
| Time | 56,723 s in training steps (15.76 h); 57,146 s wall (15.87 h) |
| Throughput | 17,630 targets/s run average; every logged 50-step window within 17,511–17,718 |
| Memory | peak 3,167 MiB allocated, 3,268 MiB reserved |
| GPU temperature (logged every 50 steps) | 53–75 °C |
| Clipping | 13.2% of updates had pre-clip gradient norm > 1.0 |
| Checkpoints | 61 saves (≤ 0.86 s each); protected milestones at steps 15,259, 30,518, 45,777 and final 61,036 |
| Stability | 0 skipped steps, 0 non-finite events, 0 resumes |
| Loss | validation 10.1737 at step 0 → **3.4574** at step 61,036 (internal perplexity **31.74**); last train step 3.5118 |

The internal validation loss uses a fixed 262,144-target subset of the FineWeb-Edu validation
split. It monitors training and is **not** the WikiText-103 result. Full log:
[`results/production/metrics.jsonl`](../results/production/metrics.jsonl).

## 6. Benchmarks

Harness settings:
- lm-evaluation-harness **0.4.13**, standard `hf` backend, loading only the local final export;
- CUDA, **float32**, `max_length` 512, batch size 8, **0-shot**;
- task YAMLs unchanged (all version 1.0), full splits.

0-shot is the project's own methodology, since the organizers have not specified a few-shot
protocol.

| Task | Dataset @ revision | Split | Examples (requests) | acc | acc_norm | Chance | Runtime |
|---|---|---|---|---|---|---|---|
| HellaSwag | `Rowan/hellaswag` @ `218ec52e09a7e7462a5400043bb9a69a41d06b76` | validation | 10,042 (40,168) | 0.2711 ± 0.0044 | 0.2806 ± 0.0045 | 0.2500 | 428.4 s |
| ARC-Easy | `allenai/ai2_arc` ARC-Easy @ `210d026faf9955653af8916fad021475a3f00453` | test | 2,376 (9,501) | 0.4280 ± 0.0102 | 0.3927 ± 0.0100 | 0.2502 | 55.0 s |
| PIQA | `baber/piqa` @ `142f6d7367fd9877f0fb3b5734ea6a545f54cdd1` | validation | 1,838 (3,676) | 0.5974 ± 0.0114 | 0.5860 ± 0.0115 | 0.5000 | 36.3 s |
| WinoGrande | `allenai/winogrande` winogrande_xl @ `01e74176c63542e6b0bcb004dcdea22d94fb67b5` | validation | 1,267 (2,534) | 0.5067 ± 0.0141 | not defined | 0.5000 | 28.0 s |

- **± values:** lm-eval standard errors.
- **Chance:** the exact expected accuracy of uniform guessing, from each document's number of
  answer options ([`chance_baselines.json`](../results/FINAL_EVALUATION_production_step_0061036/chance_baselines.json)).
  Seven ARC-Easy questions have 3 options and four have 5.
- **Command:**
  `python scripts/eval_lm_eval_final.py --export "$env:GIBC_WORK_DIR\exports\production_step_0061036" --out results\FINAL_EVALUATION_production_step_0061036`.
- **Truncation:** none needed. The longest context plus continuation is 261 tokens, well under
  512 (`results/eval_preflight/eval_request_audit_*.json`).
- **Cross-check:** 200 random requests per task were re-scored with the source checkpoint and our
  tokenizer, agreeing with lm-eval within 8.9e-5 nats.

## 7. WikiText-103

**WikiText-103 test token perplexity under the documented project methodology: 46.245**
(46.24521334355056).

- **Data:** `Salesforce/wikitext`, `wikitext-103-raw-v1`, split `test`, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`. The test split has 4,358 rows (1,467 empty) and
  1,285,622 characters; no training data is used.
- **Text:** rows are concatenated in dataset order with no separator added. Non-empty rows already
  end in `\n`, and empty rows contribute nothing.
- **Tokens:** our tokenizer, with no BOS and no EOT between articles. One EOT (id 0) is prepended as
  context only and is never scored.
- **Windows:** at most 512 input tokens, stride 256. The first window scores all its targets; each
  later window scores only its ≤ 256 new targets, with ≥ 256 tokens of unscored context. No padding.
- **Result:** 303,524 targets, each scored exactly once, in 1,185 windows. Total NLL
  1,163,698.2568611212 nats; perplexity = exp(total NLL / 303,524) = **46.245**.
- **Checks:** a repeat run was bit-identical; source and HF agree on sampled windows within
  5.7e-5 nats; scoring took 35.1 s.
- **Provenance:** the methodology was fixed in `gibc/wikitext.py` in Stage 6, before the final
  model existed, and only one variant was computed.

This is a **token** perplexity under our 24k BPE. It is not word perplexity and is not directly
comparable with numbers from other tokenizers.

## 8. HF export parity (final checkpoint)

These checks come from `hf_export_verification.json`: a fresh process, local files only,
`HF_HUB_OFFLINE=1`.

| Gate | Result |
|---|---|
| Config mapping (`rms_norm_eps` 1e-5, `tie_word_embeddings`, `bos_token_id` None, `eos_token_id` 0, `pad_token_id` None, unscaled `rope_parameters` θ = 10,000, no biases, 512 positions) | PASS |
| Unique trainable parameters after reload | 42,968,576 (PASS) |
| Weight tying after reload (object + storage) | PASS |
| All 84 weight names equal to the source (shape, dtype, exact values) | PASS |
| Tokenizer parity: 305 texts (287 real FineWeb-Edu documents) × 3 encode paths; length 24,000; EOS/PAD 0; no BOS; literal `<|endoftext|>` not special | PASS |
| Logits (fp32, 24 cases, lengths 1–512, batch 1/2/3) | CPU max \|diff\| 0.0; CUDA max \|diff\| 4.0e-5 (gate 1e-4) |
| Continuation log-likelihood (10 cases incl. a 7,573-token truncated context) | per-token 0.0; lm-eval HFLM vs ours ≤ 3.8e-6 |
| Padded-batch invariance (EOT = pad; a real EOT scored) | ≤ 5.7e-6 |

## 9. Environment

| | |
|---|---|
| Python / PyTorch / CUDA | 3.11.0 / 2.14.0+cu126 / 12.6 (cuDNN 91002) |
| Libraries | transformers 5.17.0 · lm-eval 0.4.13 · datasets 5.0.1 · tokenizers 0.23.2 · numpy 2.4.6 |
| Full freeze | [`results/env/pip-freeze.txt`](../results/env/pip-freeze.txt): training environment; sanitized public copy with one local path replaced by `-e .`, see [`results/env/README.md`](../results/env/README.md) (the `run_info.json` hash refers to the original) |

matplotlib 3.11.2 was added afterwards, for figures only.

## 10. Generation

The five fixed prompts were run once greedy and once sampled (temperature 0.8, top-k 50, seed 1234),
with nothing regenerated: [`generation_samples.json`](../results/FINAL_EVALUATION_production_step_0061036/generation_samples.json).
Greedy decoding loops. Sampled text is fluent but often factually or logically wrong (for example,
"Water boils at a lower temperature at high altitude because it is too cold").

## 11. Limitations

- Near-chance WinoGrande; HellaSwag slightly above chance; no instruction tuning or safety evaluation.
- 43M parameters, a 512-token context, and one production run with no seed variance estimate.
- Web-text contamination is not ruled out.
- The WikiText-103 number is tokenizer-specific.
- Throughput and time figures come from one laptop, with one thermal and power configuration.
