# Implementation plan

Goal: the smallest correct pipeline

```
raw data → tokenizer → tokenized dataset → model → forward/loss → backward → optimizer
        → checkpoint → reload → text generation
```

then a single main training run, evaluation, and the README. Each stage has one acceptance check.
A stage is not done until its check passes. Values marked **CANDIDATE** are not approved; they
are starting points to be confirmed or replaced by measurement.

Status legend: ✅ done · ⏳ next · ☐ planned

## Stage 0: Scaffold, config, parameter budget ✅

- `gibc/config.py`: `ModelConfig` (frozen dataclass, strict JSON keys, type/range validation)
  and `param_breakdown`.
- `gibc/paths.py`: `work_dir()`, which requires `GIBC_WORK_DIR` to be set, outside the repo and
  outside OneDrive.
- `gibc/source_filter.py`: the Wikipedia-source exclusion rule (see Stage 1).
- `configs/model/gibc_43m.json`, `scripts/param_budget.py`, tests.
- **Check:** `python -m pytest` passes; `param_budget.py` prints total 42,968,576 and OK.

### Parameter derivation (independent of model code)

With V = 24000, d = 512, L = 9, d_ff = 1536, no biases, tied head:

| component | formula | count |
|---|---|---|
| token embedding (= output head, tied) | V·d | 12,288,000 |
| attention per layer (q, k, v, o) | 4·d² | 1,048,576 |
| SwiGLU per layer (gate, up, down) | 3·d·d_ff | 2,359,296 |
| RMSNorm per layer (×2) | 2·d | 1,024 |
| **per layer** | | **3,408,896** |
| all layers | L × per layer | 30,680,064 |
| final RMSNorm | d | 512 |
| output head | tied → 0 | 0 |
| **total** | | **42,968,576** |

That leaves 7,031,424 of headroom under 50,000,000. Non-embedding params: 30,680,576.
Untied, the total would be 55,256,576, which violates the cap. **Tying is required, not optional.**

### Architecture review

I checked for correctness, memory and throughput problems. None require a change:
- head_dim = 64 is the standard fast path for SDPA.
- d_ff = 1536 = 12·128 and V = 24000 = 375·64 are tensor-core aligned.
- The logits tensor (B·T·V = B × 12.29M elements) is the largest single activation.
  Micro-batch size is decided by the Stage 6 benchmark, not by estimate.

## Setup (prerequisite) ✅

Python 3.11 venv under `$GIBC_WORK_DIR` (outside OneDrive), with `GIBC_WORK_DIR` and `HF_HOME`
set as user environment variables. Install CUDA torch from the cu126 index, then
`requirements.txt`, then `pip install -e .`.
- **Check:** `torch.cuda.is_available()` and `torch.cuda.is_bf16_supported()` are True;
  `python -m pytest` passes; `param_budget.py` prints OK.
- **Then:** commit `results/env/pip-freeze.txt`. The requirements pins move from *candidate* to
  *validated for setup*. Transformers stays a candidate until Stage 8 parity passes.

## Stage 1: Pilot data, Wikipedia-source exclusion, tokenizer ✅

### 1a. Bounded pilot acquisition — `gibc/acquire.py`, `scripts/acquire_data.py`, `configs/data/pilot.json`

- Primary dataset: **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, config `sample-10BT`, ODC-By),
  pinned at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. `sample-10BT` is 14 parquet
  files (28.5 GB) with 1,000-row row groups.
- **Never downloads or processes all of `sample-10BT`.** Only the `id`, `url` and `text` columns
  of individual row groups are fetched, via HTTP range reads (`HfFileSystem` + pyarrow).
  Row groups are visited round-robin across the 14 files, so the pilot is not one file's prefix.
- Stopping bound: accepted text ≥ `max_text_bytes` (pilot: **100 MB** of UTF-8 text) or
  `max_documents`, checked after every document, so it is deterministic.
- The "10BT" in the name counts GPT-2 tokens. It is not used for any of our budgets. All token
  counts use **our** tokenizer.
- Wikipedia-source exclusion (1b) is applied as documents are read, so nothing downstream sees
  excluded documents. Empty/whitespace-only texts are also rejected and counted.
- Validation split: 1% of documents, decided by `sha256(salt, id)`, independent of read order.
  It is never used for tokenizer training.
- Outputs: `$GIBC_WORK_DIR/data/pilot/{train,val}.jsonl` and `manifest.json` (dataset, config,
  revision, access method, settings, timestamps, stop reason, counts by reason, text bytes,
  every row group consumed, output sha256s, library versions). The manifest is copied to
  `results/data/pilot_manifest.json`. That position record lets production acquisition
  (Stage 7) continue from where the pilot stopped.
- The script checks `GIBC_WORK_DIR` and `HF_HOME` (outside the repo and OneDrive) before any
  HF import.

### 1b. Wikipedia-source exclusion (rule implemented in `gibc/source_filter.py`)

- Parse the document's `url` with `urllib.parse.urlsplit` and normalize the hostname: lowercase,
  no port, no userinfo, no trailing dot. **Reject** if the host is `wikipedia.org` or ends in
  `.wikipedia.org`. Anything else is kept, including `wikimedia.org`, `wikibooks.org`, and
  look-alikes such as `notwikipedia.org` or `wikipedia.org.example.com`.
- Documents with a missing or unparseable URL are kept and counted separately in the manifest.
- Applied **before tokenizer training** and **before production corpus tokenization**.
- **Scope, stated as-is in the README:** this reduces direct Wikipedia-source overlap. It does
  not guarantee the absence of mirrored or quoted Wikipedia, WikiText or benchmark content.
  We make no contamination-free claim. It is not "decontamination", and no general
  benchmark-decontamination system is planned.

### 1c. Tokenizer — `gibc/tokenizer.py`, `scripts/train_tokenizer.py`

Contract:
- **Training data:** only our selected training corpus (the pilot's training portion, after
  exclusion). No pretrained tokenizer files, vocab or merges.
- **Model:** byte-level BPE (HF `tokenizers`). Byte-level pre-tokenizer, no prefix space, no
  normalizer (no Unicode normalization, no lowercasing), and a matching byte-level decoder.
- **Byte coverage:** the trainer's initial alphabet is the full 256-symbol byte-level alphabet,
  so any input string is representable without an UNK token.
- **Size:** exactly **24,000 total entries**, special tokens included. The test asserts
  `get_vocab_size(with_added_tokens=True) == 24000` and fails if training produced fewer.
- **Special tokens (minimal):**
  - one end-of-text token (`<|endoftext|>`), which serves as the document boundary and EOS;
  - no automatic BOS, and no post-processor that inserts tokens;
  - no UNK (byte coverage makes it unnecessary);
  - no PAD unless a later stage proves one is needed. If a pad id is needed for batched
    evaluation, the first option is reusing the end-of-text id (padded positions are masked).
    A dedicated PAD token is added only if that proves insufficient; it would sit inside the
    24,000 and be recorded like any other special token.
  - ids are chosen explicitly at training time, not copied from Llama. They are saved in
    `tokenizer_meta.json` alongside the tokenizer sha256.
- **Document boundaries:** `encode_documents` appends the end-of-text id exactly once after each
  document, so there is exactly one between consecutive documents.
- **Literal `<|endoftext|>` in raw text:** by default HF `tokenizers` maps that substring to the
  special id (verified). We set `encode_special_tokens = True`, which encodes it as ordinary
  bytes. That flag is **not persisted in tokenizer.json**, so tokenizers must come from
  `gibc.tokenizer.load_tokenizer`/`train_tokenizer`. The HF export in Stage 8 must reproduce
  this behavior (transformers' `split_special_tokens=True`), and the tokenizer-id parity check
  must cover it.
- **Artifacts:** `tokenizer.json` + `tokenizer_meta.json` + `pilot_token_stats.json` →
  `results/tokenizer/` (small; committed). `.gitattributes` marks tokenizer.json `-text` so
  `core.autocrlf` cannot change its bytes, and with them the recorded sha256.
- **Tests (`tests/test_tokenizer.py`, run against the committed artifact):** exact vocab 24000
  and equal to the model config; sha256 matches metadata; the only special token is EOT, with
  the recorded id; all 256 byte symbols present; nothing added automatically; round trip for
  English, leading/trailing/repeated spaces, tabs, `\n`, `\r\n`, blank lines, accented Latin,
  CJK, emoji/ZWJ, combining marks, RTL, math, punctuation, code, HTML, control characters, and
  empty/one-character strings; literal EOT text never yields the EOT id; one boundary per
  document; save → reload gives identical ids; training fails loudly if 24000 is not reached.

## Stage 2: Tokenized dataset format + sampler ⏳ — `gibc/data.py`, `scripts/tokenize_data.py`

- Tokenize excluded-and-kept documents in streaming batches, append the boundary token, and
  write `uint16` token files plus `meta.json` (our-tokenizer token counts, document counts,
  source manifest, tokenizer sha256) under `$GIBC_WORK_DIR/tokens/`.
- Sampler: the corpus is viewed as non-overlapping chunks of `context_length + 1` tokens. A
  seeded permutation of chunk indices defines the order, and step s reads a fixed slice of it.
  One pass, no repeats, exact resume from `step` alone.
- Built and tested on pilot data first. Production data (Stage 7) uses the same code.
- **Tests:** synthetic corpus: `y` is `x` shifted by one; all ids < vocab_size; same seed
  gives the same batches; a resumed sampler equals a continuous one.

## Stage 3: Model ☐ — `gibc/model.py`

Llama-style decoder with HF-compatible module names. CANDIDATE init: normal(0, 0.02), with
residual output projections scaled by 1/√(2L).
- **Tests:**
  - trainable parameter count == `param_breakdown(cfg)["total"]` == 42,968,576, and ≤ limit
  - `lm_head.weight is embed_tokens.weight`
  - causality: perturbing token t leaves logits at positions < t unchanged
  - loss at init ≈ ln(24000) ≈ 10.09
  - a tiny config overfits one batch on CPU

## Stage 4: Training loop + checkpointing ☐ — `gibc/train.py`, `gibc/checkpoint.py`, `scripts/train.py`, `configs/train/*.json`

- Mechanics (fixed): AdamW, weight decay applied to ≥ 2-D weights only, gradient clipping,
  warmup followed by decay, bf16 autocast, TF32, gradient accumulation, JSONL logging including
  CUDA allocated/reserved/peak memory, periodic val loss, and an atomic checkpoint (temp file +
  `os.replace`) every ~30 min and at the end. `--resume` picks up the latest checkpoint.
- **CANDIDATE hyperparameters (not approved):** peak LR 1e-3; ~131k tokens per optimizer
  update; ~500 warmup steps; cosine decay to 10% of peak; AdamW β = (0.9, 0.95), weight decay
  0.1, clip 1.0. Final values are chosen in Stage 6.
- **Tests (CPU, tiny config):** a short smoke run lowers loss; save → reload gives identical
  logits; interrupted + resumed training matches uninterrupted training at the same step (fp32).

## Stage 5: Generation ☐ — `scripts/generate.py`

Load a checkpoint and sample from a prompt (temperature, top-k). No KV cache.
- **Check:** runs on a smoke checkpoint and prints text. This closes the pipeline.

## Stage 6: Tiny real-data training + RTX 3050 benchmark → hyperparameter selection ☐

- Tiny real-data training on pilot tokens: check that loss decreases sensibly and that the
  candidate LR/warmup are stable. Adjust if not; no sweeps.
- Forward/backward memory benchmark of the real model on GPU at several micro-batch sizes:
  CUDA allocated/reserved/peak and OOM boundary.
- Sustained throughput (≥ 10 min per setting).
- Results in `results/benchmark.json`.
- **Output:** final micro-batch, accumulation, LR, warmup and schedule, plus the token budget
  `D = measured_tok_per_s × planned_train_seconds × safety_factor`, all written into
  `configs/train/main.json`.

## Stage 7: Production data acquisition ☐

- Continue from the pilot (same dataset, config and revision; the pilot documents are part of
  the corpus) until **our-tokenizer** training tokens reach D plus a modest reserve (CANDIDATE:
  10–20%), plus the held-out validation portion. Nothing beyond that is downloaded.
- Same Wikipedia-source exclusion and manifest fields as Stage 1.
- **Check:** the manifest's our-tokenizer token count is ≥ D × (1 + reserve).

## Stage 8: Evaluation pipeline, validated before the main run ☐

**Primary route:** export to HF `LlamaForCausalLM` + tokenizer, then evaluate with lm-eval's HF
backend. **The export is accepted only after every parity check below passes.** A custom
`lm_eval.api.model.LM` adapter is the fallback only, used if parity cannot be reached.

Required architecture mapping:

| HF field | value |
|---|---|
| vocab_size | 24000 |
| hidden_size | 512 |
| intermediate_size | 1536 |
| num_hidden_layers | 9 |
| num_attention_heads / num_key_value_heads | 8 / 8 (head dim 64) |
| hidden_act | `silu` (gate/up/down = SwiGLU) |
| rms_norm_eps | identical to our `norm_eps` |
| attention_bias / mlp_bias | False / False |
| tie_word_embeddings | True |
| RoPE | θ matches our `rope_theta`, no scaling, same rotate-half convention |
| max_position_embeddings | 512 |

Check the exact field names against the pinned transformers version when this is implemented.

Required parity checks (a test for each):
1. tokenizer ids identical before and after export, special-token ids included
2. model config values identical to the table above
3. weights identical, and the head shares the embedding tensor after reload
4. logits parity
5. per-token log-probability parity
6. continuation log-likelihood parity (context + continuation, as lm-eval scores it)
7. all of the above at multiple sequence lengths, up to 512
8. batch size 1 vs batched evaluation (padding must not change scores)
9. save → reload parity of the exported model

Then run `lm_eval --model hf ... --tasks hellaswag,arc_easy,piqa,winogrande` end-to-end on a
smoke checkpoint. Scores will be at chance; the point is that the pipeline runs.

**WikiText-103:** the competition asks for WikiText-103 perplexity, and the organizer protocol is
unclear. lm-eval's default `wikitext` task is not treated as the official metric. We will seek
organizer clarification. If none arrives, `scripts/eval_wikitext103.py` implements one
WikiText-103 **test-split** methodology, documented precisely (dataset/config, split,
preprocessing, tokenizer, window/stride, normalization unit). The result is labeled with that
methodology and not presented as a universal standard. One methodology, fixed before looking
at the result.

## Stage 9: Main training run ☐

A single run of `configs/train/main.json`. Monitor the logs; on interruption, resume.

## Stage 10: Final evaluation and README ☐

Export and parity-check the final checkpoint, run the evals, commit `results/`, and write the
README with hardware, training time, compute (formula in CLAUDE.md), parameter count (from
script output), data (including the manifest and the exclusion rule with its caveats), setup,
results, reproduction commands, and AI assistance. Every number traces to a file.

## Rough timeline (hours from start, ~56 total)

| hours | work |
|---|---|
| 0–2 | Setup; Stage 1a pilot acquisition |
| 2–8 | Stages 1c–5 (tokenizer, data, model, train loop, generation) with tests |
| 8–11 | Stage 6 tiny run + benchmark; Stage 7 production acquisition (in background); Stage 8 |
| 11–~42 | Stage 9 main run (length fixed by the measured budget, ≤ ~30 h) |
| ~42–50 | Stage 10 |
| 50–56 | buffer |

## Risks

| risk | mitigation |
|---|---|
| WikiText-103 protocol unclear | seek clarification; otherwise one precisely documented test-split method |
| Throughput lower than hoped | budget from measurement; schedule fixed before launch |
| VRAM overflow | benchmarked micro-batch; memory stats logged; OOM → config change |
| HF export mismatch | parity gates acceptance; custom-LM adapter fallback |
| Unvalidated dependency pins | pins are candidates until setup (and, for transformers, parity) passes; freeze committed |
| Interruption (sleep, updates, thermals) | 30-min atomic checkpoints, exact resume |
| OneDrive file locks / sync load | `GIBC_WORK_DIR` + `HF_HOME` outside OneDrive, enforced by `work_dir()` |
| Wikipedia/benchmark overlap in web data | Wikipedia-source exclusion; residual overlap disclosed, no contamination-free claim |
