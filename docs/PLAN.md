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
  Micro-batch size is decided by the Stage 4 benchmark, not by estimate.

## Setup (prerequisite) ✅

Python 3.11 venv under `$GIBC_WORK_DIR` (outside OneDrive), with `GIBC_WORK_DIR` and `HF_HOME`
set as user environment variables. Install CUDA torch from the cu126 index, then
`requirements.txt`, then `pip install -e .`.
- **Check:** `torch.cuda.is_available()` and `torch.cuda.is_bf16_supported()` are True;
  `python -m pytest` passes; `param_budget.py` prints OK.
- **Then:** commit `results/env/pip-freeze.txt`. The requirements pins move from *candidate* to
  *validated for setup*. Transformers stays a candidate until Stage 6 parity passes.

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
  (Stage 5) continue from where the pilot stopped.
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
  `gibc.tokenizer.load_tokenizer`/`train_tokenizer`. The HF export in Stage 6 must reproduce
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

## Stage 2: Model ✅ — `gibc/model.py`, `gibc/generate.py`

- Llama-style decoder: token embedding, 9 pre-norm blocks (RMSNorm → full 8-head causal MHA with
  RoPE via `F.scaled_dot_product_attention(is_causal=True)` → residual; RMSNorm → SwiGLU
  `down(silu(gate(x)) * up(x))` → residual), final RMSNorm, output head **tied** to the
  embedding (one Parameter). No biases, no dropout, no learned positions.
- Module/parameter names mirror HF `LlamaForCausalLM`. RoPE: unscaled, θ = 10000, rotate-half
  layout, cos/sin as non-persistent buffers (not parameters, not in the state dict).
- RMSNorm: one scale vector (init 1), normalization computed in fp32 then cast back, then scaled.
- **Target convention:** `forward(input_ids, targets)` takes targets **already shifted**
  (`targets[:, t]` labels position t). No internal shift; the Stage 3 data loader must produce
  `x = chunk[:-1]`, `y = chunk[1:]`. There is no ignore-index: -100 or any id outside
  [0, 24000) raises. Sequences must satisfy 1 ≤ T ≤ 512.
- Init: all 2-D weights N(0, 0.02²), except every layer's `self_attn.o_proj.weight` and
  `mlp.down_proj.weight`, which use std 0.02/√18 ≈ 0.00471. RMSNorm scales = 1.
- `generate()`: greedy (temperature 0) or temperature/top-k sampling, optional stop at EOT,
  context cropped to 512, no KV cache.
- **Verified:** `scripts/verify_params.py` counts 42,968,576 unique trainable parameters on
  the instantiated model and checks the tie by object and storage identity.
  `scripts/smoke_model.py` runs the CUDA forward/backward, tokenizer integration and a
  random-weight generation, and writes `results/smoke/model_smoke.json`. `tests/test_model.py`
  covers counts, tying, shapes, loss, gradients, causality, RoPE, length bounds, determinism,
  no bias, no RoPE parameters, CUDA, and generation.

## Stage 3: Training system ✅

### 3a. Tokenized storage + batcher — `gibc/data.py`, `scripts/prepare_tokens.py`

- Each document is encoded and followed by exactly one EOT (id 0). The documents are
  concatenated into flat `uint16` files, `$GIBC_WORK_DIR/tokens/<name>/{train,val}.bin`, plus
  `meta.json` (dtype, vocab, EOT policy, tokenizer sha256, source manifest sha256, per-split
  documents/tokens/EOT count/bytes/sha256, command). The meta is copied to
  `results/data/<name>_tokens_meta.json`. No padding.
- Token ids are range-checked when written and again when each file is opened (memmap). This is
  why the training forward pass runs `validate_targets=False`, which avoids a GPU sync per
  microstep. The strict check stays the model's default for tests and debugging.
- Batcher: random contiguous windows of `seq_len + 1` tokens (windows may cross EOT
  boundaries), giving `x = w[:-1]` and `y = w[1:]`, the only shift anywhere. The dedicated numpy
  PCG64 RNG has a checkpointable state. (This replaces the earlier plan of a one-pass
  permutation.)
- Validation: fixed, evenly spaced windows (`eval_batches × micro_batch_size`), identical on
  every call.

### 3b. Training loop + checkpointing — `gibc/train.py`, `gibc/checkpoint.py`, `scripts/train.py`

- AdamW with two groups: ≥ 2-D weights (the tied embedding appears exactly once) with decay,
  and 1-D RMSNorm scales without. Coverage is checked. Fused on CUDA.
- Precision: `fp32` / `bf16` autocast / `fp16` autocast + `torch.amp.GradScaler("cuda")`.
- Accumulation: loss / `grad_accum_steps` per microstep. After the last microstep: unscale
  (fp16) → clip → one GPU sync to read loss and grad norm → non-finite check → step → update
  → `zero_grad(set_to_none=True)`. Tokens count the actual inputs.
- LR: linear warmup (step 0 → lr/warmup), then cosine to `lr × min_lr_ratio` at `max_steps`.
  It is a pure function of the global optimizer step, so it resumes exactly.
- Logging: JSONL (`metrics.jsonl`) with step, microsteps, tokens, loss, lr, grad norm,
  tokens/s, elapsed/train seconds and CUDA allocated/reserved/peak memory, plus val, checkpoint,
  resume and non-finite events. Also a readable console line.
- Checkpoints: model, optimizer, scaler, schedule position, counters, batcher RNG,
  Python/NumPy/torch CPU/CUDA RNG, configs, data identity (tokenizer and token-file hashes), git
  commit/dirty flag. Written as a `.tmp` beside the target, then `os.replace`. Loaded with
  `weights_only=True`. The newest `keep_checkpoints` are kept.
- Resume: `--resume <ckpt>`. The configs come from the checkpoint; data identity must match.
  `--stop-at-step N` checkpoints and exits (used to test the resume).
- A non-finite loss, or a non-finite grad norm outside fp16, writes a `non_finite` event and
  raises. No checkpoint is written, so the last valid one is preserved.
- Configs: `configs/train/smoke.json` (runnable). `benchmark.json` and `production.json` are
  **`"placeholder": true`**, and `scripts/train.py` refuses to run them.
- **CANDIDATE hyperparameters (not approved):** peak LR 1e-3; ~131k tokens per optimizer
  update; ~500 warmup steps; cosine decay to 10% of peak; AdamW β = (0.9, 0.95), weight decay
  0.1, clip 1.0. Final values are chosen in Stage 4.

### 3c. Generation from a checkpoint — `scripts/generate.py`

Loads a checkpoint (after checking the tokenizer hash) and samples through
`gibc.generate.generate`.

**Verified:** `tests/test_data.py` and `tests/test_train.py` cover:
- storage, the single shift, batcher determinism and state restore;
- optimizer coverage and the LR boundaries;
- step/microstep/token accounting, and accumulation ≡ one large batch;
- checkpoint contents; resume vs uninterrupted training (bit-identical on CPU);
- config drift rejection, RNG restore, evaluation leaving parameters unchanged;
- non-finite failure;
- bf16/fp16 CUDA runs with resume.

The GPU smoke run (fresh → exit at step 30 → fresh-process resume → step 60 → generation)
passed. Its metrics are copied to `results/smoke/train_smoke_metrics.jsonl`.

## Stage 4: RTX 3050 benchmark → hardware configuration ✅ (LR/warmup still open)

- Pre-fix: an fp16 GradScaler overflow skip is no longer counted as an optimizer step. The
  global step, LR, logging, validation and checkpoint triggers stay put; microsteps and attempted
  tokens are counted; a `scaler_skip` event is logged. More than 20 consecutive skips raise.
  Detection uses the public API (`get_scale()` falls in `update()`).
- `scripts/benchmark.py` runs one configuration per process: real fused-AdamW updates on pilot
  tokens at seq 512, 3 warmup updates, then ≥ 15 timed updates, each ending in a GPU sync.
  Results: `results/benchmark/stage4_runs.jsonl`, `stage4_sustained.jsonl`, and
  `stage4_summary.json` (`scripts/summarize_benchmark.py`, with time estimates).
- **Measured** at 16,384 tokens/update, bf16:
  - micro-batch 1 / 2 / 4 / 6 / 8: 11,972 / 14,477 / 16,505 / 17,435 / 17,730 tokens/s;
  - peak reserved memory: 1.1 / 1.4 / 2.1 / 2.8 / 3.5 GiB;
  - no OOM, and the gains flatten above micro-batch 6.
  - fp16 is within ±3% of bf16 with identical memory, so **bf16** is used (no GradScaler).
  - fp32 at micro-batch 4 runs at 11,247 tokens/s.
  - Activation checkpointing was not needed: memory isn't constrained and throughput has flattened.
- **Sustained** 120 s at bf16, micro-batch 8, accumulation 8: 17,823 tokens/s mean, every 20 s
  window within 17,814–17,837. The GPU went 59 → 71 °C, SM clock 1,755 → 1,747 MHz, ~50 W.
- Hardware config proposed in Stage 4: 8 × 512 × accumulation 8 = 32,768 tokens/update.
  **Superseded in Stage 5** by the locked 8 × 512 × **4** = 16,384 (max_steps 61,036).
- **Recommended** (approved in Stage 5): 1.0B training tokens. At the Stage 5 corrected rate,
  that is an ESTIMATE of 15.7 h raw.
- Written into `configs/train/production.json`. It stays `"placeholder": true` (refused)
  until LR/warmup are fixed by **one** short optimization sanity experiment, the token target
  is approved, and the production tokens exist.
- Before production: implement the shuffled one-pass window sampler (below) and milestone
  checkpoints.

## Stage 5: Production corpus, sampler, integrity, LR sanity ✅

- **Benchmark timing fix:** the in-update sync happens before the optimizer step. Timing now
  brackets with `torch.cuda.synchronize()` (whole interval + per update). Only the locked shape
  is rerun: `results/benchmark/stage5_corrected_timing.jsonl`.
- **Token-file integrity:** a streaming SHA-256 (`gibc.data.verify_token_files`) runs once at
  every training-process start and resume, before the corpus is used. A mismatch raises.
- **Acquisition continuation** (`gibc/production_data.py`, `scripts/acquire_production.py`,
  `configs/data/production.json`):
  - The cursor is derived from the pilot manifest and verified against the real parquet layout
    (round-robin order, full row groups, row sum): position 21 = `007_00000.parquet`, row group 1,
    next zero-based row **148** (rows 0–147 consumed).
  - Same revision, file list, round-robin policy, filters, salt and tokenizer.
  - Chunks of ≤ 40 row groups are written as `.partial` and renamed only once `chunk.json` is
    complete. A restart re-verifies completed chunks and redoes partial ones.
  - Stops at the document where pilot + continuation train tokens (our tokenizer, incl. EOT) first
    reach 1.1B.
  - Disk check: refuses if < 5 GiB would remain.
- **Build** (`scripts/build_production_tokens.py`):
  - Re-verifies every chunk and pilot hash.
  - Exact-text SHA-256 audit: counts duplicates and removes validation copies that also occur in
    train (recorded in `results/data/production_duplicate_audit.json`). This is not a
    decontamination claim.
  - Tokenizes with the frozen tokenizer into read-only `tokens/production/{train,val}.bin`, with
    meta copied to `results/data/`.
- **Sampler `shuffled_windows_v1`** (`gibc.data.ShuffledWindowSampler`):
  - W = ⌊(N−1)/512⌋ windows, window i = tokens[512i : 512i+513], so each target is predicted
    exactly once. Position 0 is context only; the final (N−1) − 512W targets are dropped.
  - Order: `PCG64(seed).permutation(W)`.
  - The checkpoint stores version, seed, W, the order hash and the cursor, and restore verifies
    them. Exhaustion raises; training checks there are enough windows before step 1.
- **Checkpoints:** rolling `step_*` files (keep N), plus protected `milestone_step_*` files (at
  `milestone_fractions`) and `final_step_*`. `run_until_step` stops early without shortening
  the LR schedule.
- **LR sanity** (exactly two runs, 6e-4 and 1e-3; `configs/train/lr_*.json`; decision by
  `scripts/lr_decision.py` using the predeclared rule) → `results/lr/lr_decision.json`.

### Stage 5 results (measured)

- **Corrected throughput** (bf16, 8 × 512 × 4): 17,685 tokens/s over a synced 60-update
  interval; per-update synced 17,679 tokens/s. Stage 4's figure was 17,730.
- **Corpus:**
  - Pilot + 27 continuation chunks; 1,043,530 continuation documents read, 9,661 excluded as
    Wikipedia-source.
  - Exact-text audit: 4,821 duplicate train documents (kept); 87 validation documents removed as
    train overlap.
  - Train: 1,044,378 documents, **1,100,000,632** tokens, sha256 `efb63e4f…`.
  - Val: 10,356 documents, **10,753,165** tokens, sha256 `f4340757…`.
- **Sampler:** W = 2,148,438 windows; production needs 1,953,152; 375 tail targets dropped.
- **LR runs** (both passed the rule):

  | peak LR | val@0 | @256 | @512 | @768 | @1024 |
  |---|---|---|---|---|---|
  | 6e-4 | 10.1761 | 6.0780 | 5.4730 | 5.1535 | 4.9451 |
  | 1e-3 | 10.1761 | 5.9812 | 5.3949 | 5.0932 | 4.8853 |

  Late-mean margin 0.0601 ≥ 0.02 and a better val@1024, so **1e-3 is selected**. The 6e-4 run
  was stopped at step 512 and resumed in a fresh process. Its consumed-window hash at step 1024
  equals the uninterrupted 1e-3 run's.
- `configs/train/production.json` holds the proposal. It is still `placeholder` /
  `TRAINING_NOT_STARTED` until final preflight approval (`scripts/verify_production_data.py` passes).

## Stage 6: Evaluation pipeline, validated before the main run ✅

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

### Stage 6 implementation and results ✅ (on the 1e-3 LR-sanity checkpoint; NOT the submitted model)

- **Export:** `gibc/hf_export.py` + `scripts/export_hf.py --checkpoint <any compatible .pt>`
  write `$GIBC_WORK_DIR/exports/<run>_step_<N>/`. The HF model is built locally from
  `LlamaConfig` (no download) and loaded only with our weights. The mapping is the identity
  (same names) and is verified tensor by tensor.
  - Non-default config: `rms_norm_eps 1e-5`, `tie_word_embeddings True`, `bos_token_id None`,
    `eos_token_id 0`, `pad_token_id None` (avoids `nn.Embedding(padding_idx)`), and
    `rope_parameters {"rope_type": "default", "rope_theta": 10000.0}`.
  - Tokenizer: `TokenizersBackend` from our exact tokenizer.json. EOS = PAD = `<|endoftext|>`
    (id 0); BOS and UNK are None; `split_special_tokens=True`; `model_max_length` 512. `len` is
    24,000.
- **Verification:** `scripts/verify_hf_export.py` runs in a fresh process, with local files only
  and `HF_HUB_OFFLINE=1`. All gates PASS, in
  `results/eval_preflight/hf_export_verification_lr_1e-3_step_0001024.json`:
  - config; 42,968,576 unique trainable parameters; tied after reload (safetensors has no
    `lm_head`); all 84 tensors equal;
  - tokenizer on 305 texts, including 287 real FineWeb-Edu docs;
  - logits: CPU max |diff| 0.0, CUDA ≤ 1.43e-5;
  - log-likelihoods ours vs HF vs lm-eval's HFLM ≤ 4.6e-6, including a truncated 7,573-token
    context;
  - padded batches ≤ 4.1e-6.
- **lm-eval smoke** (0-shot = OUR methodology, fp32, max_length 512, batch 8, limit 20, **NOT
  OFFICIAL**): `results/eval_preflight/SMOKE_NOT_OFFICIAL_lm_eval_limit20_lr_1e-3_step_0001024/`.
  All four tasks ran. The harness log-likelihoods match our own re-scoring within 2.1e-5.
- **Request audit** (`scripts/audit_eval_requests.py`): 55,879 requests over the four full
  tasks; max context+continuation is 261 tokens, so **no truncation is ever needed**. Scoring is
  ESTIMATED at ~3.3 min.
- **WikiText-103 methodology** is fixed in `gibc/wikitext.py`:
  - revision `b08601e04326c79dfdd32d625aee71d232d685c3`; rows concatenated unchanged;
  - one EOT prefix used as context only; 303,524 scored tokens;
  - windows of 512 with stride 256; exp(total NLL / N).
  - The smoke run (first 20,000 targets, NOT OFFICIAL) is deterministic, with source/HF window
    parity 2.2e-5 nats. The full run is ESTIMATED at ~26 s.

## Stage 7: Main training run ☐

A single run of `configs/train/production.json` (once approved). Monitor the logs; on
interruption, resume with `--resume <latest checkpoint>`.

## Stage 8: Final evaluation and README ☐

Export and parity-check the final checkpoint, run the evals, commit `results/`, and write the
README with hardware, training time, compute (formula in CLAUDE.md), parameter count (from
script output), data (including the manifest and the exclusion rule with its caveats), setup,
results, reproduction commands, and AI assistance. Every number traces to a file.

## Rough timeline (hours from start, ~56 total)

| hours | work |
|---|---|
| 0–2 | Setup; Stage 1a pilot acquisition |
| 2–8 | Stages 1c–3 (tokenizer, model, data, train loop, generation) with tests |
| 8–11 | Stage 4 tiny run + benchmark; Stage 5 production acquisition (in background); Stage 6 |
| 11–~42 | Stage 7 main run (length fixed by the measured budget, ≤ ~30 h) |
| ~42–50 | Stage 8 |
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
