# GIBC V2 — Track 01 submission

A decoder-only language model trained **from random initialization** under a hard
**50,000,000 trainable-parameter** cap, on one laptop GPU. Deadline ≈ 2026-10-01 21:00 local
(56 h from 2026-09-29 13:00; confirm the exact time). Optimize for a legitimate, working,
reproducible submission, not for architectural experimentation.

The step-by-step plan and status live in [docs/PLAN.md](docs/PLAN.md).

## Competition constraints (hard rules)

- Model trained from random initialization. No pretrained weights, no fine-tuning of an
  existing model, no knowledge distillation (no teacher logits, no synthetic data from another LM).
- ≤ 50,000,000 total trainable parameters. Token embeddings count. Output head counts.
  A tied embedding/head is one tensor and is counted once.
- Allowed: standard libraries (PyTorch, HF `tokenizers`, etc.) and public datasets.
  The tokenizer is trained from scratch on our own corpus. No pretrained tokenizer assets.
- Official evaluation: HellaSwag, ARC-Easy, PIQA and WinoGrande via `lm-evaluation-harness`,
  plus WikiText-103 held-out perplexity. The organizers' exact WikiText-103 protocol is currently
  unclear; see "WikiText-103 perplexity" below.
- README must report: hardware, total training time, approximate compute, parameter count,
  data, training setup, benchmark results, and AI assistance.

## Architecture (frozen)

Config: [configs/model/gibc_43m.json](configs/model/gibc_43m.json), loaded by `gibc.config.ModelConfig`.

| field | value | why |
|---|---|---|
| vocab_size | 24000 | total entries, special tokens included; multiple of 64; < 65,536, so token ids fit uint16 |
| d_model / n_heads | 512 / 8 | head_dim 64, the fast path for PyTorch SDPA kernels |
| n_layers | 9 | |
| d_ff | 1536 | SwiGLU (gate, up, down), so 3·d·d_ff = 9·d² per MLP |
| context_length | 512 | fits 6 GB VRAM; long enough for the four lm-eval tasks |
| norm | RMSNorm, pre-norm, eps 1e-5 | |
| position | RoPE, θ = 10000, no scaling | |
| biases | none | |
| embeddings | **tied** input/output | **mandatory**: untied is 55,256,576 params, over the cap |
| dropout | none | single-pass pretraining |

Do not change the architecture unless there is a specific, demonstrated correctness, memory,
or throughput problem; record the evidence in docs/PLAN.md before changing anything.

Planned, not yet implemented: module and parameter names mirror HF `LlamaForCausalLM`, and RoPE
uses the same (non-interleaved, rotate-half) convention, so a Llama-format export is a key
rename. That export is accepted only after the parity tests in PLAN.md Stage 8 pass.

### Parameter count

```
embedding   V·d               = 24000·512              = 12,288,000
per layer   4d² + 3·d·d_ff + 2d = 1,048,576 + 2,359,296 + 1,024 = 3,408,896
layers      9 · 3,408,896                               = 30,680,064
final norm  d                                           =        512
lm_head     tied → 0
total                                                   = 42,968,576   (headroom 7,031,424)
```

**All parameter counts must be verified programmatically, never typed from memory:**
- `gibc.config.param_breakdown` is the analytic count, written independently of the model code.
- The future model test must assert that
  `sum(p.numel() for p in model.parameters() if p.requires_grad)` equals the analytic total,
  is ≤ `PARAM_LIMIT`, and that `lm_head.weight is embed_tokens.weight`.
  (`model.parameters()` de-duplicates shared tensors; that is the correct count for tied weights.)
- The exported HF model's parameter count must match too.
- Every parameter must be trainable (no frozen params). RoPE tables must be non-persistent
  buffers, not parameters.
- The number in the README is copied from script output (`scripts/param_budget.py`, the training
  log), and the run log must record it.

## Tokenizer contract (Stage 1; details in PLAN.md)

Byte-level BPE trained only on our selected training corpus (after Wikipedia-source exclusion),
with a full 256-symbol initial byte alphabet. Fresh vocab and merges. No normalization. Matching
byte-level decoder. Exactly 24,000 total entries, special tokens included. Minimal special
tokens: one end-of-text/document-boundary token, appended exactly once after each document.
There is no automatic BOS, no UNK, and no PAD unless a later stage proves one is needed.
Special-token ids are chosen explicitly, saved in the tokenizer metadata, and must be identical
after save/reload and after export. Do not copy Llama's BOS/EOS ids.

## Hardware constraints

Windows 11, RTX 3050 Laptop GPU (6 GB, Ampere sm_86, 55 W power cap per `nvidia-smi`),
Ryzen 5 6600H (6C/12T), 16 GB RAM.

- **Precision:** bf16 autocast with fp32 master weights and fp32 AdamW state; enable TF32.
  bf16 needs no GradScaler.
- **VRAM:** weights+grads+AdamW ≈ 43M × 16 B ≈ 0.69 GB. Activations and the 24000-way logits
  dominate. Micro-batch size is chosen by benchmark.
- **Memory monitoring:** training logs `torch.cuda.memory_allocated`, `memory_reserved` and
  `max_memory_allocated`. Handle OOM or a sudden throughput collapse by changing the training
  configuration (micro-batch, accumulation), never the driver.
- **NVIDIA driver/Control Panel settings stay unchanged.** Do not ask the user to modify them.
- **torch.compile:** off by default. Triton is not officially supported on Windows. Use plain
  eager mode with `F.scaled_dot_product_attention(is_causal=True)`.
- **Data loading:** no `DataLoader` workers (Windows uses spawn, which is slow and fragile).
  Batches come from `np.memmap` of uint16 tokens in the main process.
- **RAM:** never load the full corpus into memory. Process data in streaming shards.
- **Long runs:** plug into AC power, disable sleep, and pause Windows Update. Checkpoint about
  every 30 min; resume must be exact. Throughput is thermally limited, so measure it over
  ≥ 10 min, not in a short burst.

## Work directory (required)

The repo is inside OneDrive. Large artifacts must live outside it:
- **`GIBC_WORK_DIR` must be set explicitly.** `gibc.paths.work_dir()` raises if it is unset,
  inside the repo, or inside OneDrive. There is no fallback into the repo.
- Under it: raw/pilot data, tokenized data, checkpoints, HF exports, run directories.
- **`HF_HOME`** is set to `$GIBC_WORK_DIR\hf_home`, so the Hugging Face hub, datasets and
  lm-eval caches go there too. It must be set before any HF library is imported, i.e. in the
  environment, not in code.
- The venv lives outside OneDrive too (`$GIBC_WORK_DIR\venv`).

## Repository layout

```
configs/model/     model configs (JSON, strict keys)
gibc/              library code (config, paths, source_filter; later tokenizer, data, model, train)
scripts/           thin argparse entry points
tests/             pytest; CPU-only, fast
docs/PLAN.md       implementation plan and status
results/           small committed artifacts: env freeze, tokenizer, run summaries, eval JSON
$GIBC_WORK_DIR/    NOT in git: data, tokens, checkpoints, exports, HF cache, venv
```

## Coding conventions

- Python 3.11 with type hints and `from __future__ import annotations`. Use dataclasses for configs.
- Library code goes in `gibc/`. Scripts only parse args and call library functions.
- Configs are JSON with strict keys (unknown keys raise). Every run saves its fully resolved
  config into its run directory.
- **Always pass `encoding="utf-8"`** to text I/O. The Windows default codec is cp1252 and breaks
  on real data.
- Use `pathlib` and no hard-coded absolute paths. Large outputs go under `work_dir()`.
- Keep the code minimal: no placeholder/stub modules, no commented-out code, no speculative
  options. Do not add a dependency without a stated reason; pin new dependencies in
  requirements.txt.
- Comments explain *why*, not *what*. Match the surrounding style.
- Tests run on CPU in well under a minute. `python -m pytest` must pass before any training
  run is launched.
- Docs must not describe unimplemented behavior as if it exists. Mark it *planned* or *candidate*.
- Git: small commits. Never commit data, checkpoints, or weights.

## Reproducibility requirements

- **Environment:** requirements.txt pins are *candidates* until setup is validated on this
  machine. After validation, commit the full resolved environment as
  `results/env/pip-freeze.txt` (`python -m pip freeze`). Transformers counts as validated only
  once the export parity tests pass.
- **Data:** each acquisition writes a manifest with the dataset id, config name, pinned revision
  SHA, access method, files/order, documents read/kept/excluded (with reason), bytes, and
  timestamps. Token counts always use **our** tokenizer, never the dataset's GPT-2-based figures.
- **Runs:** each run directory holds the resolved configs, seed, git commit hash plus a dirty-tree
  flag, the env freeze, torch/CUDA versions, the GPU name, the tokenizer sha256, the data manifest
  reference, a JSONL log (step, loss, lr, grad norm, tokens seen, tokens/s, CUDA memory stats,
  wall time; val loss at intervals), and checkpoints.
- Data order is a deterministic function of (seed, step), so resume is exact. Bitwise GPU
  determinism is not a goal. State this in the README.
- **Eval:** each eval run records the lm-eval version, task versions, num_fewshot, batch size,
  and the full results JSON.

## Results integrity (strict)

- **Never invent, guess, extrapolate, or "fill in"** any benchmark score, perplexity, loss,
  throughput, training time, token count, or compute figure, in code, docs, README or chat.
- Every number in the README must trace to a file in `results/` or a run log, produced by a
  command the README documents.
- If a number has not been measured, write `TBD (not yet run)`. If you give an estimate, label
  it an estimate and show its formula.
- Report the final checkpoint, or state the selection rule up front. No silent best-of-N.
- Test sets are never used for any decision.
- Do not train on WikiText, Wikipedia dumps, or benchmark data.
- **Wikipedia-source exclusion** (`gibc/source_filter.py`) drops documents whose normalized URL
  hostname is `wikipedia.org` or a subdomain. It runs before tokenizer training and before
  corpus tokenization. It reduces direct Wikipedia-source overlap. It does **not** guarantee
  the absence of mirrored or quoted Wikipedia, WikiText or benchmark content. Never call it
  "decontamination", and make no contamination-free claim.
- Report failures and partial runs as what they are.

Approximate compute for the README: `C ≈ D · (6N + 12·L·d·T)`, where D is the number of
training tokens actually processed (from the log), N = 42,968,576, L = 9, d = 512, T = 512.
That gives ≈ 2.86e8 FLOPs per token (6N alone ≈ 2.58e8). Also report GPU-hours of wall time.

## WikiText-103 perplexity

The competition asks for WikiText-103 held-out perplexity, and the exact organizer protocol is
unclear. lm-eval's default `wikitext` task is **not** assumed to be the official metric.
Seek organizer clarification. If no evaluator or protocol is provided, implement one
WikiText-103 **test-split** methodology and document it precisely: dataset/config, split, text
preprocessing, tokenizer, window length and stride, and the unit perplexity is normalized by.
Label the reported number with that exact methodology; do not present it as a universal
standard. Do not compute several perplexity variants to pick a favorable one.

## AI assistance disclosure

This project is built with Claude Code (Anthropic) as the implementation assistant. The README
must include an **AI assistance** section stating which parts were AI-written (code, docs,
plans), what the human author did (decisions, review, running experiments), and that every
reported number comes from actual runs, not from the AI. Commits authored with Claude carry a
`Co-Authored-By` trailer.

## Out of scope

Web UI. Distributed or multi-GPU training. Optional research ideas (MoE, multi-token prediction,
exotic optimizers or schedules, architecture search). Any use of pretrained models or tokenizers.
General benchmark-decontamination systems. Hyperparameter sweeps beyond what the plan names.

## Commands

```
python -m pytest                                          # all tests (CPU)
python scripts/param_budget.py configs/model/gibc_43m.json  # analytic parameter budget
```
