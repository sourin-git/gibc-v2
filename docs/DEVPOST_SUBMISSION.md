# Devpost submission copy

Paste-ready sections. Every number is from
`results/FINAL_EVALUATION_production_step_0061036/final_results.json` or the production training
log.

---

## Inspiration

Most language-model work assumes large clusters and pretrained starting points. GIBC V2 Track 01
asks something narrower: what can a model with **at most 50 million trainable parameters**, trained
**from random initialization**, actually learn? I wanted to answer that on ordinary hardware, a
6 GB laptop GPU, and to be able to prove every number I report.

## What it does

GIBC-43M is a decoder-only language model with **42,968,576 unique trainable parameters** (output
head tied to the embedding), built entirely from scratch:
- its own tokenizer;
- its own data pipeline;
- its own training loop and weights.

It continues English text, and it is evaluated on HellaSwag, ARC-Easy, PIQA, WinoGrande and
WikiText-103. A terminal demo loads the final checkpoint, verifies its hash, reports the parameter
count and generates text locally, with no external model or API.

## How I built it

- **Data:** FineWeb-Edu `sample-10BT` at a pinned revision. I read only the rows needed (1.1B
  tokens' worth), never the full 28 GB sample. I excluded Wikipedia-sourced pages, audited exact
  duplicates, and wrote immutable, SHA-256-verified token files.
- **Tokenizer:** a 24,000-entry byte-level BPE trained on the corpus. Its one special token is
  `<|endoftext|>`, and literal occurrences of that string in text are never treated as the token.
- **Model:** a 9-layer Transformer (d_model 512, 8 heads, SwiGLU 1,536, RoPE, RMSNorm, 512-token
  context) with the output head tied to the embedding. Without tying it would have 55.3M
  parameters and break the limit.
- **Training:** plain PyTorch, with:
  - bf16, 16,384 targets per update, fused AdamW;
  - a peak LR of 1e-3, chosen by a predeclared rule from two short runs, with cosine decay;
  - hash-verified data, a one-pass shuffled sampler, and exact-resume checkpoints.
- **Evaluation:** the checkpoint is exported to a Hugging Face Llama model and verified against the
  source model before any benchmark runs: identical weights, tokenizer parity, logits within 4e-5,
  and harness likelihoods within 4e-6. Then lm-evaluation-harness runs 0-shot, and a WikiText-103
  evaluator applies a methodology fixed in advance.

## Challenges

- **The parameter budget:** a 24k vocabulary at d_model 512 is already 12.3M parameters.
  Tying the output head was mandatory, and the tie had to survive export and reload.
- **One 6 GB GPU:** benchmarking micro-batches, bf16 vs fp16, and throughput led to micro-batch 8
  at about 3.1 GiB peak memory.
- **Correctness at scale:** before production I fixed an fp16 GradScaler step-accounting bug, a
  benchmark-timing bug and a checkpoint-retention edge case, each found by review or a test. I also
  derived the data-continuation cursor from the manifest instead of trusting an audit note, which
  turned out to be off by one row.
- **Honest evaluation:** WikiText-103 perplexity depends on the tokenizer and the protocol, so I
  fixed one documented method before seeing the final model and did not compute alternatives.

## Accomplishments that I'm proud of

- A complete from-scratch pipeline that trained **1,000,013,824 targets in 15.76 hours** on a laptop
  GPU, with zero skipped or non-finite steps.
- **Verifiable numbers:** pinned revisions, SHA-256 hashes for data, tokenizer and checkpoint,
  218 automated tests, and an HF export proven numerically equivalent before evaluation.
- Clearly above-chance results on ARC-Easy (42.8% vs 25%) and PIQA (59.7% vs 50%) from a
  43M-parameter model.

## What I learned

- A tied output head is essential under a small parameter budget, and counting tied weights by
  name silently gives the wrong total.
- The most valuable safeguards were cheap: hash checks, exact-resume tests, and parity gates
  before evaluation.
- In this run, the model moved clearly above chance on ARC-Easy and PIQA while WinoGrande stayed at
  chance, so a single aggregate score would hide very different behaviour across tasks.

## Results

| Benchmark (0-shot, full set) | acc | acc_norm | Chance |
|---|---|---|---|
| HellaSwag (10,042) | 0.2711 ± 0.0044 | 0.2806 ± 0.0045 | 0.250 |
| ARC-Easy (2,376) | 0.4280 ± 0.0102 | 0.3927 ± 0.0100 | 0.250 |
| PIQA (1,838) | 0.5974 ± 0.0114 | 0.5860 ± 0.0115 | 0.500 |
| WinoGrande (1,267) | 0.5067 ± 0.0141 | n/a | 0.500 |

**WikiText-103 test token perplexity under the documented project methodology: 46.245** (303,524
tokens under the project's 24k tokenizer; not word perplexity).

The internal FineWeb-Edu validation perplexity is 31.74 (a training monitor, not a benchmark).

## Limitations

- **Generation:** text is weak, and greedy decoding loops; sampled text is fluent but often wrong.
- **Benchmarks:** WinoGrande is at chance, and HellaSwag is only slightly above it.
- **Scale:** a small model with a 512-token context; one training run, with no seed repeats.
- **Contamination:** Wikipedia-source filtering is not complete decontamination of web text.
- **WikiText-103:** the figure is specific to the tokenizer.

## What's next

- More training tokens: the prepared corpus still has about 100M unused tokens, and longer runs are
  cheap relative to the setup.
- Architecture trade-offs within the same 50M budget, such as a smaller vocabulary for more depth.
- Longer context, and seed repeats to measure run-to-run variance.
- A small instruction-tuning stage, with a proper safety evaluation before any interactive use.

## Built with

- **Languages and frameworks:** Python 3.11, PyTorch 2.14 (CUDA 12.6)
- **Libraries:** Hugging Face Transformers 5.17, Hugging Face `tokenizers` 0.23, Hugging Face
  `datasets` 5.0, `huggingface_hub`, lm-evaluation-harness 0.4.13, NumPy, PyArrow, safetensors,
  pytest, matplotlib (figures)
- **Data:** FineWeb-Edu (sample-10BT), WikiText-103, HellaSwag, ARC-Easy, PIQA, WinoGrande
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB), AMD Ryzen 5 6600H, 16 GB RAM, Windows 11
- **Development AI:** Claude Code (Opus 5.5), ChatGPT, Codex / GPT "Astra", Context7 MCP. These
  were used for development only. They did not produce the model, its weights, or any result.
