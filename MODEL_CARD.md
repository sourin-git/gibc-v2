# Model card: GIBC-43M

## Model description

GIBC-43M is a small decoder-only causal language model trained **from random initialization**
for the GIBC V2 Track 01 challenge (at most 50M trainable parameters). It is a base model: plain
next-token prediction, with no instruction tuning and no preference or safety tuning.

- **Unique trainable parameters:** 42,968,576. The output head is tied to the token embedding.
- **Checkpoint:** `final_step_0061036.pt`, SHA-256
  `4f28fc9d3ebfa97a415afe27115aac485496681d4fc324f91889f5b938c3e349`.
- **HF format:** also exported as a local Hugging Face `LlamaForCausalLM` with a
  `TokenizersBackend` tokenizer, verified numerically equivalent to the source model.
- **Developer:** the project author (see the repository), with AI coding assistants used for
  development ([docs/AI_ASSISTANCE.md](docs/AI_ASSISTANCE.md)).

## Intended use

- Research and education on training small language models under tight parameter and hardware
  limits.
- Reproducing and inspecting the reported benchmark and perplexity measurements.
- A baseline or starting point for experiments with small models.

## Out-of-scope use

- Any use that depends on factual accuracy, such as answering questions, advice, or medical,
  legal or financial information. The model is often wrong.
- Chat or assistant use. It is not instruction-tuned and does not follow instructions reliably.
- Generating content for publication without human review.
- Safety-critical or high-stakes decisions of any kind.

## Architecture

| | |
|---|---|
| Type | decoder-only Transformer, pre-norm (Llama-style) |
| Layers / hidden / heads | 9 / 512 / 8 (head dim 64, full multi-head attention) |
| MLP | SwiGLU, intermediate size 1,536 |
| Positions | RoPE θ = 10,000 (unscaled) |
| Norm | RMSNorm, eps 1e-5 |
| Vocabulary / context | 24,000 / 512 tokens |
| Biases | none |
| Output head | tied to the input embedding |

## Training data

FineWeb-Edu `sample-10BT` (HuggingFaceFW), revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
It is English educational web text filtered from Common Crawl and released under ODC-By 1.0.

- **Tokens:** 1,100,000,632 train tokens were prepared; 1,000,013,824 targets were consumed, each
  window at most once.
- **Wikipedia-source exclusion:** documents from `wikipedia.org` or its subdomains were excluded
  (9,857 documents).
- **Duplicates:** exact duplicates within train were kept (4,821). Validation documents whose exact
  text also appears in train were removed from validation (87).
- **Caveat:** this is not complete decontamination. The data may still contain text overlapping
  with benchmarks or WikiText, and it inherits the biases and errors of web text.

## Tokenizer

A byte-level BPE trained from scratch on the project corpus, with exactly 24,000 entries. The only
special token is `<|endoftext|>` (id 0), used as the end-of-document marker; there is no BOS, PAD
or UNK. Literal `<|endoftext|>` characters in input text are encoded as ordinary bytes.
**Always load it through `gibc.tokenizer.load_tokenizer` or the exported HF tokenizer**; both
preserve this behaviour.

## Training procedure

- **Hardware:** a single NVIDIA RTX 3050 6GB Laptop GPU.
- **Precision:** bf16 autocast with fp32 master weights.
- **Batching:** micro-batch 8 × 512 tokens × gradient accumulation 4 = 16,384 targets per update.
- **Optimizer:** fused AdamW (β 0.9/0.95, weight decay 0.1 on 2-D weights, clip 1.0).
- **Learning rate:** peak 1e-3 with 256 warmup steps, then cosine decay to 1e-4.
- **Length and time:** 61,036 steps and 1.0B targets, in 15.76 h of training-step time
  (15.87 h wall), at an average of 17.6k targets/s.
- **Stability:** 0 skipped or non-finite steps.
- **Final internal validation:** loss 3.4574 and perplexity 31.74 on held-out FineWeb-Edu. This is
  not a benchmark.

## Evaluation

The model was evaluated 0-shot with lm-evaluation-harness 0.4.13 on full splits, in float32 with a
512-token context, using only the local final model.

| Task | acc | acc_norm | Random chance |
|---|---|---|---|
| HellaSwag (10,042) | 0.2711 ± 0.0044 | 0.2806 ± 0.0045 | 0.2500 |
| ARC-Easy (2,376) | 0.4280 ± 0.0102 | 0.3927 ± 0.0100 | 0.2502 |
| PIQA (1,838) | 0.5974 ± 0.0114 | 0.5860 ± 0.0115 | 0.5000 |
| WinoGrande (1,267) | 0.5067 ± 0.0141 | n/a | 0.5000 |

**WikiText-103 test token perplexity under the documented project methodology: 46.245** (303,524
tokens under this model's tokenizer, rolling 512/256 windows). This is a per-token figure, not word
perplexity. The full methodology is in [docs/RESULTS.md](docs/RESULTS.md#7-wikitext-103).

## Limitations

- **Weak generation:** greedy decoding repeats itself. Sampled text is fluent-looking but
  frequently factually or logically wrong; examples are in `generation_samples.json`.
- **Commonsense tasks:** WinoGrande is at chance and HellaSwag only slightly above it.
- **Context:** 512 tokens, so the model cannot use longer contexts.
- **Language:** English-centric data. Other languages are tokenized (byte-level) but not
  meaningfully modelled.
- **Scope of results:** one training run and one seed. The reported numbers carry no estimate of
  run-to-run variance.

## Ethical and safety considerations

- **No safety evaluation was performed.** The model has no content filtering and no alignment
  training.
- **Web-data risks:** as a model trained on web text, it may reproduce biased, offensive or false
  statements present in its data. These behaviours were not measured.
- **Plausible but wrong:** its outputs can look plausible while being wrong, so they should not be
  presented as factual.
- **Environmental cost:** small, at about 15.9 h on a single laptop GPU with a 55 W power cap.
  Total energy was not measured.

## Reproducibility

All steps are scripted, with pinned dataset revisions, recorded SHA-256 hashes for the corpus,
tokenizer and checkpoint, and a deterministic sampler and seed. Resume after interruption is
exact. See [README.md](README.md#reproducibility) and [docs/PLAN.md](docs/PLAN.md). Bitwise GPU
determinism is not guaranteed.

## License

- **Model weights and code:** no license has been chosen yet; it will be set by the author before
  public release.
- **Training data:** FineWeb-Edu is ODC-By 1.0, subject to Common Crawl's terms of use.
- **Evaluation data:** the evaluation datasets carry their own licenses.
