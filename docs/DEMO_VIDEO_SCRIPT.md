# Demo video script (about 3.5 minutes)

**Tone:** plain and factual. Present generation as proof that the pipeline works end to end, not as
proof of quality. Say every number exactly as written here; each one comes from
`results/FINAL_EVALUATION_production_step_0061036/final_results.json` or
`results/production/metrics.jsonl`.

**Before recording:**
- open the three figures in `results/figures/`, `README.md` (rendered), and a PowerShell window in
  the repo;
- activate the venv and run once:
  `$env:GIBC_WORK_DIR="C:\gibc-work"; $env:HF_HOME="$env:GIBC_WORK_DIR\hf_home"`.

---

### 0:00–0:25 · The challenge
**Screen:** README title and first paragraph.

> "The challenge: build a foundation language model completely from scratch, with no pretrained
> weights, no fine-tuning and no distillation, under a hard limit of fifty million trainable
> parameters. I trained it on the hardware I have: one six-gigabyte RTX 3050 laptop GPU."

### 0:25–0:55 · The model
**Screen:** `results/figures/system_summary.png`, then the README Architecture table.

> "GIBC-43M is a nine-layer decoder-only Transformer: RoPE positions, RMSNorm, SwiGLU, and a
> twenty-four-thousand-token byte-level tokenizer that I trained myself. The output layer shares
> its weights with the input embedding. That matters: without tying, the model would have
> fifty-five million parameters and break the limit. With tying it has exactly 42,968,576 unique
> trainable parameters, which I verify in code on the trained model and on the exported one."

### 0:55–1:30 · Data and training
**Screen:** `results/figures/training_curve.png`.

> "The training data is FineWeb-Edu at a pinned revision. I read only the rows I needed, dropped
> Wikipedia-sourced pages to reduce overlap with WikiText, and stored one point one billion tokens
> as hash-verified files. The run processed one billion prediction targets in 61,036 steps: fifteen
> point seven six hours at about seventeen thousand six hundred targets per second, with peak GPU
> memory of 3,167 mebibytes, about half the card. Zero skipped steps, zero numerical failures.
> This curve is the raw
> logged loss, not smoothed. Validation loss went from 10.17 to 3.46 on held-out FineWeb-Edu."

### 1:30–2:05 · Live local generation
**Screen:** terminal.

```powershell
python scripts/demo.py --prompt "Photosynthesis is the process by which"
```

> "This loads the final checkpoint, checks its SHA-256 against the recorded result, and prints the
> parameter count. Everything runs locally on the laptop; there's no external model or API."

*(Wait for the output, then read it as it is.)*

> "It's fluent and on topic, but a 43-million-parameter model is often wrong. Here's a question it
> gets wrong:"

```powershell
python scripts/demo.py --prompt "If a train travels 60 miles in one hour, then in three hours it travels" --temperature 0
```

> "It answers 60 miles. I'm showing this to prove the pipeline runs end to end, not to claim the
> model reasons well."

### 2:05–2:45 · Evaluation
**Screen:** `results/figures/benchmark_results.png`, then the README Results tables.

> "Benchmarks were run zero-shot with lm-evaluation-harness on the full sets, and I report both
> raw and length-normalized accuracy. ARC-Easy: 42.8 percent against a 25 percent chance level.
> PIQA: 59.7 against 50. HellaSwag: 27.1, only slightly above its 25 percent chance level. And
> WinoGrande: 50.7, which is chance. On WikiText-103 the token perplexity is 46.2, under a method I
> fixed in advance and documented. It's per token under my own tokenizer, so it isn't a word-level
> perplexity."

### 2:45–3:20 · Engineering and reproducibility
**Screen:** `docs/RESULTS.md` sections 1 and 8; optionally `python -m pytest -q` finishing with
"218 passed".

> "What I'm proudest of is that every number can be checked. The data, tokenizer and checkpoint
> are hashed. Training resumes exactly after an interruption, and that's tested. Before any
> benchmark ran, the Hugging Face export had to match my model: identical weights, the same
> tokenizer, and logits within four times ten to the minus five on the GPU. The repository has 218
> tests and step-by-step commands, including a short smoke reproduction that takes roughly a
> quarter of an hour."

### 3:20–3:40 · Limitations and close
**Screen:** README Limitations.

> "The limits are real. It's a small model with a 512-token context and one training run. It
> generates weak text and is at chance on WinoGrande, and web data can't be fully decontaminated.
> What this project shows is a complete, verified, reproducible pipeline for training a language
> model from scratch under fifty million parameters on a laptop GPU. Thanks for watching."

---

**Disclosure line** (on screen at the end, or in the video description): "Built with AI coding
assistance (Claude Code, ChatGPT, Codex/GPT Astra, Context7). The model, its weights and all
results come from the author's own training and evaluation runs."
