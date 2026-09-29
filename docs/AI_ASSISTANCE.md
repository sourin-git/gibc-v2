# AI assistance and development tools

A running record for the README's "AI assistance" section. None of these tools trained the
submitted language model or supplied its weights, training data or tokenizer assets. The model
is trained from random initialization by our own code, and every reported number comes from
runs on our hardware.

## AI tools

| Tool | Role | Stages |
|---|---|---|
| Claude Code (Anthropic; model Claude Opus 5.5) | Primary implementation agent: wrote the code, tests, configs and docs (CLAUDE.md, PLAN.md), ran the commands, and reported their outputs. | 0– |
| ChatGPT (OpenAI) | Technical lead / orchestration: planning, stage specifications and review of each stage's results (as reported by the project author). | 0– |
| Codex / GPT "Astra" (OpenAI) | Independent architecture, compliance and code review pass (e.g. the Stage 0 audit), as reported by the project author. | 0– |
| Context7 MCP (via Claude Code) | Documentation/reference lookup only: lm-evaluation-harness docs (the `wikitext` task definition, the HF backend, the custom `LM` interface) in Stage 0, HF `tokenizers` docs (special-token encoding) in Stage 1, and the lm-evaluation-harness Python API (`simple_evaluate`, `TaskManager`) in Stage 6. Findings were checked against the installed libraries where it mattered (e.g. `encode_special_tokens` was verified empirically, and PyTorch 2.14 APIs were checked by introspection in Stage 3). | 0, 1, 6 |

The human author set the requirements, chose between options, reviewed each stage and decided
what to run.

## Infrastructure (not AI tools)

| Tool | Role |
|---|---|
| Hugging Face Hub (`huggingface_hub`, `datasets`) | Dataset infrastructure: FineWeb-Edu metadata (revision SHA, file list, dataset card) and the pilot data itself. |
| PyPI / PyTorch wheel index | Dependency installation (pinned in requirements.txt; resolved set in results/env/pip-freeze.txt). |
