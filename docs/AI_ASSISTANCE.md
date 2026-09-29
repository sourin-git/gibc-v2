# AI assistance and development tools

A running record for the README's "AI assistance" section. No tool listed here supplied model
weights, training data, tokenizer assets or reported results. The model is trained from random
initialization, and every number comes from runs on our hardware.

| Tool | Used for | Stages |
|---|---|---|
| Claude Code (Anthropic; model Claude Opus 5.5) | Writing code, tests, configs and docs (CLAUDE.md, PLAN.md); running the commands; reporting their outputs. The human author set the requirements and reviewed each stage. | 0– |
| Context7 MCP (via Claude Code) | Documentation/reference lookup only: lm-evaluation-harness docs (the `wikitext` task definition, the HF backend, the custom `LM` interface) in Stage 0, and HF `tokenizers` docs (special-token encoding behavior) in Stage 1. Findings were checked against the installed libraries where it mattered (e.g. `encode_special_tokens` was verified empirically). | 0, 1 |
| Hugging Face Hub API (`huggingface_hub`) | Reading FineWeb-Edu metadata (revision SHA, file list, dataset card) and the pilot data itself. A data source, not an AI tool. | 1 |
