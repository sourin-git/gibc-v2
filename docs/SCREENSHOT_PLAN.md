# Screenshot plan (Devpost gallery)

All images are generated from committed artifacts by `python scripts/make_figures.py` (1800 × 1000
px). Upload the PNGs directly rather than screen-capturing them, so they stay sharp.

| # | File / window | Crop | Caption |
|---|---|---|---|
| 1 (required) | `results/figures/system_summary.png` | none (full image) | "GIBC-43M at a glance: 42,968,576 unique trainable parameters, 1.0B training targets, one RTX 3050 6 GB laptop GPU, 15.76 h of training." |
| 2 (required) | `results/figures/training_curve.png` | none | "Production pretraining loss, as logged (unsmoothed): train loss per 50-step window and held-out FineWeb-Edu validation loss every 1,000 steps (10.17 → 3.46). Internal validation, not WikiText-103." |
| 3 (required) | `results/figures/benchmark_results.png` | none | "0-shot results on full evaluation sets (lm-eval 0.4.13): raw accuracy (acc) and length-normalized accuracy (acc_norm) side by side, with ±1 standard error and exact random-chance lines. WinoGrande is at chance." |
| 4 (optional) | Terminal after `python scripts/demo.py --prompt "Photosynthesis is the process by which"` | from the `====` banner through the OUTPUT block; hide the prompt path if it shows a user name | "Local inference from the final checkpoint: SHA-256 verified, 42,968,576 parameters, tied head, generated on the laptop GPU. Output quality is limited; this shows the pipeline running end to end." |

Notes:
- Don't crop the y-axis or the chance lines out of screenshots 2 and 3. The zero-based axes are
  deliberate.
- If a WikiText-103 number appears in a caption, write it in full: "WikiText-103 test token
  perplexity under the documented project methodology: 46.245".
- Never use images from `results/eval_preflight/SMOKE_NOT_OFFICIAL_*`.
