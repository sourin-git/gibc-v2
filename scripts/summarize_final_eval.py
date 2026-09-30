"""Collect the FINAL evaluation artifacts into one record: results/FINAL_EVALUATION_<export>/final_results.json.

Usage: python scripts/summarize_final_eval.py --dir results/FINAL_EVALUATION_production_step_0061036 --export <export dir>
Every value is read from the files produced by the evaluation scripts; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize FINAL evaluation artifacts.")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--export", required=True)
    args = parser.parse_args()

    final_dir, export_dir = Path(args.dir), Path(args.export)
    shutil.copyfile(export_dir / "verification_report.json", final_dir / "hf_export_verification.json")
    shutil.copyfile(export_dir / "export_manifest.json", final_dir / "hf_export_manifest.json")
    ckpt = load(final_dir / "checkpoint_verification.json")
    verify = load(final_dir / "hf_export_verification.json")
    lm = load(final_dir / "lm_eval_summary.json")
    wt = load(final_dir / "wikitext103_test_token_perplexity.json")
    result = {
        "status": "FINAL",
        "checkpoint": {"path": ckpt["checkpoint"], "sha256": ckpt["sha256"], "step": ckpt["state"]["step"],
                       "tokens_trained": ckpt["state"]["tokens"], "skipped_steps": ckpt["state"]["skipped_steps"],
                       "source_commit": ckpt["source"]},
        "parameters": {"source_unique_trainable": ckpt["unique_trainable_parameters"],
                       "source_weights_tied": ckpt["weights_tied"],
                       "hf_unique_trainable": verify["parameters"]["hf_unique_trainable"],
                       "hf_weights_tied_after_reload": verify["parameters"]["hf_weights_tied"],
                       "hf_named_parameters_with_duplicates": verify["parameters"]["hf_named_parameters_with_duplicates"]},
        "export": {"path": str(export_dir.resolve()), "verification_passed": verify["passed"], "gates": verify["gates"]},
        "lm_eval": {"settings": lm["settings"], "command": lm["command"], "tasks": lm["tasks"]},
        "wikitext103": {"label": wt["label"], "dataset": wt["dataset"], "scored_tokens": wt["scored_tokens"],
                        "full_test_tokens": wt["full_test_tokens"], "windows": wt["windows"], "total_nll": wt["total_nll"],
                        "token_perplexity": wt["token_perplexity"], "seconds": wt["seconds"],
                        "deterministic_repeat": wt["deterministic_repeat"], "window": wt["window"],
                        "command": ("python scripts/eval_wikitext103.py --export " + wt["export"] +
                                    " --revision " + wt["dataset"]["revision"])},
    }
    (final_dir / "final_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    (final_dir / "README_FINAL.txt").write_text(
        "FINAL evaluation of the submitted checkpoint (production step 61,036). Source of truth: final_results.json.\n"
        "Benchmarks: lm-eval 0.4.13, 0-shot (project methodology), float32, max_length 512, full test/validation sets.\n"
        f"WikiText-103: {wt['label']} (see gibc/wikitext.py). Not word perplexity.\n"
        "Smoke outputs elsewhere (results/eval_preflight/SMOKE_NOT_OFFICIAL_*) are NOT these results.\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
