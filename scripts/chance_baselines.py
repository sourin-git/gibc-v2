"""Exact random-guess accuracy for each final benchmark, from the logged lm-eval samples.

Usage: python scripts/chance_baselines.py --samples-dir <dir> --out <json>
Chance accuracy = mean over documents of 1 / (number of answer options), where the number of options is
the number of log-likelihood requests lm-eval issued for that document. No model scores are involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact chance baselines from lm-eval sample logs.")
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = {"method": "mean over documents of 1/(number of answer options = lm-eval requests per document)"}
    for task in TASKS:
        samples = json.loads((Path(args.samples_dir) / f"{task}_samples.json").read_text(encoding="utf-8"))
        options = [len(s["arguments"]) for s in samples]
        result[task] = {"documents": len(samples), "chance_accuracy": sum(1 / n for n in options) / len(options),
                        "options_histogram": {str(k): options.count(k) for k in sorted(set(options))}}
        print(task, result[task])
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
