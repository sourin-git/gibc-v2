"""Instantiate the real model and count its UNIQUE trainable parameters by component.

Usage: python scripts/verify_params.py configs/model/gibc_43m.json
Exits with status 1 unless the count equals the analytic total, is within PARAM_LIMIT, and the
output head is the same Parameter object as the token embedding.
"""

from __future__ import annotations

import argparse
import sys

from gibc.config import PARAM_LIMIT, ModelConfig, param_breakdown
from gibc.model import CausalLM, parameter_report, weights_are_tied


def main() -> int:
    parser = argparse.ArgumentParser(description="Instantiated-model parameter verification.")
    parser.add_argument("config", help="path to a model config JSON")
    args = parser.parse_args()

    cfg = ModelConfig.from_json(args.config)
    model = CausalLM(cfg)
    report = parameter_report(model)
    analytic = param_breakdown(cfg)["total"]
    tied = weights_are_tied(model)

    print(f"config: {args.config}")
    for group, count in report.items():
        print(f"  {group:<42} {count:>12,}")
    total = report["total unique trainable"]
    print(f"analytic total: {analytic:,}   instantiated == analytic: {total == analytic}")
    print(f"output head tied to embedding (same Parameter object and storage): {tied}")
    ok = tied and total == analytic and total <= PARAM_LIMIT
    print(f"limit {PARAM_LIMIT:,}  headroom {PARAM_LIMIT - total:,}  -> {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
