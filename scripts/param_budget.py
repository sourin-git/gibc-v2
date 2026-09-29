"""Print the analytic parameter budget of a model config and check it against the limit.

Usage: python scripts/param_budget.py configs/model/gibc_43m.json
Exits with status 1 if the total exceeds PARAM_LIMIT.
"""

from __future__ import annotations

import argparse
import sys

from gibc.config import PARAM_LIMIT, ModelConfig, param_breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Analytic parameter budget for a model config.")
    parser.add_argument("config", help="path to a model config JSON")
    args = parser.parse_args()

    cfg = ModelConfig.from_json(args.config)
    counts = param_breakdown(cfg)

    print(f"config: {args.config}")
    for key, value in cfg.to_dict().items():
        print(f"  {key} = {value}")
    print(f"  head_dim = {cfg.head_dim}")
    print("analytic trainable parameters:")
    for key, value in counts.items():
        print(f"  {key:<16} {value:>12,}")

    headroom = PARAM_LIMIT - counts["total"]
    verdict = "OK" if headroom >= 0 else "OVER LIMIT"
    print(f"limit {PARAM_LIMIT:,}  headroom {headroom:,}  -> {verdict}")
    return 0 if headroom >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
