"""Build the presentation figures from tracked result artifacts (no numbers are typed in by hand).

Usage: python scripts/make_figures.py
Reads results/production/{metrics.jsonl,run_info.json} and results/FINAL_EVALUATION_production_step_0061036/
{final_results.json,chance_baselines.json}; writes results/figures/{training_curve,benchmark_results,
system_summary}.png. The training curve plots the logged values unsmoothed on a zero-based y axis.
Palette: validated categorical slots 1-2 (dataviz reference palette, light mode).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / "results" / "FINAL_EVALUATION_production_step_0061036"
OUT = REPO / "results" / "figures"

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]  # validated: adjacent CVD dE 24.7, normal-vision dE 33.6 (light)
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10, "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def training_curve(events: list[dict]) -> None:
    train = [e for e in events if e["event"] == "train"]
    val = [e for e in events if e["event"] == "val"]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    style(ax)
    ax.plot([e["step"] for e in train], [e["train_loss"] for e in train], color=SERIES[0], linewidth=1.2,
            label="Train loss (mean of each logged 50-step window)")
    ax.plot([e["step"] for e in val], [e["val_loss"] for e in val], color=SERIES[1], linewidth=0, marker="o",
            markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8,
            label="Validation loss (fixed 262,144 held-out targets, every 1,000 steps)")
    last_t, last_v = train[-1], val[-1]
    ax.annotate(f"validation {last_v['val_loss']:.3f}", (last_v["step"], last_v["val_loss"]), xytext=(-8, -18),
                textcoords="offset points", ha="right", color=INK2, fontsize=9)
    ax.annotate(f"start {val[0]['val_loss']:.2f}", (0, val[0]["val_loss"]), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, last_t["step"] * 1.01)
    ax.set_ylim(0, 11)
    ax.set_xlabel("Optimizer step  (1 step = 16,384 prediction targets; 61,036 steps = 1.0B targets)")
    ax.set_ylabel("Cross-entropy loss (nats per token)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_title("GIBC-43M production pretraining: logged loss, unsmoothed", loc="left", fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK2)
    fig.text(0.01, 0.01, "Internal FineWeb-Edu validation split (not WikiText-103). Source: results/production/metrics.jsonl",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "training_curve.png")
    plt.close(fig)


def benchmark_results(final: dict, chance: dict) -> None:
    tasks = [("hellaswag", "HellaSwag"), ("arc_easy", "ARC-Easy"), ("piqa", "PIQA"), ("winogrande", "WinoGrande")]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    style(ax)
    width = 0.36
    for i, (key, name) in enumerate(tasks):
        m = final["lm_eval"]["tasks"][key]["metrics"]
        for j, (metric, label) in enumerate((("acc", "acc (raw log-likelihood)"), ("acc_norm", "acc_norm (length-normalized)"))):
            if f"{metric},none" not in m:
                ax.text(i + (j - 0.5) * width, 0.02, "acc_norm\nnot defined", ha="center", va="bottom",
                        fontsize=7.5, color=MUTED)
                continue
            x = i + (j - 0.5) * (width + 0.02)
            value, err = m[f"{metric},none"], m[f"{metric}_stderr,none"]
            ax.bar(x, value, width, color=SERIES[j], label=label if i == 0 else None, zorder=2)
            ax.errorbar(x, value, yerr=err, color=INK2, linewidth=0.9, capsize=3, zorder=3)
            ax.text(x, value + err + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        c = chance[key]["chance_accuracy"]
        ax.hlines(c, i - 0.44, i + 0.44, colors=INK, linewidth=1.2, zorder=4,
                  label="exact random-guess accuracy" if i == 0 else None)
        n = final["lm_eval"]["tasks"][key]["examples"]
        tasks[i] = (key, f"{name}\n{n:,} examples\nchance {c:.3f}")
    ax.set_xticks(range(len(tasks)), [n for _, n in tasks], color=INK2)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Accuracy (0-shot)")
    ax.set_title("GIBC-43M final benchmarks: full evaluation sets, 0-shot, lm-eval 0.4.13", loc="left",
                 fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
    fig.text(0.01, 0.01, "Error bars: ±1 standard error (lm-eval). Horizontal line: exact random-guess accuracy. "
             "Source: final_results.json, chance_baselines.json", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "benchmark_results.png")
    plt.close(fig)


def system_summary(final: dict, events: list[dict], run_info: dict) -> None:
    end = [e for e in events if e["event"] == "end"][-1]
    train = [e for e in events if e["event"] == "train"]
    cfg, env = run_info["model_config"], run_info["environment"]
    peak_alloc = max(e["cuda_peak_allocated_mib"] for e in train)
    peak_reserved = max(e["cuda_peak_reserved_mib"] for e in train)
    tiles = [
        (f"{final['parameters']['source_unique_trainable']:,}", "unique trainable parameters\n(limit 50,000,000; tied embedding/head)"),
        (f"{end['tokens']:,}", f"training prediction targets\n({end['step']:,} steps x 16,384)"),
        (env["gpu"].replace("NVIDIA GeForce ", "").replace(" Laptop GPU", " Laptop"), "single consumer GPU\nbf16 autocast"),
        (f"{end['train_seconds'] / 3600:.2f} h", f"training-step time ({end['train_seconds']:,.0f} s)\nwall {end['wall_seconds'] / 3600:.2f} h"),
        (f"{end['tokens'] / end['train_seconds'] / 1000:.1f}k / s", "prediction targets per second\n(run average)"),
        (f"{peak_alloc:,.0f} MiB", f"peak allocated VRAM\n({peak_reserved:,.0f} MiB reserved)"),
    ]
    fig = plt.figure(figsize=(9, 5), dpi=200)
    fig.text(0.04, 0.93, "GIBC-43M: trained from scratch on one 6 GB laptop GPU", fontsize=14, color=INK, weight="bold")
    for k, (value, label) in enumerate(tiles):
        col, row = k % 3, k // 3
        x, y = 0.04 + col * 0.32, 0.62 - row * 0.33
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (x, y - 0.06), 0.29, 0.28, boxstyle="round,pad=0.005,rounding_size=0.015", transform=fig.transFigure,
            facecolor="#ffffff", edgecolor=GRID, linewidth=0.8))
        fig.text(x + 0.02, y + 0.12, value, fontsize=17 if len(value) < 14 else 14, color=INK, weight="bold")
        fig.text(x + 0.02, y - 0.02, label, fontsize=8.5, color=INK2, va="bottom", linespacing=1.4)
    fig.text(0.04, 0.07, f"Decoder-only Transformer · {cfg['n_layers']} layers · d_model {cfg['d_model']} · "
             f"{cfg['n_heads']} heads · SwiGLU d_ff {cfg['d_ff']} · RoPE · RMSNorm · "
             f"{cfg['vocab_size']:,}-token byte-level BPE · context {cfg['context_length']}",
             fontsize=8.5, color=INK2)
    fig.text(0.04, 0.03, "Source: results/production/metrics.jsonl, run_info.json, FINAL_EVALUATION_*/final_results.json",
             fontsize=7.5, color=MUTED)
    fig.savefig(OUT / "system_summary.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = jsonl(REPO / "results" / "production" / "metrics.jsonl")
    final = json.loads((FINAL / "final_results.json").read_text(encoding="utf-8"))
    chance = json.loads((FINAL / "chance_baselines.json").read_text(encoding="utf-8"))
    run_info = json.loads((REPO / "results" / "production" / "run_info.json").read_text(encoding="utf-8"))
    training_curve(events)
    benchmark_results(final, chance)
    system_summary(final, events, run_info)
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))} to {OUT}")


if __name__ == "__main__":
    main()
