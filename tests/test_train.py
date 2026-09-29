import dataclasses
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import gibc.train as train_mod
from gibc.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state
from gibc.config import ModelConfig
from gibc.data import fixed_validation_batches
from gibc.model import CausalLM, weights_are_tied
from gibc.train import (
    NonFiniteError,
    TrainConfig,
    build_optimizer,
    check_optimizer_coverage,
    evaluate,
    lr_at,
    train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_CFG = ModelConfig.from_json(REPO_ROOT / "configs" / "model" / "gibc_43m.json")
TINY = ModelConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=2, d_ff=48, context_length=32)


def tiny_train_cfg(**overrides) -> TrainConfig:
    base = dict(
        run_name="t", placeholder=False, notes="", model_config="unused", tokens="tiny",
        seq_len=16, micro_batch_size=2, grad_accum_steps=3, max_steps=6, precision="fp32",
        lr=3e-3, min_lr_ratio=0.1, warmup_steps=2, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8,
        grad_clip=1.0, seed=11, log_every=1, eval_every=3, eval_batches=2, checkpoint_every_steps=2,
        checkpoint_every_minutes=0, keep_checkpoints=10, allow_tf32=False, fused_adamw=False, device="cpu",
    )
    return TrainConfig(**{**base, **overrides})


@pytest.fixture
def tokens_dir(tmp_path) -> Path:
    d = tmp_path / "tokens" / "tiny"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    splits = {}
    for split, n in (("train", 20_000), ("val", 2_000)):
        arr = rng.integers(0, TINY.vocab_size, n).astype(np.uint16)
        arr.tofile(d / f"{split}.bin")
        splits[split] = {"tokens": n, "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    meta = {"vocab_size": TINY.vocab_size, "tokenizer_sha256": "test-tokenizer", "splits": splits}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def run(cfg, tokens_dir, run_dir, **kwargs):
    return train(cfg, TINY, tokens_dir, run_dir, REPO_ROOT, print_fn=lambda s: None, **kwargs)


def final_params(run_dir: Path) -> dict[str, torch.Tensor]:
    latest = sorted((run_dir / "checkpoints").glob("step_*.pt"))[-1]
    return load_checkpoint(latest)["model"]


def events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


# Optimizer coverage -------------------------------------------------------------------------

def test_optimizer_covers_each_trainable_parameter_once():
    model = CausalLM(MAIN_CFG)
    opt = build_optimizer(model, tiny_train_cfg(), torch.device("cpu"))
    decay, no_decay = opt.param_groups
    all_ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids)) == len({id(p) for p in model.parameters()})
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == 42_968_576
    assert all(p.dim() == 2 for p in decay["params"]) and decay["weight_decay"] == 0.1
    assert all(p.dim() == 1 for p in no_decay["params"]) and no_decay["weight_decay"] == 0.0
    emb = model.model.embed_tokens.weight
    assert sum(p is emb for g in opt.param_groups for p in g["params"]) == 1  # tied tensor once
    assert weights_are_tied(model)


def test_coverage_check_rejects_duplicates_and_omissions():
    model = CausalLM(TINY)
    params = list(model.parameters())
    duplicated = torch.optim.AdamW(params)
    duplicated.param_groups[0]["params"].append(params[0])  # AdamW's constructor would reject this itself
    with pytest.raises(ValueError, match="more than once"):
        check_optimizer_coverage(model, duplicated)
    with pytest.raises(ValueError, match="cover"):
        check_optimizer_coverage(model, torch.optim.AdamW(params[1:]))


# LR schedule ---------------------------------------------------------------------------------

def test_lr_schedule_boundaries():
    cfg = tiny_train_cfg(lr=1.0, warmup_steps=10, max_steps=110, min_lr_ratio=0.1)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(9, cfg) == pytest.approx(1.0)  # end of warmup reaches peak
    assert lr_at(10, cfg) == pytest.approx(1.0)  # cosine starts at peak
    assert lr_at(60, cfg) == pytest.approx(0.55)  # halfway: (peak + min) / 2
    assert lr_at(110, cfg) == pytest.approx(0.1) and lr_at(500, cfg) == pytest.approx(0.1)
    decay = [lr_at(s, cfg) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))
    assert lr_at(0, tiny_train_cfg(lr=1.0, warmup_steps=0, max_steps=5)) == pytest.approx(1.0)


# Accumulation, counters ----------------------------------------------------------------------

def test_step_microstep_and_token_accounting(tokens_dir, tmp_path):
    cfg = tiny_train_cfg(max_steps=4, grad_accum_steps=3)
    summary = run(cfg, tokens_dir, tmp_path / "run")
    assert summary["step"] == 4 and summary["microsteps"] == 12
    assert summary["tokens"] == 12 * cfg.micro_batch_size * cfg.seq_len == 4 * cfg.tokens_per_step
    train_events = [e for e in events(tmp_path / "run") if e["event"] == "train"]
    assert [e["step"] for e in train_events] == [1, 2, 3, 4]
    assert [e["microsteps"] for e in train_events] == [3, 6, 9, 12]
    assert all(e["lr"] == pytest.approx(lr_at(e["step"] - 1, cfg)) for e in train_events)


def test_accumulation_matches_single_large_batch(tokens_dir, tmp_path):
    """2 microsteps x 2 sequences must give the same update as 1 microstep x 4 sequences (same data)."""
    common = dict(max_steps=1, warmup_steps=1)
    accum = run(tiny_train_cfg(micro_batch_size=2, grad_accum_steps=2, **common), tokens_dir, tmp_path / "a")
    single = run(tiny_train_cfg(micro_batch_size=4, grad_accum_steps=1, **common), tokens_dir, tmp_path / "b")
    assert accum["tokens"] == single["tokens"]
    pa, pb = final_params(tmp_path / "a"), final_params(tmp_path / "b")
    for name in pa:
        torch.testing.assert_close(pa[name], pb[name], rtol=1e-5, atol=1e-6)


# Checkpoint / resume -------------------------------------------------------------------------

def test_checkpoint_contents_roundtrip(tokens_dir, tmp_path):
    cfg = tiny_train_cfg(max_steps=2)
    run(cfg, tokens_dir, tmp_path / "run")
    ckpt = load_checkpoint(tmp_path / "run" / "checkpoints" / "step_0000002.pt")
    for key in ("model", "optimizer", "scaler", "scheduler", "state", "batcher", "rng", "train_config",
                "model_config", "data", "source"):
        assert key in ckpt, key
    assert ckpt["state"]["step"] == 2 and ckpt["state"]["microsteps"] == 6
    assert ckpt["state"]["tokens"] == 2 * cfg.tokens_per_step
    assert ckpt["scheduler"]["next_step"] == 2 and ckpt["train_config"] == cfg.to_dict()
    assert ckpt["data"]["tokenizer_sha256"] == "test-tokenizer"
    model = CausalLM(TINY)
    model.load_state_dict(ckpt["model"])
    assert weights_are_tied(model)
    assert not list((tmp_path / "run" / "checkpoints").glob("*.tmp"))


def test_resume_continues_counters_and_matches_uninterrupted_run(tokens_dir, tmp_path):
    cfg = tiny_train_cfg(max_steps=6)
    full = run(cfg, tokens_dir, tmp_path / "full")

    first = run(cfg, tokens_dir, tmp_path / "split", stop_at_step=3)
    assert first["step"] == 3 and not first["finished"]
    ckpt_path = tmp_path / "split" / "checkpoints" / "step_0000003.pt"
    assert ckpt_path.exists()
    resumed = run(cfg, tokens_dir, tmp_path / "split", resume=ckpt_path)

    assert resumed["step"] == 6 and resumed["finished"]
    assert resumed["microsteps"] == full["microsteps"] == 18
    assert resumed["tokens"] == full["tokens"]
    # CPU fp32 is deterministic, so a restored logical state reproduces the uninterrupted run exactly.
    pf, ps = final_params(tmp_path / "full"), final_params(tmp_path / "split")
    for name in pf:
        assert torch.equal(pf[name], ps[name]), name
    ev = events(tmp_path / "split")
    assert [e["event"] for e in ev].count("resume") == 1
    assert [e["step"] for e in ev if e["event"] == "train"] == [1, 2, 3, 4, 5, 6]


def test_resume_rejects_changed_config(tokens_dir, tmp_path):
    cfg = tiny_train_cfg(max_steps=4)
    run(cfg, tokens_dir, tmp_path / "run", stop_at_step=2)
    with pytest.raises(ValueError, match="config differs"):
        run(dataclasses.replace(cfg, lr=1.0), tokens_dir, tmp_path / "run",
            resume=tmp_path / "run" / "checkpoints" / "step_0000002.pt")


def test_rng_state_capture_restore():
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.random(), np.random.rand(), torch.rand(3)
    restore_rng_state(state)
    got = (random.random(), np.random.rand(), torch.rand(3))
    assert got[0] == expected[0] and got[1] == expected[1] and torch.equal(got[2], expected[2])


# Validation ----------------------------------------------------------------------------------

def test_validation_does_not_update_parameters():
    torch.manual_seed(0)
    model = CausalLM(TINY).train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    tokens = np.random.default_rng(0).integers(0, TINY.vocab_size, 3000).astype(np.uint16)
    batches = fixed_validation_batches(tokens, 16, 2, 3)
    a = evaluate(model, batches, "fp32", torch.device("cpu"))
    b = evaluate(model, batches, "fp32", torch.device("cpu"))
    assert model.training
    assert a == b and a["val_tokens"] == 3 * 2 * 16
    assert a["val_ppl"] == pytest.approx(math.exp(a["val_loss"]))
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    assert all(p.grad is None for p in model.parameters())


# Non-finite safety ---------------------------------------------------------------------------

def test_non_finite_loss_fails_and_keeps_last_checkpoint(tokens_dir, tmp_path, monkeypatch):
    cfg = tiny_train_cfg(max_steps=6, grad_accum_steps=1, checkpoint_every_steps=2)

    class PoisonAfterTwoSteps(CausalLM):
        calls = 0

        def forward(self, input_ids, targets=None, validate_targets=True):
            logits, loss = super().forward(input_ids, targets, validate_targets)
            if self.training:
                PoisonAfterTwoSteps.calls += 1
                if PoisonAfterTwoSteps.calls > 2:
                    loss = loss * float("nan")
            return logits, loss

    monkeypatch.setattr(train_mod, "CausalLM", PoisonAfterTwoSteps)
    with pytest.raises(NonFiniteError, match="step 2"):
        run(cfg, tokens_dir, tmp_path / "run")
    ckpts = sorted((tmp_path / "run" / "checkpoints").glob("step_*.pt"))
    assert [p.name for p in ckpts] == ["step_0000002.pt"]
    failure = [e for e in events(tmp_path / "run") if e["event"] == "non_finite"]
    assert len(failure) == 1 and failure[0]["latest_checkpoint"].endswith("step_0000002.pt")


# CUDA mixed precision ------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_cuda_mixed_precision_training_and_resume(tokens_dir, tmp_path, precision):
    cfg = tiny_train_cfg(device="cuda", precision=precision, fused_adamw=True, max_steps=4)
    first = run(cfg, tokens_dir, tmp_path / "run", stop_at_step=2)
    ckpt = load_checkpoint(tmp_path / "run" / "checkpoints" / "step_0000002.pt")
    assert (ckpt["scaler"] != {}) == (precision == "fp16")  # GradScaler state only for fp16
    resumed = run(cfg, tokens_dir, tmp_path / "run", resume=tmp_path / "run" / "checkpoints" / "step_0000002.pt")
    assert first["step"] == 2 and resumed["step"] == 4 and resumed["tokens"] == 4 * cfg.tokens_per_step
    assert math.isfinite(resumed["last_step_loss"]) and resumed["last_val"]["val_tokens"] > 0


# fp16 GradScaler skips -----------------------------------------------------------------------

def overflow_model(poison_calls: set[int] | None = None, poison_from: int | None = None):
    """CausalLM whose Nth training forward returns loss x 1e30: finite in fp32, but the scaled fp16
    backward overflows, so GradScaler detects inf gradients and skips the update."""

    class OverflowingLM(CausalLM):
        calls = 0

        def forward(self, input_ids, targets=None, validate_targets=True):
            logits, loss = super().forward(input_ids, targets, validate_targets)
            if self.training:
                OverflowingLM.calls += 1
                n = OverflowingLM.calls
                if (poison_calls and n in poison_calls) or (poison_from is not None and n >= poison_from):
                    loss = loss * 1e30
            return logits, loss

    return OverflowingLM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp16_normal_updates_increment_global_step(tokens_dir, tmp_path):
    cfg = tiny_train_cfg(device="cuda", precision="fp16", fused_adamw=True, max_steps=4, grad_accum_steps=1)
    summary = run(cfg, tokens_dir, tmp_path / "run")
    assert summary["step"] == 4
    assert summary["microsteps"] == 4 + summary["skipped_steps"]  # every non-skipped attempt is a step
    assert [e["step"] for e in events(tmp_path / "run") if e["event"] == "train"] == [1, 2, 3, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp16_scaler_skip_is_not_an_optimizer_step(tokens_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod, "CausalLM", overflow_model(poison_calls={3}))
    cfg = tiny_train_cfg(device="cuda", precision="fp16", fused_adamw=True, max_steps=5, grad_accum_steps=1,
                         checkpoint_every_steps=2, eval_every=100)
    summary = run(cfg, tokens_dir, tmp_path / "run")
    ev = events(tmp_path / "run")
    skips = [e for e in ev if e["event"] == "scaler_skip"]

    # The poisoned 3rd attempt (while 2 updates were done) was skipped and not counted as a step.
    assert any(e["step"] == 2 and e["microsteps"] == 3 and e["scale_after"] < e["scale_before"] for e in skips)
    # Training continued and still reached max_steps real updates; nothing terminated early.
    assert summary["step"] == 5 and summary["finished"]
    assert summary["skipped_steps"] == len(skips) >= 1
    assert summary["microsteps"] == 5 + len(skips)
    assert summary["tokens"] == summary["microsteps"] * cfg.micro_batch_size * cfg.seq_len  # attempted tokens
    # Scheduler did not advance on the skip: logged real steps 1..5 each used lr_at(step - 1).
    train_ev = [e for e in ev if e["event"] == "train"]
    assert [e["step"] for e in train_ev] == [1, 2, 3, 4, 5]
    assert all(e["lr"] == pytest.approx(lr_at(e["step"] - 1, cfg)) for e in train_ev)
    # Skips trigger no validation/checkpoint of their own.
    assert [e["step"] for e in ev if e["event"] == "val"] == [0, 5]
    assert [e["step"] for e in ev if e["event"] == "checkpoint"] == [2, 4, 5]
    # Checkpoint counters are logically consistent.
    state = load_checkpoint(tmp_path / "run" / "checkpoints" / "step_0000004.pt")["state"]
    skips_before_4 = sum(1 for e in skips if e["step"] < 4)
    assert state["step"] == 4 and state["skipped_steps"] == skips_before_4
    assert state["microsteps"] == 4 + skips_before_4
    assert state["tokens"] == state["microsteps"] * cfg.micro_batch_size * cfg.seq_len


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp16_persistent_overflow_fails_instead_of_looping(tokens_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod, "CausalLM", overflow_model(poison_from=2))
    cfg = tiny_train_cfg(device="cuda", precision="fp16", fused_adamw=True, max_steps=5, grad_accum_steps=1)
    with pytest.raises(NonFiniteError, match="consecutive GradScaler overflow skips at step 1"):
        run(cfg, tokens_dir, tmp_path / "run")
    assert len([e for e in events(tmp_path / "run") if e["event"] == "scaler_skip"]) == train_mod.MAX_CONSECUTIVE_SCALER_SKIPS


# Configs -------------------------------------------------------------------------------------

def test_shipped_train_configs():
    configs = {p.stem: TrainConfig.from_json(p) for p in (REPO_ROOT / "configs" / "train").glob("*.json")}
    assert set(configs) >= {"smoke", "benchmark", "production"}
    assert not configs["smoke"].placeholder
    assert configs["benchmark"].placeholder and configs["production"].placeholder
    for cfg in configs.values():
        assert cfg.seq_len <= MAIN_CFG.context_length
    prod = configs["production"]
    assert prod.status["APPROVED_BY_BENCHMARK"] and prod.status["TRAINING_NOT_STARTED"]
    assert prod.status["LR_WARMUP_FINAL"] is False
    assert prod.seq_len == 512 and prod.tokens_per_step == 8 * 512 * 8 == 32_768


def test_old_checkpoint_config_without_status_still_matches():
    cfg = tiny_train_cfg()
    legacy = {k: v for k, v in cfg.to_dict().items() if k != "status"}
    assert TrainConfig.from_dict(legacy) == cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_benchmark_harness_runs_real_updates():
    from gibc.benchmark import run_benchmark

    tokens = np.random.default_rng(0).integers(0, TINY.vocab_size, 5000).astype(np.uint16)
    result = run_benchmark(tokens, TINY, tiny_train_cfg(), "bf16", micro_batch=2, grad_accum=2,
                           warmup_updates=1, measured_updates=3)
    assert result["status"] == "ok" and result["updates_timed"] == 3
    assert result["seq_len"] == TINY.context_length and result["tokens_per_update"] == 2 * 2 * TINY.context_length
    assert result["tokens_per_s_mean"] > 0 and result["all_losses_finite"]
    assert result["allocated_after_cleanup_mib"] - result["allocated_before_mib"] < 1.0  # nothing leaked
