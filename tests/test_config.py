import dataclasses
import json
from pathlib import Path

import pytest

from gibc.config import PARAM_LIMIT, ModelConfig, param_breakdown

MAIN_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model" / "gibc_43m.json"

# Hand-derived in docs/PLAN.md. Pinned so an accidental config or formula change fails loudly.
EXPECTED_TOTAL = 42_968_576


def main_config_dict() -> dict:
    return json.loads(MAIN_CONFIG.read_text(encoding="utf-8"))


def test_main_config_matches_hand_derived_count():
    counts = param_breakdown(ModelConfig.from_json(MAIN_CONFIG))
    assert counts["embedding"] == 24_000 * 512
    assert counts["per_layer"] == 4 * 512 * 512 + 3 * 512 * 1536 + 2 * 512
    assert counts["lm_head"] == 0
    assert counts["total"] == EXPECTED_TOTAL
    assert counts["total"] <= PARAM_LIMIT


def test_untied_head_would_exceed_limit():
    cfg = dataclasses.replace(ModelConfig.from_json(MAIN_CONFIG), tie_embeddings=False)
    total = param_breakdown(cfg)["total"]
    assert total == EXPECTED_TOTAL + 24_000 * 512
    assert total > PARAM_LIMIT


def test_json_roundtrip(tmp_path):
    cfg = ModelConfig.from_json(MAIN_CONFIG)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    assert ModelConfig.from_json(path) == cfg


def test_unknown_key_rejected():
    data = main_config_dict()
    data["dropout"] = 0.1
    with pytest.raises(ValueError, match="unknown ModelConfig keys"):
        ModelConfig.from_dict(data)


def test_indivisible_heads_rejected():
    data = main_config_dict()
    data["n_heads"] = 7
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig.from_dict(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("d_model", 0),
        ("n_layers", -1),
        ("vocab_size", 24000.0),
        ("d_ff", True),
        ("context_length", "512"),
        ("norm_eps", 0.0),
        ("norm_eps", -1e-5),
        ("norm_eps", float("nan")),
        ("rope_theta", float("inf")),
        ("rope_theta", True),
        ("tie_embeddings", 1),
        ("tie_embeddings", "true"),
    ],
)
def test_invalid_field_values_rejected(field, value):
    data = main_config_dict()
    data[field] = value
    with pytest.raises(ValueError, match=field):
        ModelConfig.from_dict(data)
