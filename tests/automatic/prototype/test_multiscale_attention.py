"""Regression checks for stage-specific memory, caching, and legacy checkpoints."""

import pytest
import torch

from nnInteractive.automatic.prototype.checkpoint import build_model, load_model
from nnInteractive.automatic.prototype.engine import loss_function
from nnInteractive.automatic.prototype.model import InContextUNet3D


@pytest.mark.parametrize("stages", [(-1,), (-1, -2), (-1, -2, -3), (-2,)])
def test_stage_memory_is_used_trainable_and_permutation_invariant(stages):
    torch.manual_seed(31)
    model = InContextUNet3D(width=4, memory_grid=2, attention_stages=stages).eval()
    query = torch.randn(1, 1, 16, 16, 16)
    supports = torch.randn(1, 2, 1, 16, 16, 16)
    masks = (supports > 0).float()
    with torch.no_grad():
        context = model.encode_context(supports, masks)
        cached = model.predict_with_context(query, None, context)
        direct = model(query, supports, masks)
        permuted = model(query, supports.flip(1), masks.flip(1))
        torch.testing.assert_close(cached, direct, rtol=0, atol=0)
        torch.testing.assert_close(cached, permuted, rtol=1e-5, atol=1e-5)
        for stage in stages:
            key = f"minus{-stage}"
            modified = {**context, "stage_tokens": dict(context["stage_tokens"])}
            if stage == -1:
                modified["tokens"] = torch.zeros_like(context["tokens"])
            else:
                modified["stage_tokens"][key] = torch.zeros_like(context["stage_tokens"][key])
            assert (model.predict_with_context(query, None, modified) - cached).abs().max() > 1e-4
    prediction = model(query, supports, masks)
    loss_function(prediction, (query > 0).float(), torch.ones_like(query, dtype=torch.bool)).backward()
    for stage in stages:
        block = model if stage == -1 else model.additional_attention[f"minus{-stage}"]
        for parameter in [block.mask_embedding.weight, block.attention.in_proj_weight, block.mlp[0].weight]:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
            assert torch.isfinite(parameter.grad).all()


def test_common_initialization_and_local_branch_are_identical():
    torch.manual_seed(2026)
    baseline = InContextUNet3D(width=4, memory_grid=2).eval()
    torch.manual_seed(2026)
    multiscale = InContextUNet3D(width=4, memory_grid=2, attention_stages=[-3, -1, -2]).eval()
    assert multiscale.attention_stages == (-1, -2, -3)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, multiscale.state_dict()[key], rtol=0, atol=0)
    image = torch.randn(1, 1, 16, 16, 16)
    prompts = torch.randn(1, 7, 16, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(
            baseline(image, query_prompts=prompts), multiscale(image, query_prompts=prompts), rtol=0, atol=0
        )


@pytest.mark.parametrize("stages", [None, [-1, -2, -3]])
def test_stage_configuration_roundtrips_and_legacy_config_loads(tmp_path, stages):
    config = {"width": 4, "mode": "memory", "memory_grid": 2, "backbone": None}
    if stages is not None:
        config["attention_stages"] = stages
    model = build_model(config).eval()
    path = tmp_path / "model.pt"
    torch.save({"format_version": 1, "model_config": config, "model": model.state_dict()}, path)
    restored, _ = load_model(path)
    restored.eval()
    assert restored.attention_stages == tuple(stages or [-1])
    image = torch.randn(1, 1, 16, 16, 16)
    supports = torch.randn(1, 2, 1, 16, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(
            model(image, supports, (supports > 0).float()),
            restored(image, supports, (supports > 0).float()),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("stages", [[], [0], [1], [-5], [-1, -1], [-1.0]])
def test_invalid_stage_configuration_is_rejected(stages):
    with pytest.raises(ValueError):
        InContextUNet3D(width=4, attention_stages=stages)
