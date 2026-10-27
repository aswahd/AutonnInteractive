"""Refinement GT loss must train the student through a frozen soft-prior network."""

import copy
from types import SimpleNamespace

import torch
from test_model import inputs, model  # noqa: F401

from nnInteractive.automatic.distillation import state_digest
from nnInteractive.automatic.refinement import (
    FrozenRefiner,
    RefinementConfig,
)


def test_refinement_only_gradients_and_frozen_state(model):  # noqa: F811
    refiner = object.__new__(FrozenRefiner)
    refiner.config = RefinementConfig()
    refiner.source = {"prompt_channels": 7, "channel_mapping": {"prev_seg": 0}}
    refiner.network = copy.deepcopy(model.backbone).eval().requires_grad_(False)
    before = state_digest(refiner.network)
    query, supports, prompts = inputs()
    supports.requires_grad_()
    batch = SimpleNamespace(
        query_images=query, query_targets=(query > 0).float(), query_valid=torch.ones_like(query, dtype=torch.bool)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        supports.grad = None
        logits = model(query, supports, prompts)["query_logits"]
        logits.retain_grad()
        loss, _ = refiner.loss(logits, batch)
        loss.backward()
        assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
        assert next(model.backbone.encoder.parameters()).grad.abs().sum() > 0
        assert model.backbone.decoder.seg_layers[-1].weight.grad.abs().sum() > 0
        if step:
            assert supports.grad.abs().sum() > 0
            assert all(b.support_in.weight.grad.abs().sum() > 0 for b in model.context.values())
        optimizer.step()
    assert state_digest(refiner.network) == before
    assert all(p.grad is None and not p.requires_grad for p in refiner.network.parameters())


def test_soft_prior_contract_and_sensitivity():
    refiner = object.__new__(FrozenRefiner)
    refiner.source = {"prompt_channels": 7, "channel_mapping": {"prev_seg": 0}}
    refiner.network = torch.nn.Conv3d(8, 2, 1).eval().requires_grad_(False)
    query = torch.randn(1, 1, 3, 3, 3)
    prior = torch.full_like(query, 0.3, requires_grad=True)
    received = []
    hook = refiner.network.register_forward_pre_hook(lambda m, a: received.append(a[0]))
    soft = refiner.predict_logits(query, prior)
    torch.testing.assert_close(received[0][:, :1], query)
    torch.testing.assert_close(received[0][:, 1:2], prior)
    assert not received[0][:, 2:].any() and received[0].requires_grad
    hook.remove()
    binary = refiner.predict_logits(query, (prior > 0.5).float())
    assert (soft - binary).abs().sum() > 0
    soft.square().mean().backward()
    assert torch.isfinite(prior.grad).all() and prior.grad.abs().sum() > 0
