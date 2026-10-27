"""Object identity, first-update gradient activation, and attention probabilities."""

import torch
from test_model import inputs, model  # noqa: F401

from nnInteractive.automatic.config import TrainConfig
from nnInteractive.automatic.losses import (
    binary_logits,
    segmentation_loss,
)
from nnInteractive.automatic.optimization import make_optimizer


def test_parameter_identity_and_optimizer_coverage(model):  # noqa: F811
    encoder, decoder = model.backbone.encoder, model.backbone.decoder
    assert decoder.encoder is encoder
    optimizer = make_optimizer(model, TrainConfig())
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    assert all(p.requires_grad for p in parameters)
    encodes, decodes = [], []
    handles = [
        encoder.stem.register_forward_pre_hook(lambda m, a: encodes.append(id(next(m.parameters())))),
        decoder.seg_layers[-1].register_forward_pre_hook(lambda m, a: decodes.append(id(m.weight))),
    ]
    model.eval()
    with torch.no_grad():
        model(*inputs(supports=4))
    for handle in handles:
        handle.remove()
    assert encodes == [id(next(encoder.stem.parameters()))] * 5
    assert decodes == [id(decoder.seg_layers[-1].weight)] * 5


def test_query_gradient_reaches_support_after_first_optimizer_update(model):  # noqa: F811
    model.train()
    query, support, prompts = [x.requires_grad_() for x in inputs()]
    optimizer = make_optimizer(model, TrainConfig())
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        query.grad, support.grad, prompts.grad = None, None, None
        output = model(query, support, prompts)
        loss = segmentation_loss(
            binary_logits(output["query_logits"]),
            (query.detach() > 0).float(),
            torch.ones_like(query, dtype=torch.bool),
        )[0]
        loss.backward()
        assert query.grad.abs().sum() > 0
        if step == 0:
            assert not support.grad.any() and not prompts.grad.any()
            assert all(block.output.weight.grad.abs().sum() > 0 for block in model.context.values())
        else:
            assert support.grad.abs().sum() > 0 and prompts.grad.abs().sum() > 0
            assert torch.isfinite(support.grad).all() and torch.isfinite(prompts.grad).all()
            assert all(block.attention.in_proj_weight.grad.abs().sum() > 0 for block in model.context.values())
        optimizer.step()


def test_inactive_attention_probability_is_exactly_zero(model):  # noqa: F811
    active = torch.tensor([[True, False, False, False], [False, True, True, False]])
    token_active = active.repeat_interleave(8, dim=1)
    for block in model.context.values():
        raw = torch.randn(2, 5, block.query_in.in_features)
        memory = torch.randn(2, 32, block.support_in.in_features)
        query = block.query_norm(block.query_in(raw))
        supports = block.support_norm(block.support_in(memory))
        with torch.no_grad():
            _, weights = block.attention(
                query, supports, supports, key_padding_mask=~token_active, need_weights=True, average_attn_weights=False
            )
        assert not weights.masked_select(~token_active[:, None, None, :]).any()
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
