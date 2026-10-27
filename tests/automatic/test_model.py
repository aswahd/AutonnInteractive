"""V2 architecture contracts and independence, exercised on a small six-stage backbone."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from torch import nn

from nnInteractive.automatic.config import ModelConfig, TrainConfig
from nnInteractive.automatic.losses import (
    binary_logits,
    joint_segmentation_loss,
    segmentation_loss,
)
from nnInteractive.automatic.model import PromptEncodedICL3D


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    channels = (4, 8, 12, 16, 20, 24)
    backbone = ResidualEncoderUNet(
        input_channels=8,
        n_stages=6,
        features_per_stage=channels,
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=[1, 2, 2, 2, 2, 2],
        n_blocks_per_stage=[1] * 6,
        num_classes=2,
        n_conv_per_stage_decoder=[1] * 5,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
        deep_supervision=False,
    )
    config = ModelConfig(
        patch_size=(64,) * 3,
        context_patch_size=(64,) * 3,
        expected_channels=channels,
        memory_grids=((2,) * 3, None, None),
        attention_dim=16,
        attention_heads=4,
        ffn_hidden=32,
    )
    return PromptEncodedICL3D(backbone, config)


def inputs(batch=1, supports=1):
    return (
        torch.randn(batch, 1, 64, 64, 64),
        torch.randn(batch, supports, 1, 64, 64, 64),
        torch.randn(batch, supports, 7, 64, 64, 64),
    )


def activate_context(model):
    with torch.no_grad():
        for block in model.context.values():
            block.output.weight.normal_(0, 0.03)


def test_initial_parity_shapes_and_public_contract(model):
    model.eval()
    query, support, prompts = inputs()
    with torch.no_grad():
        expected = model.backbone(torch.cat((query, torch.zeros(1, 7, 64, 64, 64)), 1))
        actual = model(query, support, prompts)
    assert set(actual) == {"support_logits", "query_logits"}
    torch.testing.assert_close(actual["query_logits"], expected, rtol=0, atol=0)
    assert actual["query_logits"].shape == (1, 2, 64, 64, 64)
    assert actual["support_logits"].shape == (1, 1, 2, 64, 64, 64)
    with torch.no_grad():
        original_support = model.backbone(torch.cat((support[:, 0], prompts[:, 0]), 1))
        interactive = model.interactive(support[:, 0], prompts[:, 0])
        query_only = model.predict_query(query, support, prompts)
    torch.testing.assert_close(actual["support_logits"][:, 0], original_support, rtol=0, atol=0)
    torch.testing.assert_close(interactive, original_support, rtol=0, atol=0)
    torch.testing.assert_close(query_only, actual["query_logits"], rtol=0, atol=0)
    assert list(inspect.signature(model.forward).parameters) == [
        "query_images",
        "support_images",
        "support_prompts",
        "support_active_mask",
    ]
    assert all(p.requires_grad for p in model.parameters())
    assert all(isinstance(h, nn.Identity) for h in model.backbone.decoder.seg_layers[:-1])
    with pytest.raises(ValueError):
        model(query, support, prompts[:, :, :6])
    with pytest.raises(ValueError):
        model(query[:, :, :-1], support, prompts)


def test_episode_isolation_and_support_permutation(model):
    model.eval()
    activate_context(model)
    query, support, prompts = inputs(2, 2)
    changed_support, changed_prompts = support.clone(), prompts.clone()
    changed_support[0].zero_()
    changed_prompts[0].zero_()
    with torch.no_grad():
        original = model(query, support, prompts)
        changed = model(query, changed_support, changed_prompts)
        permuted = model(query, support.flip(1), prompts.flip(1))
    for key in original:
        torch.testing.assert_close(original[key][1], changed[key][1], rtol=0, atol=0)
    assert (original["query_logits"][0] - changed["query_logits"][0]).abs().max() > 1e-4
    torch.testing.assert_close(original["query_logits"], permuted["query_logits"], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(original["support_logits"].flip(1), permuted["support_logits"], rtol=0, atol=0)


def test_gradients_both_branches_and_all_active_parameters(model):
    model.train()
    activate_context(model)
    query, support, prompts = inputs()
    query.requires_grad_()
    support.requires_grad_()
    logits = model(query, support, prompts)
    loss, _ = segmentation_loss(
        binary_logits(logits["query_logits"]), (query.detach() > 0).float(), torch.ones_like(query, dtype=torch.bool)
    )
    loss.backward()
    assert query.grad.abs().sum() > 0 and support.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.requires_grad and p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    for stage, block in model.context.items():
        assert block.support_in.weight.grad.abs().sum() > 0, stage
        assert block.attention.in_proj_weight.grad.abs().sum() > 0, stage
    assert model.backbone.decoder.seg_layers[-1].weight.grad.abs().sum() > 0


def test_query_input_is_zero_prompted(model):
    model.eval()
    observed = []
    handle = model.backbone.encoder.stem.register_forward_pre_hook(lambda module, args: observed.append(args[0]))
    query, support, prompts = inputs()
    with torch.no_grad():
        model(query, support, prompts)
    handle.remove()
    assert not observed[0][:, 1:].any()
    torch.testing.assert_close(observed[1][:, 1:], prompts[:, 0])


def test_binary_logit_loss_and_invalid_voxels():
    two = torch.randn(2, 2, 3, 4, 5)
    binary = two[:, 1:2] - two[:, :1]
    torch.testing.assert_close(binary.sigmoid(), two.softmax(1)[:, 1:2])
    target = (torch.rand_like(binary) > 0.7).float()
    target[1].zero_()
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, :, 0] = False
    loss, metrics = segmentation_loss(binary, target, valid)
    changed_logits, changed_target = binary.clone(), target.clone()
    changed_logits[~valid] = float("nan")
    changed_target[~valid] = -999
    changed_loss, _ = segmentation_loss(changed_logits, changed_target, valid)
    torch.testing.assert_close(loss, changed_loss)
    assert metrics["dice_loss"][1] == 0
    expected = torch.nn.functional.softplus(binary[1][valid[1]]).mean()
    torch.testing.assert_close(metrics["loss_per_episode"][1], expected)
    separated = torch.stack(
        [segmentation_loss(binary[i : i + 1], target[i : i + 1], valid[i : i + 1])[0] for i in range(2)]
    ).mean()
    torch.testing.assert_close(loss, separated)
    with pytest.raises(ValueError, match="zero valid"):
        segmentation_loss(binary, target, torch.zeros_like(valid))


def test_support_path_has_no_attention_or_duplicate_encoding(model):
    model.eval()
    query, support, prompts = inputs(supports=2)
    encodes, attention_calls = [], []
    handles = [model.backbone.encoder.stem.register_forward_pre_hook(lambda m, a: encodes.append(a[0].shape[0]))]
    handles += [b.register_forward_hook(lambda m, a, o: attention_calls.append(1)) for b in model.context.values()]
    with torch.no_grad():
        before = model(query, support, prompts)
    assert encodes == [1, 1, 1] and len(attention_calls) == 3
    for handle in handles:
        handle.remove()
    activate_context(model)
    with torch.no_grad():
        after = model(query + 2, support, prompts)
    torch.testing.assert_close(before["support_logits"], after["support_logits"], rtol=0, atol=0)
    assert not torch.equal(before["query_logits"], after["query_logits"])


def test_support_loss_updates_same_backbone_without_query_attention(model):
    model.train()
    activate_context(model)
    query, support, prompts = inputs()
    query.requires_grad_()
    support.requires_grad_()
    outputs = model(query, support, prompts)
    loss, _ = segmentation_loss(
        binary_logits(outputs["support_logits"].flatten(0, 1)),
        (support.detach().flatten(0, 1) > 0).float(),
        torch.ones_like(support.flatten(0, 1), dtype=torch.bool),
    )
    loss.backward()
    assert query.grad is None
    assert support.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.context.parameters())
    for module in [model.backbone.encoder, model.backbone.decoder]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_joint_loss_eligible_mean_k_invariance_and_masking():
    torch.manual_seed(9)
    q = torch.randn(2, 2, 3, 4, 5, requires_grad=True)
    s = torch.randn(2, 2, 2, 3, 4, 5, requires_grad=True)
    qt = (torch.rand(2, 1, 3, 4, 5) > 0.6).float()
    st = (torch.rand(2, 2, 1, 3, 4, 5) > 0.6).float()
    valid = torch.ones_like(st, dtype=torch.bool)
    valid[:, :, :, 0] = False
    st[~valid] = float("nan")
    eligible = torch.tensor([[True, False], [False, False]])
    batch = SimpleNamespace(
        query_targets=qt,
        query_valid=torch.ones_like(qt, dtype=torch.bool),
        support_targets=st,
        support_valid=valid,
        support_eligible=eligible,
    )
    outputs = {"query_logits": q, "support_logits": s}
    loss, parts = joint_segmentation_loss(outputs, batch, TrainConfig())
    qloss = segmentation_loss(binary_logits(q), qt, batch.query_valid)[0]
    sloss = segmentation_loss(binary_logits(s[:1, 0]), st[:1, 0], valid[:1, 0])[0]
    torch.testing.assert_close(loss, qloss + sloss / 2)
    assert parts["support_loss_per_episode"][1] == 0
    loss.backward()
    assert s.grad[0, 0].abs().sum() > 0
    assert not s.grad[0, 1].any() and not s.grad[1].any()
    assert not s.grad[0, 0, :, 0].any()
    repeated = SimpleNamespace(
        query_targets=qt,
        query_valid=batch.query_valid,
        support_targets=st.repeat_interleave(3, dim=1),
        support_valid=valid.repeat_interleave(3, dim=1),
        support_eligible=eligible.repeat_interleave(3, dim=1),
    )
    repeated_loss, _ = joint_segmentation_loss(
        {"query_logits": q, "support_logits": s.repeat_interleave(3, dim=1)}, repeated, TrainConfig()
    )
    torch.testing.assert_close(loss, repeated_loss)
    batch.support_eligible.zero_()
    zero_loss, parts = joint_segmentation_loss(outputs, batch, TrainConfig())
    torch.testing.assert_close(zero_loss, qloss)
    assert not parts["support_loss_per_episode"].any()
