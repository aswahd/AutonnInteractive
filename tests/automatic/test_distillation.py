"""Plain KD: detached soft targets, no GT gate, independent sampling, native inputs."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from nnInteractive.automatic.distillation import (
    FrozenTeacher,
    TeacherConfig,
    distillation_loss,
    selected_episodes,
    state_digest,
    summarize_teacher,
)


def test_soft_targets_valid_reduction_and_detach():
    logits = torch.tensor([[[[[0.0, 0.0, float("nan")]]], [[[1.0, -1.0, float("nan")]]]]], requires_grad=True)
    target = torch.tensor([[[[[0.8, 0.3, float("nan")]]]]], requires_grad=True)
    valid = torch.tensor([[[[[True, True, False]]]]])
    loss = distillation_loss(logits, target, valid).sum()
    expected = F.binary_cross_entropy_with_logits(torch.tensor([1.0, -1.0]), torch.tensor([0.8, 0.3]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert target.grad is None and not logits.grad[..., 2].any()


def teacher_stub():
    teacher = object.__new__(FrozenTeacher)
    teacher.config = TeacherConfig(call_probability=1)
    teacher.source = {"prompt_channels": 7, "channel_mapping": {"prev_seg": 0}}
    teacher.network = torch.nn.Conv3d(8, 2, 1).eval().requires_grad_(False)
    return teacher


def test_native_input_gt_independence_empty_skip_and_batch_reduction():
    teacher = teacher_stub()
    inputs = []
    handle = teacher.network.register_forward_pre_hook(lambda m, a: inputs.append(a[0].clone()))
    batch = SimpleNamespace(
        query_images=torch.randn(2, 1, 2, 2, 2),
        query_targets=torch.ones(2, 1, 2, 2, 2),
        query_valid=torch.ones(2, 1, 2, 2, 2, dtype=torch.bool),
        metadata=[{"global_episode_index": i} for i in range(2)],
    )
    logits = torch.zeros(2, 2, 2, 2, 2)
    logits[0, 1] = 1
    logits[1, 1] = -1
    logits.requires_grad_()
    before = state_digest(teacher.network)
    loss, stats = teacher.loss(logits, batch, 2026)
    assert inputs[0].shape == (1, 8, 2, 2, 2) and not inputs[0].requires_grad
    torch.testing.assert_close(inputs[0][:, :1], batch.query_images[:1])
    assert inputs[0][:, 1].eq(1).all() and not inputs[0][:, 2:].any()
    p = teacher.predict(batch.query_images[:1], torch.ones_like(batch.query_images[:1]))
    torch.testing.assert_close(loss, distillation_loss(logits[:1], p, batch.query_valid[:1]).sum() / 2)
    batch.query_targets.zero_()
    changed_gt_loss, _ = teacher.loss(logits, batch, 2026)
    torch.testing.assert_close(loss, changed_gt_loss)
    optimizer = torch.optim.SGD([logits], lr=0.1)
    loss.backward()
    assert logits.grad[0].abs().sum() > 0 and not logits.grad[1].any()
    optimizer.step()
    assert state_digest(teacher.network) == before
    assert all(p.grad is None and not p.requires_grad for p in teacher.network.parameters())
    summary = summarize_teacher(stats)
    assert summary["train/teacher_call_fraction"] == 0.5
    assert summary["train/teacher_empty_prior_fraction"] == 0.5
    assert summary["train/teacher_acceptance_fraction"] == 1
    teacher.config = replace(teacher.config, call_probability=0)
    zero, no_calls = teacher.loss(logits, batch, 2026)
    assert zero == 0 and no_calls[3] == 0
    handle.remove()


def test_teacher_loading_preserves_rng_and_freezes(tmp_path):
    source = {
        "checkpoint_sha256": "a" * 64,
        "prompt_channels": 7,
        "native_patch_size": [192] * 3,
        "channel_mapping": {"prev_seg": 0},
    }
    config = TeacherConfig(
        enabled=True,
        checkpoint=str(tmp_path / "fold_0/checkpoint_final.pth"),
        checkpoint_sha256=source["checkpoint_sha256"],
    )
    state = torch.get_rng_state()

    def loader(*args):
        return torch.nn.Conv3d(8, 2, 1), source

    with (
        patch("nnInteractive.automatic.distillation.sha256", return_value="a" * 64),
        patch("nnInteractive.automatic.distillation.load_original", side_effect=loader),
    ):
        teacher = FrozenTeacher(config, source, "cpu")
    assert torch.equal(state, torch.get_rng_state())
    assert not teacher.network.training and not any(p.requires_grad for p in teacher.network.parameters())


def test_selection_replays_without_consuming_student_rng():
    state = torch.get_rng_state()
    a = selected_episodes(range(10000), 2026, 0.25, "cpu")
    b = torch.cat([selected_episodes(range(i, i + 100), 2026, 0.25, "cpu") for i in range(0, 10000, 100)])
    assert torch.equal(a, b) and 0.23 < a.float().mean() < 0.27
    assert torch.equal(state, torch.get_rng_state())


def test_weight_ramp():
    config = TeacherConfig()
    assert [config.weight(i) for i in (0, 1000, 2000, 50000)] == [0, 0.1, 0.2, 0.2]
    assert replace(config, ramp_updates=0).weight(0) == 0.2
    with pytest.raises(ValueError):
        TeacherConfig(max_weight=float("nan"))
