"""Whole-context loss, coordinate alignment, padding, and frozen-teacher gradient contracts."""

from types import SimpleNamespace

import numpy as np
import torch
from test_data import METADATA, manifest  # noqa: F401
from torch import nn
from torch.nn import functional as F

from nnInteractive.automatic.config import ModelConfig, TrainConfig
from nnInteractive.automatic.data import (
    EpisodeSampler,
    collate_episodes,
)
from nnInteractive.automatic.distillation import (
    FrozenTeacher,
    TeacherConfig,
)
from nnInteractive.automatic.infer import predict_volume
from nnInteractive.automatic.losses import segmentation_loss
from nnInteractive.automatic.refinement import FrozenRefiner
from nnInteractive.automatic.spatial import (
    resize_volume,
    support_at_resolution,
    teacher_inputs,
)


def test_full_context_loss_matches_explicit_resize_and_global_dice():
    torch.manual_seed(4)
    coarse = torch.randn(2, 1, 4, 4, 4, requires_grad=True)
    truth = torch.zeros(2, 1, 8, 12, 16)
    truth[0, :, :2, :2, :2] = 1  # Foreground far outside any central-only supervision region.
    valid = torch.ones_like(truth, dtype=torch.bool)
    valid[:, :, -2:] = False
    expected, _ = segmentation_loss(
        F.interpolate(coarse, size=truth.shape[-3:], mode="trilinear", align_corners=False), truth, valid
    )
    actual, _ = segmentation_loss(coarse, truth, valid)
    torch.testing.assert_close(actual, expected)
    compact, _ = segmentation_loss(coarse, truth.bool(), valid)
    torch.testing.assert_close(compact, actual, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(compact, coarse, retain_graph=True)[0],
        torch.autograd.grad(actual, coarse, retain_graph=True)[0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.autograd.grad(actual, coarse, retain_graph=True)[0], torch.autograd.grad(expected, coarse)[0]
    )
    original = actual.detach()
    truth[~valid] = float("nan")
    torch.testing.assert_close(segmentation_loss(coarse, truth, valid)[0], original)
    truth[0, :, :2, :2, :2] = 0
    assert abs(float(segmentation_loss(coarse, truth, valid)[0].detach() - original)) > 1e-3


def test_tiny_foreground_prompt_seed_is_mapped_not_erased():
    image = torch.arange(8 * 16 * 16).reshape(1, 8, 16, 16).float()
    target = torch.zeros_like(image)
    target[0, 0, 15, 15] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    student, mask, known, rescued = support_at_resolution(image, target, valid, (4, 4, 4))
    assert rescued and mask[0, 0, 3, 3] == 1 and known[0, 0, 3, 3]
    torch.testing.assert_close(student, F.interpolate(image[None], (4, 4, 4), mode="trilinear", align_corners=False)[0])


def test_sampler_retains_full_labels_and_context_with_padding(manifest):  # noqa: F811
    sampler = EpisodeSampler(
        manifest,
        METADATA,
        TrainConfig(augment_intensity=False, perturb_dense_supports=False, crop_jitter=2),
        patch_size=(48, 64, 64),
        student_input_size=(16,) * 3,
    )
    episode = sampler.sample_episode(
        np.random.default_rng(42), target="union", query_kind="positive", prompt_type="dense_mask"
    )
    batch = collate_episodes([episode]).to("cpu")
    assert batch.support_contexts.shape == (1, 4, 1, 48, 64, 64)
    assert batch.query_targets.shape == batch.query_context.shape == (1, 1, 48, 64, 64)
    assert batch.support_images.shape == (1, 4, 1, 16, 16, 16)
    assert batch.support_prompts.shape == (1, 4, 7, 16, 16, 16)
    torch.testing.assert_close(batch.query_images, resize_volume(batch.query_context, (16,) * 3))
    assert batch.query_valid.any() and not batch.query_valid.all()
    assert not batch.support_eligible.any()
    assert batch.support_prompts[:, :, 0].flatten(2).any(2).all()
    assert not batch.support_prompts[:, :, 1:].any()
    sampler.retain_support_contexts = False
    compact = sampler.sample_episode(
        np.random.default_rng(42), target="union", query_kind="positive", prompt_type="dense_mask"
    )
    assert compact.support_contexts is None and compact.metadata == episode.metadata
    for name in vars(episode):
        if isinstance(getattr(episode, name), torch.Tensor) and name != "support_contexts":
            torch.testing.assert_close(getattr(compact, name), getattr(episode, name), rtol=0, atol=0)
    starts = [sampler._random_start((32, 32, 32), np.random.default_rng(i)) for i in range(100)]
    assert len(set(starts)) > 90
    assert all(
        all(n - p <= start <= 0 for n, p, start in zip((32,) * 3, sampler.patch_size, origin)) for origin in starts
    )


def test_resized_teacher_paths_are_aligned_and_have_distinct_gradients():
    torch.manual_seed(8)
    batch = SimpleNamespace(
        query_context=torch.randn(1, 1, 8, 16, 16),
        query_images=torch.randn(1, 1, 4, 4, 4),
        query_targets=torch.ones(1, 1, 8, 16, 16),
        query_valid=torch.ones(1, 1, 8, 16, 16, dtype=torch.bool),
        metadata=[{"global_episode_index": 0}],
    )
    logits = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    for cls in (FrozenTeacher, FrozenRefiner):
        teacher = cls.__new__(cls)
        teacher.source = METADATA
        teacher.input_size = (8,) * 3
        teacher.config = TeacherConfig(soft_prior=True, call_probability=1, skip_empty_initial_mask=False)
        teacher.network = nn.Conv3d(8, 2, 1).eval().requires_grad_(False)
        captured = []
        hook = teacher.network.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        loss, _ = teacher.loss(logits, batch, 2026)
        gradient = torch.autograd.grad(loss, logits)[0]
        assert gradient.isfinite().all() and gradient.norm() > 0
        assert all(p.grad is None and not p.requires_grad for p in teacher.network.parameters())
        ct, teacher_logits, _ = teacher_inputs(logits, batch, teacher.input_size)
        torch.testing.assert_close(captured[0][:, :1], ct)
        torch.testing.assert_close(captured[0][:, 1:2], teacher_logits.float().softmax(1)[:, 1:2])
        assert not captured[0][:, 2:].any()
        hook.remove()


def test_unpooled_configuration_and_complete_resized_volume_inference():
    config = ModelConfig(context_patch_size=(192, 384, 384), student_input_size=(96,) * 3, memory_grids=(None,) * 3)
    config.validate()
    assert config.patch_size == (96,) * 3 and config.context_size == (192, 384, 384)

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(patch_size=(4,) * 3, context_size=(8, 12, 16), teacher_input_size=(8,) * 3)

        def predict_query(self, query, images, prompts):
            assert query.shape == (1, 1, 4, 4, 4)
            return torch.cat((query * 0, query * 0 + 2), 1)

    class Refiner:
        def predict_logits(self, query, prior):
            assert query.shape == prior.shape == (1, 1, 8, 8, 8)
            torch.testing.assert_close(prior, torch.full_like(prior, torch.sigmoid(torch.tensor(2.0)).item()))
            return torch.cat((query * 0, query * 0 + 1), 1)

    result = predict_volume(
        Student(),
        np.ones((1, 9, 10, 17), np.float32),
        torch.ones(4, 1, 4, 4, 4),
        torch.zeros(4, 7, 4, 4, 4),
        mode="resized_context",
        refiner=Refiner(),
    )
    assert result.mask.shape == (9, 10, 17) and result.coverage_min > 0
    assert result.windows == 4 and result.student_prediction is not None
    np.testing.assert_allclose(result.probabilities, torch.sigmoid(torch.tensor(1.0)).item(), rtol=1e-6)
    np.testing.assert_allclose(
        result.student_prediction.probabilities, torch.sigmoid(torch.tensor(2.0)).item(), rtol=1e-6
    )
