"""Support subset semantics, gradients, distribution, and resumable evaluation."""

import itertools

import pytest
import torch
from test_data import METADATA, manifest  # noqa: F401
from test_model import activate_context, inputs, model  # noqa: F401

from nnInteractive.automatic.config import TrainConfig
from nnInteractive.automatic.data import EpisodeSampler
from nnInteractive.automatic.evaluate_support_counts import (
    evaluate_support_counts,
    summarize_support_counts,
)
from nnInteractive.automatic.losses import (
    binary_logits,
    segmentation_loss,
)
from nnInteractive.automatic.optimization import restore_rng, rng_state
from nnInteractive.automatic.support_dropout import (
    active_support_statistics,
    preserve_foreground_supports,
    sample_support_active_mask,
)
from nnInteractive.automatic.validation_state import TaskJournal


def test_count_distribution_uniform_subsets_and_rng_resume():
    torch.manual_seed(917)
    state = rng_state()
    mask = sample_support_active_mask(40000, 4, "cpu")
    restore_rng(state)
    assert torch.equal(mask, sample_support_active_mask(40000, 4, "cpu"))
    assert mask.shape == (40000, 4) and mask.dtype == torch.bool and mask.any(1).all()
    counts = mask.sum(1)
    histogram = torch.bincount(counts - 1, minlength=4)
    assert ((histogram.float() / len(mask) - 0.25).abs() < 0.012).all()
    for k in range(1, 5):
        group = mask[counts == k]
        subsets = list(itertools.combinations(range(4), k))
        for subset in subsets:
            expected = torch.zeros(4, dtype=torch.bool)
            expected[list(subset)] = True
            frequency = (group == expected).all(1).float().mean()
            assert abs(frequency - 1 / len(subsets)) < 0.02
    metrics = active_support_statistics(histogram)
    assert metrics["train/active_support_count_min"] == 1 and metrics["train/active_support_count_max"] == 4
    assert abs(sum(metrics[f"train/active_support_fraction_{k}"] for k in range(1, 5)) - 1) < 1e-6
    selected = sample_support_active_mask(
        8, 4, "cpu", {"sampling_strategy": "categorical_active_count", "count_probabilities": [0, 1, 0, 0]}
    )
    assert (selected.sum(1) == 2).all()
    before = torch.get_rng_state()
    assert sample_support_active_mask(8, 4, "cpu", {"enabled": False}).all()
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("channel", [0, 1, 3, 5])
def test_dropout_restores_all_supports_only_when_foreground_is_lost(channel):
    prompts = torch.zeros(2, 4, 7, 2, 2, 2)
    prompts[:, :, [2, 4, 6]] = 1  # Negative cues do not define foreground.
    prompts[0, 2, channel, 0, 0, 0] = 1
    prompts[1, 1, channel, 0, 0, 0] = 1
    active = torch.tensor([[True, False, False, False], [False, True, False, False]])
    corrected = preserve_foreground_supports(active, prompts, METADATA["channel_mapping"])
    assert corrected[0].all()
    assert torch.equal(corrected[1], active[1])
    assert active.sum() == 2
    assert preserve_foreground_supports(torch.ones_like(active), prompts, METADATA["channel_mapping"]).all()


def test_no_foreground_support_is_rejected_even_without_dropout():
    prompts = torch.zeros(2, 4, 7, 2, 2, 2)
    prompts[0, 0, 0] = 1  # One valid episode cannot rescue another.
    prompts[1, :, [2, 4, 6]] = 1
    with pytest.raises(ValueError, match="foreground support"):
        preserve_foreground_supports(torch.ones(2, 4, dtype=torch.bool), prompts, METADATA["channel_mapping"])


def test_mask_validation(model):  # noqa: F811
    q, s, p = inputs(supports=4)
    for mask in [torch.ones(1, 4), torch.ones(4, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool), [[True] * 4]]:
        with pytest.raises(ValueError):
            model(q, s, p, mask)
    with pytest.raises(ValueError, match="device"):
        model(q, s, p, torch.ones(1, 4, dtype=torch.bool, device="meta"))


def test_dropped_invariance_physical_subset_permutation_and_support_predictions(model):  # noqa: F811
    model.eval()
    activate_context(model)
    q, s, p = inputs(supports=4)
    mask = torch.tensor([[False, True, False, False]])
    changed_s, changed_p = s.clone(), p.clone()
    changed_s[:, [0, 2, 3]] = torch.randn_like(changed_s[:, [0, 2, 3]]) * 3
    changed_p[:, [0, 2, 3]] = torch.randn_like(changed_p[:, [0, 2, 3]]) * 2
    perm = [3, 1, 0, 2]
    with torch.no_grad():
        original = model(q, s, p, mask)
        changed = model(q, changed_s, changed_p, mask)
        removed = model(q, s[:, 1:2], p[:, 1:2])
        permuted = model(q, s[:, perm], p[:, perm], mask[:, perm])
        unmasked = model(q, s, p)
    for other in [changed, removed, permuted]:
        torch.testing.assert_close(original["query_logits"], other["query_logits"], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(original["support_logits"], unmasked["support_logits"], rtol=0, atol=0)
    assert (changed["support_logits"][:, 0] - original["support_logits"][:, 0]).abs().max() > 1e-3
    assert (unmasked["query_logits"] - original["query_logits"]).abs().max() > 1e-5


def test_masked_batch_episode_isolation(model):  # noqa: F811
    model.eval()
    activate_context(model)
    q, s, p = inputs(2, 4)
    mask = torch.tensor([[True, False, False, False], [False, True, True, False]])
    with torch.no_grad():
        original = model.predict_query(q, s, p, mask)
        s[0].normal_()
        p[0].normal_()
        mask[0] = ~mask[0]
        changed = model.predict_query(q, s, p, mask)
    torch.testing.assert_close(original[1], changed[1], rtol=0, atol=0)
    assert (original[0] - changed[0]).abs().max() > 1e-5


def test_masked_query_and_support_gradients_are_separate(model):  # noqa: F811
    model.train()
    activate_context(model)
    q, s, p = inputs(supports=4)
    q.requires_grad_()
    s.requires_grad_()
    p.requires_grad_()
    mask = torch.tensor([[False, True, False, False]])
    output = model(q, s, p, mask)
    loss = segmentation_loss(
        binary_logits(output["query_logits"]), (q.detach() > 0).float(), torch.ones_like(q, dtype=torch.bool)
    )[0]
    loss.backward()
    assert q.grad.abs().sum() > 0 and s.grad[:, 1].abs().sum() > 0 and p.grad[:, 1].abs().sum() > 0
    assert not s.grad[:, [0, 2, 3]].any() and not p.grad[:, [0, 2, 3]].any()
    for block in model.context.values():
        assert block.attention.in_proj_weight.grad.abs().sum() > 0
    for module in [model.backbone.encoder, model.backbone.decoder]:
        assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in module.parameters())
    model.zero_grad(set_to_none=True)
    q.grad, s.grad, p.grad = None, None, None
    output = model(q, s, p, mask)
    loss = segmentation_loss(
        binary_logits(output["support_logits"].flatten(0, 1)),
        (s.detach().flatten(0, 1) > 0).float(),
        torch.ones_like(s.flatten(0, 1), dtype=torch.bool),
    )[0]
    loss.backward()
    assert q.grad is None and all(p.grad is None for p in model.context.parameters())
    for k in range(4):
        assert s.grad[:, k].abs().sum() > 0
    for module in [model.backbone.encoder, model.backbone.decoder]:
        assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in module.parameters())


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_finite_forward_backward_all_stages_and_actual_counts(model, count):  # noqa: F811
    model.train()
    activate_context(model)
    q, s, p = inputs(supports=4)
    mask = torch.arange(4)[None] < count
    output = model(q, s, p, mask)
    loss = sum(x.square().mean() for x in output.values())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    model.eval()
    with torch.no_grad():
        actual = model(q, s[:, :count], p[:, :count])
    assert actual["support_logits"].shape[:3] == (1, count, 2)
    torch.testing.assert_close(actual["query_logits"], output["query_logits"], rtol=2e-4, atol=2e-4)


def test_fixed_bank_encodes_once_and_nested_predictions_match(model):  # noqa: F811
    model.eval()
    activate_context(model)
    q, s, p = inputs(supports=4)
    masks = [torch.arange(4)[None] < k for k in range(1, 5)]
    encodes, observed = [], []
    handles = [model.backbone.encoder.stem.register_forward_pre_hook(lambda m, a: encodes.append(a[0].shape[0]))]
    handles += [
        block.attention.register_forward_pre_hook(
            lambda m, a, kw: observed.append(kw["key_padding_mask"]), with_kwargs=True
        )
        for block in model.context.values()
    ]
    with torch.no_grad():
        results = list(model.iter_query_subsets(q, s, p, masks))
    for h in handles:
        h.remove()
    assert encodes == [1] * 5 and len(observed) == 12
    for k in range(4):
        for stage_mask in observed[k * 3 : k * 3 + 3]:
            tokens = stage_mask.shape[1] // 4
            assert torch.equal(stage_mask, ~masks[k].repeat_interleave(tokens, dim=1))
    with torch.no_grad():
        for k, actual in enumerate(results, 1):
            torch.testing.assert_close(actual, model.predict_query(q, s[:, :k], p[:, :k]), rtol=2e-4, atol=2e-4)


def test_fixed_bank_records_and_journal_resume(manifest, tmp_path):  # noqa: F811
    class CountEcho(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def iter_query_subsets(self, q, s, p, masks):
            for mask in masks:
                yield torch.cat((torch.zeros_like(q), q + mask.sum().float() + self.anchor), 1)

    sampler = EpisodeSampler(manifest, METADATA, TrainConfig(crop_jitter=2), split="val", patch_size=(16,) * 3)
    echo_model = CountEcho()
    uninterrupted = evaluate_support_counts(echo_model, sampler)
    binding = {"checkpoint": "fixed"}
    first = TaskJournal(tmp_path / "journal", binding, max_tasks=2)
    evaluate_support_counts(echo_model, sampler, progress=first)
    assert len(first.rows()) == 8
    resumed = TaskJournal(tmp_path / "journal", binding)
    evaluate_support_counts(echo_model, sampler, progress=resumed)
    assert resumed.rows() == uninterrupted
    for start in range(0, len(uninterrupted), 4):
        rows = uninterrupted[start : start + 4]
        assert [r["active_support_count"] for r in rows] == [1, 2, 3, 4]
        assert all(r["fixed_input_sha256"] == rows[0]["fixed_input_sha256"] for r in rows)
        assert all(r["active_support_indices"] == rows[-1]["active_support_indices"][:k] for k, r in enumerate(rows, 1))
    summary = summarize_support_counts(uninterrupted)
    assert all(f"val/query_dice_k{k}" in summary for k in range(1, 5))
    with pytest.raises(ValueError, match="mismatch"):
        TaskJournal(tmp_path / "journal", {"checkpoint": "different"})
