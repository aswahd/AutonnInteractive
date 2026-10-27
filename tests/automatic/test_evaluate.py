"""Target-macro aggregation and same-image prompt-swap control integrity."""

import numpy as np
import pytest
import torch
from test_data import METADATA, manifest  # noqa: F401
from torch import nn

from nnInteractive.automatic.config import TrainConfig
from nnInteractive.automatic.data import EpisodeSampler
from nnInteractive.automatic.evaluate import binary_metrics, summarize
from nnInteractive.automatic.swaps import (
    evaluate_prompt_swaps,
    make_swap_episodes,
    summarize_swaps,
)


def test_macro_and_empty_strata():
    rows = []
    for target, dice in [("large", 1.0), ("large", 1.0), ("small", 0.0)]:
        rows.append({"target": target, "prompt_type": "point", "dice": dice})
    negative = binary_metrics(np.array([1, 0]), np.array([0, 0]), np.array([True, True]))
    rows.append({"target": "small", "prompt_type": "point", **negative})
    scores = summarize(rows)
    assert scores["positive_macro_dice"] == 0.5
    assert scores["by_prompt_type"]["point"]["positive_macro_dice"] == 0.5
    assert scores["by_target"]["small"]["negative_mean_predicted_voxels"] == 1
    assert scores["by_prompt_type"]["point"]["negative_mean_false_positive_fraction"] == 0.5
    assert negative["dice"] is None


def test_swap_only_changes_prompts_and_scores_both_targets(manifest):  # noqa: F811
    sampler = EpisodeSampler(manifest, METADATA, TrainConfig(support_count=4), split="val", patch_size=(64,) * 3)
    episodes, reason = make_swap_episodes(sampler, ["union", "second"], "dense_mask", np.random.default_rng(12))
    assert reason is None
    first, second = episodes
    assert torch.equal(first.query_images, second.query_images)
    assert torch.equal(first.support_images, second.support_images)
    assert not torch.equal(first.support_prompts, second.support_prompts)
    assert first.metadata["query_id"] == "volume6"
    ids = [s["volume_id"] for s in first.metadata["supports"]]
    assert len(set(ids)) == 4 and "volume6" not in ids

    class PromptEcho(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def predict_query(self, query_images, support_images, support_prompts):
            return torch.cat((torch.zeros_like(query_images), support_prompts[:, 0, :1] * 20 - 10), dim=1)

    rows = evaluate_prompt_swaps(PromptEcho(), sampler, [["union", "second"]])
    dense = [r for r in rows if r["prompt_type"] == "dense_mask"]
    assert len(dense) == 2
    for row in dense:
        assert row["correct_prompt"]["dice"] == 1
        assert row["swapped_original_target"]["dice"] == 0
        assert row["swapped_replacement_target"]["dice"] == 1
        assert row["prediction_changed_voxels"] > 0
    assert summarize_swaps(rows)["controls"] == 10
    with pytest.raises(ValueError, match="two different"):
        make_swap_episodes(sampler, ["union", "union"], "dense_mask", np.random.default_rng(0))


def test_interactive_baseline_matches_crops_and_prompts_and_excludes_exact_dense(manifest):  # noqa: F811
    from nnInteractive.automatic.evaluate import (
        evaluate_interactive,
        summarize_interactive,
    )

    sampler = EpisodeSampler(
        manifest, METADATA, TrainConfig(support_count=4, crop_jitter=2), split="val", patch_size=(32,) * 3
    )

    class LocalPromptModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def interactive(self, image, prompts):
            return torch.cat((torch.zeros_like(image), prompts[:, :1] * 20 - 10 + self.anchor), 1)

    model = LocalPromptModel()
    baseline = evaluate_interactive(model, sampler)
    after = evaluate_interactive(model, sampler)
    assert baseline == after and len(baseline) == 12
    for target in sampler.targets:
        rows = [r for r in baseline if r["target"] == target]
        assert len({(r["query_id"], tuple(r["query_start"])) for r in rows}) == 1
        assert not next(r for r in rows if r["prompt_type"] == "dense_mask")["interactive_eligible"]
    scores = summarize_interactive(after, baseline)
    assert scores["eligible"]["records"] == 10
    assert scores["paired_dice_change_from_pretrained"]["positive_macro_dice"] == 0
    after[0] = {**after[0], "prompt_sha256": "changed"}
    with pytest.raises(ValueError, match="identical"):
        summarize_interactive(after, baseline)


def test_explicit_full_volume_prompt_protocol(manifest, monkeypatch):  # noqa: F811
    from types import SimpleNamespace

    from nnInteractive.automatic import evaluate

    sampler = EpisodeSampler(manifest, METADATA, TrainConfig(crop_jitter=2), split="val", patch_size=(16,) * 3)
    observed = []

    def predict(model, image, images, prompts, **kwargs):
        observed.append(prompts)
        return SimpleNamespace(mask=np.zeros(image.shape[1:], dtype=bool), windows=1, coverage_min=1.0)

    monkeypatch.setattr(evaluate, "predict_volume", predict)
    rows = evaluate.evaluate_volumes(
        SimpleNamespace(config=SimpleNamespace(patch_size=(16,) * 3)),
        sampler,
        ["volume6"],
        prompt_type="dense_mask",
    )
    assert len(rows) == 2 and len(observed) == 2
    assert all(r["prompt_type"] == "dense_mask" for r in rows)
    assert all(p[:, 0].any() and not p[:, 1:].any() for p in observed)
    with pytest.raises(ValueError, match="unknown"):
        evaluate.evaluate_volumes(None, sampler, ["volume6"], prompt_type="invalid")
