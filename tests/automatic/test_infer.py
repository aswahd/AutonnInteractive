"""Every original voxel is covered; probabilities, not logits, are blended."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from nnInteractive.automatic.infer import (
    axis_starts,
    gaussian_weights,
    predict_volume,
)


class IdentityLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(patch_size=(8, 8, 8))
        self.calls = 0

    def predict_query(self, query_images, support_images, support_prompts):
        self.calls += 1
        return torch.cat((torch.zeros_like(query_images), query_images + self.anchor), dim=1)


def test_native_grid_coverage_probability_blending_and_small_images():
    for shape in [(19, 13, 9), (3, 6, 5)]:
        image = np.random.default_rng(8).normal(size=(1, *shape)).astype(np.float32)
        model = IdentityLogitModel()
        result = predict_volume(model, image, torch.zeros(1, 1, 8, 8, 8), torch.zeros(1, 7, 8, 8, 8), stride=(4,) * 3)
        expected = torch.from_numpy(image[0]).sigmoid().numpy()
        np.testing.assert_allclose(result.probabilities, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(result.mask, expected > 0.5)
        assert result.coverage_min > 0 and result.windows == model.calls
        assert result.probabilities.shape == shape and model.training
    assert axis_starts(19, 8, 4) == [0, 4, 8, 11]
    assert float(gaussian_weights((8,) * 3).min()) >= 1e-6 - 1e-12


def test_overlap_averages_probabilities_not_logits():
    class ConstantPatchModel(IdentityLogitModel):
        def predict_query(self, query_images, support_images, support_prompts):
            self.calls += 1
            return torch.cat(
                (torch.zeros_like(query_images), torch.ones_like(query_images) * query_images.mean()), dim=1
            )

    query = np.zeros((1, 12, 8, 8), dtype=np.float32)
    query[:, 4:8] = 2
    query[:, 8:] = 4
    prediction = predict_volume(
        ConstantPatchModel(), query, torch.zeros(1, 1, 8, 8, 8), torch.zeros(1, 7, 8, 8, 8), stride=(4,) * 3
    )
    weights = np.exp(-0.5 * np.array([1.5, -2.5]) ** 2)
    expected = np.dot(weights, 1 / (1 + np.exp(-np.array([1.0, 3.0])))) / weights.sum()
    logit_average = 1 / (1 + np.exp(-np.dot(weights, [1.0, 3.0]) / weights.sum()))
    assert abs(float(prediction.probabilities[5, 3, 3]) - expected) < 1e-6
    assert abs(float(prediction.probabilities[5, 3, 3]) - logit_average) > 1e-3
