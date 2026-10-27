"""Stateless, per-episode mixing compatible with indexed EpisodeStream RNGs."""

import numpy as np


class MixedEpisodeDataset:
    def __init__(self, datasets, probabilities):
        self.datasets = tuple(datasets)
        weights = np.asarray(probabilities, dtype=float)
        if (
            not self.datasets
            or weights.shape != (len(self.datasets),)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError("provide one nonnegative probability per dataset, summing to one")
        self.probabilities = weights / weights.sum()

    def sample_episode(self, rng):
        dataset = self.datasets[int(rng.choice(len(self.datasets), p=self.probabilities))]
        return dataset.sample_episode(rng)
