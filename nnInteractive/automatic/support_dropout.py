"""Episode-wise support selection."""

import math

import torch

DEFAULTS = {
    "enabled": True,
    "min_active_supports": 1,
    "max_active_supports": 4,
    "sampling_strategy": "uniform_active_count",
    "per_episode": True,
    "count_probabilities": None,
}


def validate_dropout_config(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("unknown support dropout configuration")
    config = {**DEFAULTS, **config}
    low, high = config["min_active_supports"], config["max_active_supports"]
    if type(config["enabled"]) is not bool or config["per_episode"] is not True:
        raise ValueError("support dropout must use a boolean enabled flag and independent episodes")
    if type(low) is not int or type(high) is not int or not 1 <= low <= high <= 4:
        raise ValueError("active support count range must lie within 1..4")
    probabilities = config["count_probabilities"]
    strategy = config["sampling_strategy"]
    if strategy == "uniform_active_count":
        if probabilities is not None:
            raise ValueError("use categorical_active_count for nonuniform count probabilities")
    elif strategy == "categorical_active_count":
        if not isinstance(probabilities, (list, tuple)) or len(probabilities) != high - low + 1:
            raise ValueError("provide one probability per active count")
        if any(not math.isfinite(p) or p < 0 for p in probabilities) or not math.isclose(sum(probabilities), 1.0):
            raise ValueError("count probabilities must be finite, nonnegative, and sum to one")
    else:
        raise ValueError("unsupported support dropout sampling strategy")
    return config


def sample_support_active_mask(batch_size, num_supports, device, config=None):
    config = validate_dropout_config({} if config is None else config)
    if batch_size < 1 or num_supports < 1:
        raise ValueError("positive batch and support counts required")
    if not config["enabled"]:
        return torch.ones(batch_size, num_supports, dtype=torch.bool, device=device)
    low, high = config["min_active_supports"], config["max_active_supports"]
    if high > num_supports:
        raise ValueError("active count exceeds the supplied support bank")
    if config["sampling_strategy"] == "uniform_active_count":
        counts = torch.randint(low, high + 1, (batch_size,), device=device)
    else:
        probabilities = torch.tensor(config["count_probabilities"], device=device, dtype=torch.float32)
        counts = torch.multinomial(probabilities, batch_size, replacement=True) + low
    ranks = torch.rand(batch_size, num_supports, device=device).argsort(dim=1).argsort(dim=1)
    return ranks < counts[:, None]


def preserve_foreground_supports(active, prompts, channel_mapping):
    """Undo dropout per episode if it removes every foreground support prompt."""
    channels = [channel_mapping["prev_seg"]] + [
        channel_mapping[key][0] for key in ("points", "scribble", "bbox2d", "lasso")
    ]
    foreground = prompts.flatten(3).amax(3)[..., channels].gt(0).any(2)
    if not foreground.any(1).all():
        raise ValueError("every episode requires a foreground support prompt before dropout")
    return active | ~(active & foreground).any(1, keepdim=True)


def active_support_statistics(histogram):
    """Summarize all episodes since the preceding log, aggregated across ranks."""
    histogram = histogram.detach().cpu().tolist()
    total = sum(histogram)
    if not total:
        raise ValueError("cannot summarize an empty support-count histogram")
    observed = [i + 1 for i, n in enumerate(histogram) if n]
    return {
        "train/active_support_count_mean": sum((i + 1) * n for i, n in enumerate(histogram)) / total,
        "train/active_support_count_min": min(observed),
        "train/active_support_count_max": max(observed),
        "train/active_support_episodes": total,
        **{f"train/active_support_fraction_{i + 1}": n / total for i, n in enumerate(histogram)},
    }
