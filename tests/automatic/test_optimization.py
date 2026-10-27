"""Optimizer grouping, schedule, full state, and next independent episode replay."""

import random
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from nnInteractive.automatic.config import ModelConfig, TrainConfig
from nnInteractive.automatic.optimization import (
    episode_rng,
    lr_factor,
    make_optimizer,
    make_scheduler,
    restore_rng,
    rng_state,
    save_checkpoint,
)


def test_resume_allows_only_runtime_prefetch_changes():
    original = TrainConfig()
    legacy = original.to_dict()
    legacy.pop("num_workers")
    legacy.pop("prefetch_factor")
    assert TrainConfig(**legacy).resume_settings() == TrainConfig(num_workers=4, prefetch_factor=3).resume_settings()
    assert original.resume_settings() != TrainConfig(updates=1000).resume_settings()
    assert original.resume_settings() != TrainConfig(effective_batch=12).resume_settings()
    assert original.resume_settings() != TrainConfig(seed=99).resume_settings()


def test_optimizer_groups_warmup_and_resume(tmp_path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
            self.context = nn.Linear(4, 1)
            self.config = ModelConfig()

        def forward(self, x):
            return self.context(self.backbone(x))

    config = TrainConfig(updates=10, warmup_updates=2)
    assert config.accumulation_steps(1) == 4 and config.accumulation_steps(4) == 1
    assert lr_factor(0, config) > 0 and lr_factor(9, config) == config.final_lr_fraction
    torch.manual_seed(7)
    model = Tiny()
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    for group in optimizer.param_groups:
        if "no_decay" in group["group_name"]:
            assert group["weight_decay"] == 0
            assert all(p.ndim <= 1 for p in group["params"])
        assert group["lr"] > 0
    x = torch.randn(2, 3)
    model(x).square().mean().backward()
    optimizer.step()
    scheduler.step()
    saved_rng = rng_state()
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, scheduler, 1, 0.2, config, {"test": True}, {"split": "sha256"}, [saved_rng])
    state = torch.load(path, weights_only=False)
    assert state["next_global_episode_index"] == 4
    assert state["train_config"] == asdict(config) and state["manifests"] == {"split": "sha256"}
    next_episode = episode_rng(config.seed, 4).integers(100000, size=10)
    np.testing.assert_array_equal(
        next_episode,
        episode_rng(state["train_config"]["seed"], state["next_global_episode_index"]).integers(100000, size=10),
    )
    expected_random = (random.random(), float(np.random.random()), torch.rand(3))
    restore_rng(state["rank_rng_states"][0])
    actual_random = (random.random(), float(np.random.random()), torch.rand(3))
    assert expected_random[:2] == actual_random[:2]
    torch.testing.assert_close(expected_random[2], actual_random[2])
    restored = Tiny()
    restored.load_state_dict(state["model"], strict=True)
    restored_optimizer = make_optimizer(restored, config)
    restored_scheduler = make_scheduler(restored_optimizer, config)
    restored_optimizer.load_state_dict(state["optimizer"])
    restored_scheduler.load_state_dict(state["scheduler"])
    for network, opt, sched in [(model, optimizer, scheduler), (restored, restored_optimizer, restored_scheduler)]:
        opt.zero_grad()
        network(x).square().mean().backward()
        opt.step()
        sched.step()
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert scheduler.get_last_lr() == restored_scheduler.get_last_lr()
