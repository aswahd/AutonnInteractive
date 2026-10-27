"""Prefetch and resume preserve the exact independently indexed episode sequence."""

import pytest
import torch
from test_data import METADATA, manifest  # noqa: F401

from nnInteractive.automatic.config import TrainConfig
from nnInteractive.automatic.data import EpisodeSampler
from nnInteractive.automatic.stream import EpisodeStream


@pytest.mark.parametrize("workers,prefetch", [(1, 1), (2, 2), (4, 2)])
@pytest.mark.parametrize("microbatch", [1, 2])
@pytest.mark.parametrize("backend", ["thread", "process"])
def test_prefetch_matches_serial_and_restart(manifest, tmp_path, workers, prefetch, microbatch, backend):  # noqa: F811
    config = TrainConfig(
        microbatch=microbatch,
        worker_backend=backend,
        effective_batch=4 * microbatch,
        support_count=2,
        crop_jitter=2,
        augment_intensity=True,
        num_workers=workers,
        prefetch_factor=prefetch,
    )
    sampler = EpisodeSampler(manifest, METADATA, config, patch_size=(16,) * 3)
    serial = EpisodeStream(sampler, config, 0, 2, 2, tmp_path / "serial.jsonl", prefetch=False)
    parallel = EpisodeStream(sampler, config, 0, 2, 2, tmp_path / "parallel.jsonl", prefetch=True)
    try:
        expected_next = None
        for update, micro in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            expected, _ = serial.get(update, micro)
            observed, _ = parallel.get(update, micro)
            assert [e.metadata for e in observed] == [e.metadata for e in expected]
            for name in [
                "query_images",
                "support_images",
                "support_prompts",
                "query_targets",
                "query_valid",
                "support_targets",
                "support_valid",
                "support_eligible",
            ]:
                for a, b in zip(observed, expected):
                    torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
            if (update, micro) == (0, 1):
                parallel.quiesce()
            if (update, micro) == (1, 0):
                expected_next = expected[0]
    finally:
        serial.close()
        parallel.close()
    resumed = EpisodeStream(sampler, config, 0, 2, 2, tmp_path / "resumed.jsonl", prefetch=True)
    try:
        observed, _ = resumed.get(1, 0)
        assert observed[0].metadata == expected_next.metadata
        torch.testing.assert_close(observed[0].support_prompts, expected_next.support_prompts, rtol=0, atol=0)
    finally:
        resumed.close()


def test_background_collation_matches_serial(manifest, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    pinned = []
    original_empty = torch.empty

    def allocate(*args, **kwargs):
        if kwargs.pop("pin_memory", False):
            pinned.append(True)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    config = TrainConfig(support_count=2, microbatch=2, effective_batch=4, crop_jitter=2, num_workers=2)
    sampler = EpisodeSampler(manifest, METADATA, config, patch_size=(16,) * 3)
    serial = EpisodeStream(sampler, config, 0, 1, 2, tmp_path / "serial.jsonl", prefetch=False, collate=True)
    parallel = EpisodeStream(sampler, config, 0, 1, 2, tmp_path / "parallel.jsonl", collate=True)
    try:
        for update, micro in ((0, 0), (0, 1), (1, 0), (1, 1)):
            expected, _ = serial.get(update, micro)
            observed, _ = parallel.get(update, micro)
            assert observed.metadata == expected.metadata
            for name, value in vars(expected).items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(getattr(observed, name), value, rtol=0, atol=0)
            assert len(parallel.ready) <= 1
            if (update, micro) == (0, 0):
                parallel.quiesce()
        parallel.quiesce()
        assert len(pinned) == 8 * sum(isinstance(value, torch.Tensor) for value in vars(expected).values())
    finally:
        serial.close()
        parallel.close()
