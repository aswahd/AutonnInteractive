"""Online coarse crops read full volumes, retain foreground, and replay by episode."""

import json

import numpy as np
import torch

from nnInteractive.automatic.prototype.data import MixedEpisodes
from nnInteractive.automatic.prototype.online import OnlineVolumeStore


def test_full_read_random_crop_resize_and_episode_replay(tmp_path, monkeypatch):
    cases = [f"case{i}" for i in range(7)]
    manifest = dict(
        dataset=str(tmp_path),
        configuration=".",
        regions={"target": [1]},
        splits={"train": cases[:6], "val": cases[6:]},
        cases={case: {"present": ["target"], "shape": [24, 40, 40]} for case in cases},
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    store = OnlineVolumeStore(path, (12, 24, 24), (6, 6, 6))
    raw = np.arange(24 * 40 * 40, dtype=np.float32).reshape(1, 24, 40, 40)
    labels = np.zeros_like(raw, dtype=np.int16)
    labels[0, 12, 20, 20] = 1
    labels[:, :2] = -1
    reads = []

    class FullRead:
        def __init__(self, array):
            self.array = array

        def __getitem__(self, key):
            assert key == slice(None), "The requested pipeline must read the full source volume"
            reads.append(1)
            return self.array.copy()

    monkeypatch.setattr(store, "arrays", lambda case: (FullRead(raw), FullRead(labels)))
    monkeypatch.setattr(store, "foreground_locations", lambda case, target: np.array([[12, 20, 20]]))
    dataset = MixedEpisodes(None, "train", ["target"], 4, episodes=8, seed=2026, store=store)
    first, replay = dataset[0], dataset[0]
    assert len(reads) == 20 and dataset.retriever is None
    assert first["query"].shape == (1, 6, 6, 6)
    assert first["supports"].shape == (4, 1, 6, 6, 6)
    assert len(set(first["support_ids"])) == 4 and first["query_id"] not in first["support_ids"]
    assert (first["support_masks"] > 0).flatten(1).any(1).all()
    torch.testing.assert_close(first["supports"], replay["supports"], rtol=0, atol=0)
    torch.testing.assert_close(first["query"], replay["query"], rtol=0, atol=0)
    assert not torch.equal(first["crop_starts"], dataset[1]["crop_starts"])
    before = store.sample(cases[0], np.random.default_rng(4))
    labels[:] = 0
    after = store.sample(cases[0], np.random.default_rng(4))
    assert before["start"] == after["start"]
    torch.testing.assert_close(before["image"], after["image"], rtol=0, atol=0)
    start = before["start"]
    expected = (start[0] + 0.5) * 1600 + (start[1] + 1.5) * 40 + start[2] + 1.5
    assert before["image"][0, 0, 0, 0] == expected
