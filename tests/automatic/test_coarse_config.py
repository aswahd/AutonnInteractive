"""Configured coarse budgets, fixed cohort selection, and unchanged input intensities."""

import hashlib
import json
from types import SimpleNamespace

import blosc2
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from nnInteractive.automatic.prototype import run_config
from nnInteractive.automatic.prototype.data import VolumeStore, prepare


def test_coarse_configuration_sets_budget_and_verifies_initialization(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model/fold_0/checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    spec = dict(
        cache="cache",
        checkpoint_dir=str(tmp_path / "model"),
        targets=["organ"],
        pretrained_sha256=hashlib.sha256(b"fixture").hexdigest(),
        model=dict(mode="prototype"),
        training=dict(updates=50000, batch_size=12, shots=4),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec))
    argv = ["coarse", "--config", str(path), "--output", str(tmp_path / "run")]
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr("sys.argv", argv)
    captured = []
    monkeypatch.setattr(run_config, "run_prototype", lambda: captured.extend(run_config.sys.argv))
    run_config.main()
    assert captured[captured.index("--steps") + 1] == "50000"
    assert captured[captured.index("--batch-size") + 1] == "12"
    spec["pretrained_sha256"] = "incorrect"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(ValueError, match="checkpoint hash"):
        run_config.main()


def test_coarse_preparation_respects_split_intensities_and_local_cache(tmp_path, monkeypatch):
    source, arrays, output = (tmp_path / name for name in ("source", "arrays", "cache"))
    source.mkdir()
    arrays.mkdir()
    (source / "dataset.json").write_text(json.dumps({"labels": {"background": 0, "organ": 1}}))
    (source / "splits_final.json").write_text(json.dumps([{"train": ["a"], "val": ["b"]}]))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["a"], "val": []}))
    image = np.arange(12 * 16 * 20, dtype=np.float32).reshape(1, 12, 16, 20)
    labels = np.ones_like(image, dtype=np.int16)
    labels[:, :3] = -1
    for suffix, data in (("", image), ("_seg", labels)):
        blosc2.asarray(data, urlpath=str(arrays / f"a{suffix}.b2nd"), mode="w")
    prepare(
        SimpleNamespace(
            dataset=str(source),
            output=str(output),
            configuration="unused",
            fold=0,
            seed=2026,
            shape=[8] * 3,
            train_cases=0,
            val_cases=0,
            patient_map=None,
            shard_size=16,
            threads=1,
            split=str(split),
            array_dir=str(arrays),
            normalization="none",
        )
    )
    store = VolumeStore(output)
    assert store.manifest["splits"] == {"train": ["a"], "val": []}
    assert store.manifest["normalization"] == "none"
    expected = F.interpolate(torch.from_numpy(image)[None], (8,) * 3, mode="trilinear", align_corners=False)[0].half()
    torch.testing.assert_close(store.load("a")["image"], expected, rtol=0, atol=0)
    assert (store.load("a")["seg"] < 0).any()
    local = tmp_path / "local"
    local.mkdir()
    (local / "manifest.json").write_text("{}")
    monkeypatch.setenv("NNI_COARSE_CACHE", str(local))
    with pytest.raises(ValueError, match="does not match"):
        VolumeStore(output)
