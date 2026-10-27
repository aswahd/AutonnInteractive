"""Array identity, sampler leakage, native prompt semantics, and true-label negatives."""

import pickle

import blosc2
import numpy as np
import pytest
import torch

from nnInteractive.automatic.config import TrainConfig
from nnInteractive.automatic.data import (
    ArrayStore,
    EpisodeSampler,
    augment_native_image,
    collate_episodes,
)
from nnInteractive.automatic.prompts import (
    PROMPT_TYPES,
    NativePromptGenerator,
)

METADATA = {
    "prompt_channels": 7,
    "channel_mapping": {"prev_seg": 0, "points": [3, 4], "scribble": [5, 6], "bbox2d": [1, 2], "lasso": [1, 2]},
    "point_radius": 4,
    "point_distance_transform": True,
    "scribble_thickness": [2, 2, 2],
}


@pytest.fixture
def manifest(tmp_path):
    cases = {}
    regions = {"union": [1, 2], "second": [3]}
    for index in range(7):
        case = f"volume{index}"
        image = np.arange(32**3, dtype=np.float32).reshape(1, 32, 32, 32) + index
        seg = np.zeros_like(image, dtype=np.int16)
        seg[:, 3:9, 4:9, 4:9] = 1
        seg[:, 9:12, 4:9, 4:9] = 2
        seg[:, 22:28, 22:28, 22:28] = 3
        seg[:, :2] = -1
        for suffix, array in [("", image), ("_seg", seg)]:
            blosc2.asarray(array, urlpath=str(tmp_path / f"{case}{suffix}.b2nd"), mode="w")
        locations = {(1, 2): np.argwhere(np.isin(seg, [1, 2])), 3: np.argwhere(seg == 3)}
        with (tmp_path / f"{case}.pkl").open("wb") as handle:
            pickle.dump({"class_locations": locations}, handle)
        cases[case] = {"volume_id": case, "shape": [32] * 3, "present": list(regions), "annotated": list(regions)}
    return {
        "dataset": str(tmp_path),
        "configuration": ".",
        "regions": regions,
        "cases": cases,
        "splits": {"train": [f"volume{i}" for i in range(6)], "val": ["volume6"]},
        "image_padding": 0.0,
    }


def test_native_crop_preserves_values_and_masks_padding(manifest):
    store = ArrayStore(manifest)
    image, target, valid = store.crop("volume0", (-4, 0, 0), (16,) * 3, "union")
    assert not valid[:, :6].any()
    assert not image[:, :4].any()
    original = store.arrays("volume0")[0][:, :12, :16, :16]
    np.testing.assert_array_equal(image[:, 4:].numpy(), original)
    assert target[:, 7:16, 4:9, 4:9].sum() == 9 * 5 * 5


def test_all_models_read_full_arrays_before_cropping(manifest, monkeypatch):
    store = ArrayStore(manifest)
    arrays = store.arrays("volume0")
    reads = []

    class FullRead:
        def __init__(self, array):
            self.array = array

        def __getitem__(self, key):
            assert key == slice(None)
            reads.append(key)
            return self.array[:]

    monkeypatch.setattr(store, "arrays", lambda case: tuple(FullRead(array) for array in arrays))
    store.crop("volume0", (0, 0, 0), (16,) * 3, "union")
    assert len(reads) == 2


@pytest.mark.parametrize("kind", PROMPT_TYPES)
def test_prompts_use_only_allowed_channels_and_known_voxels(kind):
    target = torch.zeros(1, 24, 24, 24)
    target[:, 7:17, 7:17, 7:17] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, :3] = False
    prompts = NativePromptGenerator(METADATA).render(
        torch.zeros_like(target), target, valid, kind, np.random.default_rng(9)
    )
    assert not prompts[:, ~valid[0]].any()
    if kind == "dense_mask":
        torch.testing.assert_close(prompts[0], target[0])
        assert not prompts[1:].any()
    else:
        assert not prompts[0].any()
        channels = {"point": [3, 4], "scribble": [5, 6], "box": [1, 2], "lasso": [1, 2]}[kind]
        assert prompts[channels[0]].any()
        assert not prompts[[c for c in range(7) if c not in channels]].any()
        if kind != "point":
            positions = torch.nonzero(prompts[channels[0]])
            assert any(len(torch.unique(positions[:, axis])) == 1 for axis in range(3))


def test_independent_sampling_distinct_volumes_and_replay(manifest):
    config = TrainConfig(support_count=4, crop_jitter=2, augment_intensity=False)
    sampler = EpisodeSampler(manifest, METADATA, config, split="train", patch_size=(16,) * 3)
    first = sampler.sample_episode(np.random.default_rng(42), target="union", query_kind="hard_negative")
    replay = sampler.sample_episode(np.random.default_rng(42), target="union", query_kind="hard_negative")
    torch.testing.assert_close(first.support_images, replay.support_images)
    assert first.metadata == replay.metadata
    ids = [s["volume_id"] for s in first.metadata["supports"]]
    assert len(set(ids)) == 4 and first.metadata["query_id"] not in ids
    if first.metadata["query_kind"] == "hard_negative_verified":
        assert not first.query_targets.any()
    second = sampler.sample_episode(np.random.default_rng(43), target="second", query_kind="positive")
    batch = collate_episodes([first, second])
    assert batch.support_images.shape == (2, 4, 1, 16, 16, 16)
    assert batch.metadata[0]["target"] != batch.metadata[1]["target"]
    validation = EpisodeSampler(manifest, METADATA, config, split="val", patch_size=(16,) * 3)
    episode = validation.sample_episode(np.random.default_rng(7))
    assert episode.metadata["query_id"] == "volume6"
    assert all(s["volume_id"] in manifest["splits"]["train"] for s in episode.metadata["supports"])


def test_failed_negative_search_preserves_true_foreground(manifest):
    config = TrainConfig(support_count=1, crop_jitter=0, negative_attempts=2, augment_intensity=False)
    sampler = EpisodeSampler(manifest, METADATA, config, patch_size=(64,) * 3)
    episode = sampler.sample_episode(np.random.default_rng(2), target="union", query_kind="hard_negative")
    assert episode.metadata["query_kind"] == "random_fallback_true_label"
    assert episode.query_targets.any()


@pytest.mark.parametrize("seed", range(4))
def test_dense_augmentation_preserves_tiny_foreground(seed):
    target = torch.zeros(1, 1, 1, 1)
    target.fill_(1)
    valid = torch.ones_like(target, dtype=torch.bool)
    generator = NativePromptGenerator(METADATA)
    prompts = generator.render(target, target, valid, "dense_mask", np.random.default_rng(seed), perturb_dense=True)
    assert prompts[METADATA["channel_mapping"]["prev_seg"]].any()
    assert not generator.support_loss_eligible(prompts, target, valid)


def test_background_query_keeps_foreground_supports(manifest):
    from nnInteractive.automatic.support_dropout import (
        preserve_foreground_supports,
    )

    sampler = EpisodeSampler(
        manifest, METADATA, TrainConfig(crop_jitter=0, augment_intensity=False), patch_size=(16,) * 3
    )
    episode = sampler.sample_episode(
        np.random.default_rng(42), target="union", query_kind="hard_negative", prompt_type="dense_mask"
    )
    assert not episode.query_targets.any()
    active = torch.tensor([[True, False, False, False]])
    corrected = preserve_foreground_supports(active, episode.support_prompts[None], METADATA["channel_mapping"])
    assert torch.equal(corrected, active)
    assert episode.support_prompts[0, METADATA["channel_mapping"]["prev_seg"]].any()


def test_negative_point_keeps_original_stencil_near_target_boundary():
    # Point centers specify foreground/background. The native radius-four stencil
    # is not shaped using the hidden dense mask around the selected center.
    target = torch.ones(1, 24, 24, 24)
    target[0, 12, 12, 12] = 0
    valid = torch.ones_like(target, dtype=torch.bool)
    generator = NativePromptGenerator(METADATA)
    prompts = generator.render(torch.zeros_like(target), target, valid, "point", np.random.default_rng(8))
    expected = torch.zeros(24, 24, 24)
    generator.point.place_point((12, 12, 12), expected)
    torch.testing.assert_close(prompts[4], expected, rtol=0, atol=0)
    assert not prompts[0].any()


def test_intensity_augmentation_preserves_unlabelled_image_and_only_resets_padding(manifest):
    store = ArrayStore(manifest)
    origin, size = (-4, 24, -3), (16,) * 3
    image, _, valid = store.crop("volume0", origin, size, "union")
    rng = np.random.default_rng(1729)
    scale, offset = float(rng.uniform(0.9, 1.1)), float(rng.normal(0, 0.05))
    expected = image * scale + offset
    augmented = augment_native_image(image, origin, (32,) * 3, np.random.default_rng(1729))
    # Original source spans D=[4,16), H=[0,8), W=[3,16) in this crop.
    geometric = torch.zeros_like(valid)
    geometric[:, 4:, :8, 3:] = True
    assert (geometric & ~valid).any()  # Unknown source labels, not padding.
    torch.testing.assert_close(augmented[geometric], expected[geometric], rtol=0, atol=0)
    assert not augmented[~geometric].any()
    assert augmented[geometric & ~valid].abs().sum() > 0


def test_dense_supervision_exclusion_and_perturbation_preserve_sampling(manifest):
    arguments = dict(support_count=4, crop_jitter=2, augment_intensity=False)
    clean = EpisodeSampler(
        manifest, METADATA, TrainConfig(**arguments, perturb_dense_supports=False), patch_size=(16,) * 3
    )
    noisy = EpisodeSampler(
        manifest, METADATA, TrainConfig(**arguments, perturb_dense_supports=True), patch_size=(16,) * 3
    )
    a = clean.sample_episode(np.random.default_rng(42), target="union", query_kind="positive", prompt_type="dense_mask")
    b = noisy.sample_episode(np.random.default_rng(42), target="union", query_kind="positive", prompt_type="dense_mask")
    assert not a.support_eligible.any() and b.support_eligible.all()
    for field in ["query_images", "query_targets", "query_valid", "support_images", "support_targets", "support_valid"]:
        torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
    assert [r["start"] for r in a.metadata["supports"]] == [r["start"] for r in b.metadata["supports"]]
    assert not b.support_prompts[:, 1:].any()
    assert not b.support_prompts[:, :1][~b.support_valid].any()
    validation = EpisodeSampler(manifest, METADATA, TrainConfig(**arguments), split="val", patch_size=(16,) * 3)
    exact = validation.sample_episode(np.random.default_rng(8), prompt_type="dense_mask")
    sparse = validation.sample_episode(np.random.default_rng(8), prompt_type="point")
    assert not exact.support_eligible.any() and sparse.support_eligible.all()
    prompt = exact.support_prompts[0].clone()
    prompt[3, 5, 5, 5] = 1
    assert not validation.prompts.support_loss_eligible(prompt, exact.support_targets[0], exact.support_valid[0])


def test_held_out_labels_cannot_guide_training_negative_crops(manifest, monkeypatch):
    config = TrainConfig(crop_jitter=2, augment_intensity=False)
    sampler = EpisodeSampler(manifest, METADATA, config, targets=["union"], patch_size=(16,) * 3)
    requested = []
    original = sampler._positive_start

    def record(case, target, rng):
        requested.append(target)
        assert target == "union", "held-out annotation used to center a training crop"
        return original(case, target, rng)

    monkeypatch.setattr(sampler, "_positive_start", record)
    for seed in range(6):
        episode = sampler.sample_episode(np.random.default_rng(seed), target="union", query_kind="hard_negative")
        assert episode.metadata["target"] == "union"
    assert requested and set(requested) == {"union"}
    with pytest.raises(ValueError, match="ineligible"):
        sampler.sample_episode(np.random.default_rng(4), target="second")


def test_overlapping_region_targets_match_nnunet_and_dense_prompts(manifest):
    from batchgeneratorsv2.transforms.utils.seg_to_regions import (
        ConvertSegmentationToRegionsTransform,
    )

    manifest["regions"].update(child1=[1], child2=[2])
    store = ArrayStore(manifest)
    segmentation = torch.from_numpy(store.arrays("volume0")[1][:])
    reference = ConvertSegmentationToRegionsTransform([(1, 2), 1, 2])(segmentation=segmentation)["segmentation"]
    masks = []
    for index, target in enumerate(("union", "child1", "child2")):
        image, mask, valid = store.crop("volume0", (0, 0, 0), (32,) * 3, target)
        torch.testing.assert_close(mask[0].bool(), reference[index], rtol=0, atol=0)
        prompt = NativePromptGenerator(METADATA).render(image, mask, valid, "dense_mask", np.random.default_rng(0))
        torch.testing.assert_close(prompt[0], mask[0].float(), rtol=0, atol=0)
        masks.append(mask.bool())
    assert torch.equal(masks[0], masks[1] | masks[2])
    assert (masks[0] & masks[1]).any() and (masks[0] & masks[2]).any()
