"""Independent native-array episodes with paired image/label augmentation."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import blosc2
import numpy as np
import torch

from .augmentation import SpatialAugmentation
from .backbone import sha256
from .config import TrainConfig
from .prompts import NativePromptGenerator
from .spatial import resize_volume, support_at_resolution

SUPPORT_CROP_POLICY = "foreground_containing_v1"


def support_crop_policy(full_volume=False, fixed_support_crop=False):
    if fixed_support_crop:
        return "fixed_positive_crop_v1"
    return "full_volume_resize_v1" if full_volume else SUPPORT_CROP_POLICY


def foreground_window(foreground, preferred_size):
    """Contain the complete foreground, shifting/expanding inside the available scan."""
    if not foreground.any():
        raise ValueError("support annotation has no foreground")
    bounds = [np.flatnonzero(foreground.any(axis=tuple(d for d in range(3) if d != axis))) for axis in range(3)]
    size = tuple(min(n, max(p, int(b[-1] - b[0] + 1))) for b, p, n in zip(bounds, preferred_size, foreground.shape))
    start = tuple(
        int(np.clip((b[0] + b[-1] + 1 - length) // 2, max(0, b[-1] + 1 - length), min(b[0], n - length)))
        for b, length, n in zip(bounds, size, foreground.shape)
    )
    return start, size


def region_key(values):
    return values[0] if len(values) == 1 else tuple(values)


def padding_fraction(start, shape, size):
    inside = [max(0, min(n, a + p) - max(0, a)) for a, n, p in zip(start, shape, size)]
    return 1 - float(np.prod(inside) / np.prod(size))


def build_manifest(dataset, configuration, output, selected_split=None):
    dataset, output = Path(dataset), Path(output)
    definition = json.loads((dataset / "dataset.json").read_text())
    original_split = json.loads((dataset / "splits_final.json").read_text())[0]
    split = original_split if selected_split is None else selected_split
    if set(split["train"]) & set(split["val"]):
        raise ValueError("query/train split overlap")
    for name in ("train", "val"):
        if len(set(split[name])) != len(split[name]) or not set(split[name]) <= set(original_split[name]):
            raise ValueError("selected split must be a unique subset of its original partition")
    regions = {
        name: (value if isinstance(value, list) else [value])
        for name, value in definition["labels"].items()
        if name not in {"background", "ignore"}
    }
    folder = dataset / configuration
    cases = {}
    for index, case in enumerate(split["train"] + split["val"]):
        path = folder / f"{case}.pkl"
        payload = path.read_bytes()
        properties = pickle.loads(payload)
        locations = properties["class_locations"]
        shape = list(blosc2.open(str(folder / f"{case}.b2nd"), mode="r").shape)
        seg_shape = list(blosc2.open(str(folder / f"{case}_seg.b2nd"), mode="r").shape)
        if len(shape) != 4 or shape[0] != 1 or seg_shape != shape:
            raise ValueError(f"incompatible source array shape for {case}")
        missing = [name for name, values in regions.items() if region_key(values) not in locations]
        if missing:
            raise ValueError(f"cannot infer annotation availability for {case}: missing {missing}")
        cases[case] = {
            "volume_id": case,
            "shape": shape[1:],
            "properties_sha256": hashlib.sha256(payload).hexdigest(),
            "present": [name for name, values in regions.items() if len(locations[region_key(values)])],
            "annotated": list(regions),
        }
        if definition.get("native_spacing"):
            cases[case]["spacing"] = list(properties["spacing"])
        if (index + 1) % 100 == 0:
            print(json.dumps({"indexed_volumes": index + 1}), flush=True)
    manifest = {
        "format_version": 2,
        "dataset": str(dataset.resolve()),
        "configuration": configuration,
        "native_spacing": definition.get("native_spacing", False),
        "splits": split,
        "regions": regions,
        "cases": cases,
        "image_padding": 0.0,
        "array_transform": "none; existing preprocessed arrays, crop/pad only",
        "split_unit": "volume ID; patient identity is not inferred",
        "patient_identity_audited": False,
        "annotation_policy": "Dataset's complete categorical task views; negative labels/ignore/padding invalid. No claim to recover source mask overlaps.",
        "ignore_label": definition["labels"].get("ignore", -1),
        "source_sha256": {name: sha256(dataset / name) for name in ["dataset.json", "splits_final.json"]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output)
    return manifest


class InvalidCropError(ValueError):
    """An attempted crop contains no valid supervised voxels."""


class ArrayStore:
    def __init__(self, manifest, *, use_array_override=True):
        self.manifest = json.loads(Path(manifest).read_text()) if not isinstance(manifest, dict) else manifest
        self.folder = Path(
            (os.environ.get("PROMPT_ICL_ARRAY_DIR") if use_array_override else None)
            or Path(self.manifest["dataset"]) / self.manifest["configuration"]
        )
        marker = self.folder / "cache.json"
        self.decoded_cache = marker.is_file()
        if self.decoded_cache:
            cached = json.loads(marker.read_text())
            source = (Path(self.manifest["dataset"]) / self.manifest["configuration"]).resolve()
            if cached["source"] != str(source) or not set(self.manifest["cases"]) <= set(cached["cases"]):
                raise ValueError("decoded cache does not match the source dataset")

    @lru_cache(maxsize=8)
    def locations(self, case):
        with (self.folder / f"{case}.pkl").open("rb") as handle:
            return pickle.load(handle)["class_locations"]

    @lru_cache(maxsize=16)
    def arrays(self, case):
        if self.decoded_cache:
            return tuple(
                np.load(self.folder / f"{case}{suffix}.npy", mmap_mode="r", allow_pickle=False)
                for suffix in ("", "_seg")
            )
        return tuple(
            blosc2.open(
                str(self.folder / f"{case}{suffix}.b2nd"),
                mode="r",
                dparams={"nthreads": 1},
            )
            for suffix in ["", "_seg"]
        )

    def foreground_locations(self, case, target):
        values = self.manifest["regions"][target]
        locations = np.asarray(self.locations(case)[region_key(values)])
        return locations[:, -3:] if locations.size else np.empty((0, 3), dtype=np.int64)

    def read_volume(self, case):
        """Read original full arrays for each sample; never persist resized volumes."""
        return tuple(array[:] for array in self.arrays(case))

    def spacing(self, case=None):
        if self.manifest.get("native_spacing"):
            spacings = {tuple(self.manifest["cases"][case]["spacing"])}
        else:
            spacings = {
                tuple(config["spacing"])
                for path in Path(self.manifest["dataset"]).glob("*Plans*.json")
                for config in json.loads(path.read_text())["configurations"].values()
                if config.get("data_identifier") == self.manifest["configuration"]
            }
        if len(spacings) != 1:
            raise ValueError("augmentation requires one verified spacing for the preprocessed data identifier")
        spacing = spacings.pop()
        if len(spacing) != 3 or not all(np.isfinite(s) and s > 0 for s in spacing):
            raise ValueError("invalid preprocessed voxel spacing")
        return spacing

    def foreground(self, labels, target):
        return np.isin(labels, self.manifest["regions"][target])

    def valid_labels(self, labels):
        return (labels >= 0) & (labels != self.manifest.get("ignore_label", -1))

    def has_target(self, case, target):
        return target in self.manifest["cases"][case]["present"]

    def foreground_crop(self, case, size, target):
        volume = self.read_volume(case)
        foreground = np.isin(volume[1][0], self.manifest["regions"][target])
        foreground &= (volume[1][0] >= 0) & (volume[1][0] != self.manifest.get("ignore_label", -1))
        start, size = foreground_window(foreground, size)
        return self.crop(case, start, size, target, volume=volume), start

    def crop(self, case, start, size, target, *, volume=None):
        shape = self.manifest["cases"][case]["shape"]
        source = tuple(slice(max(0, int(a)), min(n, int(a) + s)) for a, s, n in zip(start, size, shape))
        destination = tuple(
            slice(max(0, -int(a)), max(0, -int(a)) + src.stop - src.start) for a, src in zip(start, source)
        )
        if any(src.stop <= src.start for src in source):
            raise InvalidCropError("reject crop with no image overlap")
        image = np.full((1, *size), self.manifest.get("image_padding", 0.0), dtype=np.float32)
        labels = np.full((1, *size), -1, dtype=np.int16)
        raw, seg = self.read_volume(case) if volume is None else volume
        image[(slice(None), *destination)] = raw[(slice(None), *source)]
        labels[(slice(None), *destination)] = seg[(slice(None), *source)]
        valid = self.valid_labels(labels)
        foreground = self.foreground(labels, target) & valid
        if not np.isfinite(image).all():
            raise ValueError("nonfinite source image")
        if not valid.any():
            raise InvalidCropError("reject all-invalid query crop")
        return (
            torch.from_numpy(image),
            torch.from_numpy(foreground),
            torch.from_numpy(valid),
        )


def augment_native_image(image, start, source_shape, rng, padding_value=0.0, strength=1.0):
    """Apply intensity jitter inside the image extent, independently of annotation availability."""
    augmented = image * float(rng.uniform(1 - 0.1 * strength, 1 + 0.1 * strength)) + float(
        rng.normal(0, 0.05 * strength)
    )
    for axis, (origin, extent, length) in enumerate(zip(start, source_shape, image.shape[1:]), start=1):
        low, high = max(0, -int(origin)), min(length, int(extent) - int(origin))
        if low:
            padding = [slice(None)] * 4
            padding[axis] = slice(0, low)
            augmented[tuple(padding)] = padding_value
        if high < length:
            padding = [slice(None)] * 4
            padding[axis] = slice(high, length)
            augmented[tuple(padding)] = padding_value
    return augmented


@dataclass
class Episode:
    query_images: torch.Tensor
    support_images: torch.Tensor
    support_prompts: torch.Tensor
    query_targets: torch.Tensor
    query_valid: torch.Tensor
    metadata: dict
    support_targets: torch.Tensor | list[torch.Tensor]
    support_valid: torch.Tensor | list[torch.Tensor]
    support_eligible: torch.Tensor
    query_context: torch.Tensor | None = None
    support_contexts: torch.Tensor | list[torch.Tensor] | None = None


@dataclass
class EpisodeBatch:
    query_images: torch.Tensor
    support_images: torch.Tensor
    support_prompts: torch.Tensor
    query_targets: torch.Tensor
    query_valid: torch.Tensor
    metadata: list[dict]
    support_targets: torch.Tensor | list[list[torch.Tensor]]
    support_valid: torch.Tensor | list[list[torch.Tensor]]
    support_eligible: torch.Tensor
    query_context: torch.Tensor | None = None
    support_contexts: torch.Tensor | list[list[torch.Tensor]] | None = None

    def to(self, device):
        return EpisodeBatch(
            **{
                name: (
                    getattr(self, name).to(device, non_blocking=True)
                    if isinstance(getattr(self, name), torch.Tensor)
                    else getattr(self, name)
                )
                for name in [
                    "query_images",
                    "support_images",
                    "support_prompts",
                    "query_targets",
                    "query_valid",
                    "support_targets",
                    "support_valid",
                    "support_eligible",
                    "query_context",
                    "support_contexts",
                ]
            },
            metadata=self.metadata,
        )


def collate_episodes(episodes, pin_memory=False):
    def stack(name):
        value = getattr(episodes[0], name)
        if value is None:
            return None
        if isinstance(value, list):  # Variable-sized validation references stay on the CPU.
            return [getattr(episode, name) for episode in episodes]
        output = torch.empty((len(episodes), *value.shape), dtype=value.dtype, pin_memory=True) if pin_memory else None
        return torch.stack([getattr(episode, name) for episode in episodes], out=output)

    return EpisodeBatch(
        **{
            name: stack(name)
            for name in [
                "query_images",
                "support_images",
                "support_prompts",
                "query_targets",
                "query_valid",
                "support_targets",
                "support_valid",
                "support_eligible",
                "query_context",
                "support_contexts",
            ]
        },
        metadata=[episode.metadata for episode in episodes],
    )


class EpisodeDataset:
    """Shared native crops, augmentation, prompts, and episode assembly."""

    store_class = ArrayStore

    def __init__(
        self,
        manifest,
        prompt_metadata,
        config: TrainConfig,
        split="train",
        targets=None,
        patch_size=(192,) * 3,
        student_input_size=None,
        retain_support_contexts=True,
        full_volume=False,
        fixed_support_crop=False,
    ):
        self.store, self.config = self.store_class(manifest), config
        self.manifest = self.store.manifest
        self.split, self.patch_size = split, tuple(patch_size)
        self.student_input_size = tuple(student_input_size or patch_size)
        self.full_volume, self.fixed_support_crop = full_volume, fixed_support_crop
        self.mixed = None
        if split == "train" and config.spatial_sampling is not None:
            from .spatial_sampling import MixedSpatialSampler

            self.mixed = MixedSpatialSampler(self)
        if full_volume and config.sample_positive_queries and config.spatial_sampling is None:
            raise ValueError("positive query volume sampling requires crop-mode queries")
        self.crop_policy = support_crop_policy(full_volume, fixed_support_crop)
        self.resized_context = not self.mixed and not full_volume and self.student_input_size != self.patch_size
        self.retain_support_contexts = retain_support_contexts
        self.augmentation = (
            SpatialAugmentation(**config.spatial_augmentation)
            if split == "train" and config.spatial_augmentation is not None
            else None
        )
        self.spacing = (
            {case: self.store.spacing(case) for case in self.manifest["cases"]}
            if self.augmentation and not self.mixed and self.manifest.get("native_spacing")
            else self.store.spacing() if self.augmentation and not self.mixed else None
        )
        self.prompts = NativePromptGenerator(prompt_metadata)
        if split not in {"train", "val"}:
            raise ValueError("unknown query split")
        if not np.isclose(sum(config.prompt_probabilities.values()), 1) or any(
            v < 0 for v in config.prompt_probabilities.values()
        ):
            raise ValueError("invalid prompt probabilities")

        self._initialize(targets)

    def _random_start(self, shape, rng):
        if self.resized_context:
            return tuple(int(rng.integers(min(0, n - p), max(0, n - p) + 1)) for n, p in zip(shape, self.patch_size))
        return tuple(
            int(rng.integers(0, max(1, n - p + 1))) if n >= p else (n - p) // 2 for n, p in zip(shape, self.patch_size)
        )

    def _augment(self, image, target, valid, case, start, rng, support=False):
        source_shape = self.manifest["cases"][case]["shape"]
        spacing = self.spacing[case] if isinstance(self.spacing, dict) else self.spacing
        if self.full_volume and self.split == "train" and not (support and self.fixed_support_crop):
            image, target, valid, _ = support_at_resolution(image, target, valid, self.student_input_size)
            target = target.bool()
            if not valid.any():
                raise InvalidCropError("whole-volume resize removed all valid voxels")
            if spacing is not None:
                spacing = np.asarray(spacing) * np.asarray(source_shape) / self.student_input_size
            source_shape = self.student_input_size
        if self.split == "train" and self.config.augment_intensity:
            image = augment_native_image(
                image,
                start,
                source_shape,
                rng,
                self.manifest.get("image_padding", 0.0),
                strength=2.5 if self.augmentation else 1.0,
            )
        if self.augmentation:
            return self.augmentation(image, target, valid, rng, spacing, self.manifest.get("image_padding", 0.0))
        return image, target, valid

    def _positive_start(self, case, target, rng, size=None, jitter=True):
        locations = self.store.foreground_locations(case, target)
        if not len(locations):
            raise ValueError("positive crop requested for target-absent volume")
        center = locations[int(rng.integers(len(locations)))].astype(int)
        if jitter:
            center += rng.integers(-self.config.crop_jitter, self.config.crop_jitter + 1, size=3)
        return tuple(int(c - p // 2) for c, p in zip(center, size or self.patch_size))

    def _crop(self, case, target, kind, rng):
        shape = self.manifest["cases"][case]["shape"]
        if kind == "support_fixed":
            for attempt in range(17):
                start = self._positive_start(case, target, rng, self.student_input_size, jitter=attempt < 16)
                try:
                    crop = self.store.crop(case, start, self.student_input_size, target)
                except InvalidCropError:
                    continue
                if crop[1].any():
                    return crop, start, kind
            raise ValueError("fixed support crop could not retain its foreground")
        if self.full_volume:
            return self.store.crop(case, (0, 0, 0), tuple(shape), target), (0, 0, 0), "full_volume"
        if kind == "support":
            crop, start = self.store.foreground_crop(case, self.patch_size, target)
            return crop, start, kind
        if not self.store.has_target(case, target):
            kind = "random_absent_volume"
        if kind == "positive":
            for _ in range(16):
                start = self._positive_start(case, target, rng)
                crop = self.store.crop(case, start, self.patch_size, target)
                if bool(crop[1].any()):
                    return crop, start, "positive"
            raise ValueError("foreground sampling failed; inspect the location metadata")
        if kind == "hard_negative":
            others = self._negative_targets(case, target)
            for attempt in range(self.config.negative_attempts):
                if attempt % 2 == 0 and others:
                    start = self._positive_start(case, str(rng.choice(others)), rng)
                else:
                    start = self._random_start(shape, rng)
                crop = self.store.crop(case, start, self.patch_size, target)
                if not bool(crop[1].any()):
                    return crop, start, "hard_negative_verified"
            kind = "random_fallback_true_label"
        start = self._random_start(shape, rng)
        return self.store.crop(case, start, self.patch_size, target), start, kind

    def sample_episode(self, rng, K=None, target=None, query_id=None, query_kind=None, prompt_type=None):
        # One separate stream, advanced independently for the query and every support.
        augmentation_rng = np.random.default_rng(rng.integers(2**63)) if self.augmentation else rng
        count = self.config.support_count if K is None else K
        if count != self.config.support_count:
            raise ValueError("K is fixed per run")
        query_record = {}
        if self.mixed:
            query_mode = query_kind or self.mixed.policy.draw(rng)
            query_mode = {"positive": "targeted"}.get(query_mode, query_mode)
            if query_mode not in {"targeted", "random", "hard_negative", "full_volume"}:
                raise ValueError("unknown mixed spatial query mode")
            query_kind = "positive" if query_mode == "targeted" else query_mode
        query, crop_target, support_ids, query_kind, selection_metadata = self._select_episode(
            rng, count, target, query_id, query_kind
        )
        if self.mixed:
            (image, truth, valid), query_record = self.mixed.view(query, crop_target, query_mode, rng, augmentation_rng)
            start, actual_kind = query_record["start"], query_record["actual_mode"]
        else:
            (image, truth, valid), start, actual_kind = self._crop(query, crop_target, query_kind, rng)
            image, truth, valid = self._augment(image, truth, valid, query, start, augmentation_rng)
        support_images, support_prompts, supports, support_contexts = [], [], [], []
        support_targets, support_valid, support_eligible = [], [], []
        for case in support_ids:
            support_record = {}
            if self.mixed:
                mode = self.mixed.policy.draw(rng, support=True)
                if len(supports) < self.mixed.policy.broad_supports and mode == "random":
                    mode = "targeted"
                (support, mask, known), support_record = self.mixed.view(
                    case, crop_target, mode, rng, augmentation_rng, support=True
                )
                origin, crop_size = support_record["start"], support_record["crop_size"]
            else:
                (support, mask, known), origin, _ = self._crop(
                    case,
                    crop_target,
                    "support_fixed" if self.fixed_support_crop else "support" if self.split == "val" else "positive",
                    rng,
                )
                crop_size = list(support.shape[-3:])
                support, mask, known = self._augment(support, mask, known, case, origin, augmentation_rng, support=True)
            kind = prompt_type or str(
                rng.choice(
                    list(self.config.prompt_probabilities),
                    p=list(self.config.prompt_probabilities.values()),
                )
            )
            if (self.resized_context or self.split == "val") and self.retain_support_contexts:
                support_contexts.append(support)
            student_support, prompt_target, prompt_valid, rescued = support_at_resolution(
                support, mask, known, self.student_input_size
            )
            prompts = self.prompts.render(
                student_support,
                prompt_target,
                prompt_valid,
                kind,
                rng,
                perturb_dense=self.split == "train" and self.config.perturb_dense_supports,
            )
            eligible = self.prompts.support_loss_eligible(prompts, prompt_target, prompt_valid)
            support_targets.append(mask)
            support_valid.append(known)
            support_eligible.append(eligible)
            support_images.append(student_support)
            support_prompts.append(prompts)
            supports.append(
                {
                    "volume_id": case,
                    "start": list(origin),
                    **(
                        {
                            "crop_size": crop_size,
                            "crop_policy": self.crop_policy,
                        }
                        if self.split == "val" or self.full_volume or self.fixed_support_crop
                        else {}
                    ),
                    "prompt_type": kind,
                    "interactive_eligible": eligible,
                    "prompt_foreground_rescued": rescued,
                    "valid_fraction": float(known.float().mean()),
                    "padding_fraction": (
                        0.0
                        if self.full_volume and not self.fixed_support_crop
                        else padding_fraction(origin, self.manifest["cases"][case]["shape"], mask.shape[-3:])
                    ),
                    "dense_prompt": ("perturbed" if eligible else "exact") if kind == "dense_mask" else None,
                    **support_record,
                }
            )
        pack_supports = list if self.split == "val" else torch.stack
        return Episode(
            resize_volume(image[None], self.student_input_size)[0],
            torch.stack(support_images),
            torch.stack(support_prompts),
            truth,
            valid,
            {
                "query_id": query,
                **selection_metadata,
                "query_start": list(start),
                "requested_query_kind": query_kind,
                "query_kind": actual_kind,
                "positive_voxels": int(truth.sum()),
                "query_valid_fraction": float(valid.float().mean()),
                "query_padding_fraction": (
                    0.0
                    if self.full_volume
                    else padding_fraction(start, self.manifest["cases"][query]["shape"], self.patch_size)
                ),
                "context_patch_size": list(
                    self.manifest["cases"][query]["shape"] if self.full_volume else self.patch_size
                ),
                **({"input_policy": support_crop_policy(True)} if self.full_volume else {}),
                **(
                    {
                        "spatial_sampling": query_record,
                        "query_start": query_record["start"],
                        "context_patch_size": query_record["crop_size"],
                        "query_padding_fraction": query_record["padding_fraction"],
                        "input_policy": query_record["crop_policy"],
                    }
                    if self.mixed
                    else {}
                ),
                "student_input_size": list(self.student_input_size),
                "supports": supports,
                "support_patch_count": count,
                "support_unique_volume_count": len(set(support_ids)),
            },
            pack_supports(support_targets),
            pack_supports(support_valid),
            torch.tensor(support_eligible, dtype=torch.bool),
            image if self.resized_context else None,
            pack_supports(support_contexts) if support_contexts else None,
        )


class EpisodeSampler(EpisodeDataset):
    """Real semantic targets with distinct support volumes."""

    def _initialize(self, targets):
        self.targets = list(targets or self.manifest["regions"])
        self.queries, self.support_pool = {}, {}
        for target in self.targets:
            self.queries[target] = [
                c for c in self.manifest["splits"][self.split] if target in self.manifest["cases"][c]["annotated"]
            ]
            self.support_pool[target] = [
                c for c in self.manifest["splits"]["train"] if target in self.manifest["cases"][c]["present"]
            ]
        self.targets = [
            t
            for t in self.targets
            if self.queries[t] and len(self.support_pool[t]) >= self.config.support_count + (self.split == "train")
        ]
        if not self.targets:
            raise ValueError("no target has enough distinct eligible support volumes")

    def _negative_targets(self, case, target):
        return [t for t in self.manifest["cases"][case]["present"] if t != target and t in self.targets]

    def _select_episode(self, rng, count, target, query_id, query_kind):
        target = str(rng.choice(self.targets)) if target is None else target
        if target not in self.targets:
            raise ValueError("target is ineligible for this run")
        query_pool = self.queries[target]
        if self.split == "train" and (self.mixed or (query_id is None and self.config.sample_positive_queries)):
            query_kind = query_kind or str(
                rng.choice(["positive", "hard_negative", "random"], p=self.config.query_probabilities)
            )
            if query_kind == "positive":
                query_pool = [case for case in query_pool if target in self.manifest["cases"][case]["present"]]
        query = str(rng.choice(query_pool)) if query_id is None else query_id
        if query not in self.queries[target]:
            raise ValueError("query not in the configured partition")
        pool = [c for c in self.support_pool[target] if c != query]
        if len(pool) < count:
            raise ValueError("insufficient distinct support volumes after excluding query")
        support_ids = rng.choice(pool, size=count, replace=False).tolist()
        query_kind = (
            "full_volume"
            if self.full_volume and not self.mixed
            else query_kind
            or str(
                rng.choice(
                    ["positive", "hard_negative", "random"],
                    p=self.config.query_probabilities,
                )
            )
        )
        return query, target, support_ids, query_kind, {"target": target, "episode_type": "real_label"}

    def sample_supports(self, rng, target, query_id, prompt_type=None):
        """Support selection for inference: never opens or inspects query labels."""
        if target not in self.targets:
            raise ValueError("target is ineligible")
        pool = [case for case in self.support_pool[target] if case != query_id]
        if len(pool) < self.config.support_count:
            raise ValueError("insufficient distinct support volumes")
        support_ids = rng.choice(pool, size=self.config.support_count, replace=False).tolist()
        images, prompts, records = [], [], []
        for case in support_ids:
            (image, mask, valid), start, _ = self._crop(
                case, target, "support_fixed" if self.fixed_support_crop else "support", rng
            )
            crop_size = list(image.shape[-3:])
            kind = prompt_type or str(
                rng.choice(
                    list(self.config.prompt_probabilities),
                    p=list(self.config.prompt_probabilities.values()),
                )
            )
            image, mask, valid, _ = support_at_resolution(image, mask, valid, self.student_input_size)
            images.append(image)
            prompts.append(self.prompts.render(image, mask, valid, kind, rng))
            records.append(
                {
                    "volume_id": case,
                    "start": list(start),
                    "prompt_type": kind,
                    "crop_policy": self.crop_policy,
                    "crop_size": crop_size,
                }
            )
        return torch.stack(images), torch.stack(prompts), records
