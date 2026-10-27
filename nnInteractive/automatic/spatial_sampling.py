"""Independent training views on a canonical S/A/R grid, with bounded crop retries."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .spatial import resize_volume, support_at_resolution


@dataclass
class SpatialSampling:
    full_volume_probability: float = 0.30
    targeted_probability_given_patch: float = 0.50
    target_crop_scale_range: tuple[float, float] = (0.30, 1.00)
    hard_negative_probability_given_random_query_patch: float = 0.50
    context_margin_mm: float = 0.0
    broad_supports: int = 0

    def __post_init__(self):
        probabilities = (
            self.full_volume_probability,
            self.targeted_probability_given_patch,
            self.hard_negative_probability_given_random_query_patch,
        )
        if any(not 0 <= p <= 1 for p in probabilities):
            raise ValueError("spatial sampling probabilities must lie in [0, 1]")
        if not np.isfinite(self.context_margin_mm) or self.context_margin_mm < 0:
            raise ValueError("context margin must be finite and nonnegative")
        if type(self.broad_supports) is not int or not 0 <= self.broad_supports <= 4:
            raise ValueError("broad support count must lie within 0..4")
        low, high = self.target_crop_scale_range
        if not 0 < low <= high <= 1:
            raise ValueError("target crop scales must satisfy 0 < low <= high <= 1")
        self.target_crop_scale_range = (low, high)

    def draw(self, rng, support=False):
        if rng.random() < self.full_volume_probability:
            return "full_volume"
        if rng.random() < self.targeted_probability_given_patch:
            return "targeted"
        if not support and rng.random() < self.hard_negative_probability_given_random_query_patch:
            return "hard_negative"
        return "random"


class MixedSpatialSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.geometries = {}
        self.bounds = {}
        self.policy = SpatialSampling(**dataset.config.spatial_sampling)
        self.size = np.array(dataset.student_input_size)
        manifest = dataset.manifest
        plans = [json.loads(p.read_text()) for p in Path(manifest["dataset"]).glob("*.json")]
        self.plans = [
            p
            for p in plans
            if isinstance(p, dict)
            and any(c.get("data_identifier") == manifest["configuration"] for c in p.get("configurations", {}).values())
        ]

    def geometry(self, case):
        """Use closest SAR axes without interpolation; preserve and report acquisition obliquity."""
        if case in self.geometries:
            return self.geometries[case]
        d = self.dataset
        properties = pickle.loads((d.store.folder / f"{case}.pkl").read_bytes())
        obliquity = np.zeros(3)
        if "array_axis_codes" in d.manifest:
            orientation = nib.orientations.axcodes2ornt(d.manifest["array_axis_codes"])
        else:
            orders = {tuple(p["transpose_forward"]) for p in self.plans}
            if len(orders) != 1:
                raise ValueError("spatial sampling requires one preprocessing axis order or explicit array_axis_codes")
            readers = {p.get("image_reader_writer") for p in self.plans}
            if not readers <= {"NibabelIO", "SimpleITKIO"}:
                raise ValueError("unsupported orientation reader; provide explicit stored array_axis_codes")
            order = np.array([2, 1, 0])[list(orders.pop())]
            if "nibabel_stuff" in properties:
                directions = properties["nibabel_stuff"]["original_affine"][:3, order]
            elif "sitk_stuff" in properties:
                directions = (
                    np.diag([-1, -1, 1]) @ np.array(properties["sitk_stuff"]["direction"]).reshape(3, 3)[:, order]
                )
            else:
                raise ValueError("spatial sampling requires image orientation metadata")
            lengths = np.linalg.norm(directions, axis=0)
            if not np.isfinite(directions).all() or (lengths <= 0).any():
                raise ValueError(f"invalid image orientation for {case}")
            directions = directions / lengths
            if abs(np.linalg.det(directions)) < 1e-6:
                raise ValueError(f"degenerate image orientation for {case}")
            obliquity = np.degrees(np.arccos(np.clip(np.abs(directions).max(axis=0), 0, 1)))
            affine = np.eye(4)
            affine[:3, :3] = directions
            orientation = nib.orientations.io_orientation(affine)
        transform = nib.orientations.ornt_transform(orientation, nib.orientations.axcodes2ornt(("S", "A", "R")))
        permutation = np.argsort(transform[:, 0])
        flips = transform[permutation, 1] < 0
        spacings = {
            tuple(c["spacing"])
            for p in self.plans
            for c in p["configurations"].values()
            if c.get("data_identifier") == d.manifest["configuration"]
        }
        if d.manifest.get("native_spacing"):
            spacings = {tuple(d.manifest["cases"][case]["spacing"])}
        if len(spacings) != 1:
            raise ValueError("spatial sampling requires one verified preprocessed spacing")
        spacing = np.array(spacings.pop())[permutation]
        if not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("invalid preprocessed spacing")
        shape = np.array(d.manifest["cases"][case]["shape"])[permutation]
        self.geometries[case] = (shape, spacing, permutation, flips, obliquity[permutation])
        return self.geometries[case]

    def target_bounds(self, case, target, permutation, flips, shape):
        """Cache exact annotation bounds, never extrema of sampled foreground locations."""
        key = (case, target["path"] if isinstance(target, dict) else target)
        if key not in self.bounds:
            from .data import foreground_window

            source_shape = tuple(self.dataset.manifest["cases"][case]["shape"])
            _, truth, _ = self.dataset.store.crop(case, (0, 0, 0), source_shape, target)
            start, size = foreground_window(truth[0].numpy(), (1, 1, 1))
            low, size = np.array(start)[permutation], np.array(size)[permutation]
            low = np.where(flips, shape - low - size, low)
            self.bounds[key] = (low, low + size)
        return self.bounds[key]

    def view(self, case, target, mode, rng, augmentation_rng, support=False):
        from .data import InvalidCropError, augment_native_image, padding_fraction

        d = self.dataset
        shape, spacing, permutation, flips, obliquity = self.geometry(case)
        requested, fallback, rescued = mode, None, False
        for attempt in range(17):
            if attempt == 16:
                if support and mode == "random":
                    mode, fallback = "targeted", "support_foreground"
                elif mode == "hard_negative":
                    mode, fallback = "random", "hard_negative_unavailable"
                else:
                    raise InvalidCropError("mixed spatial view exhausted 16 valid/foreground attempts")
            scale = float(rng.uniform(*self.policy.target_crop_scale_range)) if mode == "targeted" else None
            size = (
                shape.copy() if mode == "full_volume" else np.maximum(1, np.rint(self.size * (scale or 1))).astype(int)
            )
            anchor = target if mode == "targeted" else None
            if mode == "hard_negative" and attempt % 2 == 0 and hasattr(d, "_negative_targets"):
                others = d._negative_targets(case, target)
                if others:
                    anchor = str(rng.choice(others))
            if anchor is not None:
                locations = d.store.foreground_locations(case, anchor)
                if not len(locations):
                    raise InvalidCropError("target-centered crop requested from target-absent volume")
                center = locations[int(rng.integers(len(locations)))][permutation]
                center = np.where(flips, shape - 1 - center, center)
                start = np.where(
                    shape >= size, np.clip(center - size // 2, 0, np.maximum(0, shape - size)), (shape - size) // 2
                )
            else:
                start = np.array(
                    [int(rng.integers(n - s + 1)) if n >= s else (n - s) // 2 for n, s in zip(shape, size)]
                )
            if mode == "targeted" and self.policy.context_margin_mm:
                low, high = self.target_bounds(case, target, permutation, flips, shape)
                margin = np.ceil(self.policy.context_margin_mm / spacing).astype(int)
                size = np.minimum(shape, np.maximum(size, high - low + 2 * margin))
                start = np.clip((low + high - size) // 2, 0, shape - size)
            # Crop on stored axes, then apply the exact permutation/flips to all three tensors.
            source_start, source_size = np.empty(3, int), np.empty(3, int)
            source_start[permutation] = np.where(flips, shape - start - size, start)
            source_size[permutation] = size
            try:
                arrays = d.store.crop(case, tuple(source_start), tuple(source_size), target)
            except InvalidCropError:
                continue
            arrays = [
                a.permute(0, *(permutation + 1)).flip(tuple(np.flatnonzero(flips) + 1)).contiguous() for a in arrays
            ]
            image, truth, valid = arrays
            foreground = int(truth.sum())
            if (support and not foreground) or (mode == "hard_negative" and foreground):
                continue
            if support:
                image, truth, valid, rescued = support_at_resolution(image, truth, valid, self.size)
                truth = truth.bool()
            else:
                image = resize_volume(image[None], self.size)[0]
                truth = resize_volume(truth[None], self.size, "nearest-exact")[0].bool()
                valid = resize_volume(valid[None], self.size, "nearest-exact")[0].bool()
            effective_spacing = spacing * size / self.size
            if not valid.any() or (mode == "targeted" and not truth.any()):
                continue
            if d.config.augment_intensity:
                image = augment_native_image(
                    image,
                    (0, 0, 0),
                    self.size,
                    augmentation_rng,
                    d.manifest.get("image_padding", 0.0),
                    strength=2.5 if d.augmentation else 1.0,
                )
                for axis in range(3):
                    centers = (torch.arange(int(self.size[axis])) + 0.5) * size[axis] / self.size[axis] + start[axis]
                    outside = (centers < 0) | (centers >= shape[axis])
                    slices = [slice(None)] * 4
                    slices[axis + 1] = outside
                    image[tuple(slices)] = d.manifest.get("image_padding", 0.0)
            if d.augmentation:
                image, truth, valid = d.augmentation(
                    image, truth, valid, augmentation_rng, effective_spacing, d.manifest.get("image_padding", 0.0)
                )
            if not valid.any() or ((support or mode == "targeted") and not truth.any()):
                continue
            return (image, truth, valid), {
                "requested_mode": requested,
                "actual_mode": mode,
                "start": start.tolist(),
                "crop_size": size.tolist(),
                "source_start": source_start.tolist(),
                "source_crop_size": source_size.tolist(),
                "scale": scale,
                "foreground_voxels": foreground,
                "model_foreground_voxels": int(truth.sum()),
                "effective_spacing": effective_spacing.tolist(),
                "orientation": "SAR",
                "orientation_method": "closest_canonical_axes",
                "obliquity_degrees": obliquity.tolist(),
                "fallback": fallback,
                "attempts": attempt + 1,
                "padding_fraction": padding_fraction(start, shape, size),
                "prompt_foreground_rescued": rescued,
                "crop_policy": "context_preserving_v2" if self.policy.context_margin_mm else "mixed_scale_v1",
                **(
                    {"augmentation": d.augmentation.last_result.copy()}
                    if d.augmentation and getattr(d.augmentation, "min_foreground_retention", 0)
                    else {}
                ),
            }
        raise InvalidCropError("mixed spatial fallback could not produce a valid view")
