"""Read full preprocessed volumes, crop in source space, then resize each episode."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..data import ArrayStore
from ...utils.crop import crop_and_pad_into_buffer


class OnlineVolumeStore(ArrayStore):
    def __init__(self, manifest, crop_shape=(192, 384, 384), input_shape=(96, 96, 96)):
        super().__init__(manifest)
        self.manifest_path = Path(manifest)
        self.crop_shape, self.input_shape = tuple(crop_shape), tuple(input_shape)
        if min(*self.crop_shape, *self.input_shape) < 1:
            raise ValueError("crop and input dimensions must be positive")
        self.manifest["shape"] = list(self.input_shape)
        for case, record in self.manifest["cases"].items():
            record["patient"] = record.get("patient", case)

    def sample(self, case, rng, target=None):
        # Deliberately read the full arrays for every sampled volume; no resized cache.
        raw, labels = self.read_volume(case)
        shape = raw.shape[-3:]
        start = tuple(int(rng.integers(min(0, n - p), max(0, n - p) + 1)) for n, p in zip(shape, self.crop_shape))
        if target is not None:
            locations = self.foreground_locations(case, target)
            if not len(locations):
                raise ValueError("support volume has no annotated target foreground")
            point = locations[int(rng.integers(len(locations)))].astype(int)
            # Randomize the foreground's output position while keeping that real voxel
            # on the nearest-exact grid, so tiny targets survive support resizing.
            offset = [int((rng.integers(s) + 0.5) * p / s) for p, s in zip(self.crop_shape, self.input_shape)]
            start = tuple(int(c - d) for c, d in zip(point, offset))
        bbox = [(a, a + p) for a, p in zip(start, self.crop_shape)]
        image = torch.full((1, *self.crop_shape), self.manifest.get("image_padding", 0.0), dtype=torch.float32)
        segmentation = torch.full((1, *self.crop_shape), -1, dtype=torch.int16)
        crop_and_pad_into_buffer(image, bbox, raw)
        crop_and_pad_into_buffer(segmentation, bbox, labels)
        segmentation[segmentation == self.manifest.get("ignore_label", -1)] = -1
        image = F.interpolate(image[None], self.input_shape, mode="trilinear", align_corners=False)[0]
        segmentation = F.interpolate(segmentation[None].float(), self.input_shape, mode="nearest-exact")[0].short()
        if not torch.isfinite(image).all() or not (segmentation >= 0).any():
            raise ValueError("sampled crop has nonfinite intensities or no valid labels")
        if target is not None and not np.isin(segmentation.numpy(), self.manifest["regions"][target]).any():
            raise ValueError("foreground location metadata does not match support segmentation")
        return {"image": image, "seg": segmentation, "start": start}
