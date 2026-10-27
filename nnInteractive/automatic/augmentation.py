"""Independent physical-space augmentation, with aligned categorical supervision."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import affine_transform, map_coordinates, zoom
from scipy.spatial.transform import Rotation


@dataclass
class SpatialAugmentation:
    probability: float = 0.75
    rotation_degrees: float = 20.0
    scale_min: float = 0.8
    scale_max: float = 1.2
    flip_probability: float = 0.5
    anisotropic_scale: bool = False
    translation_fraction: float = 0.0
    elastic_displacement_mm: float = 0.0
    elastic_grid_size: int = 5
    min_foreground_retention: float = 0.0
    boundary_margin_voxels: int = 0

    def __post_init__(self):
        if not (0 <= self.probability <= 1 and 0 <= self.flip_probability <= 1):
            raise ValueError("augmentation probabilities must lie in [0, 1]")
        if not (0 < self.scale_min <= self.scale_max and 0 <= self.rotation_degrees <= 180):
            raise ValueError("invalid augmentation scale or rotation range")
        if not (0 <= self.translation_fraction <= 0.5 and 0 <= self.elastic_displacement_mm < np.inf):
            raise ValueError("invalid translation or elastic displacement")
        if not 0 <= self.min_foreground_retention <= 1:
            raise ValueError("foreground retention must lie within [0, 1]")
        if self.min_foreground_retention and self.elastic_displacement_mm:
            raise ValueError("foreground retention accounting requires affine-only augmentation")
        if type(self.boundary_margin_voxels) is not int or self.boundary_margin_voxels < 0:
            raise ValueError("boundary margin must be a nonnegative integer")
        if self.elastic_grid_size < 2:
            raise ValueError("elastic control grid requires at least two points per axis")

    def __call__(self, image, target, valid, rng, spacing, padding_value=0.0):
        matrix = np.eye(3)
        if rng.random() < self.probability:
            matrix = Rotation.from_euler(
                "xyz", rng.uniform(-self.rotation_degrees, self.rotation_degrees, 3), degrees=True
            ).as_matrix()
            matrix /= rng.uniform(self.scale_min, self.scale_max, size=3 if self.anisotropic_scale else None)
        matrix *= np.where(rng.random(3) < self.flip_probability, -1, 1)
        # scipy maps output coordinates to input coordinates; rotations act in mm.
        spacing = np.asarray(spacing)
        matrix = matrix * spacing[None, :] / spacing[:, None]
        center = (np.asarray(image.shape[1:]) - 1) / 2
        offset = center - matrix @ center
        if self.translation_fraction:
            offset += rng.uniform(-self.translation_fraction, self.translation_fraction, 3) * image.shape[1:]

        def warp(array, order, cval):
            return affine_transform(array, matrix, offset, order=order, cval=cval, prefilter=False)

        if self.elastic_displacement_mm:
            shape = image.shape[1:]
            coordinates = np.indices(shape, dtype=np.float32)
            for axis in range(3):
                control = rng.uniform(-1, 1, (self.elastic_grid_size,) * 3)
                control = torch.from_numpy(zoom(control, 4, order=3).astype(np.float32))[None, None]
                field = torch.nn.functional.interpolate(control, size=shape, mode="trilinear", align_corners=True)[
                    0, 0
                ].numpy()
                field *= self.elastic_displacement_mm / spacing[axis] / max(float(np.abs(field).max()), 1e-6)
                # Bound displacement derivatives before composing with the affine map.
                slope = sum(float(np.abs(np.diff(field, axis=d)).max()) for d in range(3))
                coordinates[axis] += field * min(1.0, 0.35 / max(slope, 1e-6))
            coordinates = np.einsum("ij,jdhw->idhw", matrix.astype(np.float32), coordinates)
            coordinates += offset.astype(np.float32)[:, None, None, None]

            def warp(array, order, cval):
                return map_coordinates(array, coordinates, order=order, cval=cval, prefilter=False)

        labels = np.where(valid[0].numpy(), target[0].numpy(), -1).astype(np.int8)
        labels = warp(labels, 0, -1)
        count = int(target.sum())
        retention = float((labels > 0).sum() * abs(np.linalg.det(matrix)) / max(count, 1))
        clipped = False
        if self.boundary_margin_voxels and count:
            margin = min(self.boundary_margin_voxels, (min(labels.shape) - 1) // 2)
            inner = tuple(slice(margin, -margin) if margin else slice(None) for _ in range(3))
            before = 1 - float(target[0][inner].sum()) / count
            after = 1 - float((labels[inner] > 0).sum()) / max(int((labels > 0).sum()), 1)
            clipped = after > before + 0.02
        rejected = (count > 0 and retention < self.min_foreground_retention) or clipped
        self.last_result = {
            "proposed_foreground_retention": retention if count else None,
            "rejected": bool(rejected),
            "introduced_boundary_contact": bool(clipped),
        }
        if rejected:
            return image, target, valid
        # Do not erase a tiny positive support or turn a crop into wholly unknown data.
        if not (labels >= 0).any() or (bool(target.any()) and not (labels > 0).any()):
            return image, target, valid
        transformed = warp(image[0].numpy(), 1, padding_value)
        return tuple(torch.from_numpy(x[None]) for x in (transformed, labels > 0, labels >= 0))
