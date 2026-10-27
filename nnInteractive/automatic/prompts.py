"""Render support prompts in the original checkpoint's channel/numeric conventions."""

from copy import deepcopy

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from nnInteractive.interaction.point import PointInteraction_stub

PROMPT_TYPES = ("dense_mask", "point", "scribble", "box", "lasso")


class NativePromptGenerator:
    def __init__(self, metadata):
        self.channels = metadata["prompt_channels"]
        self.mapping = metadata["channel_mapping"]
        self.point = PointInteraction_stub(metadata["point_radius"], metadata["point_distance_transform"])
        thickness = metadata["scribble_thickness"]
        self.thickness = thickness if isinstance(thickness, list) else [thickness] * 3

    def support_loss_eligible(self, prompts, target, valid):
        """Exclude an exact previous-mask target, including when other cues coexist."""
        prior = prompts[self.mapping["prev_seg"]]
        return not torch.equal(prior[valid[0]], target[0][valid[0]])

    def render(self, image, target, valid, prompt_type, rng, perturb_dense=False):
        if prompt_type not in PROMPT_TYPES or target.shape != valid.shape or image.shape != target.shape:
            raise ValueError("expected aligned 1DHW image/target/valid and a known prompt type")
        if target.ndim != 4 or target.shape[0] != 1:
            raise ValueError("expected single-channel 3D crop")
        known = valid[0].numpy().astype(bool)
        fg = (target[0].numpy() > 0.5) & known
        if not fg.any():
            raise ValueError("support crop must contain valid target foreground")
        prompts = torch.zeros(self.channels, *fg.shape, dtype=torch.float32)
        if prompt_type == "dense_mask":
            previous = fg
            if perturb_dense:
                # Reuse the prototype's initial_mask simulation (two-voxel morphology,
                # 2/3-voxel shift). Clip translation instead of wrapping across borders.
                # A copied RNG preserves the established support/query crop sequence.
                local = deepcopy(rng)
                previous = (
                    binary_dilation(fg, iterations=2) if local.random() < 0.5 else binary_erosion(fg, iterations=2)
                )
                offset, axis = int(local.choice([-3, -2, 2, 3])), int(local.integers(3))
                previous = np.roll(previous, offset, axis=axis)
                edge = [slice(None)] * 3
                edge[axis] = slice(0, offset) if offset > 0 else slice(offset, None)
                previous[tuple(edge)] = False
                previous &= known
                if not previous.any():
                    previous = fg
            prompts[self.mapping["prev_seg"]] = torch.from_numpy(previous.astype(np.float32))
            return prompts
        distance = distance_transform_edt(fg)
        candidates = np.argwhere(distance >= max(1.0, float(distance.max()) * 0.5))
        center = candidates[int(rng.integers(len(candidates)))]
        key = {"point": "points", "scribble": "scribble", "box": "bbox2d", "lasso": "lasso"}[prompt_type]
        pos, neg = self.mapping[key]
        if prompt_type == "point":
            self.point.place_point(tuple(int(c) for c in center), prompts[pos])
        else:
            axis = int(rng.integers(3))
            plane_slice = [slice(None)] * 3
            plane_slice[axis] = int(center[axis])
            plane_slice = tuple(plane_slice)
            plane = fg[plane_slice]
            if prompt_type == "box":
                coordinates = np.argwhere(plane)
                low, high = coordinates.min(0), coordinates.max(0) + 1
                raster = np.zeros_like(plane)
                raster[tuple(slice(int(a), int(b)) for a, b in zip(low, high))] = True
            elif prompt_type == "lasso":
                # Original add_lasso_interaction consumes a filled planar raster.
                raster = plane.copy()
            else:
                current = np.delete(center, axis)
                raster = np.zeros_like(plane)
                for _ in range(40):
                    raster[tuple(current)] = True
                    candidate = current + rng.integers(-1, 2, size=2)
                    if np.all(candidate >= 0) and np.all(candidate < plane.shape) and plane[tuple(candidate)]:
                        current = candidate
                radius = max(0, int(round(max(self.thickness) / 2)))
                if radius:
                    raster = binary_dilation(raster, iterations=radius) & plane
            prompts[(pos, *plane_slice)] = torch.from_numpy(raster.astype(np.float32))
        # A background point is an allowed additional cue for point prompts.
        # The original point rasterizer supplies its distance-transform intensity.
        if prompt_type == "point":
            background = known & ~fg
            coords = np.argwhere(background)
            if len(coords):
                center = coords[int(rng.integers(len(coords)))]
                self.point.place_point(tuple(int(c) for c in center), prompts[neg])
        prompts *= torch.from_numpy(known)[None]
        assert not prompts[self.mapping["prev_seg"]].any()
        return prompts
