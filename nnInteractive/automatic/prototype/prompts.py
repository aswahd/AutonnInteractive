"""Synthetic supervision in nnInteractive's seven-channel interaction layout."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from nnInteractive.interaction.point import PointInteraction_stub

PROMPT_TYPES = ("point", "scribble", "lasso", "bbox", "initial_mask")
# Existing session layout: previous mask, +/- box or lasso, +/- point, +/- scribble.
PREVIOUS, BOX_POS, BOX_NEG, POINT_POS, POINT_NEG, SCRIBBLE_POS, SCRIBBLE_NEG = range(7)


def draw_point(channel: torch.Tensor, center, radius: int = 2) -> None:
    """A clipped spherical point, in voxel coordinates (D,H,W)."""
    PointInteraction_stub(radius, use_distance_transform=True).place_point(
        tuple(int(c) for c in center), channel
    )


def simulate_prompt(
    target: torch.Tensor, kind: str, seed: int, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Use annotation ONLY for supervised interactive training/evaluation prompts.

    target: CPU (1,D,H,W) binary mask. Returns CPU (7,D,H,W).
    Lasso and boxes are drawn on one plane, not as full 3D target masks.
    """
    if target.ndim != 4 or target.shape[0] != 1 or kind not in PROMPT_TYPES:
        raise ValueError("expected target (1,D,H,W) and a supported prompt type")
    rng = np.random.default_rng(seed)
    known = (target >= 0) if valid is None else valid.bool()
    if known.shape != target.shape:
        raise ValueError("validity mask must match target shape")
    known = known[0].numpy()
    foreground = (target[0].numpy() > 0.5) & known
    prompts = torch.zeros(7, *foreground.shape)
    positions = np.argwhere(foreground)
    if len(positions) == 0:
        center = tuple(s // 2 for s in foreground.shape)
        if known.any():
            if not known[center]:
                center = tuple(np.argwhere(known)[0])
            draw_point(prompts[POINT_NEG], center)
        return prompts * torch.from_numpy(known)[None]
    # Sample from the object interior so spherical positives do not straddle borders unnecessarily.
    interior = distance_transform_edt(foreground)
    candidates = np.argwhere(interior >= max(1, float(interior.max()) * 0.5))
    center = candidates[int(rng.integers(len(candidates)))]
    if kind == "point":
        draw_point(prompts[POINT_POS], center)
    elif kind == "initial_mask":
        altered = (
            binary_dilation(foreground, iterations=2)
            if rng.random() < 0.5
            else binary_erosion(foreground, iterations=2)
        )
        # This is a noisy prior, never a perfectly copied query target.
        altered = np.roll(
            altered, int(rng.choice([-3, -2, 2, 3])), axis=int(rng.integers(3))
        )
        prompts[PREVIOUS] = torch.from_numpy(altered.astype(np.float32))
        draw_point(prompts[POINT_POS], center)
    else:
        axis = int(rng.integers(3))
        slicer = [slice(None)] * 3
        slicer[axis] = int(center[axis])
        plane = foreground[tuple(slicer)]
        if kind == "lasso":
            # Filled approximate contour; positive/negative semantics match add_lasso_interaction.
            plane = binary_dilation(plane, iterations=int(rng.integers(1, 3)))
            prompts[(BOX_POS, *slicer)] = torch.from_numpy(plane.astype(np.float32))
        elif kind == "bbox":
            coordinates = np.argwhere(plane)
            lower, upper = coordinates.min(0), coordinates.max(0) + 1
            box = np.zeros_like(plane, dtype=np.float32)
            box[
                tuple(
                    slice(max(0, lo - 1), min(size, hi + 1))
                    for lo, hi, size in zip(lower, upper, plane.shape)
                )
            ] = 1
            prompts[(BOX_POS, *slicer)] = torch.from_numpy(box)
        else:
            # A short connected random walk constrained to the target, on a single plane.
            current = np.delete(center, axis)
            scribble = np.zeros_like(plane)
            for _ in range(20):
                scribble[tuple(current)] = True
                candidate = current + rng.integers(-1, 2, size=2)
                if (
                    np.all(candidate >= 0)
                    and np.all(candidate < np.array(plane.shape))
                    and plane[tuple(candidate)]
                ):
                    current = candidate
            scribble = binary_dilation(scribble) & plane
            prompts[(SCRIBBLE_POS, *slicer)] = torch.from_numpy(
                scribble.astype(np.float32)
            )
    # Negative feedback is sampled near the object, with all three negative interfaces represented.
    ring = binary_dilation(foreground, iterations=5) & ~binary_dilation(
        foreground, iterations=2
    )
    negative = np.argwhere(ring)
    if len(negative):
        point = negative[int(rng.integers(len(negative)))]
        if kind in {"lasso", "bbox"}:
            slicer = tuple(
                slice(int(c), int(c) + (1 if i == 0 else 2))
                for i, c in enumerate(point)
            )
            prompts[(BOX_NEG, *slicer)] = 1
        elif kind == "scribble":
            draw_point(prompts[SCRIBBLE_NEG], point, radius=1)
        else:
            draw_point(prompts[POINT_NEG], point)
    return prompts * torch.from_numpy(known)[None]


def add_error_correction(
    prompts: torch.Tensor,
    probability: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Detached previous prediction plus a positive/negative corrective point per case."""
    revised = prompts.detach().clone() * 0.98
    revised[:, PREVIOUS : PREVIOUS + 1] = (probability.detach() > 0.5).float()
    for index in range(len(prompts)):
        prediction = probability[index, 0].detach().float().cpu().numpy() > 0.5
        truth = target[index, 0].detach().float().cpu().numpy() > 0.5
        known = (
            np.ones_like(truth)
            if valid is None
            else valid[index, 0].detach().cpu().numpy().astype(bool)
        )
        errors = [
            (truth & ~prediction & known, POINT_POS),
            (prediction & ~truth & known, POINT_NEG),
        ]
        error, channel = max(errors, key=lambda pair: pair[0].sum())
        if error.any():
            distance = distance_transform_edt(error)
            center = np.unravel_index(np.argmax(distance), distance.shape)
            draw_point(revised[index, channel], center)
            revised[index, channel] *= torch.as_tensor(known, device=revised.device)
    return revised
