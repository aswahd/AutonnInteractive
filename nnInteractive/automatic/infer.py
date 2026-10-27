"""Standalone native-preprocessed-grid inference with probability blending."""

import time
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from .losses import binary_logits
from .spatial import resize_volume


@dataclass
class VolumePrediction:
    probabilities: np.ndarray | None
    mask: np.ndarray
    coverage_min: float
    windows: int
    mode: str = "sliding_window"
    student_prediction: "VolumePrediction | None" = None
    native_forward_passes: int = 0
    refinement_seconds: float = 0.0


def axis_starts(length, patch, stride):
    if length < 1 or not 0 < stride <= patch:
        raise ValueError("positive image extent and 0 < stride <= patch required")
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def gaussian_weights(shape, sigma_fraction=0.125, minimum=1e-6):
    if sigma_fraction <= 0 or minimum <= 0:
        raise ValueError("Gaussian sigma and weight floor must be positive")
    axes = [torch.arange(n, dtype=torch.float32) - (n - 1) / 2 for n in shape]
    weights = torch.ones(shape, dtype=torch.float32)
    for axis, coordinates in enumerate(axes):
        view = [1, 1, 1]
        view[axis] = shape[axis]
        weights *= torch.exp(-0.5 * (coordinates / (shape[axis] * sigma_fraction)).square()).reshape(view)
    weights /= weights.max()
    return weights.clamp_min(minimum)


@torch.inference_mode()
def predict_volume(
    model,
    preprocessed_query,
    support_images,
    support_prompts,
    stride=None,
    padding_value=0.0,
    bf16=True,
    *,
    mode="sliding_window",
    refiner=None,
):
    """Predict on the supplied grid, optionally followed by native AutoZoom refinement.

    Inputs: query 1DHW, supports K1dhw, prompts K7dhw. No query labels or ROI.
    Whole-volume checkpoints resize the complete query and run one student forward.
    Supports are re-encoded independently for every window; no feature cache.
    AutoZoom returns a hard mask (probabilities=None), retaining the complete original
    result in student_prediction. windows/coverage describe exhaustive student tiling.
    """
    if mode not in {"sliding_window", "autozoom", "resized_context"}:
        raise ValueError("mode must be sliding_window, autozoom or resized_context")
    if mode != "sliding_window" and refiner is None:
        raise ValueError("refinement requires an original frozen nnInteractive refiner")
    query = torch.as_tensor(preprocessed_query, device="cpu", dtype=torch.float32)
    if query.ndim != 4 or query.shape[0] != 1 or not torch.isfinite(query).all():
        raise ValueError("query must be finite preprocessed 1DHW")
    device = next(model.parameters()).device
    support_images = torch.as_tensor(support_images, device=device, dtype=torch.float32)
    support_prompts = torch.as_tensor(support_prompts, device=device, dtype=torch.float32)
    if support_images.ndim != 5 or support_prompts.ndim != 5:
        raise ValueError("supports must be K1dhw and prompts K7dhw")
    student_size = tuple(model.config.patch_size)
    if getattr(model.config, "full_volume", False):
        if mode == "resized_context":
            raise ValueError("whole-volume checkpoints support student-only or autozoom inference")
        was_training = model.training
        model.eval()
        try:
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                output = model.predict_query(
                    resize_volume(query[None], student_size).to(device), support_images[None], support_prompts[None]
                )
            probability = resize_volume(binary_logits(output).float().cpu(), query.shape[1:])[0, 0].sigmoid().numpy()
        finally:
            model.train(was_training)
        if not np.isfinite(probability).all():
            raise FloatingPointError("nonfinite whole-volume prediction")
        student = VolumePrediction(probability, (probability > 0.5).astype(np.uint8), 1.0, 1, "full_volume")
        return refine_volume(student, query, refiner) if mode == "autozoom" else student
    patch = tuple(getattr(model.config, "context_size", student_size))
    stride = tuple(stride or [p // 2 for p in patch])
    shape = tuple(query.shape[1:])
    starts = [axis_starts(n, p, s) for n, p, s in zip(shape, patch, stride)]
    numerator = torch.zeros(shape, dtype=torch.float32)
    denominator = torch.zeros_like(numerator)
    refined_numerator = torch.zeros_like(numerator) if mode == "resized_context" else None
    refinement_seconds = 0.0
    weights = gaussian_weights(patch)
    was_training = model.training
    model.eval()
    windows = 0
    try:
        for origin in product(*starts):
            sizes = [min(p, n - start) for p, n, start in zip(patch, shape, origin)]
            source = tuple(slice(start, start + size) for start, size in zip(origin, sizes))
            destination = tuple(slice(0, size) for size in sizes)
            image = torch.full((1, 1, *patch), padding_value, dtype=torch.float32, device=device)
            image[(0, slice(None), *destination)] = query[(slice(None), *source)].to(device)
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                output = model.predict_query(
                    resize_volume(image, student_size), support_images[None], support_prompts[None]
                )
            logits = binary_logits(output)
            probability = resize_volume(logits.float(), patch)[0, 0].sigmoid().cpu()[destination]
            numerator[source] += probability * weights[destination]
            if mode == "resized_context":
                started = time.monotonic()
                teacher_size = tuple(model.config.teacher_input_size)
                teacher_input = resize_volume(image, teacher_size)
                prior = resize_volume(binary_logits(output.float()), teacher_size).sigmoid()
                refined_logits = refiner.predict_logits(teacher_input, prior)
                refined_probability = (
                    resize_volume(binary_logits(refined_logits.float()), patch)[0, 0].sigmoid().cpu()[destination]
                )
                refined_numerator[source] += refined_probability * weights[destination]
                refinement_seconds += time.monotonic() - started
            denominator[source] += weights[destination]
            windows += 1
    finally:
        model.train(was_training)
    if not bool((denominator > 0).all()):
        raise AssertionError("sliding window left uncovered original voxels")
    probability = (numerator / denominator).numpy()
    if not np.isfinite(probability).all():
        raise FloatingPointError("nonfinite volume prediction")
    student = VolumePrediction(probability, (probability > 0.5).astype(np.uint8), float(denominator.min()), windows)
    if refined_numerator is not None:
        refined_probability = (refined_numerator / denominator).numpy()
        if not np.isfinite(refined_probability).all():
            raise FloatingPointError("nonfinite resized-context refinement")
        return VolumePrediction(
            refined_probability,
            (refined_probability > 0.5).astype(np.uint8),
            student.coverage_min,
            windows,
            "resized_context",
            student,
            windows,
            refinement_seconds,
        )
    return refine_volume(student, query, refiner) if mode == "autozoom" else student


def refine_volume(student, preprocessed_query, refiner):
    """Reuse an existing student prediction to compare modes without repeating tiling."""
    if student.mode not in {"sliding_window", "full_volume"}:
        raise ValueError("native refinement requires the original student prediction")
    started = time.monotonic()
    mask, calls = refiner.refine(preprocessed_query, student.mask)
    return VolumePrediction(
        None, mask, student.coverage_min, student.windows, "autozoom", student, calls, time.monotonic() - started
    )
