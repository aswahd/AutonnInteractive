"""Inference from normalized nnU-Net volumes; not the raw-image interactive API."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .checkpoint import load_model


class InContextInferenceSession:
    """Binary segmentation defined entirely by labeled 3D support examples.

    Inputs are preprocessed float tensors (C,D,H,W), on CPU, with the same
    normalization/orientation as the training cache. This API does not accept a
    query mask. Outputs are probabilities in the input query's preprocessed grid.
    """

    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model, state = load_model(checkpoint, self.device)
        self.model.eval()
        self.shape = state["input_shape"]
        self.context = None

    def set_context(self, images: list[torch.Tensor], masks: list[torch.Tensor]):
        if not images or len(images) != len(masks):
            raise ValueError(
                "supply equally sized nonempty lists of support images and masks"
            )
        resized_images, resized_masks = [], []
        for image, mask in zip(images, masks):
            self._validate_image(image)
            if mask.shape != (1, *image.shape[-3:]) or mask.device.type != "cpu":
                raise ValueError("support mask must be a CPU tensor shaped (1,D,H,W)")
            if not ((mask == 0) | (mask == 1)).all() or not mask.any():
                raise ValueError(
                    "each support must contain a nonempty binary target mask"
                )
            resized_images.append(
                F.interpolate(
                    image[None].float(),
                    size=self.shape,
                    mode="trilinear",
                    align_corners=False,
                )[0]
            )
            resized = F.interpolate(
                mask[None].float(), size=self.shape, mode="nearest-exact"
            )[0]
            if not resized.any():
                raise ValueError(
                    "target disappeared during downsampling; use a finer training/inference grid"
                )
            resized_masks.append(resized)
        self.context = (
            torch.stack(resized_images)[None].to(self.device),
            torch.stack(resized_masks)[None].to(self.device),
        )

    def _validate_image(self, image):
        if (
            image.ndim != 4
            or image.shape[0] != self.model.image_channels
            or image.device.type != "cpu"
        ):
            raise ValueError(
                "image must be a CPU tensor shaped (C,D,H,W) with the model's channel count"
            )
        if not torch.isfinite(image).all():
            raise ValueError("image contains nonfinite values")

    @torch.inference_mode()
    def predict(self, query: torch.Tensor) -> torch.Tensor:
        if self.context is None:
            raise RuntimeError("call set_context before predicting")
        self._validate_image(query)
        image = F.interpolate(
            query[None].float(), size=self.shape, mode="trilinear", align_corners=False
        ).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(image, *self.context)
        return F.interpolate(
            logits.float().sigmoid(),
            size=query.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )[0, 0].cpu()
