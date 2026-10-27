"""Small multiscale mask encoder and fixed spatial support positions."""

import math

import torch
from torch import nn


class ChannelLayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.movedim(1, -1)).movedim(-1, 1)


class MaskEncoder3D(nn.Module):
    """Encode one-channel masks at the selected stride-2 backbone stages."""

    def __init__(self, expected_channels=(32, 64, 128, 256, 320, 320), stages=(-3, -2, -1)):
        super().__init__()
        self.stages = tuple(s + len(expected_channels) if s < 0 else s for s in stages)
        if (
            len(expected_channels) != 6
            or not self.stages
            or len(set(self.stages)) != len(self.stages)
            or any(s < 1 or s > 5 for s in self.stages)
        ):
            raise ValueError("mask stages must be distinct stride-2 stages of a six-stage encoder")
        widths = (1, 8, 16, 32, 64, 128)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(widths[s - 1], widths[s], 3, stride=2, padding=1),
                ChannelLayerNorm(widths[s], eps=1e-6),
                nn.GELU(),
            )
            for s in range(1, max(self.stages) + 1)
        )
        self.heads = nn.ModuleDict({str(s): nn.Conv3d(widths[s], expected_channels[s], 1) for s in self.stages})

    def forward(self, masks):
        if masks.ndim != 5 or masks.shape[1] != 1:
            raise ValueError("masks must have shape N1DHW")
        outputs = {}
        for stage, block in enumerate(self.blocks, 1):
            masks = block(masks)
            if str(stage) in self.heads:
                outputs[stage] = self.heads[str(stage)](masks)
        return outputs


def support_position_encoding(features):
    """Normalized DHW sine/cosine positions, shared across batches and supports."""
    _, channels, *spatial = features.shape
    pairs = channels // 6
    positions = torch.zeros((1, channels, *spatial), device=features.device, dtype=torch.float32)
    if pairs:
        frequencies = 10000 ** (torch.arange(pairs, device=features.device, dtype=torch.float32) / pairs)
        for axis, size in enumerate(spatial):
            shape = [1, 1, 1]
            shape[axis] = size
            coordinates = torch.arange(1, size + 1, device=features.device, dtype=torch.float32).reshape(shape)
            angles = coordinates.unsqueeze(0) * (2 * math.pi / size) / frequencies[:, None, None, None]
            values = torch.stack((angles.sin(), angles.cos()), dim=1).flatten(0, 1)
            positions[:, axis * 2 * pairs : (axis + 1) * 2 * pairs] = values
    return positions.to(dtype=features.dtype)
