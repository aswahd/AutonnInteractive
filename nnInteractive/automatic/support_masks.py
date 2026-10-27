"""Convert native support prompts to binary masks before ICL context encoding."""

from copy import deepcopy

import torch
from torch import nn


class FrozenSupportSegmenter(nn.Module):
    def __init__(self, network, mask_channel, chunk_size, require_nonempty=False, convert_dense=True):
        super().__init__()
        self.network = deepcopy(network).eval().requires_grad_(False)
        self.mask_channel, self.chunk_size = mask_channel, chunk_size
        self.require_nonempty = require_nonempty
        self.convert_dense = convert_dense
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images, prompts):
        """Preserve dense-only priors when enabled; convert sparse/mixed prompts without GT or AutoZoom."""
        shape = prompts.shape
        images, prompts = images.reshape(-1, *images.shape[-4:]), prompts.reshape(-1, *shape[-4:])
        present = prompts.flatten(2).any(2)
        if self.require_nonempty and not present.any(1).all():
            raise ValueError("unprompted support: supply a previous segmentation or a positive sparse prompt")
        converted = torch.zeros_like(prompts)
        dense = present[:, self.mask_channel] & (present.sum(1) == 1) & (not self.convert_dense)
        converted[dense, self.mask_channel] = (prompts[dense, self.mask_channel] > 0.5).to(converted.dtype)
        for selected in (~dense).nonzero().flatten().split(self.chunk_size):
            if not len(selected):
                continue
            logits = self.network(torch.cat((images[selected], prompts[selected].to(images.dtype)), 1))
            if logits.shape != (len(selected), 2, *images.shape[-3:]) or not torch.isfinite(logits).all():
                raise FloatingPointError("invalid frozen support segmentation")
            converted[selected, self.mask_channel] = logits.argmax(1).to(converted.dtype)
        if self.require_nonempty and not converted[:, self.mask_channel].flatten(1).any(1).all():
            raise ValueError("empty support segmentation: provide a usable mask or revise the support prompt")
        return converted.reshape(shape)
