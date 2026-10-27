"""A promptable 3D nnU-Net with an additional support-volume dimension."""

from __future__ import annotations

import torch
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from torch import Tensor, nn
from torch.nn import functional as F


def small_backbone(width: int) -> nn.Module:
    return ResidualEncoderUNet(
        input_channels=8,
        n_stages=4,
        features_per_stage=[width, width * 2, width * 4, width * 8],
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=[1, 2, 2, 2],
        n_blocks_per_stage=[1, 1, 2, 2],
        num_classes=2,
        n_conv_per_stage_decoder=[1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
        deep_supervision=False,
    )


def _initialize_attention(module: nn.Module, channels: int) -> None:
    """Keep layer order and names identical to the original bottleneck block."""
    if channels % 4:
        raise ValueError("attention channels must be divisible by four heads")
    module.mask_embedding = nn.Linear(2, channels)
    module.position_embedding = nn.Linear(3, channels, bias=False)
    module.query_norm, module.memory_norm = (
        nn.LayerNorm(channels),
        nn.LayerNorm(channels),
    )
    module.attention = nn.MultiheadAttention(channels, 4, batch_first=True, dropout=0)
    module.output_norm = nn.LayerNorm(channels)
    module.mlp = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(), nn.Linear(2 * channels, channels))


def _memory_tokens(module, features, foreground, known, batch, count, memory_grid):
    grid = (memory_grid,) * 3
    tokens = F.adaptive_avg_pool3d(features, grid).flatten(2).transpose(1, 2)
    occupancy = F.adaptive_avg_pool3d(foreground, grid).flatten(2).transpose(1, 2)
    coverage = F.adaptive_avg_pool3d(known, grid).flatten(2).transpose(1, 2)
    axes = [torch.linspace(-1, 1, memory_grid, device=features.device)] * 3
    positions = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    tokens = tokens + module.mask_embedding(torch.cat([occupancy, coverage - occupancy], -1))
    tokens = tokens + module.position_embedding(positions)[None]
    return module.memory_norm(tokens.reshape(batch, count * memory_grid**3, -1))


def _condition_features(module, features, memory):
    query_tokens = features.flatten(2).transpose(1, 2)
    conditioned = (
        query_tokens + module.attention(module.query_norm(query_tokens), memory, memory, need_weights=False)[0]
    )
    conditioned = conditioned + module.mlp(module.output_norm(conditioned))
    return conditioned.transpose(1, 2).reshape_as(features)


class StageCrossAttention(nn.Module):
    """Independent mask/position encoding and cross-attention at one encoder scale."""

    def __init__(self, channels: int):
        super().__init__()
        _initialize_attention(self, channels)


class PrototypeMemory(nn.Module):
    """Legacy prototype and mask-memory construction, independent of training/data."""

    def __init__(self, channels, mode="memory", memory_grid=4, attention_stages=(-1,), prototype_tokens=False):
        super().__init__()
        if mode not in {"memory", "prototype"} or memory_grid < 1:
            raise ValueError("invalid mode or memory grid")
        self.mode, self.memory_grid = mode, memory_grid
        self.prototype_tokens = prototype_tokens
        self.fusions = nn.ModuleList(nn.Conv3d(c + 2, c, 1) for c in channels)
        stages = tuple(attention_stages)
        if not stages or any(type(stage) is not int or not -len(channels) <= stage < 0 for stage in stages):
            raise ValueError("attention stages must be negative encoder indices")
        if len(set(stages)) != len(stages):
            raise ValueError("attention stages must be unique")
        if mode == "prototype" and stages != (-1,):
            raise ValueError("attention stages apply only to memory mode")
        self.attention_stages = tuple(sorted(stages, reverse=True))
        if mode == "memory":
            # Preserve historical state-dict keys and initialization for stage -1.
            if -1 in self.attention_stages:
                _initialize_attention(self, channels[-1])
            self.additional_attention = nn.ModuleDict(
                {
                    f"minus{-stage}": StageCrossAttention(channels[stage])
                    for stage in self.attention_stages
                    if stage != -1
                }
            )

    def encode_features(self, skips, masks):
        batch, count = masks.shape[:2]
        raw_masks = masks.flatten(0, 1).float()
        foreground, known = raw_masks.clamp_min(0), (raw_masks >= 0).float()
        prototypes = []
        for features in skips:
            size = features.shape[-3:]
            occupancy = F.adaptive_avg_pool3d(foreground, size).reshape(batch, count, 1, -1)
            coverage = F.adaptive_avg_pool3d(known, size).reshape(batch, count, 1, -1)
            flattened = features.reshape(batch, count, features.shape[1], -1)
            pair = []
            for weights in (occupancy, coverage - occupancy):
                prototype = (flattened * weights).sum((1, 3)) / weights.sum((1, 3)).clamp_min(1e-6)
                pair.append(F.normalize(prototype.float(), dim=1))
            prototypes.append(pair)
        tokens, stage_tokens = None, {}
        if self.mode == "memory":
            for stage in self.attention_stages:
                block = self if stage == -1 else self.additional_attention[f"minus{-stage}"]
                memory = _memory_tokens(
                    block,
                    skips[stage],
                    foreground,
                    known,
                    batch,
                    count,
                    self.memory_grid,
                )
                if self.prototype_tokens:
                    pair = torch.stack(prototypes[stage], dim=1)
                    identity = torch.eye(2, device=pair.device, dtype=pair.dtype)
                    pair = block.memory_norm(pair + block.mask_embedding(identity)[None])
                    memory = torch.cat((pair, memory), dim=1)
                if stage == -1:
                    tokens = memory
                else:
                    stage_tokens[f"minus{-stage}"] = memory
        return {
            "prototypes": prototypes,
            "tokens": tokens,
            "stage_tokens": stage_tokens,
        }

    def condition_features(self, skips, context):
        skips = list(skips)
        if context["prototypes"][0][0].shape[0] != skips[0].shape[0]:
            raise ValueError("query and context batch sizes differ")
        for i, (features, prototypes) in enumerate(zip(skips, context["prototypes"])):
            unit = F.normalize(features.float(), dim=1)
            similarities = [(unit * p[:, :, None, None, None]).sum(1, keepdim=True) for p in prototypes]
            skips[i] = features + self.fusions[i](torch.cat([features, *similarities], 1).to(features.dtype))
        if self.mode == "memory":
            for stage in self.attention_stages:
                if stage == -1:
                    skips[stage] = _condition_features(self, skips[stage], context["tokens"])
                else:
                    name = f"minus{-stage}"
                    skips[stage] = _condition_features(
                        self.additional_attention[name],
                        skips[stage],
                        context["stage_tokens"][name],
                    )
        return skips


class InContextUNet3D(PrototypeMemory):
    """Single-volume interaction and K-volume context share one encoder/decoder.

    Query: BCDHW; supports: BKCDHW; support masks: BK1DHW; prompts: B7DHW.
    Support masks may be binary annotations or predictions. -1 marks unknown voxels.
    No class ID or hidden query label enters this module. Independent supports form
    a set; context order does not change a prediction with fixed support masks.
    """

    def __init__(
        self,
        width=16,
        mode="memory",
        memory_grid=4,
        backbone=None,
        image_channels=1,
        interaction_channels=7,
        attention_stages=(-1,),
        prototype_tokens=False,
    ):
        if mode not in {"memory", "prototype"} or width < 1 or memory_grid < 1:
            raise ValueError("invalid mode, width, or memory grid")
        if interaction_channels != 7:
            raise ValueError("the prototype backend requires nnInteractive's canonical seven prompt channels")
        backbone = small_backbone(width) if backbone is None else backbone
        super().__init__(backbone.encoder.output_channels, mode, memory_grid, attention_stages, prototype_tokens)
        self.backbone = backbone
        self.image_channels, self.interaction_channels = (
            image_channels,
            interaction_channels,
        )

    def _input(self, image: Tensor, prompts: Tensor | None = None) -> Tensor:
        if image.ndim != 5 or image.shape[1] != self.image_channels:
            raise ValueError("image must be BCDHW with the configured image channel count")
        if prompts is None:
            prompts = image.new_zeros(image.shape[0], self.interaction_channels, *image.shape[2:])
        if prompts.shape != (
            image.shape[0],
            self.interaction_channels,
            *image.shape[2:],
        ):
            raise ValueError("prompt tensor must match image batch and spatial shape")
        return torch.cat([image, prompts.to(image.dtype)], dim=1)

    def interactive_logits(self, image: Tensor, prompts: Tensor | None = None) -> Tensor:
        """Original single-image path; no memory modules are invoked."""
        logits = self.backbone(self._input(image, prompts))
        return logits[:, 1:2] - logits[:, :1]

    def encode_context(self, images: Tensor, masks: Tensor) -> dict:
        """Encode supports once; inference can reuse these compact tensors for all crops."""
        if images.ndim != 6 or masks.ndim != 6:
            raise ValueError("support images/masks must be BKCDHW")
        batch, count, channels, *spatial = images.shape
        if count < 1 or channels != self.image_channels or masks.shape != (batch, count, 1, *spatial):
            raise ValueError("incompatible support/mask shapes or empty context")
        skips = self.backbone.encoder(self._input(images.flatten(0, 1)))
        return self.encode_features(skips, masks)

    def predict_with_context(self, query: Tensor, prompts: Tensor | None, context: dict) -> Tensor:
        skips = self.backbone.encoder(self._input(query, prompts))
        logits = self.backbone.decoder(self.condition_features(skips, context))
        return logits[:, 1:2] - logits[:, :1]

    def forward(self, query, support_images=None, support_masks=None, query_prompts=None):
        if support_images is None:
            if support_masks is not None:
                raise ValueError("support masks require support images")
            return self.interactive_logits(query, query_prompts)
        if support_masks is None:
            raise ValueError("support images require explicit or predicted support masks")
        return self.predict_with_context(query, query_prompts, self.encode_context(support_images, support_masks))

    def unify_context(self, images: Tensor, prompts: Tensor, supplied_masks: Tensor, dense: Tensor):
        """All support prompt types become predicted masks before entering memory.

        Dense annotated masks may bypass prediction. The hidden target of a sparse
        support must not be supplied as an annotation; it belongs only in the loss.
        """
        batch, count = images.shape[:2]
        if dense.shape != (batch, count) or dense.dtype != torch.bool:
            raise ValueError("dense must be a boolean BK annotation-presence tensor")
        logits = self.interactive_logits(images.flatten(0, 1), prompts.flatten(0, 1))
        logits = logits.reshape(batch, count, 1, *images.shape[-3:])
        memory_masks = torch.where(
            dense[:, :, None, None, None, None],
            supplied_masks,
            logits.float().sigmoid(),
        )
        return memory_masks, logits
