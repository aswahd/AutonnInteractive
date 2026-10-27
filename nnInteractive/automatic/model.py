"""Direct prompted support features, with optional interleaved query cross-attention."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .mask_encoder import MaskEncoder3D, support_position_encoding
from .prototype.model import PrototypeMemory
from .support_masks import FrozenSupportSegmenter


def apply_rope_3d(x, grid):
    """Rotate B,H,N,D projected tokens; repeat local DHW coordinates per support."""
    pairs = x.shape[-1] // 2
    sizes = (pairs // 3, pairs // 3, pairs - 2 * (pairs // 3))
    coordinates = torch.meshgrid(*(torch.arange(n, device=x.device) for n in grid), indexing="ij")
    angles = torch.cat(
        [
            coordinate.flatten()[:, None].float() * 10000.0 ** (-torch.arange(size, device=x.device).float() / size)
            for coordinate, size in zip(coordinates, sizes)
        ],
        dim=-1,
    )
    if x.shape[-2] % angles.shape[0]:
        raise ValueError("RoPE tokens must contain complete spatial grids")
    angles = angles.repeat(x.shape[-2] // angles.shape[0], 1)
    real, imaginary = x.float().unflatten(-1, (pairs, 2)).unbind(-1)
    cosine, sine = angles.cos(), angles.sin()
    return torch.stack((real * cosine - imaginary * sine, real * sine + imaginary * cosine), -1).flatten(-2).to(x.dtype)


class ContextBlock(nn.Module):
    """Condition B,C,D,H,W queries on support-major B,K*Ns,C memory tokens."""

    def __init__(self, channels, config, memory_grid=None):
        super().__init__()
        self.memory_grid = memory_grid
        self.rope_3d = config.rope_3d
        self.query_position_encoding = config.query_position_encoding
        self.support_position_encoding = config.support_position_encoding
        self.hierarchical = config.cross_attention_type == "hierarchical"
        dim = config.attention_dim
        self.query_in = nn.Linear(channels, dim)
        self.support_in = nn.Linear(channels, dim)
        self.query_norm = nn.LayerNorm(dim)
        self.support_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, config.attention_heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.ffn_hidden),
            nn.GELU(),
            nn.Linear(config.ffn_hidden, dim),
        )
        self.output = nn.Linear(dim, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.support_scorer = None
        if self.hierarchical and config.support_aggregation == "learned":
            self.support_scorer = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
            nn.init.zeros_(self.support_scorer[-1].weight)
            nn.init.zeros_(self.support_scorer[-1].bias)

    def forward(self, query, memory, token_active=None, token_foreground=None):
        if self.hierarchical and token_active is not None:
            # Exclude padded nonfinite values before projection as well as from attention below.
            memory = memory.masked_fill(~token_active[..., None], 0)
        raw = query.flatten(2).transpose(1, 2)
        u, support = self.query_in(raw), self.support_norm(self.support_in(memory))
        queries, keys = self.query_norm(u), support
        if self.query_position_encoding:
            positioned = (query + support_position_encoding(query)).flatten(2).transpose(1, 2)
            queries = self.query_norm(self.query_in(positioned))
        if self.support_position_encoding:
            grid = self.memory_grid or query.shape[2:]
            positions = support_position_encoding(query.new_empty(1, query.shape[1], *grid))
            positions = positions.flatten(2).transpose(1, 2)
            positions = positions.repeat(1, memory.shape[1] // positions.shape[1], 1)
            keys = self.support_norm(self.support_in(memory + positions))
        padding = None if token_active is None else ~token_active
        if token_foreground is not None:
            active = torch.ones_like(token_foreground) if token_active is None else token_active
            allowed = token_foreground & active
            # Mask2Former fallback for an empty foreground; never restore dropped supports.
            allowed = torch.where(allowed.any(1, keepdim=True), allowed, active)
            # Float key padding is added to QK^T before softmax, broadcast over queries and heads.
            padding = queries.new_zeros(allowed.shape).masked_fill(~allowed, -torch.inf)
        attention = (
            self._hierarchical_attention(queries, keys, support, query.shape[2:], padding)
            if self.hierarchical
            else self._spatial_attention(queries, keys, support, query.shape[2:], padding)
        )
        h = u + attention
        j = h + self.ffn(self.ffn_norm(h))
        updated = raw + self.output(j - u)
        return updated.transpose(1, 2).reshape_as(query)

    def _spatial_attention(self, queries, keys, support, query_grid, padding):
        """Shared projections/backend for flat memory or independent query-support pairs."""
        if self.rope_3d:
            mha = self.attention
            q, k, v = (
                F.linear(tokens, weight, bias).unflatten(-1, (mha.num_heads, mha.head_dim)).transpose(1, 2)
                for tokens, weight, bias in zip(
                    (queries, keys, support), mha.in_proj_weight.chunk(3), mha.in_proj_bias.chunk(3)
                )
            )
            q = apply_rope_3d(q, query_grid)
            k = apply_rope_3d(k, self.memory_grid or query_grid)
            if padding is not None:
                if padding.dtype == torch.bool:
                    padding = q.new_zeros(padding.shape).masked_fill(padding, -torch.inf)
                padding = padding[:, None, None, :].to(q.dtype)
            attention = F.scaled_dot_product_attention(q, k, v, attn_mask=padding)
            attention = mha.out_proj(attention.transpose(1, 2).flatten(2))
        else:
            attention = self.attention(queries, keys, support, key_padding_mask=padding, need_weights=False)[0]
        return attention

    def _hierarchical_attention(self, queries, keys, support, query_grid, padding):
        """Batch independent spatial attention, then reduce B,K,Nq,C responses over K."""
        batch, tokens, _ = keys.shape
        count, remainder = divmod(tokens, math.prod(self.memory_grid or query_grid))
        if not count or remainder:
            raise ValueError("hierarchical memory must contain complete spatial grids per support")
        allowed = torch.ones((batch, tokens), device=keys.device, dtype=torch.bool)
        if padding is not None:
            allowed = ~padding if padding.dtype == torch.bool else padding.isfinite()
        allowed = allowed.unflatten(1, (count, -1))
        valid = allowed.any(-1)
        if not valid.any(1).all():
            raise ValueError("hierarchical attention requires at least one valid support per episode")
        # Pack valid B*K pairs: one shared kernel, with no cross-support spatial softmax.
        rows, slots = valid.nonzero(as_tuple=True)
        paired = self._spatial_attention(
            queries[rows],
            keys.unflatten(1, (count, -1))[valid],
            support.unflatten(1, (count, -1))[valid],
            query_grid,
            ~allowed[valid],
        )
        responses = paired.new_zeros((batch, count, *paired.shape[1:])).index_put((rows, slots), paired)
        return self._aggregate_supports(queries, responses, valid)[0]

    def _aggregate_supports(self, queries, responses, valid):
        # A shared scorer and softmax over K make the weighted sum invariant to support order.
        # There are no support-index embeddings or support-specific projection parameters.
        scores = responses.new_zeros((*responses.shape[:-1], 1), dtype=torch.float32)
        if self.support_scorer is not None:
            repeated = queries[:, None].expand_as(responses)
            scores = self.support_scorer(torch.cat((repeated, responses), -1)).float()
        weights = scores.masked_fill(~valid[:, :, None, None], -torch.inf).softmax(1).to(responses.dtype)
        return (weights * responses).sum(1), weights


class ContextStack(nn.ModuleList):
    """Sequential residual cross-attention blocks sharing the same support memory."""

    def forward(self, query, memory, token_active=None, token_foreground=None):
        for block in self:
            query = block(query, memory, token_active, token_foreground)
        return query


class PromptEncodedICL3D(nn.Module):
    """Shared interactive support and in-context query segmentation.

    forward returns original two-channel logits for all K supports and the query.
    predict_query skips support decoding when the memory strategy does not need masks.
    No labels, class IDs, query ROI, or persistent feature cache enter the model.
    """

    def __init__(self, backbone, config: ModelConfig, prev_seg_channel=0):
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = backbone
        encoder, decoder = backbone.encoder, backbone.decoder
        if tuple(encoder.output_channels) != tuple(config.expected_channels):
            raise ValueError("checkpoint encoder channels differ from the declared architecture")
        expected_strides = [[1, 1, 1]] + [[2, 2, 2]] * 5
        if [list(s) for s in encoder.strides] != expected_strides:
            raise ValueError("checkpoint encoder strides differ from the declared architecture")
        if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in backbone.modules()):
            raise ValueError("batch-coupled normalization violates independent episode isolation")
        if decoder.num_classes != 2 or decoder.seg_layers[-1].out_channels != 2:
            raise ValueError("checkpoint must have the original binary two-logit head")
        first_conv = next(m for m in encoder.modules() if isinstance(m, nn.Conv3d))
        if first_conv.in_channels != 1 + config.prompt_channels:
            raise ValueError("checkpoint image/prompt input contract differs")
        decoder.deep_supervision = False
        # Strict loading happens before removing unused auxiliary head parameters.
        if not config.query_deep_supervision:
            for index in range(len(decoder.seg_layers) - 1):
                decoder.seg_layers[index] = nn.Identity()
        self.context = (
            PrototypeMemory(
                config.expected_channels,
                memory_grid=config.prototype_memory_grid,
                attention_stages=tuple(stage - len(config.expected_channels) for stage in config.stages),
                prototype_tokens=True,
            )
            if config.prototype_memory_grid is not None
            else nn.ModuleDict(
                {
                    str(s): (
                        ContextBlock(config.expected_channels[s], config, grid)
                        if config.attention_depth == 1
                        else ContextStack(
                            ContextBlock(config.expected_channels[s], config, grid)
                            for _ in range(config.attention_depth)
                        )
                    )
                    for s, grid in zip(config.stages, config.memory_grids)
                }
            )
        )
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.support_segmenter = (
            FrozenSupportSegmenter(
                backbone,
                prev_seg_channel,
                config.support_chunk_size,
                config.require_nonempty_support_masks,
                config.convert_dense_support_masks,
            )
            if config.support_mask_context
            else None
        )
        self.prev_seg_channel = prev_seg_channel
        self.mask_encoder = (
            MaskEncoder3D(config.expected_channels, config.mask_encoder_stages)
            if config.mask_encoder_stages is not None
            else None
        )
        if self.mask_encoder is not None:
            encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.mask_encoder is not None:
            self.backbone.encoder.eval()
        return self

    def _call(self, module, *args):
        if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _encode(self, x, stop=None):
        encoder = self.backbone.encoder
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.mask_encoder is None):
            if encoder.stem is not None:
                x = self._call(encoder.stem, x)
            features = []
            for stage in encoder.stages[:stop]:
                x = self._call(stage, x)
                features.append(x)
        return features

    def _encode_support(self, images, prompts):
        image_prompts = torch.zeros_like(prompts) if self.mask_encoder is not None else prompts
        features = self._encode(torch.cat((images, image_prompts.to(images.dtype)), 1))
        if self.mask_encoder is not None:
            masks = prompts[:, self.prev_seg_channel : self.prev_seg_channel + 1].to(images.dtype)
            for stage, encoded_mask in self._call(self.mask_encoder, masks).items():
                features[stage] = features[stage] + encoded_mask
        return features

    def _decode(self, skips, auxiliary=None):
        decoder = self.backbone.decoder
        x = skips[-1]
        for index in range(len(decoder.stages)):

            def step(low, skip, i=index):
                up = decoder.transpconvs[i](low)
                return decoder.stages[i](torch.cat((up, skip), dim=1))

            x = self._call(step, x, skips[-index - 2])
            if auxiliary is not None and index < len(decoder.seg_layers) - 1:
                auxiliary.append(self._call(decoder.seg_layers[index], x))
        logits = self._call(decoder.seg_layers[-1], x)
        return logits

    def _support_memories(self, support_images, support_prompts, decode_supports):
        batch, count = support_images.shape[:2]
        prototypes = self.config.prototype_memory_grid is not None
        images, prompts = support_images.flatten(0, 1), support_prompts.flatten(0, 1)
        memories = [[] for _ in self.config.stages]
        foreground = [[] for _ in self.config.stages]
        predictions = []
        chunk_size = self.config.support_encoder_chunk_size or self.config.support_chunk_size
        for start in range(0, batch * count, chunk_size):
            end = start + chunk_size
            encoded = self._encode_support(images[start:end], prompts[start:end])
            if decode_supports or prototypes:
                predictions.append(self._decode(encoded))
            if prototypes:
                continue
            for index, (features, grid) in enumerate(
                zip([encoded[s] for s in self.config.stages], self.config.memory_grids)
            ):
                if grid is not None:
                    if any(g > n for g, n in zip(grid, features.shape[2:])):
                        raise ValueError("support pooling cannot upsample a feature grid")
                    features = F.adaptive_avg_pool3d(features, grid)
                memories[index].append(features.flatten(2).transpose(1, 2))
                if self.config.support_mask_attention:
                    masks = prompts[start:end, self.prev_seg_channel : self.prev_seg_channel + 1]
                    # Preserve tiny foregrounds, including after optional memory pooling.
                    foreground[index].append(F.adaptive_max_pool3d(masks, features.shape[2:]).flatten(1) > 0.5)
        if prototypes:
            logits = torch.cat(predictions, dim=0).unflatten(0, (batch, count))
            return (support_images, support_prompts, logits), logits if decode_supports else None
        result = []
        for index, parts in enumerate(memories):
            features = torch.cat(parts, dim=0)
            tokens = features.reshape(batch, count * features.shape[1], features.shape[2])
            result.append(
                (tokens, torch.cat(foreground[index], 0).reshape(batch, -1))
                if self.config.support_mask_attention
                else tokens
            )
        logits = torch.cat(predictions, dim=0).unflatten(0, (batch, count)) if decode_supports else None
        return result, logits

    def interactive(self, images, prompts):
        """Original image + local prompts pathway, using the same encoder and decoder."""
        shape = tuple(self.config.patch_size)
        if images.ndim != 5 or tuple(images.shape[1:]) != (1, *shape):
            raise ValueError("images must be B1DHW at the configured patch size")
        if prompts.shape != (images.shape[0], self.config.prompt_channels, *shape):
            raise ValueError("prompts must be B7DHW matching images")
        if self.support_segmenter is not None:
            prompts = self.support_segmenter(images, prompts)
        return self._decode(self._encode_support(images, prompts))

    def forward(self, query_images, support_images, support_prompts, support_active_mask=None, *, decode_supports=True):
        return self._predict(
            query_images,
            support_images,
            support_prompts,
            decode_supports,
            support_active_mask,
            query_auxiliary=self.training and self.config.query_deep_supervision,
        )

    def predict_query(self, query_images, support_images, support_prompts, support_active_mask=None):
        """Two-channel query logits; re-encode raw supports for each invocation."""
        return self._predict(query_images, support_images, support_prompts, False, support_active_mask)["query_logits"]

    @staticmethod
    def _validate_mask(mask, support_images):
        if mask is None:
            return
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("support_active_mask must be a boolean tensor")
        if mask.shape != support_images.shape[:2]:
            raise ValueError("support_active_mask must have shape [B, K]")
        if mask.device != support_images.device:
            raise ValueError("support_active_mask must be on the support device")
        if not mask.any(dim=1).all():
            raise ValueError("every episode must have at least one active support")

    def _attend_query(self, stage, query, memory, count, mask):
        memory, foreground = memory if self.config.support_mask_attention else (memory, None)
        token_active = None if mask is None else mask.repeat_interleave(memory.shape[1] // count, dim=1)
        return self._call(self.context[str(stage)], query, memory, token_active, foreground)

    def _condition_query(self, skips, memories, count, mask, auxiliary=None):
        if self.config.prototype_memory_grid is not None:
            return self._prototype_query(skips, memories, mask, auxiliary)
        conditioned = list(skips)
        if self.config.interleave_query_context:
            memories = dict(zip(self.config.stages, memories))
            for stage in range(len(conditioned), len(self.backbone.encoder.stages)):
                query = self._call(self.backbone.encoder.stages[stage], conditioned[-1])
                if stage in memories:
                    query = self._attend_query(stage, query, memories[stage], count, mask)
                conditioned.append(query)
        else:
            for stage, memory in zip(self.config.stages, memories):
                conditioned[stage] = self._attend_query(stage, skips[stage], memory, count, mask)
        return self._decode(conditioned, auxiliary)

    def _prototype_memory(self, support, masks):
        zeros = support.new_zeros(support.shape[0], self.config.prompt_channels, *support.shape[2:])
        return self.context.encode_features(self.backbone.encoder(torch.cat((support, zeros), 1)), masks)

    def _prototype_query(self, skips, memories, active, auxiliary=None):
        images, prompts, logits = memories
        supplied = prompts[:, :, :1]
        dense = supplied.flatten(2).ne(0).any(-1)[:, :, None, None, None, None]
        masks = torch.where(dense, supplied, (logits[:, :, 1:2] - logits[:, :, :1]).float().sigmoid())
        conditioned = []
        for index in range(images.shape[0]):
            selected = slice(None) if active is None else active[index]
            support = images[index, selected]
            # Retain compact memory across episodes, recomputing support features in backward.
            memory = self._call(self._prototype_memory, support, masks[index : index + 1, selected])
            conditioned.append(
                self._call(self.context.condition_features, [s[index : index + 1] for s in skips], memory)
            )
        return self._decode([torch.cat(parts, 0) for parts in zip(*conditioned)], auxiliary)

    def _predict(
        self,
        query_images,
        support_images,
        support_prompts,
        decode_supports,
        support_active_mask,
        query_auxiliary=False,
    ):
        self._validate_mask(support_active_mask, support_images)
        skips, memories, support_logits = self._encode_inputs(
            query_images, support_images, support_prompts, decode_supports
        )
        auxiliary = [] if query_auxiliary else None
        query_logits = self._condition_query(skips, memories, support_images.shape[1], support_active_mask, auxiliary)
        result = {"support_logits": support_logits, "query_logits": query_logits}
        if auxiliary is not None:
            result["query_aux_logits"] = auxiliary[::-1]  # Finest to coarsest, excluding the final head.
        return result

    def iter_query_subsets(self, query_images, support_images, support_prompts, masks):
        """Compare subsets of a fixed bank without persistent cross-episode caches.

        Direct memories are reused; prototype memories are rebuilt for each subset.
        Interleaved query stages rerun per subset; only the unconditioned prefix is shared.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("subset comparison requires eval mode and no gradients")
        masks = tuple(masks)
        if not masks:
            raise ValueError("supply at least one subset mask")
        for mask in masks:
            self._validate_mask(mask, support_images)
        skips, memories, _ = self._encode_inputs(query_images, support_images, support_prompts, False)
        for mask in masks:
            yield self._condition_query(skips, memories, support_images.shape[1], mask)

    def _encode_inputs(self, query_images, support_images, support_prompts, decode_supports):
        shape = tuple(self.config.patch_size)
        if query_images.ndim != 5 or tuple(query_images.shape[1:]) != (1, *shape):
            raise ValueError("query_images must be B1DHW at the configured patch size")
        if support_images.ndim != 6 or tuple(support_images.shape[2:]) != (1, *shape):
            raise ValueError("support_images must be BK1DHW at the configured patch size")
        batch, count = support_images.shape[:2]
        if batch != query_images.shape[0] or count < 1:
            raise ValueError("support episode count must match query batch, with K >= 1")
        if support_prompts.shape != (batch, count, self.config.prompt_channels, *shape):
            raise ValueError("support_prompts must be BK7DHW matching support images")
        if self.support_segmenter is not None:
            support_prompts = self.support_segmenter(support_images, support_prompts)
        zeros = query_images.new_zeros(batch, self.config.prompt_channels, *shape)
        stop = min(self.config.stages) if self.config.interleave_query_context else None
        skips = self._encode(torch.cat((query_images, zeros), dim=1), stop)
        memories, support_logits = self._support_memories(support_images, support_prompts, decode_supports)
        return skips, memories, support_logits
