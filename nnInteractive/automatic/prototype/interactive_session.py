"""Add 3D context to the existing session without replacing its interaction engine."""

from __future__ import annotations

import numpy as np
import torch
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from torch import nn
from torch.nn import functional as F

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnInteractive.interaction.point import PointInteraction_stub

from .checkpoint import build_model, compact_configuration, load_model


class ContextNetworkAdapter(nn.Module):
    """Preserve the original network(input_image_and_prompts) -> two logits contract."""

    def __init__(self, model, native_refiner=None):
        super().__init__()
        self.model = model
        self.native_refiner = native_refiner
        self.context = None

    def forward(self, inputs):
        if self.native_refiner is not None:
            # Global learned context enters the native engine through previous-mask feedback.
            return self.native_refiner(inputs)
        if self.context is None:
            # Exact original behavior, including the original pair of decoder logits.
            return self.model.backbone(inputs)
        image = inputs[:, : self.model.image_channels]
        prompts = inputs[:, self.model.image_channels :]
        logits = self.model.predict_with_context(image, prompts, self.context)
        return torch.cat([-0.5 * logits, 0.5 * logits], 1)


class TemporalInferenceSession(nnInteractiveInferenceSession):
    """All nnInteractive prompt/undo/AutoZoom/refinement methods are inherited.

    New methods only manage the support-volume dimension. Supports survive set_image
    and reset_interactions; clear_context explicitly starts a new context task.
    Each session owns its context, so it cannot leak into other inference sessions.
    """

    def initialize_from_icl_checkpoint(self, checkpoint: str):
        model, state = load_model(checkpoint, self.device)
        self.context_shape = tuple(state["input_shape"])
        if min(self.context_shape) <= 48:
            raise ValueError(
                "native nnInteractive refinement requires patch dimensions above 48; use a grid of at least 64"
            )
        self.context_images, self.context_masks = [], []
        native_refiner = None
        if "native_refiner" in state:
            native_state = state["native_refiner"]
            if native_state["policy"] != "frozen_native_refinement":
                raise ValueError("unsupported native refinement policy")
            native_refiner = build_model(state["model_config"]).backbone
            native_refiner.load_state_dict(native_state["model"], strict=True)
            native_refiner.eval().requires_grad_(False)
        self.context_adapter = ContextNetworkAdapter(model.eval(), native_refiner).to(
            self.device
        )
        capability = self._legacy_default_capability()
        capability["interaction_channels"] = (
            6  # Metadata excludes the previous-mask channel.
        )
        self._apply_capability(capability)
        self.interaction_decay = 0.98
        self.preferred_scribble_thickness = [2, 2, 2]
        self.point_interaction = PointInteraction_stub(2, use_distance_transform=True)
        self.pad_mode_data = "constant"
        self.license = "Research checkpoint; training dataset CC-BY-NC-SA-4.0"
        plans = PlansManager({"label_manager": "LabelManager"})
        configuration = ConfigurationManager(
            compact_configuration(state["model_config"]["width"], self.context_shape)
        )
        dataset = {
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "target": 1},
        }
        if state["model_config"]["backbone"] is not None:
            metadata = state["model_config"]["backbone"]["session"]
            self._apply_capability(metadata["capability"])
            self.interaction_decay = metadata["interaction_decay"]
            self.preferred_scribble_thickness = metadata["preferred_scribble_thickness"]
            self.point_interaction = PointInteraction_stub(
                metadata["point_radius"], metadata["point_use_distance_transform"]
            )
            self.pad_mode_data, self.license = (
                metadata["pad_mode_data"],
                metadata["license"],
            )
            plans = PlansManager(metadata["plans"])
            configuration = ConfigurationManager(metadata["configuration"])
            dataset = metadata["dataset"]
        self.manual_initialization(
            self.context_adapter, plans, configuration, dataset, "ICL3D"
        )
        self.network.eval()

    @torch.inference_mode()
    def add_current_as_context(self):
        """Store the current volume and its completed mask, irrespective of prompt type.

        Call after any inherited add_*_interaction, or after supplying a full mask
        with add_initial_seg_interaction. Only the completed prediction/annotation
        enters memory; unexposed support ground truth is never read.
        """
        self._finish_preprocessing_and_initialize_interactions()
        if self.target_buffer is None or self.preprocessed_image is None:
            raise RuntimeError(
                "set an image and target buffer, then annotate or predict the support"
            )
        if tuple(self.target_buffer.shape) != tuple(self.original_image_shape[1:]):
            raise ValueError("support target buffer must match the original image grid")
        mask = torch.as_tensor(self.target_buffer, device="cpu").float()
        if not ((mask == 0) | (mask == 1)).all() or not mask.any():
            raise ValueError(
                "context requires a nonempty binary completed support segmentation"
            )
        image = F.interpolate(
            self.preprocessed_image[None].float(),
            self.context_shape,
            mode="trilinear",
            align_corners=False,
        )[0]
        mask = F.interpolate(
            mask[None, None], self.context_shape, mode="nearest-exact"
        )[0]
        if not mask.any():
            raise ValueError(
                "support target vanished on the context grid; use a finer grid"
            )
        candidate_images = [*self.context_images, image]
        candidate_masks = [*self.context_masks, mask]
        images = torch.stack(candidate_images)[None].to(self.device)
        masks = torch.stack(candidate_masks)[None].to(self.device)
        # Match the inherited session's CUDA autocast precision.
        with torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            context = self.context_adapter.model.encode_context(images, masks)
        # Commit the context only after encoding succeeds, preserving the old state on error.
        self.context_images, self.context_masks = candidate_images, candidate_masks
        self.context_adapter.context = context

    def clear_context(self):
        self.context_images.clear()
        self.context_masks.clear()
        self.context_adapter.context = None

    @torch.inference_mode()
    def predict_from_context(self, refine: bool = True):
        """Unprompted coarse query initialization, then the original refinement engine.

        Existing query prompts should be refined through add_*_interaction instead;
        this entry point is only for a fresh query with no local interactions.
        """
        self._finish_preprocessing_and_initialize_interactions()
        if self.context_adapter.context is None:
            raise RuntimeError("add a support volume before predicting from context")
        if self._dirty_channels:
            raise RuntimeError(
                "query already has interactions; use add_*_interaction to refine them"
            )
        if self.target_buffer is None:
            raise RuntimeError("set_target_buffer before predicting")
        image = F.interpolate(
            self.preprocessed_image[None].float(),
            self.context_shape,
            mode="trilinear",
            align_corners=False,
        )
        with torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            logits = self.context_adapter.model.predict_with_context(
                image.to(self.device), None, self.context_adapter.context
            )
        probability = F.interpolate(
            logits.float().sigmoid(),
            self.original_image_shape[1:],
            mode="trilinear",
            align_corners=False,
        )
        initial = (probability[0, 0] > 0.5).to(device="cpu", dtype=torch.uint8).numpy()
        # The existing initial-seg path preserves its documented reset and undo semantics.
        return self.add_initial_seg_interaction(
            initial, run_prediction=refine and bool(np.any(initial))
        )
