"""Public automatic inference on preprocessed CT, with optional frozen refinement."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .autozoom import NativeAutoZoomRefiner
from .checkpoint import load_model
from .data import foreground_window
from .distillation import FrozenTeacher, TeacherConfig
from .infer import VolumePrediction, predict_volume
from .spatial import resize_volume, support_at_resolution
from .support_dropout import preserve_foreground_supports


class AutomaticSegmentationSession:
    """A trained prompt-conditioned student; query inputs contain no target annotations.

    Supports are aligned 1DHW images and 7DHW prompts, preferably four examples.
    Dense targets receive scan-clipped 384-cubed crops, resized to the student grid.
    Queries are complete preprocessed 1DHW images. Outputs retain their input grid.
    The separate ``prototype`` backend supports historical dense-mask checkpoints.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        original_model_dir: str | Path | None = None,
        *,
        full_volume: bool = True,
    ):
        self.model, state = load_model(checkpoint, device)
        if state.get("format") != "prompt_encoded_icl3d_v2":
            raise ValueError("use automatic.prototype.session for a prototype checkpoint")
        self.model.eval()
        self.model.config.full_volume = full_volume
        self.source = dict(state["source"])
        if original_model_dir is not None:
            root = Path(original_model_dir).resolve()
            self.source.update(
                checkpoint_dir=str(root),
                checkpoint_file=str(root / f"fold_{self.source['fold']}" / self.source["checkpoint_name"]),
            )
        self.context = None
        self._refiner = None

    def set_context(
        self,
        images: torch.Tensor | np.ndarray | Sequence,
        prompts: torch.Tensor | np.ndarray | Sequence,
        *,
        support_context: int | None = 384,
    ) -> None:
        """Crop and resize dense supports; use lists for scans with different shapes.

        ``support_context=None`` accepts supports already prepared at the student grid.
        """
        size = tuple(self.model.config.patch_size)
        if len(images) < 1 or len(images) != len(prompts):
            raise ValueError("supply matching nonempty support image and prompt lists")
        if support_context is not None and (type(support_context) is not int or support_context < 1):
            raise ValueError("support_context must be a positive integer or None")
        prepared = []
        channel = self.source["channel_mapping"]["prev_seg"]
        for image, prompt in zip(images, prompts):
            image, prompt = [
                torch.as_tensor(x, dtype=torch.float32, device="cpu").detach().clone() for x in (image, prompt)
            ]
            if image.ndim != 4 or image.shape[0] != 1 or min(image.shape[1:]) < 1:
                raise ValueError("each support image must be 1DHW")
            if prompt.shape != (self.model.config.prompt_channels, *image.shape[1:]):
                raise ValueError("support prompts must be K7DHW and spatially aligned with the images")
            if not all(torch.isfinite(x).all() for x in (image, prompt)):
                raise ValueError("support images and prompts must be finite")
            mask = prompt[channel : channel + 1] > 0
            if support_context is None and tuple(image.shape[1:]) != size:
                raise ValueError("prepared supports must match the checkpoint's student input size")
            if support_context is not None and mask.any():
                start, shape = foreground_window(mask[0].numpy(), (support_context,) * 3)
                selection = (slice(None), *[slice(a, a + n) for a, n in zip(start, shape)])
                image, prompt, mask = [x[selection] for x in (image, prompt, mask)]
            image, mask, _, _ = support_at_resolution(image, mask, torch.ones_like(mask), size)
            prompt = resize_volume(prompt[None], size, "nearest-exact")[0]
            prompt[channel] = mask[0]
            prepared.append((image, prompt))
        images, prompts = [torch.stack(items) for items in zip(*prepared)]
        preserve_foreground_supports(
            torch.ones(1, len(images), dtype=torch.bool), prompts[None], self.source["channel_mapping"]
        )
        self.context = images, prompts

    @torch.inference_mode()
    def predict(
        self, query: torch.Tensor | np.ndarray, refinement: str = "autozoom", **window_options
    ) -> VolumePrediction:
        """Predict with ``none``, ``resized_context``, or native ``autozoom`` refinement."""
        if self.context is None:
            raise RuntimeError("set_context must supply prompted supports before prediction")
        device = next(self.model.parameters()).device
        if refinement == "autozoom":
            with NativeAutoZoomRefiner(self.source, device) as refiner:
                return predict_volume(
                    self.model, query, *self.context, mode="autozoom", refiner=refiner, **window_options
                )
        if refinement == "resized_context" and self._refiner is None:
            config = TeacherConfig(
                enabled=True,
                checkpoint=self.source["checkpoint_file"],
                checkpoint_sha256=self.source["checkpoint_sha256"],
                soft_prior=True,
            )
            self._refiner = FrozenTeacher(config, self.source, device, self.model.config)
        if refinement not in {"none", "resized_context"}:
            raise ValueError("refinement must be none, resized_context, or autozoom")
        return predict_volume(
            self.model,
            query,
            *self.context,
            mode="sliding_window" if refinement == "none" else refinement,
            refiner=self._refiner if refinement == "resized_context" else None,
            **window_options,
        )
