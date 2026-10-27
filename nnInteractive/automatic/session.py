"""Public automatic inference on preprocessed CT, with optional frozen refinement."""

from pathlib import Path

import numpy as np
import torch

from .autozoom import NativeAutoZoomRefiner
from .checkpoint import load_model
from .distillation import FrozenTeacher, TeacherConfig
from .infer import VolumePrediction, predict_volume
from .support_dropout import preserve_foreground_supports


class AutomaticSegmentationSession:
    """A trained prompt-conditioned student; query inputs contain no target annotations.

    Supports are K1DHW images and K7DHW prompts at the checkpoint's student grid.
    Queries are complete preprocessed 1DHW images. Outputs retain their input grid.
    The separate ``prototype`` backend supports historical dense-mask checkpoints.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cuda", original_model_dir: str | Path | None = None):
        self.model, state = load_model(checkpoint, device)
        if state.get("format") != "prompt_encoded_icl3d_v2":
            raise ValueError("use automatic.prototype.session for a prototype checkpoint")
        self.model.eval()
        self.source = dict(state["source"])
        if original_model_dir is not None:
            root = Path(original_model_dir).resolve()
            self.source.update(
                checkpoint_dir=str(root),
                checkpoint_file=str(root / f"fold_{self.source['fold']}" / self.source["checkpoint_name"]),
            )
        self.context = None
        self._refiner = None

    def set_context(self, images: torch.Tensor | np.ndarray, prompts: torch.Tensor | np.ndarray) -> None:
        images, prompts = [
            torch.as_tensor(x, dtype=torch.float32, device="cpu").detach().clone() for x in (images, prompts)
        ]
        size = tuple(self.model.config.patch_size)
        if images.ndim != 5 or images.shape[0] < 1 or tuple(images.shape[1:]) != (1, *size):
            raise ValueError("support images must be K1DHW at the checkpoint's student input size")
        if prompts.shape != (len(images), self.model.config.prompt_channels, *size):
            raise ValueError("support prompts must be K7DHW and spatially aligned with the images")
        if not all(torch.isfinite(x).all() for x in (images, prompts)):
            raise ValueError("support images and prompts must be finite")
        preserve_foreground_supports(
            torch.ones(1, len(images), dtype=torch.bool), prompts[None], self.source["channel_mapping"]
        )
        self.context = images, prompts

    @torch.inference_mode()
    def predict(self, query: torch.Tensor | np.ndarray, refinement: str = "none", **window_options) -> VolumePrediction:
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
