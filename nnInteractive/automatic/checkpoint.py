from pathlib import Path

import torch

from .backbone import reconstruct_original_architecture
from .config import ModelConfig
from .model import PromptEncodedICL3D


def load_model(path: str | Path, device: str | torch.device = "cpu"):
    """Return ``(model, state)`` from a trusted local training/inference checkpoint.

    Training checkpoints include Python/NumPy RNG state. Only load checkpoints
    from trusted sources; the original format identifiers remain compatible.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") == "prompt_encoded_icl3d_v2":
        model = PromptEncodedICL3D(
            reconstruct_original_architecture(state["source"]),
            # Missing feature flags retain their original checkpoint behavior.
            ModelConfig(
                **{
                    "support_mask_context": False,
                    "support_mask_attention": False,
                    "query_deep_supervision": False,
                    "attention_depth": 1,
                    "require_nonempty_support_masks": False,
                    "convert_dense_support_masks": True,
                    **state["model_config"],
                }
            ),
            state["source"]["channel_mapping"]["prev_seg"],
        )
    elif state.get("format_version") == 1:
        from .prototype.checkpoint import build_model

        model = build_model(state["model_config"])
    else:
        raise ValueError("unsupported automatic-segmentation checkpoint format")
    model.load_state_dict(state["model"], strict=True)
    return model.to(device), state
