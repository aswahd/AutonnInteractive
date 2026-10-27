"""Self-contained ICL checkpoints and explicit nnInteractive backbone initialization."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import InContextUNet3D


def build_model(config: dict) -> InContextUNet3D:
    if config["backbone"] is None:
        return InContextUNet3D(
            width=config["width"],
            mode=config["mode"],
            memory_grid=config["memory_grid"],
            attention_stages=config.get("attention_stages", [-1]),
            prototype_tokens=config.get("prototype_tokens", False),
        )
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    specification = config["backbone"]
    network = get_network_from_plans(
        **specification["architecture"],
        input_channels=specification["image_channels"] + specification["interaction_channels"],
        output_channels=2,
        allow_init=False,
        deep_supervision=False,
    )
    return InContextUNet3D(
        mode=config["mode"],
        memory_grid=config["memory_grid"],
        attention_stages=config.get("attention_stages", [-1]),
        prototype_tokens=config.get("prototype_tokens", False),
        backbone=network,
        image_channels=specification["image_channels"],
        interaction_channels=specification["interaction_channels"],
    )


def initialize_backbone(config: dict, model_dir: str, fold: str, checkpoint_name: str) -> InContextUNet3D:
    """Initialize through the original loader, preserving legacy and modern capabilities."""
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    loader = nnInteractiveInferenceSession(device=torch.device("cpu"), torch_n_threads=1)
    try:
        artifacts = loader._load_model_artifacts_from_disk(model_dir, fold, checkpoint_name)
        configuration = artifacts["configuration_manager"]
        canonical = {
            "prev_seg": 0,
            "bbox2d": (1, 2),
            "bbox3d": (1, 2),
            "lasso": (1, 2),
            "points": (3, 4),
            "scribble": (5, 6),
        }
        if loader.num_interaction_channels != 7 or loader.channel_mapping != canonical:
            raise ValueError("training currently requires the canonical seven-channel prompt layout")
        config["backbone"] = {
            "architecture": {
                "arch_class_name": configuration.network_arch_class_name,
                "arch_kwargs": configuration.network_arch_init_kwargs,
                "arch_kwargs_req_import": configuration.network_arch_init_kwargs_req_import,
            },
            "image_channels": len(artifacts["dataset_json"]["channel_names"]),
            "interaction_channels": loader.num_interaction_channels,
            "session": {
                "capability": artifacts["capability_content"],
                "plans": artifacts["plans_manager"].plans,
                "configuration": configuration.configuration,
                "dataset": artifacts["dataset_json"],
                "point_radius": artifacts["point_interaction"].point_radius,
                "point_use_distance_transform": artifacts["point_interaction"].use_distance_transform,
                "preferred_scribble_thickness": artifacts["preferred_scribble_thickness"],
                "interaction_decay": artifacts["interaction_decay"],
                "pad_mode_data": artifacts["pad_mode_data"],
                "license": artifacts["license"],
            },
        }
        return InContextUNet3D(
            mode=config["mode"],
            memory_grid=config["memory_grid"],
            attention_stages=config.get("attention_stages", [-1]),
            backbone=artifacts["network"],
            prototype_tokens=config.get("prototype_tokens", False),
            image_channels=config["backbone"]["image_channels"],
        )
    finally:
        loader.executor.shutdown()


def load_model(path: str | Path, device: str | torch.device = "cpu"):
    from ..checkpoint import load_model as load_automatic_model

    model, checkpoint = load_automatic_model(path, device)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported ICL checkpoint format")
    return model, checkpoint


def compact_configuration(width: int, shape: tuple | list) -> dict:
    """A complete nnU-Net configuration for the compact prototype backbone."""
    return {
        "patch_size": list(shape),
        "architecture": {
            "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
            "arch_kwargs": {
                "n_stages": 4,
                "features_per_stage": [width, width * 2, width * 4, width * 8],
                "conv_op": "torch.nn.Conv3d",
                "kernel_sizes": 3,
                "strides": [1, 2, 2, 2],
                "n_blocks_per_stage": [1, 1, 2, 2],
                "n_conv_per_stage_decoder": [1, 1, 1],
                "conv_bias": True,
                "norm_op": "torch.nn.InstanceNorm3d",
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "nonlin": "torch.nn.LeakyReLU",
                "nonlin_kwargs": {"negative_slope": 0.01, "inplace": True},
            },
            "_kw_requires_import": ["conv_op", "norm_op", "nonlin"],
        },
    }


def export_inference(checkpoint_path, output_path, native_model_dir=None, native_fold="0"):
    """Export weights and optionally pin the original native interactive/refinement path.

    The learned low-resolution model still trains interactive and context tasks.
    A frozen native refiner prevents coarse-grid fine-tuning from changing the
    original full-resolution prompt predictor. Context reaches it as a previous mask.
    """
    import copy

    from .retrieval import file_sha256

    output = Path(output_path)
    if output.exists():
        raise FileExistsError("use a new inference checkpoint path")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state["format_version"] != 1:
        raise ValueError("unsupported ICL checkpoint format")
    for key in ("optimizer", "torch_rng", "cuda_rng", "loader_rng"):
        state.pop(key, None)
    import json

    state["training_provenance"] = [
        json.loads(path.read_text()) for path in sorted(Path(checkpoint_path).parent.glob("provenance_step_*.json"))
    ]
    state["inference_export"] = {"training_checkpoint_sha256": file_sha256(checkpoint_path)}
    if native_model_dir is not None:
        config = copy.deepcopy(state["model_config"])
        native = initialize_backbone(config, native_model_dir, native_fold, "checkpoint_final.pth")
        if config["backbone"] != state["model_config"]["backbone"]:
            raise ValueError("native refiner architecture and session metadata must match the trained backbone")
        state["native_refiner"] = {
            "model": native.backbone.state_dict(),
            "source_sha256": file_sha256(Path(native_model_dir) / f"fold_{native_fold}" / "checkpoint_final.pth"),
            "policy": "frozen_native_refinement",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(output)
