import hashlib
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_original(model_dir, fold="0", checkpoint_name="checkpoint_final.pth"):
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    model_dir = Path(model_dir)
    loader = nnInteractiveInferenceSession(
        device=torch.device("cpu"), torch_n_threads=1
    )
    try:
        artifacts = loader._load_model_artifacts_from_disk(
            str(model_dir), fold, checkpoint_name
        )
        network = artifacts["network"]
        checkpoint_file = model_dir / f"fold_{fold}" / checkpoint_name
        # The original loader uses strict=True (the PyTorch default). Verify its
        # declared network contract before allowing the student wrapper to adapt it.
        encoder, decoder = network.encoder, network.decoder
        if (
            loader.num_interaction_channels != 7
            or len(artifacts["dataset_json"]["channel_names"]) != 1
        ):
            raise ValueError(
                "v2 requires one image channel and seven checkpoint prompt channels"
            )
        required = {"prev_seg", "points", "scribble", "lasso", "bbox2d"}
        if not required <= loader.channel_mapping.keys():
            raise ValueError("checkpoint lacks required prompt mappings")
        if not all(
            loader.supported_interactions[k]
            for k in ["points", "scribble", "lasso", "bbox2d"]
        ):
            raise ValueError("checkpoint does not support all baseline prompt types")
        configuration = artifacts["configuration_manager"]
        metadata = {
            "checkpoint_dir": str(model_dir.resolve()),
            "checkpoint_file": str(checkpoint_file.resolve()),
            "checkpoint_sha256": sha256(checkpoint_file),
            "checkpoint_version": model_dir.name,
            "strict_load": True,
            "fold": fold,
            "checkpoint_name": checkpoint_name,
            "trainer": artifacts["trainer_name"],
            "prompt_channels": loader.num_interaction_channels,
            "channel_mapping": loader.channel_mapping,
            "point_radius": loader.point_interaction.point_radius,
            "point_distance_transform": loader.point_interaction.use_distance_transform,
            "scribble_thickness": loader.preferred_scribble_thickness,
            "native_patch_size": list(configuration.patch_size),
            "encoder_channels": list(encoder.output_channels),
            "encoder_strides": [list(x) for x in encoder.strides],
            "normalization": str(encoder.norm_op),
            "decoder_outputs": decoder.num_classes,
            "architecture": {
                "arch_class_name": configuration.network_arch_class_name,
                "arch_kwargs": configuration.network_arch_init_kwargs,
                "arch_kwargs_req_import": configuration.network_arch_init_kwargs_req_import,
            },
            "source_metadata_sha256": {
                name: sha256(model_dir / name)
                for name in [
                    "plans.json",
                    "dataset.json",
                    "inference_session_class.json",
                    "inference_info.json",
                ]
                if (model_dir / name).exists()
            },
        }
        return network, metadata
    finally:
        loader.executor.shutdown()


def reconstruct_original_architecture(metadata):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    return get_network_from_plans(
        **metadata["architecture"],
        input_channels=1 + metadata["prompt_channels"],
        output_channels=2,
        allow_init=False,
        deep_supervision=False,
    )
