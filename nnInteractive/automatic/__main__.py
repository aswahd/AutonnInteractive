"""Command-line entry point for automatic segmentation preparation, training and inference."""

import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "train",
            "validate",
            "compare",
            "predict",
            "prototype",
        ],
    )
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    delegates = {
        "train": "train",
        "validate": "validate_cli",
        "compare": "compare_inference",
        "prototype": "prototype.__main__",
    }
    if args.command in delegates:
        sys.argv = [sys.argv[0], *remaining]
        return importlib.import_module(
            "." + delegates[args.command], __package__
        ).main()
    command = argparse.ArgumentParser(prog=f"nninteractive-automatic {args.command}")
    command.add_argument("--output", type=Path, required=True)
    if args.command == "prepare":
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--configuration", default="nnUNetPlans_3d_fullres")
        command.add_argument(
            "--split",
            type=Path,
            help="Optional JSON containing train and val case lists",
        )
        args = command.parse_args(remaining)
        from .data import build_manifest

        return build_manifest(
            args.dataset,
            args.configuration,
            args.output,
            json.loads(args.split.read_text()) if args.split else None,
        )
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument(
        "--query", type=Path, required=True, help="Preprocessed 1DHW .npy image"
    )
    command.add_argument(
        "--supports",
        type=Path,
        required=True,
        help=".npz with images=K1DHW and prompts=K7DHW",
    )
    command.add_argument("--device", default="cuda")
    command.add_argument(
        "--refinement", choices=["none", "resized_context", "autozoom"], default="none"
    )
    command.add_argument("--original-model-dir", type=Path)
    args = command.parse_args(remaining)
    import numpy as np
    from .session import AutomaticSegmentationSession

    session = AutomaticSegmentationSession(
        args.checkpoint, args.device, args.original_model_dir
    )
    with np.load(args.supports) as bank:
        session.set_context(bank["images"], bank["prompts"])
    result = session.predict(np.load(args.query), refinement=args.refinement)
    arrays = {"mask": result.mask}
    if result.probabilities is not None:
        arrays["probabilities"] = result.probabilities
    if result.student_prediction is not None:
        arrays.update(
            coarse_mask=result.student_prediction.mask,
            coarse_probabilities=result.student_prediction.probabilities,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)


if __name__ == "__main__":
    main()
