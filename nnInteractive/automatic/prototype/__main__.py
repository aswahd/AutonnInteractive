"""Run with python -m nnInteractive.automatic.prototype."""

import argparse

from .data import prepare
from .engine import evaluate_checkpoint, train


def main():
    parser = argparse.ArgumentParser(description="Prototype checkpoint compatibility and evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("prepare", help="Create sharded whole-volume caches from nnU-Net b2nd files")
    cache.add_argument("--dataset", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--configuration", default="nnUNetPlans_3d_fullres")
    cache.add_argument("--fold", type=int, default=0)
    cache.add_argument("--shape", nargs=3, type=int, default=[96, 96, 96])
    cache.add_argument(
        "--train-cases",
        type=int,
        default=64,
        help="0 selects the entire training split",
    )
    cache.add_argument(
        "--val-cases",
        type=int,
        default=16,
        help="0 selects the entire validation split",
    )
    cache.add_argument("--shard-size", type=int, default=16)
    cache.add_argument("--patient-map", help="Optional JSON mapping every case to its patient ID")
    cache.add_argument("--seed", type=int, default=2026)
    cache.add_argument("--threads", type=int, default=4)
    cache.add_argument("--split", help="JSON train/val case lists overriding the dataset fold")
    cache.add_argument("--array-dir", help="Directory containing the preprocessed b2nd arrays")
    cache.add_argument("--normalization", choices=["none", "zscore"], default="none")
    training = subparsers.add_parser("train")
    data = training.add_mutually_exclusive_group(required=True)
    data.add_argument("--cache", help="Legacy resized cache")
    data.add_argument("--data-manifest", help="Read full volumes and crop/resize on the fly")
    training.add_argument("--crop-shape", nargs=3, type=int, default=[192, 384, 384])
    training.add_argument("--input-shape", nargs=3, type=int, default=[96, 96, 96])
    training.add_argument("--output", required=True)
    training.add_argument(
        "--regions",
        nargs="+",
        required=True,
        help="Names from dataset.json, including union regions",
    )
    training.add_argument("--mode", choices=["memory", "prototype"], default="memory")
    training.add_argument("--interactive-probability", type=float, default=0.5)
    training.add_argument("--bce-balance", choices=["class", "voxel"], default="class")
    training.add_argument("--selection", choices=["joint", "context"], default="joint")
    training.add_argument("--width", type=int, default=16)
    training.add_argument("--memory-grid", type=int, default=4)
    training.add_argument(
        "--prototype-tokens", action="store_true", help="Cross-attend to FG/BG prototypes and mask memory"
    )
    training.add_argument(
        "--attention-stages",
        nargs="+",
        type=int,
        default=[-1],
        help="Encoder stages for cross-attention, e.g. -1 -2 -3; defaults to the original bottleneck",
    )
    training.add_argument("--shots", type=int, default=1)
    training.add_argument("--steps", type=int, default=10000)
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--micro-batch-size", type=int, help="Accumulate to batch-size; must divide it evenly")
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--backbone-learning-rate", type=float)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--validate-every", type=int, default=250)
    training.add_argument("--log-every", type=int, default=25)
    training.add_argument("--seed", type=int, default=2026)
    training.add_argument("--device", default="cuda")
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--resume")
    training.add_argument(
        "--backbone-model-dir",
        help="Trusted local nnInteractive model with inference_info.json",
    )
    training.add_argument("--backbone-fold", default="all")
    training.add_argument("--backbone-checkpoint", default="checkpoint_final.pth")
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--cache")
    evaluation.add_argument("--data-manifest")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--regions", nargs="+")
    evaluation.add_argument("--shots", type=int, default=1)
    evaluation.add_argument(
        "--retrieval",
        help="Feature cache from cache-retrieval; uses patient-excluding MMR",
    )
    evaluation.add_argument("--seed", type=int, default=102026)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--threads", type=int, default=4)
    retrieval = subparsers.add_parser("cache-retrieval")
    retrieval.add_argument("--checkpoint", required=True)
    retrieval.add_argument("--cache", required=True)
    retrieval.add_argument("--output", required=True)
    retrieval.add_argument("--grid", type=int, default=2)
    retrieval.add_argument("--threads", type=int, default=4)
    retrieval.add_argument("--device", default="cuda")
    export = subparsers.add_parser("export", help="Export inference weights with an optional frozen native refiner")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--native-model-dir")
    export.add_argument("--native-fold", default="0")
    args = parser.parse_args()
    if args.command == "export":
        from .checkpoint import export_inference

        export_inference(args.checkpoint, args.output, args.native_model_dir, args.native_fold)
        return
    if args.command == "cache-retrieval":
        from .retrieval import build_retrieval

        if args.grid < 1:
            parser.error("retrieval grid must be positive")
        build_retrieval(args)
        return
    if args.command == "prepare":
        if min(args.shape) < 8 or args.shard_size < 1 or min(args.train_cases, args.val_cases, args.fold) < 0:
            parser.error("invalid shape, shard size, split size, or fold")
        prepare(args)
    elif args.command == "train":
        if not 0 <= args.interactive_probability <= 1:
            parser.error("interactive probability must be in [0,1]")
        if (
            min(
                args.steps,
                args.batch_size,
                args.validate_every,
                args.log_every,
                args.threads,
            )
            < 1
        ):
            parser.error("steps, batch size, logging intervals, and threads must be positive")
        train(args)
    else:
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
