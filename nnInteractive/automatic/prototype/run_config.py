"""Reproduce historical experiments; current coarse training uses automatic.train."""

import argparse
import json
import os
import sys
from pathlib import Path

from ..backbone import sha256
from ..config import TrainConfig
from .__main__ import main as run_prototype


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The historical coarse trainer uses one GPU per experiment")
    spec = json.loads(args.config.read_text())
    if "prototype_memory_grid" in spec.get("model", {}):
        raise ValueError("Current coarse configurations must use nnInteractive.automatic train, like the other arms")
    if "array_dir" in spec:
        os.environ["PROMPT_ICL_ARRAY_DIR"] = spec["array_dir"]
    training = dict(spec["training"])
    if "steps" in training:
        raise ValueError("Use training.updates as the sole optimizer-update budget")
    training["steps"] = training.pop("updates", TrainConfig.updates)
    checkpoint = Path(spec["checkpoint_dir"]) / f"fold_{spec.get('checkpoint_fold', '0')}" / "checkpoint_final.pth"
    if sha256(checkpoint) != spec["pretrained_sha256"]:
        raise ValueError("original pretrained checkpoint hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "configuration.json").write_text(json.dumps(spec, indent=2) + "\n")
    options = {
        **(
            {
                "data_manifest": spec["data_manifest"],
                "crop_shape": spec.get("crop_shape", [192, 384, 384]),
                "input_shape": spec.get("input_shape", [96, 96, 96]),
            }
            if "data_manifest" in spec
            else {"cache": spec["cache"]}
        ),
        "output": str(args.output),
        "regions": spec["targets"],
        "backbone_model_dir": spec["checkpoint_dir"],
        "backbone_fold": spec.get("checkpoint_fold", "0"),
        **spec["model"],
        **training,
    }
    if args.resume:
        options["resume"] = str(args.resume)
    sys.argv = [sys.argv[0], "train"]
    for name, value in options.items():
        if isinstance(value, bool):
            if value:
                sys.argv.append("--" + name.replace("_", "-"))
            continue
        sys.argv.extend(["--" + name.replace("_", "-"), *map(str, value if isinstance(value, list) else [value])])
    run_prototype()


if __name__ == "__main__":
    main()
