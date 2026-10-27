"""Paired development evaluation of student-only and student-plus-native-AutoZoom."""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch

from .autozoom import NativeAutoZoomRefiner
from .backbone import sha256
from .config import TrainConfig
from .data import EpisodeSampler
from .evaluate import evaluate_volumes, summarize
from .optimization import load_checkpoint
from .refinement import evaluation_refiner
from .validation_state import TaskJournal, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-ids", nargs="+")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--refinement", choices=["autozoom", "resized_context"])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, state = load_checkpoint(args.checkpoint, "cuda")
    policy = state["manifests"]
    mode = args.refinement or ("resized_context" if model.config.resized_context else "autozoom")
    if mode == "resized_context" and not model.config.resized_context:
        raise ValueError("resized-context comparison requires a large-context student")
    manifest = Path(policy["data_manifest"])
    if sha256(manifest) != policy["data_sha256"]:
        raise ValueError("development manifest changed")
    targets = args.targets or policy["targets"]
    queries = args.query_ids or policy["full_volume_query_ids"]
    if not set(targets) <= set(policy["targets"]) or not set(queries) <= set(policy["full_volume_query_ids"]):
        raise ValueError("mode comparison is restricted to the checkpoint's development protocol")
    os.environ["PROMPT_ICL_ARRAY_DIR"] = policy["array_directory"]
    sampler = EpisodeSampler(
        manifest,
        state["source"],
        TrainConfig(**state["train_config"]),
        split="val",
        targets=targets,
        **model.config.sampler_options(),
    )
    binding = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "policy": policy,
        "support_crop_policy": sampler.crop_policy,
        "targets": targets,
        "queries": queries,
        "refinement_mode": mode,
        "native_checkpoint_sha256": state["source"]["checkpoint_sha256"],
        "native_engine_sha256": sha256(Path(__file__).parents[1] / "inference/inference_session.py"),
        "implementation": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    journal = TaskJournal(args.output / "paired", binding)
    torch.cuda.reset_peak_memory_stats()
    context = (
        nullcontext(evaluation_refiner(model, state["source"]))
        if mode == "resized_context"
        else NativeAutoZoomRefiner(state["source"], "cuda", args.threads)
    )
    with context as refiner:
        evaluate_volumes(
            model,
            sampler,
            queries,
            policy["validation_seed"] + 100000,
            progress=journal,
            prompt_type=policy.get("full_volume_prompt_type"),
            refiner=refiner,
        )
    records = journal.rows()
    key = "resized_refinement" if mode == "resized_context" else "autozoom"
    summaries = {
        "full_volume" if model.config.full_volume else "sliding_window": summarize(records),
        key: summarize([{**r, **r[key]} for r in records]),
    }
    full_protocol = (
        targets == policy["targets"]
        and queries == policy["full_volume_query_ids"]
        and policy.get("inference_support_policy") == sampler.crop_policy
    )
    report = {
        "summary": summaries,
        "full_development_protocol": full_protocol,
        "support_crop_policy": sampler.crop_policy,
        "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "selection": "Keep both modes selectable; compare development Dice, negative false positives and cost. No per-case GT oracle.",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "checkpoint_update": state["completed_updates"],
    }
    atomic_json(args.output / "records.json", records)
    atomic_json(args.output / "comparison.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
