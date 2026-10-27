import argparse
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import torch
import torch.distributed as dist

from .backbone import sha256
from .config import TrainConfig
from .data import EpisodeSampler, support_crop_policy
from .evaluate import (
    evaluate_interactive,
    evaluate_patches,
    evaluate_query_patches,
    evaluate_volumes,
    summarize,
    summarize_interactive,
    summarize_query_dice,
)
from .evaluate_support_counts import evaluate_support_counts, summarize_support_counts
from .optimization import best_validation_score, load_checkpoint
from .prompts import PROMPT_TYPES
from .refinement import evaluation_refiner
from .swaps import evaluate_prompt_swaps, summarize_swaps
from .train import gather_objects
from .transfer import final_evaluation_policy
from .validation_state import TaskJournal, atomic_json


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-tasks-per-rank", type=int, default=None, help="Bounded interruption test only")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Evaluate selected best.pt after completed training; never update selection",
    )
    args = parser.parse_args()
    rank, world, local = (
        int(os.environ.get(name, default))
        for name, default in [("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]
    )
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    torch.set_num_threads(args.threads)
    checkpoint_path = args.output / ("best.pt" if args.final_test else "last.pt")
    model, state = load_checkpoint(checkpoint_path, device)
    policy = state["manifests"]
    if not args.final_test and policy.get("inference_support_policy") != support_crop_policy(
        model.config.full_volume, model.config.fixed_support_crop
    ):
        raise ValueError(
            "Support crop policy changed; use compare with a separate output to reevaluate old checkpoints without changing selection"
        )
    cohort = None
    request = state.get("pending_validation")
    if args.final_test:
        policy, cohort = final_evaluation_policy(args.output, state)
        request = {"update": state["completed_updates"], "kinds": ["full_volume"], "final_test": True}
    if request is None:
        raise ValueError("no pending validation in this checkpoint")
    update = state["completed_updates"]
    if request["update"] != update:
        raise ValueError("pending validation must refer to exactly these weights")
    manifest_path = Path(policy["data_manifest"])
    if sha256(manifest_path) != policy["data_sha256"]:
        raise ValueError("validation manifest changed")
    os.environ["PROMPT_ICL_ARRAY_DIR"] = policy["array_directory"]
    config = TrainConfig(**state["train_config"])
    sampler = EpisodeSampler(
        manifest_path,
        state["source"],
        config,
        split="val",
        retain_support_contexts=False,
        targets=policy["targets"] if args.final_test else config.validation_targets or policy["targets"],
        **model.config.sampler_options(),
    )
    seed = policy["validation_seed"]
    baseline_path = args.output / "interactive_pretrained.json"
    baseline = json.loads(baseline_path.read_text()) if "interactive" in request["kinds"] else None
    binding = {
        "checkpoint_sha256": sha256(checkpoint_path),
        "data_sha256": policy["data_sha256"],
        "pretrained_baseline_sha256": sha256(baseline_path) if baseline is not None else None,
        "evaluation_policy": policy,
        "support_crop_policy": sampler.crop_policy,
        "request": request,
        "implementation_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    directory = args.output / ("final_test" if args.final_test else "validation") / f"update_{update}"
    model.eval()
    refiner = (
        evaluation_refiner(model, state["source"])
        if any(k in request["kinds"] for k in ["full_volume", "prompt_swap"])
        else None
    )
    target_count = len(sampler.targets)
    total_tasks = {
        "query_patch": target_count * config.validation_patches_per_target,
        "patch": target_count * len(PROMPT_TYPES) * 2,
        "support_counts": target_count * len(PROMPT_TYPES) * 2,
        "interactive": target_count * (len(PROMPT_TYPES) + 1),
        "prompt_swap": len(policy["prompt_swap_pairs"]) * len(PROMPT_TYPES),
        "full_volume": len(policy["full_volume_query_ids"]) * target_count,
    }
    complete = True
    summaries, counts = {}, {}
    for kind in request["kinds"]:
        journal = TaskJournal(
            directory / kind / "partial",
            binding,
            rank,
            world,
            deadline=started + args.max_seconds,
            max_tasks=args.max_tasks_per_rank,
        )
        if kind == "query_patch":
            evaluate_query_patches(model, sampler, seed, rank, world, progress=journal)
        elif kind == "patch":
            evaluate_patches(model, sampler, seed, rank, world, progress=journal)
        elif kind == "support_counts":
            evaluate_support_counts(model, sampler, seed, rank, world, progress=journal)
        elif kind == "interactive":
            evaluate_interactive(model, sampler, seed + 300000, rank, world, progress=journal)
        elif kind == "prompt_swap":
            evaluate_prompt_swaps(
                model,
                sampler,
                policy["prompt_swap_pairs"],
                seed + 200000,
                rank,
                world,
                progress=journal,
                refiner=refiner,
            )
        elif kind == "full_volume":
            evaluate_volumes(
                model,
                sampler,
                policy["full_volume_query_ids"],
                seed + 100000,
                rank,
                world,
                progress=journal,
                prompt_type=policy.get("full_volume_prompt_type"),
                refiner=refiner,
            )
        else:
            raise ValueError("unknown validation kind")
        finished = torch.tensor(int(journal.complete(total_tasks[kind])), device=device)
        if world > 1:
            dist.all_reduce(finished, op=dist.ReduceOp.MIN)
        complete = complete and bool(finished.item())
        gathered = gather_objects(journal.rows(), rank, world)
        task_counts = gather_objects(len(journal.tasks), rank, world)
        if rank == 0:
            rows = sorted([r for part in gathered for r in part], key=lambda r: r["index"])
            counts[kind] = {"done": sum(task_counts), "total": total_tasks[kind], "complete": bool(finished.item())}
            if bool(finished.item()):
                scores = (
                    summarize_query_dice(rows)
                    if kind == "query_patch"
                    else (
                        summarize_swaps(rows)
                        if kind == "prompt_swap"
                        else (
                            summarize_interactive(rows, baseline)
                            if kind == "interactive"
                            else summarize_support_counts(rows) if kind == "support_counts" else summarize(rows)
                        )
                    )
                )
                if cohort is not None:
                    scores = {
                        name: summarize([row for row in rows if row["target"] in cohort[targets]])
                        for name, targets in (("seen", "training_targets"), ("held_out", "held_out_targets"))
                    }
                atomic_json(directory / kind / "records.json", rows)
                atomic_json(directory / kind / "summary.json", scores)
                summaries[kind] = scores
            print("VALIDATION_PROGRESS " + json.dumps({"update": update, "kind": kind, **counts[kind]}), flush=True)
    if rank == 0:
        if complete and not args.final_test:
            selection = state.get("selection_metric", "full_volume")
            best = (
                best_validation_score(state, config)
                if "selection_metric" in state
                else state["best_full_volume_macro_dice"]
            )
            improved = False
            if selection in summaries:
                value = summaries[selection]["query_dice" if selection == "query_patch" else "positive_macro_dice"]
                if value is not None and (best is None or value > best):
                    best, improved = value, True
            state["selection_metric"], state["best_validation_dice"] = selection, best
            state["best_full_volume_macro_dice"] = best if selection == "full_volume" else None
            state["pending_validation"] = None
            state["last_completed_validation"] = request
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoint_path)
            if improved:
                temporary = args.output / "best.pt.tmp"
                torch.save(state, temporary)
                temporary.replace(args.output / "best.pt")
        report = {
            "update": update,
            "final_test": args.final_test,
            "complete": complete,
            "task_counts": counts,
            "elapsed_seconds": time.monotonic() - started,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "checkpoint_before_validation_sha256": binding["checkpoint_sha256"],
            "selection_metric": state.get("selection_metric", "full_volume"),
            "best_validation_dice": state.get("best_validation_dice", state.get("best_full_volume_macro_dice")),
        }
        invocation_id = report["job_id"] or f"local_{uuid4().hex}"
        atomic_json(directory / f"allocation_{invocation_id}.json", report)
        atomic_json(directory / "progress.json", report)
        print("VALIDATION_ALLOCATION_COMPLETE " + json.dumps(report), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
