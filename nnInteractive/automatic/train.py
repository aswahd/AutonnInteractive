import argparse
import atexit
import json
import os
import random
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .backbone import load_original, sha256
from .config import ModelConfig, TrainConfig
from .data import EpisodeSampler, support_crop_policy
from .distillation import (
    STAT_FIELDS,
    FrozenTeacher,
    TeacherConfig,
    state_digest,
    summarize_teacher,
)
from .evaluate import (
    evaluate_query_patches,
    evaluate_volumes,
    summarize,
    summarize_query_dice,
    validation_targets,
)
from .losses import joint_segmentation_loss, training_dice_totals
from .mixing import MixedEpisodeDataset
from .model import PromptEncodedICL3D
from .optimization import (
    best_validation_score,
    load_checkpoint,
    make_optimizer,
    make_scheduler,
    restore_rng,
    rng_state,
    save_checkpoint,
)
from .refinement import (
    REFINEMENT_STAT_FIELDS,
    FrozenRefiner,
    RefinementConfig,
    evaluation_refiner,
    summarize_refinement,
)
from .stream import EpisodeStream
from .supervoxel_data import SupervoxelEpisodeDataset
from .support_dropout import (
    active_support_statistics,
    preserve_foreground_supports,
    sample_support_active_mask,
)
from .swaps import DEFAULT_PAIRS
from .transfer import bind_transfer_protocol

STOP_REQUESTED = False


def format_progress(row, total_updates):
    dice = ["n/a" if row[key] is None else f"{row[key]:.3f}" for key in ("query_dice", "support_dice")]
    return (
        f"step {row['update']}/{total_updates} | loss {row['loss']:.4f} | "
        f"query Dice {dice[0]} | support Dice {dice[1]} | {row['rank0_step_seconds']:.2f}s"
    )


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def gather_objects(value, rank, world_size):
    if world_size == 1:
        return [value]
    collected = [None] * world_size if rank == 0 else None
    dist.gather_object(value, collected, dst=0)
    return collected


def sampling_resume_transition(previous, config, allow_change=False, allow_augmentation_change=False):
    """Permit an explicitly requested sampler change without relaxing other training checks."""
    old = TrainConfig(**previous)
    before, after = old.resume_settings(), config.resume_settings()
    if allow_change:
        before.pop("spatial_sampling")
        after.pop("spatial_sampling")
    if allow_augmentation_change:
        before.pop("spatial_augmentation")
        after.pop("spatial_augmentation")
    if before != after:
        raise ValueError(
            "resume requires unchanged training; explicitly allow spatial sampling/augmentation changes only"
        )
    changes = {}
    if old.spatial_sampling != config.spatial_sampling:
        changes.update(before=old.spatial_sampling, after=config.spatial_sampling)
    if old.spatial_augmentation != config.spatial_augmentation:
        changes["augmentation"] = {"before": old.spatial_augmentation, "after": config.spatial_augmentation}
    return changes or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--allow-spatial-sampling-change",
        action="store_true",
        help="Explicitly change only training.spatial_sampling on resume, retaining optimizer, scheduler and RNG state",
    )
    parser.add_argument(
        "--allow-spatial-augmentation-change",
        action="store_true",
        help="Explicitly change spatial augmentation while retaining optimizer, scheduler and RNG state",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--defer-validation",
        action="store_true",
        help="Checkpoint and return at validation boundaries; resume training after validate CLI finishes",
    )
    args = parser.parse_args()
    if (args.allow_spatial_sampling_change or args.allow_spatial_augmentation_change) and args.resume is None:
        parser.error("spatial sampling/augmentation changes require --resume")
    specification = json.loads(args.config.read_text())
    config = TrainConfig(**specification.get("training", {}))
    teacher_config = TeacherConfig(**specification.get("teacher", {}))
    refinement_config = RefinementConfig(**specification.get("refinement", {}))
    if teacher_config.enabled and refinement_config.enabled:
        raise ValueError("KD and differentiable refinement are separate experiments")
    refining = refinement_config.enabled
    if refining:
        teacher_config = refinement_config
    if teacher_config.enabled and not (
        config.supervise_query and config.supervise_supports and config.lambda_support == 1
    ):
        raise ValueError("distillation retains both GT objectives and lambda_support=1")
    if config.support_dropout["enabled"] and config.support_count != 4:
        raise ValueError("support dropout training always samples four supports")
    model_config = ModelConfig(**specification.get("model", {}))
    if model_config.query_deep_supervision and not config.supervise_query:
        raise ValueError("query deep supervision requires query supervision")
    if (
        model_config.mask_encoder_stages is not None
        and not config.supervise_supports
        and set(model_config.mask_encoder_stages) != set(model_config.stages)
    ):
        raise ValueError("query-only training requires matching mask and attention stages to avoid unused heads")
    rank, world_size, local_rank = (
        int(os.environ.get(k, default)) for k, default in [("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    accumulation = config.accumulation_steps(world_size)
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    manifest_path = Path(specification["data_manifest"])
    manifest = json.loads(manifest_path.read_text())
    swap_pairs = specification.get(
        "prompt_swap_pairs",
        [
            list(pair)
            for pair in DEFAULT_PAIRS
            if all(t in (specification.get("targets") or manifest["regions"]) for t in pair)
        ],
    )
    manifests = {
        "data_manifest": str(manifest_path.resolve()),
        "data_sha256": sha256(manifest_path),
        "source_manifest_sha256": manifest["source_sha256"],
        "split_unit": manifest["split_unit"],
        "inference_support_policy": support_crop_policy(model_config.full_volume, model_config.fixed_support_crop),
        "targets": specification.get("targets"),
        "prompt_swap_pairs": swap_pairs,
        "full_volume_query_ids": specification.get("full_volume_query_ids", []),
        "validation_seed": specification.get("validation_seed", 102026),
        "array_directory": os.environ.get(
            "PROMPT_ICL_ARRAY_DIR",
            str(Path(manifest["dataset"]) / manifest["configuration"]),
        ),
    }
    if "full_volume_prompt_type" in specification:
        manifests["full_volume_prompt_type"] = specification["full_volume_prompt_type"]
    manifests.update(bind_transfer_protocol(specification, manifest))
    supervoxel_manifest = None
    if config.supervoxel_probability:
        supervoxel_manifest = Path(specification.get("supervoxel_manifest", manifest_path))
        manifests.update(
            supervoxel_manifest=str(supervoxel_manifest.resolve()),
            supervoxel_sha256=sha256(supervoxel_manifest),
            episode_mixing_policy="independent_datasets_v1",
        )
    if teacher_config.enabled:
        manifests["refinement" if refining else "distillation"] = asdict(teacher_config)
    enabled_mask_attention = False
    sampling_transition = None
    if args.resume:
        model, state = load_checkpoint(args.resume, device)
        if "attention_depth" not in specification.get("model", {}):
            model_config.attention_depth = model.config.attention_depth
        if state.get("pending_validation") is not None:
            raise ValueError("complete pending validation before resuming optimizer updates")
        sampling_transition = sampling_resume_transition(
            state["train_config"], config, args.allow_spatial_sampling_change, args.allow_spatial_augmentation_change
        )
        if state["world_size"] != world_size or state["manifests"] != manifests:
            raise ValueError("resume requires unchanged training, world size, data, and validation policy")
        # This parameter-free transition preserves optimizer/RNG state; reject all other changes below.
        enabled_mask_attention = model_config.support_mask_attention and not model.config.support_mask_attention
        if enabled_mask_attention:
            model.config.support_mask_attention = True
            model.config.validate()
        # JSON turns tuples into lists; compare canonical encodings.
        if json.dumps(asdict(model.config), sort_keys=True) != json.dumps(asdict(model_config), sort_keys=True):
            raise ValueError("resume model configuration mismatch")
        source = state["source"]
        completed, best = (
            state["completed_updates"],
            None if enabled_mask_attention else best_validation_score(state, config),
        )
    else:
        if (args.output / "last.pt").exists():
            raise FileExistsError("existing run requires --resume")
        backbone, source = load_original(
            specification["checkpoint_dir"],
            str(specification.get("checkpoint_fold", "0")),
        )
        model = PromptEncodedICL3D(backbone, model_config, source["channel_mapping"]["prev_seg"]).to(device)
        completed, best = 0, None
    if (
        not (model_config.resized_context or model_config.full_volume)
        and list(model_config.patch_size) != source["native_patch_size"]
    ):
        raise ValueError("training must preserve the original 192-cubed patch geometry")
    frozen_class = FrozenRefiner if refining else FrozenTeacher
    teacher = frozen_class(teacher_config, source, device, model_config) if teacher_config.enabled else None
    torch.set_num_threads(args.threads)
    optimizer, scheduler = make_optimizer(model, config), None
    scheduler = make_scheduler(optimizer, config)
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rank_rng_states"][rank])
    wrapped = (
        DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world_size > 1 else model
    )
    training = EpisodeSampler(
        manifest_path,
        source,
        config,
        split="train",
        retain_support_contexts=False,
        targets=specification.get("targets"),
        **model_config.sampler_options(),
    )
    selected_validation_targets = validation_targets(manifest, training.targets, config)
    validation = EpisodeSampler(
        manifest_path,
        source,
        config,
        split="val",
        retain_support_contexts=False,
        targets=selected_validation_targets,
        **model_config.sampler_options(),
    )
    if selected_validation_targets != validation.targets:
        raise ValueError("validation must support every selected validation target")
    training_data = training
    if supervoxel_manifest is not None:
        supervoxels = SupervoxelEpisodeDataset(
            supervoxel_manifest,
            source,
            config,
            retain_support_contexts=False,
            **model_config.sampler_options(),
        )
        training_data = MixedEpisodeDataset(
            (training, supervoxels), (1 - config.supervoxel_probability, config.supervoxel_probability)
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        provenance = {
            "specification": specification,
            "resolved_training": asdict(config),
            "resolved_model": asdict(model_config),
            "source": source,
            "manifests": manifests,
            "active_targets": training.targets,
            "validation_targets": validation.targets,
            "world_size": world_size,
            "accumulation_steps": accumulation,
            "effective_batch": config.microbatch * world_size * accumulation,
            "parameters": sum(p.numel() for p in model.parameters()),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "implementation_directory": str(Path(__file__).parent.resolve()),
            "implementation_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "start_update": completed,
            "enabled_support_mask_attention_on_resume": enabled_mask_attention,
            "spatial_sampling_transition": sampling_transition,
            "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
            "learning_rates": {group["group_name"]: group["lr"] for group in optimizer.param_groups},
            "augmentation_spacing": training.spacing,
        }
        if teacher is not None:
            provenance["teacher"] = {
                "source": teacher.source,
                "configuration": asdict(teacher_config),
                "mode": "differentiable_refinement" if refining else "detached_output_distillation",
                "inference_type": "resized_full_context" if model_config.resized_context else "single_patch",
            }
            provenance["student_state_sha256"] = state_digest(model)
            print("FROZEN_TEACHER " + json.dumps(provenance["teacher"]), flush=True)
        (args.output / f"provenance_{completed}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    limit = config.updates
    if completed >= limit:
        raise ValueError("training endpoint must exceed completed updates")
    stream = EpisodeStream(
        training_data,
        config,
        rank,
        world_size,
        limit,
        args.output / f"rejected_rank{rank}.jsonl",
        config.prefetch_batches,
        collate=True,
    )
    atexit.register(stream.close)
    start_time = time.monotonic()
    active_histogram = torch.zeros(config.support_count, dtype=torch.long, device=device)
    teacher_totals = torch.zeros(
        len(REFINEMENT_STAT_FIELDS if refining else STAT_FIELDS),
        dtype=torch.float64,
        device=device,
    )
    for update in range(completed, limit):
        log_due = (update + 1) % 10 == 0 or update < 4
        dice_totals = torch.zeros(4, device=device)
        step_started = time.monotonic()
        sampling_seconds = 0.0
        data_wait_seconds = 0.0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        component_sum = torch.zeros(3, device=device)
        head_loss_sum = torch.zeros(5 if model_config.query_deep_supervision else 1, device=device)
        valid_fractions = torch.zeros(4, device=device)
        kd_sum = torch.zeros((), device=device)
        refinement_logit_grad_norm = torch.zeros((), device=device)
        episode_metadata = []
        for micro in range(accumulation):
            sampling_started = time.monotonic()
            batch, sample_duration = stream.get(update, micro)
            sampling_seconds += sample_duration
            data_wait_seconds += time.monotonic() - sampling_started
            batch = batch.to(device)
            valid_fractions += (
                torch.stack(
                    (
                        batch.query_valid.sum().float() / batch.query_valid.numel(),
                        batch.support_valid.sum().float() / batch.support_valid.numel(),
                        valid_fractions.new_tensor(np.mean([m["query_padding_fraction"] for m in batch.metadata])),
                        valid_fractions.new_tensor(
                            np.mean([s["padding_fraction"] for m in batch.metadata for s in m["supports"]])
                        ),
                    )
                )
                / accumulation
            )
            support_active = sample_support_active_mask(
                batch.support_images.shape[0],
                batch.support_images.shape[1],
                device,
                config.support_dropout,
            )
            support_active = preserve_foreground_supports(
                support_active, batch.support_prompts, training.prompts.mapping
            )
            active_histogram += torch.bincount(support_active.sum(1) - 1, minlength=config.support_count)
            for metadata, mask in zip(batch.metadata, support_active.tolist()):
                metadata["support_active_mask"] = mask
            synchronization = wrapped.no_sync() if world_size > 1 and micro < accumulation - 1 else nullcontext()
            with synchronization:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                    logits = wrapped(
                        batch.query_images,
                        batch.support_images,
                        batch.support_prompts,
                        support_active,
                        decode_supports=config.supervise_supports,
                    )
                loss, parts = joint_segmentation_loss(logits, batch, config)
                head_loss_sum += parts["query_head_losses"] / accumulation
                if teacher is not None:
                    kd, teacher_stats = teacher.loss(logits["query_logits"], batch, config.seed)
                    if refining and log_due:
                        grad = torch.autograd.grad(kd, logits["query_logits"], retain_graph=True)[0]
                        refinement_logit_grad_norm += grad.float().norm().detach() / accumulation
                    loss = loss + teacher_config.weight(update) * kd
                    teacher_totals += teacher_stats
                    kd_sum += kd.detach() / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite supervised loss")
                (loss / accumulation).backward()
            loss_sum += float(loss.detach()) / accumulation
            component_sum += (
                torch.stack(
                    (
                        parts["query_loss"],
                        parts["support_loss"],
                        parts["eligible_supports_per_episode"].float().mean(),
                    )
                )
                / accumulation
            )
            episode_metadata.extend(batch.metadata)
            if log_due:
                dice_totals += training_dice_totals(logits, batch)
            del batch, logits, loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        completed = update + 1
        if log_due:
            averaged = torch.tensor(loss_sum, device=device)
            performance = torch.tensor(
                [
                    time.monotonic() - step_started,
                    torch.cuda.max_memory_allocated() / 1024**3,
                    data_wait_seconds,
                ],
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(averaged)
                averaged /= world_size
                dist.all_reduce(component_sum)
                component_sum /= world_size
                dist.all_reduce(head_loss_sum)
                head_loss_sum /= world_size
                dist.all_reduce(active_histogram)
                dist.all_reduce(dice_totals)
                dist.all_reduce(valid_fractions)
                valid_fractions /= world_size
                dist.all_reduce(performance, op=dist.ReduceOp.MAX)
                if teacher is not None:
                    dist.all_reduce(teacher_totals)
                    dist.all_reduce(kd_sum)
                    kd_sum /= world_size
                    if refining:
                        dist.all_reduce(refinement_logit_grad_norm)
                        refinement_logit_grad_norm /= world_size
            if rank == 0:
                query_dice, support_dice = [
                    float(dice_totals[i] / dice_totals[i + 1]) if dice_totals[i + 1] > 0 else None for i in (0, 2)
                ]
                row = {
                    "update": completed,
                    "loss": float(averaged),
                    "query_dice": query_dice,
                    "support_dice": support_dice,
                    "query_dice_count": int(dice_totals[1]),
                    "support_dice_count": int(dice_totals[3]),
                    "train/query_valid_fraction": float(valid_fractions[0]),
                    "train/support_valid_fraction": float(valid_fractions[1]),
                    "train/query_padded_fraction": float(valid_fractions[2]),
                    "train/support_padded_fraction": float(valid_fractions[3]),
                    "train/total_loss": float(averaged),
                    "train/query_loss": float(component_sum[0]),
                    "train/query_head_losses": head_loss_sum.tolist(),
                    "train/support_loss": float(component_sum[1]),
                    **active_support_statistics(active_histogram),
                    "query_loss": float(component_sum[0]),
                    "support_loss": float(component_sum[1]),
                    "eligible_supports_mean": float(component_sum[2]),
                    "gradient_norm": float(norm),
                    "elapsed_seconds": time.monotonic() - start_time,
                    "rank0_step_seconds": time.monotonic() - step_started,
                    "rank0_sampling_seconds": sampling_seconds,
                    "rank0_data_wait_seconds": data_wait_seconds,
                    "performance/step_time": float(performance[0]),
                    "performance/peak_gpu_memory": float(performance[1]),
                    "performance/max_rank_data_wait_seconds": float(performance[2]),
                    "learning_rates": scheduler.get_last_lr(),
                    "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "rank0_episodes": episode_metadata,
                }
                if teacher is not None:
                    row.update(summarize_refinement(teacher_totals) if refining else summarize_teacher(teacher_totals))
                    objective = "refinement" if refining else "distillation"
                    row.update(
                        {
                            f"train/{objective}_loss": float(kd_sum),
                            f"train/{objective}_weight": teacher_config.weight(update),
                            f"train/weighted_{objective}_loss": float(kd_sum) * teacher_config.weight(update),
                        }
                    )
                    if refining:
                        row["train/refinement_only_student_logit_gradient_norm"] = float(refinement_logit_grad_norm)
                with (args.output / "training.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                suffix = (
                    ""
                    if teacher is None
                    else (
                        f" | refinement {float(kd_sum):.4f}"
                        if refining
                        else f" | KD {float(kd_sum):.4f} | teacher {int(teacher_totals[3])}/{int(teacher_totals[0])} called"
                    )
                )
                print(format_progress(row, config.updates) + suffix, flush=True)
            active_histogram.zero_()
            teacher_totals.zero_()
        stop = torch.tensor(int(STOP_REQUESTED), device=device)
        if world_size > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        stopping = bool(stop.item())
        kinds = config.validation_kinds(completed, stopping)
        improved = False
        pending_validation = {"update": completed, "kinds": kinds} if args.defer_validation and kinds else None
        if kinds and pending_validation is None:
            stream.quiesce()
            if world_size > 1:
                dist.barrier()
            model.eval()
            context_refiner = (
                (teacher or evaluation_refiner(model, source))
                if "full_volume" in kinds and model_config.resized_context
                else None
            )
            for kind in kinds:
                if kind == "query_patch":
                    rows = evaluate_query_patches(
                        model,
                        validation,
                        manifests["validation_seed"],
                        rank,
                        world_size,
                    )
                else:
                    rows = evaluate_volumes(
                        model,
                        validation,
                        specification["full_volume_query_ids"],
                        manifests["validation_seed"] + 100000,
                        rank,
                        world_size,
                        prompt_type=manifests.get("full_volume_prompt_type"),
                        refiner=context_refiner,
                    )
                gathered = gather_objects(rows, rank, world_size)
                if rank == 0:
                    rows = sorted(
                        [row for part in gathered for row in part],
                        key=lambda row: row["index"],
                    )
                    scores = summarize_query_dice(rows) if kind == "query_patch" else summarize(rows)
                    destination = args.output / "validation" / f"update_{completed}" / kind
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / "records.json").write_text(json.dumps(rows, indent=2) + "\n")
                    (destination / "summary.json").write_text(json.dumps(scores, indent=2) + "\n")
                    print(
                        "VALIDATION " + json.dumps({"update": completed, "kind": kind, **scores}),
                        flush=True,
                    )
                    if kind == config.selection_metric:
                        metric = scores["query_dice" if kind == "query_patch" else "positive_macro_dice"]
                        if metric is not None and (best is None or metric > best):
                            best, improved = metric, True
            del context_refiner
        save_due = (
            completed % config.checkpoint_every == 0
            or bool(kinds)
            or stopping
            or completed == limit
            or pending_validation is not None
        )
        if save_due:
            stream.quiesce()
            states = gather_objects(rng_state(), rank, world_size)
            if rank == 0:
                save_checkpoint(
                    args.output / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    completed,
                    best,
                    config,
                    source,
                    manifests,
                    states,
                    pending_validation=pending_validation,
                )
                if improved:
                    save_checkpoint(
                        args.output / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        completed,
                        best,
                        config,
                        source,
                        manifests,
                        states,
                    )
            if world_size > 1:
                dist.barrier()
        if stopping or pending_validation is not None:
            break
    stream.close()
    atexit.unregister(stream.close)
    if rank == 0:
        report = {
            "completed_updates": completed,
            "requested_updates": config.updates,
            "stopped_by_signal": bool(STOP_REQUESTED),
            "pending_validation": pending_validation,
            "selection_metric": config.selection_metric,
            "best_validation_dice": best,
            "elapsed_seconds": time.monotonic() - start_time,
        }
        (args.output / f"completion_{completed}.json").write_text(json.dumps(report, indent=2) + "\n")
        if completed == limit and pending_validation is None:
            (args.output / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
        print("TRAINING_STOP " + json.dumps(report), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
