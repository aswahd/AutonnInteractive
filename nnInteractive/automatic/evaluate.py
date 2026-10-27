"""Fixed patch and whole-volume development evaluations."""

import time
from collections import defaultdict

import numpy as np
import torch

from .data import collate_episodes
from .infer import predict_volume, refine_volume
from .losses import binary_logits, segmentation_loss
from .optimization import episode_rng
from .prompts import PROMPT_TYPES
from .spatial import resize_volume, support_at_resolution
from .validation_state import commit_task, task_indices


def binary_metrics(prediction, target, valid):
    prediction = np.asarray(prediction, dtype=bool) & valid
    truth = np.asarray(target, dtype=bool) & valid
    target_count, predicted_count = int(truth.sum()), int(prediction.sum())
    intersection = int((prediction & truth).sum())
    negative_count = int((valid & ~truth).sum())
    false_positive = int((prediction & ~truth).sum())
    return {
        "dice": 2 * intersection / (predicted_count + target_count) if target_count else None,
        "target_voxels": target_count,
        "predicted_voxels": predicted_count,
        "valid_voxels": int(valid.sum()),
        "false_positive_voxels": false_positive,
        "background_false_positive_fraction": false_positive / negative_count if negative_count else None,
        "empty_query": target_count == 0,
    }


def _group_summary(records):
    positive = defaultdict(list)
    negative = []
    for record in records:
        if record["dice"] is not None:
            positive[record["target"]].append(record["dice"])
        else:
            negative.append(record)
    return {
        "positive_macro_dice": float(np.mean([np.mean(v) for v in positive.values()])) if positive else None,
        "positive_targets": len(positive),
        "positive_queries": sum(len(v) for v in positive.values()),
        "records": len(records),
        "positive_by_target": {k: float(np.mean(v)) for k, v in positive.items()},
        "negative_queries": len(negative),
        "negative_mean_predicted_voxels": (
            float(np.mean([r["predicted_voxels"] for r in negative])) if negative else None
        ),
        "negative_mean_false_positive_fraction": (
            float(np.mean([r["background_false_positive_fraction"] for r in negative])) if negative else None
        ),
        "negative_total_predicted_voxels": sum(r["predicted_voxels"] for r in negative),
        "negative_total_valid_voxels": sum(r["valid_voxels"] for r in negative),
    }


def summarize(records):
    by_target, by_prompt = defaultdict(list), defaultdict(list)
    for record in records:
        by_target[record["target"]].append(record)
        by_prompt[record["prompt_type"]].append(record)
    result = {
        **_group_summary(records),
        "by_target": {k: _group_summary(v) for k, v in by_target.items()},
        "by_prompt_type": {k: _group_summary(v) for k, v in by_prompt.items()},
        "policy": "Target-macro Dice over nonempty references; empty-query false positives reported separately. Prompt strata use the same target-macro reduction.",
    }
    refined = [{**r, **r["resized_refinement"]} for r in records if "resized_refinement" in r]
    if refined:
        result["resized_refinement"] = _group_summary(refined)
    return result


def summarize_query_dice(records):
    by_target = defaultdict(list)
    for row in records:
        by_target[row["target"]].append(row["dice"])
    means = {target: float(np.mean(values)) for target, values in by_target.items()}
    return {"query_dice": float(np.mean(list(means.values()))) if means else None, "query_dice_by_target": means}


def validation_targets(manifest, training_targets, config):
    """Resolve periodic validation targets and reject missing positives before training."""
    targets = training_targets if config.validation_targets is None else config.validation_targets
    if not set(targets) <= set(training_targets):
        raise ValueError("validation_targets must be active training targets")
    missing = [t for t in targets if not any(t in manifest["cases"][c]["present"] for c in manifest["splits"]["val"])]
    if missing:
        raise ValueError(f"query Dice validation requires positive held-out volumes for {missing}")
    return targets


@torch.no_grad()
def evaluate_query_patches(model, sampler, seed=102026, rank=0, world_size=1, progress=None):
    """Fixed positive queries per organ, using the checkpoint's crop or full-volume policy."""
    device = next(model.parameters()).device
    model.eval()
    records = []
    total = len(sampler.targets) * sampler.config.validation_patches_per_target
    for index in task_indices(total, rank, world_size, progress):
        target = sampler.targets[index % len(sampler.targets)]
        rng = episode_rng(seed, index)
        candidates = [c for c in sampler.queries[target] if target in sampler.manifest["cases"][c]["present"]]
        if not candidates:
            raise ValueError(f"query Dice validation requires a positive held-out volume for {target}")
        episode = sampler.sample_episode(
            rng, target=target, query_id=str(rng.choice(candidates)), query_kind="positive"
        )
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()
        ):
            logits = model.predict_query(
                episode.query_images[None].to(device),
                episode.support_images[None].to(device),
                episode.support_prompts[None].to(device),
            )
        logits = binary_logits(logits).float()
        if sampler.full_volume:
            logits = logits.cpu()
        logits = resize_volume(logits, episode.query_targets.shape[-3:])
        prediction = (logits[0, 0] > 0).cpu() & episode.query_valid[0]
        truth = episode.query_targets[0].bool() & episode.query_valid[0]
        dice = float(2 * (prediction & truth).sum() / (prediction.sum() + truth.sum()))
        records.append({"index": index, **episode.metadata, "dice": dice})
        commit_task(progress, index, records[-1:])
    return records


@torch.no_grad()
def evaluate_patches(model, sampler, seed=102026, rank=0, world_size=1, progress=None):
    """Two fixed query regimes per target/prompt: positive and attempted hard negative."""
    device = next(model.parameters()).device
    model.eval()
    records = []
    tasks = [
        (target, prompt, kind)
        for target in sampler.targets
        for prompt in PROMPT_TYPES
        for kind in ["positive", "hard_negative"]
    ]
    for index in task_indices(len(tasks), rank, world_size, progress):
        target, prompt, kind = tasks[index]
        episode = sampler.sample_episode(episode_rng(seed, index), target=target, query_kind=kind, prompt_type=prompt)
        batch = collate_episodes([episode]).to(device)
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            outputs = model(batch.query_images, batch.support_images, batch.support_prompts)
            logits = binary_logits(outputs["query_logits"])
        loss, _ = segmentation_loss(logits, batch.query_targets, batch.query_valid)
        logits = resize_volume(logits.float(), batch.query_targets.shape[-3:])
        metrics = binary_metrics(
            logits[0, 0].float().sigmoid().cpu().numpy() > 0.5,
            episode.query_targets[0].numpy(),
            episode.query_valid[0].numpy(),
        )
        support_records = []
        for k, truth in enumerate(episode.support_targets):
            prediction = (
                resize_volume(binary_logits(outputs["support_logits"][0, k : k + 1]).float(), truth.shape[-3:])[0, 0]
                .sigmoid()
                .cpu()
                .numpy()
                > 0.5
            )
            support_records.append(
                {
                    **episode.metadata["supports"][k],
                    **binary_metrics(
                        prediction,
                        truth[0].numpy(),
                        episode.support_valid[k][0].numpy(),
                    ),
                }
            )
        records.append(
            {
                "index": index,
                "prompt_type": prompt,
                "loss": float(loss),
                **episode.metadata,
                **metrics,
                "interactive_support_predictions": support_records,
            }
        )
        commit_task(progress, index, records[-1:])
        del outputs, logits, batch
    return records


def evaluate_volumes(
    model,
    sampler,
    query_ids,
    seed=202026,
    rank=0,
    world_size=1,
    progress=None,
    prompt_type=None,
    refiner=None,
):
    """Primary checkpoint criterion. Every query voxel is tiled without reference ROIs."""
    if not set(query_ids) <= set(sampler.manifest["splits"]["val"]):
        raise ValueError("full-volume queries must be in the held-out volume partition")
    if prompt_type is not None and prompt_type not in PROMPT_TYPES:
        raise ValueError("unknown full-volume support prompt type")
    tasks = [(case, target) for case in query_ids for target in sampler.targets]
    records = []
    loaded_case, image, segmentation = None, None, None
    for index in task_indices(len(tasks), rank, world_size, progress):
        case, target = tasks[index]
        # Balance fixed prompt-type assignments over the query/target task list.
        prompt = prompt_type or PROMPT_TYPES[index % len(PROMPT_TYPES)]
        images, prompts, support_records = sampler.sample_supports(episode_rng(seed, index), target, case, prompt)
        if loaded_case != case:
            arrays = sampler.store.arrays(case)
            image, segmentation = arrays[0][:], arrays[1][0]
            loaded_case = case
        started = time.monotonic()
        resized_refinement = hasattr(refiner, "predict_logits") and getattr(model.config, "resized_context", False)
        result = predict_volume(
            model,
            image,
            images,
            prompts,
            stride=tuple(p // 2 for p in getattr(model.config, "context_size", model.config.patch_size)),
            padding_value=sampler.manifest.get("image_padding", 0.0),
            mode="resized_context" if resized_refinement else "sliding_window",
            refiner=refiner if resized_refinement else None,
        )
        student_seconds = time.monotonic() - started
        if resized_refinement:
            refined, result = result, result.student_prediction
            student_seconds -= refined.refinement_seconds
        else:
            refined = refine_volume(result, image, refiner) if refiner is not None else None
        # The reference is used only after both complete label-free predictions.
        valid = (segmentation >= 0) & (segmentation != sampler.manifest.get("ignore_label", -1))
        truth = np.isin(segmentation, sampler.manifest["regions"][target]) & valid
        records.append(
            {
                "index": index,
                "query_id": case,
                "target": target,
                "prompt_type": prompt,
                "supports": support_records,
                "support_patch_count": len(support_records),
                "support_unique_volume_count": len({r["volume_id"] for r in support_records}),
                "windows": result.windows,
                "coverage_min": result.coverage_min,
                **binary_metrics(result.mask, truth, valid),
            }
        )
        if refined is not None:
            records[-1]["student_seconds"] = student_seconds
            records[-1]["resized_refinement" if resized_refinement else "autozoom"] = {
                **binary_metrics(refined.mask, truth, valid),
                (
                    "frozen_forward_passes" if resized_refinement else "native_forward_passes"
                ): refined.native_forward_passes,
                "refinement_seconds": refined.refinement_seconds,
                "total_seconds": student_seconds + refined.refinement_seconds,
            }
        commit_task(progress, index, records[-1:])
        del result, images, prompts, truth, valid
    return records


@torch.no_grad()
def evaluate_interactive(model, sampler, seed=402026, rank=0, world_size=1, progress=None):
    """Fixed held-out prompted crops, identical before and after fine-tuning.

    Each target uses the same image crop across prompt types. Exact dense prompts
    are reported as a separate trivial-input control, excluded from retention means.
    """
    device = next(model.parameters()).device
    model.eval()
    protocols = [*PROMPT_TYPES, "perturbed_mask"]
    records = []
    for index in task_indices(len(sampler.targets) * len(protocols), rank, world_size, progress):
        target_index, prompt_index = divmod(index, len(protocols))
        target, protocol = sampler.targets[target_index], protocols[prompt_index]
        rng = episode_rng(seed, target_index)
        candidates = [c for c in sampler.queries[target] if target in sampler.manifest["cases"][c]["present"]]
        if not candidates:
            raise ValueError("interactive validation requires a positive held-out volume per target")
        case = str(rng.choice(candidates))
        (image, truth, valid), origin, _ = sampler._crop(case, target, "positive", rng)
        image, prompt_target, prompt_valid, _ = support_at_resolution(image, truth, valid, sampler.student_input_size)
        prompt = sampler.prompts.render(
            image,
            prompt_target,
            prompt_valid,
            "dense_mask" if protocol == "perturbed_mask" else protocol,
            episode_rng(seed + 1, index),
            perturb_dense=protocol == "perturbed_mask",
        )
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            logits = model.interactive(image[None].to(device), prompt[None].to(device))
        prediction = resize_volume(binary_logits(logits.float()), truth.shape[-3:])[0, 0].sigmoid().cpu().numpy() > 0.5
        import hashlib

        records.append(
            {
                "index": index,
                "query_id": case,
                "query_start": list(origin),
                "target": target,
                "prompt_type": protocol,
                "prompt_sha256": hashlib.sha256(prompt.numpy().tobytes()).hexdigest(),
                "interactive_eligible": sampler.prompts.support_loss_eligible(prompt, prompt_target, prompt_valid),
                **binary_metrics(prediction, truth[0].numpy(), valid[0].numpy()),
            }
        )
        commit_task(progress, index, records[-1:])
    return records


def summarize_interactive(records, baseline=None):
    eligible = [r for r in records if r["interactive_eligible"]]
    result = {
        "eligible": summarize(eligible),
        "all_protocols": summarize(records),
        "policy": "Held-out prompted patches; identical crop per target across protocols. Exact dense masks excluded from eligible retention mean.",
    }
    if baseline is not None:
        original = {r["index"]: r for r in baseline}
        deltas = []
        for row in records:
            before = original[row["index"]]
            for field in [
                "query_id",
                "query_start",
                "target",
                "prompt_type",
                "prompt_sha256",
                "interactive_eligible",
            ]:
                if row[field] != before[field]:
                    raise ValueError("interactive comparison requires identical original and fine-tuned protocols")
            if row["interactive_eligible"]:
                deltas.append({**row, "dice": row["dice"] - before["dice"]})
        result["paired_dice_change_from_pretrained"] = summarize(deltas)
        result["pretrained_eligible"] = summarize([r for r in baseline if r["interactive_eligible"]])
    return result
