"""Held-out paired-target controls: identical CT crops, changed support prompts only."""

from collections import defaultdict

import numpy as np
import torch

from .data import Episode, collate_episodes, foreground_window
from .evaluate import binary_metrics
from .losses import binary_logits
from .optimization import episode_rng
from .prompts import PROMPT_TYPES
from .spatial import full_query, resize_volume, support_at_resolution
from .validation_state import commit_task, task_indices

DEFAULT_PAIRS = (("kidney_left", "kidney_right"), ("adrenal_gland_left", "adrenal_gland_right"))


def paired_crop(store, case, targets, patch_size, fit_foreground=False, full_volume=False):
    """Reference-centered control patches only; never used by whole-volume inference."""
    volume = None
    if full_volume:
        volume = store.read_volume(case)
        origin, patch_size = (0, 0, 0), tuple(volume[0].shape[-3:])
    elif fit_foreground:
        volume = store.read_volume(case)
        labels = volume[1][0]
        union = np.isin(labels, [v for target in targets for v in store.manifest["regions"][target]])
        union &= (labels >= 0) & (labels != store.manifest.get("ignore_label", -1))
        origin, patch_size = foreground_window(union, patch_size)
    else:
        centers = [np.median(store.foreground_locations(case, target), axis=0) for target in targets]
        origin = tuple(int(c - p // 2) for c, p in zip(np.mean(centers, axis=0), patch_size))
    crops = [store.crop(case, origin, patch_size, target, volume=volume) for target in targets]
    if not all(bool(crop[1].any()) for crop in crops):
        return None
    return origin, crops


def make_swap_episodes(sampler, targets, prompt_type, rng):
    """Return both target definitions on exactly the same query and support images."""
    if len(targets) != 2 or targets[0] == targets[1]:
        raise ValueError("swap requires two different targets")
    if any(target not in sampler.targets for target in targets):
        raise ValueError("swap targets must be active targets")
    if set(sampler.manifest["regions"][targets[0]]) & set(sampler.manifest["regions"][targets[1]]):
        raise ValueError("use disjoint target definitions for an interpretable swap control")
    selected = {}
    for split, count in [("val", 1), ("train", sampler.config.support_count)]:
        eligible = [
            case
            for case in sampler.manifest["splits"][split]
            if all(target in sampler.manifest["cases"][case]["present"] for target in targets)
        ]
        chosen = []
        for index in rng.permutation(len(eligible))[:64]:
            case = eligible[int(index)]
            pair = paired_crop(
                sampler.store,
                case,
                targets,
                sampler.patch_size,
                fit_foreground=split == "train",
                full_volume=sampler.full_volume,
            )
            if pair is not None:
                chosen.append((case, *pair))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            return None, f"fewer than {count} eligible {split} volumes with both targets inside a control patch"
        selected[split] = chosen
    query, query_start, query_crops = selected["val"][0]
    episodes = []
    for target_index, target in enumerate(targets):
        images, prompts, support_records, contexts = [], [], [], []
        support_targets, support_valid, eligible = [], [], []
        for case, origin, crops in selected["train"]:
            image, mask, valid = crops[target_index]
            contexts.append(image)
            student_image, prompt_target, prompt_valid, _ = support_at_resolution(
                image, mask, valid, sampler.student_input_size
            )
            images.append(student_image)
            prompts.append(sampler.prompts.render(student_image, prompt_target, prompt_valid, prompt_type, rng))
            support_targets.append(mask)
            support_valid.append(valid)
            eligible.append(sampler.prompts.support_loss_eligible(prompts[-1], prompt_target, prompt_valid))
            support_records.append(
                {
                    "volume_id": case,
                    "start": list(origin),
                    "prompt_type": prompt_type,
                    "crop_size": list(mask.shape[-3:]),
                    "crop_policy": sampler.crop_policy,
                }
            )
        image, mask, valid = query_crops[target_index]
        episodes.append(
            Episode(
                resize_volume(image[None], sampler.student_input_size)[0],
                torch.stack(images),
                torch.stack(prompts),
                mask,
                valid,
                {
                    "query_id": query,
                    "query_start": list(query_start),
                    "target": target,
                    "supports": support_records,
                    "support_patch_count": len(support_records),
                    "support_unique_volume_count": len({r["volume_id"] for r in support_records}),
                },
                support_targets,
                support_valid,
                torch.tensor(eligible, dtype=torch.bool),
                image if sampler.resized_context else None,
                contexts,
            )
        )
    assert torch.equal(episodes[0].query_images, episodes[1].query_images)
    assert torch.equal(episodes[0].support_images, episodes[1].support_images)
    assert torch.equal(episodes[0].query_valid, episodes[1].query_valid)
    return episodes, None


@torch.no_grad()
def evaluate_prompt_swaps(model, sampler, pairs, seed=302026, rank=0, world_size=1, progress=None, refiner=None):
    device = next(model.parameters()).device
    model.eval()
    tasks = [(pair, prompt) for pair in pairs for prompt in PROMPT_TYPES]
    records = []
    for index in task_indices(len(tasks), rank, world_size, progress):
        pair, prompt = tasks[index]
        episodes, reason = make_swap_episodes(sampler, pair, prompt, episode_rng(seed, index))
        if episodes is None:
            records.append({"index": 2 * index, "pair": pair, "prompt_type": prompt, "skipped": reason})
            commit_task(progress, index, records[-1:])
            continue
        predictions, refined_predictions = [], []
        for episode in episodes:
            batch = collate_episodes([episode]).to(device)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()
            ):
                logits = binary_logits(
                    model.predict_query(batch.query_images, batch.support_images, batch.support_prompts)
                )
            predictions.append(
                resize_volume(logits.float(), episode.query_targets.shape[-3:])[0, 0].sigmoid().cpu().numpy() > 0.5
            )
            if refiner is not None:
                size = model.config.teacher_input_size
                refined = refiner.predict_logits(
                    resize_volume(full_query(batch), size), resize_volume(logits.float(), size).sigmoid()
                )
                refined_predictions.append(
                    resize_volume(binary_logits(refined.float()), episode.query_targets.shape[-3:])[0, 0]
                    .sigmoid()
                    .cpu()
                    .numpy()
                    > 0.5
                )
            del batch, logits
        for direction, original in enumerate(episodes):
            replacement = episodes[1 - direction]
            valid = original.query_valid[0].numpy()
            original_truth = original.query_targets[0].numpy()
            replacement_truth = replacement.query_targets[0].numpy()
            records.append(
                {
                    "index": 2 * index + direction,
                    "prompt_type": prompt,
                    **original.metadata,
                    "replacement_target": replacement.metadata["target"],
                    "correct_prompt": binary_metrics(predictions[direction], original_truth, valid),
                    "swapped_original_target": binary_metrics(predictions[1 - direction], original_truth, valid),
                    "swapped_replacement_target": binary_metrics(predictions[1 - direction], replacement_truth, valid),
                    "prediction_changed_voxels": int(((predictions[0] != predictions[1]) & valid).sum()),
                    "skipped": None,
                }
            )
            for prefix, masks in [("coarse", predictions), ("refined", refined_predictions)]:
                if not masks:
                    continue
                prediction = masks[direction] & valid
                requested = original_truth.astype(bool) & valid
                opposite = replacement_truth.astype(bool) & valid
                own = binary_metrics(prediction, requested, valid)
                other = binary_metrics(prediction, opposite, valid)
                own_coverage = float((prediction & requested).sum() / max(1, requested.sum()))
                opposite_coverage = float((prediction & opposite).sum() / max(1, opposite.sum()))
                records[-1][prefix + "_laterality"] = {
                    "requested_dice": own["dice"],
                    "contralateral_dice": other["dice"],
                    "contralateral_false_positive_voxels": int((prediction & opposite).sum()),
                    "contralateral_coverage": opposite_coverage,
                    "both_structures_coverage_above_10_percent": own_coverage > 0.1 and opposite_coverage > 0.1,
                    "requested_target_preferred": own["dice"] > other["dice"],
                }
        commit_task(progress, index, records[-2:])
    return records


def summarize_swaps(records):
    groups = defaultdict(list)
    valid = [r for r in records if not r["skipped"]]
    for row in valid:
        groups[row["prompt_type"]].append(row)

    def scores(rows):
        result = {
            key + "_mean_dice": float(np.mean([r[key]["dice"] for r in rows])) if rows else None
            for key in ["correct_prompt", "swapped_original_target", "swapped_replacement_target"]
        }
        for prefix in ("coarse", "refined"):
            values = [r[prefix + "_laterality"] for r in rows if prefix + "_laterality" in r]
            if values:
                result[prefix + "_laterality"] = {key: float(np.mean([v[key] for v in values])) for key in values[0]}
        return result

    return {
        "controls": len(valid),
        "skipped_tasks": sum(bool(r["skipped"]) for r in records),
        **scores(valid),
        "by_prompt_type": {k: scores(v) for k, v in groups.items()},
        "policy": "Fixed local control crops containing two disjoint targets. Images stay identical; only support prompt channels change. Separate from full-volume checkpoint selection.",
    }
