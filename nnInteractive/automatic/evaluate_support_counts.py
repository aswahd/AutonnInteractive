"""Controlled nested subsets of one fixed support bank for each validation query."""

import hashlib

import torch

from .data import collate_episodes
from .evaluate import binary_metrics, summarize
from .losses import binary_logits, segmentation_loss
from .optimization import episode_rng
from .prompts import PROMPT_TYPES
from .spatial import resize_volume
from .validation_state import commit_task, task_indices


def summarize_support_counts(records):
    by_count = {
        str(k): summarize([r for r in records if r["active_support_count"] == k])
        for k in range(1, 5)
    }
    return {
        "by_support_count": by_count,
        **{
            f"val/query_dice_k{k}": by_count[str(k)]["positive_macro_dice"]
            for k in range(1, 5)
        },
        "policy": "Fixed four-support bank, one seeded permutation per episode, nested subsets. Patch evaluation; no monotonicity assumption. One ordering does not quantify support-selection uncertainty.",
    }


@torch.no_grad()
def evaluate_support_counts(
    model, sampler, seed=102026, rank=0, world_size=1, progress=None
):
    if sampler.config.support_count != 4:
        raise ValueError("controlled count validation requires a bank of four supports")
    device = next(model.parameters()).device
    model.eval()
    tasks = [
        (t, p, k)
        for t in sampler.targets
        for p in PROMPT_TYPES
        for k in ["positive", "hard_negative"]
    ]
    records = []
    for index in task_indices(len(tasks), rank, world_size, progress):
        target, prompt, kind = tasks[index]
        # Match the existing patch-validation bank exactly; subset RNG is separate.
        episode = sampler.sample_episode(
            episode_rng(seed, index), target=target, query_kind=kind, prompt_type=prompt
        )
        batch = collate_episodes([episode]).to(device)
        permutation = episode_rng(seed + 400000, index).permutation(4).tolist()
        masks = []
        for count in range(1, 5):
            mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
            mask[:, permutation[:count]] = True
            masks.append(mask)
        hashes = {
            name: hashlib.sha256(
                getattr(episode, name).contiguous().numpy().tobytes()
            ).hexdigest()
            for name in [
                "query_images",
                "support_images",
                "support_prompts",
                "query_targets",
                "query_valid",
            ]
        }
        rows = []
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            predictions = model.iter_query_subsets(
                batch.query_images, batch.support_images, batch.support_prompts, masks
            )
            for count, output in enumerate(predictions, 1):
                logits = binary_logits(output)
                loss, _ = segmentation_loss(
                    logits, batch.query_targets, batch.query_valid
                )
                logits = resize_volume(logits.float(), batch.query_targets.shape[-3:])
                metrics = binary_metrics(
                    logits[0, 0].float().sigmoid().cpu().numpy() > 0.5,
                    episode.query_targets[0].numpy(),
                    episode.query_valid[0].numpy(),
                )
                rows.append(
                    {
                        "index": 4 * index + count - 1,
                        "episode_index": index,
                        **episode.metadata,
                        "prompt_type": prompt,
                        "active_support_count": count,
                        "active_support_indices": permutation[:count],
                        "support_permutation": permutation,
                        "fixed_input_sha256": hashes,
                        "loss": float(loss),
                        **metrics,
                    }
                )
        records.extend(rows)
        commit_task(progress, index, rows)
        del batch, predictions, output, logits, episode
    return records
