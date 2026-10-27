import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .spatial import resize_volume


def segmentation_loss(logits, targets, valid, epsilon=1e-5, reduction="mean"):
    if logits.shape[:2] == targets.shape[:2] and logits.shape[2:] != targets.shape[2:]:
        # Recompute each whole-context loss during backward; retain global Dice, not tilewise Dice.
        def full_loss(z, y, known):
            loss, parts = segmentation_loss(resize_volume(z, y.shape[-3:]), y, known, epsilon, "none")
            return torch.stack((loss, parts["bce"], parts["dice_loss"]), 1)

        rows = []
        for z, y, known in zip(logits.split(1), targets.split(1), valid.split(1)):
            rows.append(
                checkpoint(full_loss, z, y, known, use_reentrant=False)
                if torch.is_grad_enabled() and z.requires_grad
                else full_loss(z, y, known)
            )
        values = torch.cat(rows)
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be mean or none")
        return (values[:, 0].mean() if reduction == "mean" else values[:, 0]), {
            "loss_per_episode": values[:, 0].detach(),
            "bce": values[:, 1].detach(),
            "dice_loss": values[:, 2].detach(),
        }
    if logits.shape != targets.shape or valid.shape != targets.shape:
        raise ValueError("logits, targets, and valid must have identical B1DHW shapes")
    if logits.ndim != 5 or logits.shape[1] != 1 or valid.dtype != torch.bool:
        raise ValueError("expected B1DHW logits/targets and boolean validity")
    z, y = logits.float(), targets.float()
    counts = valid.flatten(1).sum(1)
    if not bool((counts > 0).all()):
        raise ValueError("reject episode with zero valid supervised voxels")
    # Mask before BCE so ignored NaN/sentinel targets cannot poison reductions.
    y = torch.where(valid, y, 0.0)
    z = torch.where(valid, z, 0.0)
    if bool(((y[valid] < 0) | (y[valid] > 1)).any()):
        raise ValueError("valid targets must lie in [0,1]")
    voxel_bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    foreground, background = valid & (y > 0.5), valid & (y <= 0.5)
    nfg, nbg = foreground.flatten(1).sum(1), background.flatten(1).sum(1)
    fg = (voxel_bce * foreground).flatten(1).sum(1) / nfg.clamp_min(1)
    bg = (voxel_bce * background).flatten(1).sum(1) / nbg.clamp_min(1)
    bce = (fg + bg) / ((nfg > 0).float() + (nbg > 0).float())
    probability = z.sigmoid() * valid
    intersection = (probability * y).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + (y * valid).flatten(1).sum(1)
    dice = torch.where(nfg > 0, 1 - (2 * intersection + epsilon) / (denominator + epsilon), 0.0)
    per_episode = bce + dice
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be mean or none")
    return (per_episode.mean() if reduction == "mean" else per_episode), {
        "loss_per_episode": per_episode.detach(),
        "bce": bce.detach(),
        "dice_loss": dice.detach(),
    }


def binary_logits(logits, channel_dim=1):
    """Foreground minus background; sigmoid equals foreground softmax probability."""
    if logits.shape[channel_dim] != 2:
        raise ValueError("expected the pretrained two-channel segmentation head")
    return logits.narrow(channel_dim, 1, 1) - logits.narrow(channel_dim, 0, 1)


@torch.no_grad()
def training_dice_totals(outputs, batch):
    """Hard Dice sums/counts for positive queries and eligible positive supports."""
    totals = []
    paths = [(outputs["query_logits"], batch.query_targets, batch.query_valid, None)]
    if outputs["support_logits"] is not None:
        paths.append(
            (
                outputs["support_logits"].flatten(0, 1),
                batch.support_targets.flatten(0, 1),
                batch.support_valid.flatten(0, 1),
                batch.support_eligible.flatten(),
            )
        )
    for logits, targets, valid, eligible in paths:
        prediction = (resize_volume(binary_logits(logits.float()), targets.shape[-3:])[:, 0] > 0) & valid[:, 0]
        truth = (targets[:, 0] > 0.5) & valid[:, 0]
        positives = truth.flatten(1).sum(1)
        included = positives > 0
        if eligible is not None:
            included &= eligible
        denominator = prediction.flatten(1).sum(1) + positives
        dice = 2 * (prediction & truth).flatten(1).sum(1) / denominator.clamp_min(1)
        totals.extend(((dice * included).sum(), included.sum()))
    if len(paths) == 1:
        totals.extend((totals[0].new_zeros(()), totals[0].new_zeros(())))
    return torch.stack(totals)


def joint_segmentation_loss(outputs, batch, config):
    predictions = (outputs["query_logits"], *outputs.get("query_aux_logits", ()))
    query_losses = torch.stack(
        [
            segmentation_loss(binary_logits(z), batch.query_targets, batch.query_valid, reduction="none")[0]
            for z in predictions
        ]
    )
    weights = query_losses.new_tensor([2.0**-i for i in range(len(predictions))])
    query = (query_losses * weights[:, None]).sum(0) / weights.sum()
    head_losses = query_losses.detach().mean(1)
    if not config.supervise_supports:
        return query.mean() * float(config.supervise_query), {
            "query_loss": query.detach().mean(),
            "query_head_losses": head_losses,
            "support_loss": query.new_zeros(()),
            "support_loss_per_episode": torch.zeros_like(query),
            "eligible_supports_per_episode": torch.zeros_like(query, dtype=torch.long),
        }
    supports = outputs["support_logits"]
    batch_size, count = supports.shape[:2]
    if batch.support_eligible.shape != (batch_size, count) or batch.support_eligible.dtype != torch.bool:
        raise ValueError("support eligibility must be a boolean B,K tensor")
    eligible = batch.support_eligible.flatten()
    # Keep a zero gradient edge to support outputs when all dense masks are exact.
    # Only eligible items enter BCE/Dice, so excluded items cannot affect normalization.
    per_episode = supports[:, :, :, 0, 0, 0].float().sum((1, 2)) * 0
    counts = batch.support_eligible.sum(1)
    if bool(eligible.any()):
        selected, _ = segmentation_loss(
            binary_logits(supports.flatten(0, 1)[eligible]),
            batch.support_targets.flatten(0, 1)[eligible],
            batch.support_valid.flatten(0, 1)[eligible],
            reduction="none",
        )
        episode_ids = torch.arange(batch_size, device=supports.device).repeat_interleave(count)[eligible]
        per_episode = per_episode.scatter_add(0, episode_ids, selected) / counts.clamp_min(1)
    total = query * float(config.supervise_query)
    total = total + config.lambda_support * float(config.supervise_supports) * per_episode
    return total.mean(), {
        "query_loss": query.detach().mean(),
        "query_head_losses": head_losses,
        "support_loss": per_episode.detach().mean(),
        "support_loss_per_episode": per_episode.detach(),
        "eligible_supports_per_episode": counts.detach(),
    }
