import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model


def make_optimizer(model, config):
    groups = {}
    frozen_encoder = (
        {id(p) for p in model.backbone.encoder.parameters()}
        if getattr(model, "mask_encoder", None) is not None
        else set()
    )
    for name, parameter in model.named_parameters():
        if name.startswith("support_segmenter.") or id(parameter) in frozen_encoder:
            if parameter.requires_grad:
                raise ValueError(f"frozen model parameter became trainable: {name}")
            continue

        if not parameter.requires_grad:
            raise ValueError(f"active student parameter is frozen: {name}")

        branch = "context" if name.startswith(("context.", "mask_encoder.")) else "backbone"
        decay = not (parameter.ndim <= 1 or name.endswith(".bias"))
        key = (branch, decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": config.context_lr if branch == "context" else config.backbone_lr,
                "weight_decay": config.weight_decay if decay else 0.0,
                "group_name": branch + ("_decay" if decay else "_no_decay"),
            }
        groups[key]["params"].append(parameter)
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999))


def lr_factor(completed_updates, config):
    next_update = completed_updates + 1
    if next_update <= config.warmup_updates:
        return next_update / max(1, config.warmup_updates)
    fraction = min(
        1.0,
        (next_update - config.warmup_updates) / max(1, config.updates - config.warmup_updates),
    )
    return config.final_lr_fraction + (1 - config.final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * fraction))


def make_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda completed: lr_factor(completed, config))


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def episode_rng(seed, global_episode_index, attempt=0):
    return np.random.default_rng(np.random.SeedSequence([seed, global_episode_index, attempt]))


def best_validation_score(state, config):
    """Reset selection when switching metrics or changing the sampled patch set."""
    if state.get("selection_metric", "full_volume") != config.selection_metric:
        return None
    if state["train_config"].get("validation_targets") != config.validation_targets:
        return None
    if config.selection_metric == "query_patch" and (
        state["train_config"].get("validation_patches_per_target") != config.validation_patches_per_target
    ):
        return None
    return state.get("best_validation_dice", state.get("best_full_volume_macro_dice"))


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    update,
    best,
    train_config,
    source,
    manifests,
    rank_rng_states,
    pending_validation=None,
):
    state = {
        "format": "prompt_encoded_icl3d_v2",
        "model": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "completed_updates": update,
        "next_global_episode_index": update * train_config.effective_batch,
        "selection_metric": train_config.selection_metric,
        "best_validation_dice": best,
        "best_full_volume_macro_dice": best if train_config.selection_metric == "full_volume" else None,
        "train_config": asdict(train_config),
        "source": source,
        "manifests": manifests,
        "rank_rng_states": rank_rng_states,
        "world_size": len(rank_rng_states),
        "pending_validation": pending_validation,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    model, state = load_model(path, device)
    if state.get("format") != "prompt_encoded_icl3d_v2":
        raise ValueError("incompatible checkpoint format")
    return model, state
