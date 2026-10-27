import json
from pathlib import Path

import torch

from .backbone import sha256
from .config import TrainConfig
from .optimization import best_validation_score


def bind_transfer_protocol(specification, manifest):
    if manifest.get("evaluation_only"):
        raise ValueError("inference-only support volumes cannot be optimizer training data")
    if "final_evaluation_manifest" not in specification:
        return {}
    paths = {key: Path(specification[key]) for key in ("cohort_policy_file", "final_evaluation_manifest")}
    policy, final = (json.loads(path.read_text()) for path in paths.values())
    seen, held_out = policy["training_targets"], policy["held_out_targets"]
    regions = policy["target_regions"]
    pools = [
        policy[key]
        for key in (
            "training_volumes",
            "seen_target_development_query_volumes",
            "inference_support_volumes",
            "final_query_volumes",
        )
    ]
    if (
        not seen
        or not held_out
        or len(set(seen + held_out)) != len(seen) + len(held_out)
        or set(seen + held_out) != set(regions)
    ):
        raise ValueError("target-transfer protocol requires nonempty, disjoint targets covering all regions")
    if set().union(*(set(regions[t]) for t in seen)) & set().union(*(set(regions[t]) for t in held_out)):
        raise ValueError("source labels overlap across seen and held-out target families")
    if any(len(ids) != len(set(ids)) for ids in pools) or any(
        set(a) & set(b) for i, a in enumerate(pools) for b in pools[i + 1 :]
    ):
        raise ValueError("training, development, support, and final query volumes must be disjoint")
    if specification["targets"] != seen or specification["full_volume_query_ids"] != pools[1]:
        raise ValueError("only seen targets and development queries may select checkpoints")
    for data, expected in ((manifest, pools[:2]), (final, pools[2:])):
        if data["regions"] != regions or any(
            set(data["splits"][key]) != set(ids) for key, ids in zip(("train", "val"), expected)
        ):
            raise ValueError("data manifest does not match the frozen cohort policy")
    if not final.get("evaluation_only") or manifest["source_sha256"] != final["source_sha256"]:
        raise ValueError("final manifest must be inference-only and use the same source dataset")
    if specification.get("full_volume_prompt_type") != "dense_mask":
        raise ValueError("primary target-transfer evaluation requires dense support prompts")
    return {
        field: value
        for key, path in paths.items()
        for field, value in ((key, str(path.resolve())), (key + "_sha256", sha256(path)))
    }


def final_evaluation_policy(output, selected):
    latest = torch.load(Path(output) / "last.pt", map_location="cpu", weights_only=False)
    if latest["completed_updates"] != latest["train_config"]["updates"] or latest.get("pending_validation"):
        raise ValueError("finish training and development selection before opening the final test")
    for key in ("manifests", "source"):
        if latest[key] != selected[key]:
            raise ValueError("selected checkpoint does not match the completed training run")
    config = TrainConfig(**latest["train_config"])
    if TrainConfig(**selected["train_config"]).resume_settings() != config.resume_settings():
        raise ValueError("selected checkpoint does not match the completed training run")
    # Older checkpoints always selected on full volumes, even when the field was omitted.
    if "selection_metric" not in latest:
        config.full_validation_every = 5000
    best = best_validation_score(selected, config)
    if selected.get("pending_validation") or best is None or best != best_validation_score(latest, config):
        raise ValueError("final evaluation requires a checkpoint selected by completed development evaluation")
    policy = dict(selected["manifests"])
    for key in ("cohort_policy_file", "final_evaluation_manifest"):
        if sha256(policy[key]) != policy[key + "_sha256"]:
            raise ValueError("target-transfer protocol changed after training began")
    cohort = json.loads(Path(policy["cohort_policy_file"]).read_text())
    final = json.loads(Path(policy["final_evaluation_manifest"]).read_text())
    if not final.get("evaluation_only") or policy["targets"] != cohort["training_targets"]:
        raise ValueError("checkpoint is not from the prescribed seen-target training run")
    policy.update(
        data_manifest=policy["final_evaluation_manifest"],
        data_sha256=policy["final_evaluation_manifest_sha256"],
        targets=cohort["training_targets"] + cohort["held_out_targets"],
        full_volume_query_ids=cohort["final_query_volumes"],
    )
    return policy, cohort
