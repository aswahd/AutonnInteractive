from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoint import build_model, initialize_backbone, load_model
from .data import Episodes, MixedEpisodes, region_mask
from .prompts import PROMPT_TYPES, add_error_correction, simulate_prompt


def loss_function(logits, target, valid, balance="class"):
    valid = valid.float()
    dims = tuple(range(1, target.ndim))
    terms = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    if balance == "voxel":
        bce = (terms * valid).sum() / valid.sum().clamp_min(1)
    elif balance == "class":
        foreground, background = target * valid, (1 - target) * valid
        positives, negatives = foreground.sum(dims), background.sum(dims)
        positive_loss = (terms * foreground).sum(dims) / positives.clamp_min(1)
        negative_loss = (terms * background).sum(dims) / negatives.clamp_min(1)
        present_classes = (positives > 0).float() + (negatives > 0).float()
        bce = ((positive_loss + negative_loss) / present_classes.clamp_min(1)).mean()
    else:
        raise ValueError("BCE balance must be class or voxel")
    probability = logits.float().sigmoid() * valid
    target = target * valid
    dice = (2 * (probability * target).sum(dims) + 1e-5) / (probability.sum(dims) + target.sum(dims) + 1e-5)
    return bce + 1 - dice.mean()


def score(logits, target, valid):
    prediction = (logits > 0) & valid
    truth = (target > 0.5) & valid
    intersection = int((prediction & truth).sum())
    predicted, actual = int(prediction.sum()), int(truth.sum())
    return {
        "dice": 2 * intersection / (predicted + actual) if predicted + actual else 1.0,
        "iou": intersection / (predicted + actual - intersection) if predicted + actual else 1.0,
        "target_voxels": actual,
        "predicted_voxels": predicted,
        "false_positive_fraction": (predicted - intersection) / max(1, int(valid.sum() - truth.sum())),
    }


def summarize(records):
    result = {}
    for control in sorted({r["control"] for r in records}):
        rows = [r for r in records if r["control"] == control]
        present = [r for r in rows if r["target_voxels"] > 0]
        absent = [r for r in rows if r["target_voxels"] == 0]
        per_region = defaultdict(list)
        for row in present:
            per_region[row["region"]].append(row["dice"])
        result[control] = {
            "positive_macro_dice": float(np.mean([np.mean(v) for v in per_region.values()])) if present else None,
            "all_episode_mean_dice": float(np.mean([r["dice"] for r in rows])),
            "per_region_positive_dice": {k: float(np.mean(v)) for k, v in per_region.items()},
            "positive_episodes": len(present),
            "absent_episodes": len(absent),
            "absent_mean_predicted_voxels": float(np.mean([r["predicted_voxels"] for r in absent])) if absent else None,
        }
    return result


@torch.inference_mode()
def evaluate(model, dataset, device, controls=("correct",)):
    model.eval()
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        query = item["query"][None].to(device)
        images = item["supports"][None].to(device)
        masks = item["support_masks"][None].to(device)
        target, valid = item["target"][None].to(device), item["valid"][None].to(device)
        for control in controls:
            support_masks = masks
            alternative = None
            if control == "empty":
                support_masks = torch.where(masks < 0, -1.0, 0.0)
            elif control == "wrong":
                # Rotate requested regions, keeping the support images and query identical.
                alternative = dataset.regions[(dataset.regions.index(item["region"]) + 1) % len(dataset.regions)]
                values = dataset.store.manifest["regions"][alternative]
                support_masks = torch.stack(
                    [
                        torch.where(
                            segmentation >= 0,
                            region_mask(segmentation, values),
                            -1.0,
                        )
                        for segmentation in item["support_segs"]
                    ]
                )[None].to(device)
            elif control not in {"correct", "mask_average"}:
                raise ValueError(f"unknown control {control}")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = (
                    (support_masks.clamp_min(0).mean(1) - 0.5)
                    if control == "mask_average"
                    else model(query, images, support_masks)
                )
            record = {
                "query_id": item["query_id"],
                "support_ids": item["support_ids"],
                "region": item["region"],
                "control": control,
                "alternative_region": alternative,
                "positive_supports": int((support_masks > 0).flatten(2).any(2).sum()),
                **score(logits, target, valid),
            }
            if alternative is not None:
                alternative_target = region_mask(
                    item["query_seg"],
                    dataset.store.manifest["regions"][alternative],
                )[
                    None
                ].to(device)
                record["alternative_target_dice"] = score(logits, alternative_target, valid)["dice"]
            records.append(record)
    return records, summarize(records)


def save_checkpoint(path, model, optimizer, step, best, config, args, generator, provenance):
    state = {
        "format_version": 1,
        "model_config": config,
        "model": model.state_dict(),
        "input_shape": (
            args.input_shape
            if args.data_manifest
            else json.loads((Path(args.cache) / "manifest.json").read_text())["shape"]
        ),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best": best,
        "arguments": vars(args),
        "provenance": provenance,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_rng": generator.get_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(args):
    args.data_manifest = getattr(args, "data_manifest", None)
    args.crop_shape = getattr(args, "crop_shape", [192, 384, 384])
    args.input_shape = getattr(args, "input_shape", [96, 96, 96])
    micro_batch = getattr(args, "micro_batch_size", None)
    micro_batch = args.batch_size if micro_batch is None else micro_batch
    if micro_batch < 1 or args.batch_size % micro_batch:
        raise ValueError("micro-batch-size must be positive and divide batch-size")
    accumulation = args.batch_size // micro_batch
    store = None
    if args.data_manifest:
        from .online import OnlineVolumeStore

        store = OnlineVolumeStore(args.data_manifest, args.crop_shape, args.input_shape)
    if args.backbone_learning_rate is None:
        args.backbone_learning_rate = 1e-5 if args.backbone_model_dir else args.learning_rate

    def objective(logits, target, valid):
        return loss_function(logits, target, valid, balance=args.bce_balance)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("run already exists; specify --resume or a new output directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    requested_stages = sorted(getattr(args, "attention_stages", [-1]), reverse=True)
    config = {
        "width": args.width,
        "mode": args.mode,
        "memory_grid": args.memory_grid,
        "backbone": None,
        "attention_stages": requested_stages,
        "prototype_tokens": getattr(args, "prototype_tokens", False),
    }
    if args.resume:
        model, checkpoint = load_model(args.resume, device)
        config = checkpoint["model_config"]
        if config.get("prototype_tokens", False) != getattr(args, "prototype_tokens", False):
            raise ValueError("resume configuration mismatch: prototype_tokens")
        if sorted(config.get("attention_stages", [-1]), reverse=True) != requested_stages:
            raise ValueError("resume configuration mismatch: attention_stages")
        for name in (
            "cache",
            "data_manifest",
            "crop_shape",
            "input_shape",
            "regions",
            "shots",
            "seed",
            "batch_size",
            "learning_rate",
            "backbone_learning_rate",
            "mode",
            "width",
            "memory_grid",
            "interactive_probability",
            "bce_balance",
            "selection",
        ):
            if checkpoint["arguments"].get(name, vars(args)[name]) != vars(args)[name]:
                raise ValueError(f"resume configuration mismatch: {name}")
    elif args.backbone_model_dir:
        model = initialize_backbone(
            config,
            args.backbone_model_dir,
            args.backbone_fold,
            args.backbone_checkpoint,
        ).to(device)
    else:
        model = build_model(config).to(device)
    dataset = MixedEpisodes(
        args.cache,
        "train",
        args.regions,
        args.shots,
        args.steps * args.batch_size,
        args.seed,
        True,
        store=store,
    )
    validation = Episodes(args.cache, "val", args.regions, args.shots, seed=args.seed + 100000, store=store)
    torch.set_num_threads(args.threads)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.backbone.parameters()),
                "lr": args.backbone_learning_rate,
            },
            {
                "params": [p for name, p in model.named_parameters() if not name.startswith("backbone.")],
                "lr": args.learning_rate,
            },
        ],
        weight_decay=1e-4,
    )
    start, best = 0, -1.0
    generator = torch.Generator().manual_seed(args.seed)
    if args.resume:
        start, best = checkpoint["step"], checkpoint["best"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        generator.set_state(checkpoint["loader_rng"])
    if args.resume and not (output / "best.pt").exists():
        shutil.copyfile(Path(args.resume).with_name("best.pt"), output / "best.pt")
    if start >= args.steps:
        raise ValueError("--steps must exceed the resumed step")
    # Per-index sampling permits continuation without replaying earlier data loading.
    sampler = range(start * args.batch_size, args.steps * args.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        sampler=sampler,
        num_workers=args.workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    git = {"git_commit": None, "git_status": None}
    source_root = Path(__file__).resolve().parents[3]
    if (source_root / ".git").exists():
        try:
            for key, command in (
                ("git_commit", ["rev-parse", "HEAD"]),
                ("git_status", ["status", "--short"]),
            ):
                git[key] = subprocess.check_output(["git", *command], cwd=source_root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass  # Installed packages retain source hashes without requiring Git.
    provenance = {
        "arguments": vars(args),
        "model_config": config,
        "torch": str(torch.__version__),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "manifest_sha256": hashlib.sha256(
            (Path(args.data_manifest) if args.data_manifest else Path(args.cache) / "manifest.json").read_bytes()
        ).hexdigest(),
        **git,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
        },
    }
    (output / f"provenance_step_{start}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    start_time = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    update_loss = 0.0
    with (output / "training.jsonl").open("a") as log:
        for micro_step, batch in enumerate(loader, start=start * accumulation + 1):
            step = (micro_step - 1) // accumulation + 1
            model.train()
            query, supports, masks, target, valid = [
                batch[k].to(device, non_blocking=True)
                for k in ("query", "supports", "support_masks", "target", "valid")
            ]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                local_task = np.random.default_rng(args.seed + step).random() < args.interactive_probability
                if local_task:
                    prompts = batch["query_prompts"].to(device)
                    logits = model(query, query_prompts=prompts)
                    revised = add_error_correction(prompts, logits.float().sigmoid(), target, valid)
                    corrected = model(query, query_prompts=revised)
                    loss = 0.5 * (objective(logits, target, valid) + objective(corrected, target, valid))
                else:
                    support_prompts = batch["support_prompts"].to(device)
                    dense = batch["dense_support"].to(device)
                    supplied = torch.where(dense[:, :, None, None, None, None], masks, 0.0)
                    memory_masks, support_logits = model.unify_context(supports, support_prompts, supplied, dense)
                    logits = model(query, supports, memory_masks)
                    loss = objective(logits, target, valid)
                    loss = loss + 0.5 * objective(
                        support_logits.flatten(0, 1),
                        masks.clamp_min(0).flatten(0, 1),
                        (masks >= 0).flatten(0, 1),
                    )
                    if step % 3 == 0:
                        empty_prompts = query.new_zeros(query.shape[0], 7, *query.shape[-3:])
                        correction = add_error_correction(empty_prompts, logits.float().sigmoid(), target, valid)
                        corrected = model(query, supports, memory_masks, query_prompts=correction)
                        loss = loss + 0.5 * objective(corrected, target, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            (loss / accumulation).backward()
            update_loss += float(loss.detach()) / accumulation
            if micro_step % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % args.log_every == 0:
                row = {
                    "step": step,
                    "loss": update_loss,
                    "task": "interactive" if local_task else "context",
                    "elapsed_seconds": time.monotonic() - start_time,
                }
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
            update_loss = 0.0
            if step % args.validate_every == 0 or step == args.steps:
                _, summary = evaluate(model, validation, device)
                context_metric = summary["correct"]["positive_macro_dice"]
                local_records = evaluate_local_probe(model, validation, device)
                local_summary = summarize(local_records)
                local_metric = local_summary["interactive_probe"]["positive_macro_dice"]
                metric = (context_metric + local_metric) / 2 if args.selection == "joint" else context_metric
                if metric is None:
                    raise ValueError("validation has no positive targets")
                row = {
                    "step": step,
                    "validation": summary,
                    "interactive_validation": local_summary,
                    "selection_score": metric,
                }
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if metric > best:
                    best = metric
                    save_checkpoint(
                        output / "best.pt",
                        model,
                        optimizer,
                        step,
                        best,
                        config,
                        args,
                        generator,
                        provenance,
                    )
                save_checkpoint(
                    output / "last.pt",
                    model,
                    optimizer,
                    step,
                    best,
                    config,
                    args,
                    generator,
                    provenance,
                )
    model, _ = load_model(output / "best.pt", device)
    records, summary = evaluate(model, validation, device, ("correct", "empty", "wrong", "mask_average"))
    (output / "validation_records.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"final_context_controls": summary}), flush=True)
    prompt_records, prompt_summary = evaluate_prompts(model, validation, device)
    (output / "prompt_records.json").write_text(json.dumps(prompt_records, indent=2) + "\n")
    (output / "prompt_summary.json").write_text(json.dumps(prompt_summary, indent=2) + "\n")
    print(json.dumps({"final_prompt_controls": prompt_summary}), flush=True)


def evaluate_checkpoint(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    model, checkpoint = load_model(args.checkpoint, device)
    regions = args.regions if args.regions else checkpoint["arguments"]["regions"]
    previous = checkpoint.get("arguments", {})
    manifest = args.data_manifest or previous.get("data_manifest")
    store = None
    if manifest:
        from .online import OnlineVolumeStore

        if args.retrieval:
            raise ValueError("online coarse episodes use random supports, not retrieval")
        store = OnlineVolumeStore(manifest, previous.get("crop_shape", (192, 384, 384)), checkpoint["input_shape"])
    dataset = Episodes(
        args.cache or previous.get("cache"),
        "val",
        regions,
        args.shots,
        seed=args.seed,
        retrieval=args.retrieval,
        store=store,
    )
    if dataset.retriever is not None:
        from .retrieval import file_sha256

        if dataset.retriever.checkpoint_sha256 != file_sha256(args.checkpoint):
            raise ValueError("retrieval cache was built with a different checkpoint")
    records, summary = evaluate(model, dataset, device, ("correct", "empty", "wrong", "mask_average"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "evaluation.json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256((Path(args.cache) / "manifest.json").read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)


@torch.inference_mode()
def evaluate_prompts(model, dataset, device):
    """Compare local interaction and prompt-derived support memory for every interface."""
    model.eval()
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        query = item["query"][None].to(device)
        target, valid = item["target"][None].to(device), item["valid"][None].to(device)
        supports = item["supports"][None].to(device)
        for kind in PROMPT_TYPES:
            local = simulate_prompt(item["target"], kind, dataset.seed + index, item["valid"])[None].to(device)
            support_prompts = torch.stack(
                [
                    simulate_prompt(mask, kind, dataset.seed + index + i + 1)
                    for i, mask in enumerate(item["support_masks"])
                ]
            )[None].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                local_logits = model(query, query_prompts=local)
                corrected_prompts = add_error_correction(local, local_logits.float().sigmoid(), target, valid)
                corrected_logits = model(query, query_prompts=corrected_prompts)
                # No dense support annotations are exposed in this evaluation.
                empty_masks = supports.new_zeros(1, supports.shape[1], 1, *supports.shape[-3:])
                dense = torch.zeros(1, supports.shape[1], device=device, dtype=torch.bool)
                memory_masks, _ = model.unify_context(supports, support_prompts, empty_masks, dense)
                context_logits = model(query, supports, memory_masks)
            for name, logits in (
                (f"interactive_{kind}", local_logits),
                (f"interactive_{kind}_corrected", corrected_logits),
                (f"context_{kind}", context_logits),
            ):
                records.append(
                    {
                        "query_id": item["query_id"],
                        "region": item["region"],
                        "support_ids": item["support_ids"],
                        "control": name,
                        **score(logits, target, valid),
                    }
                )
    return records, summarize(records)


@torch.inference_mode()
def evaluate_local_probe(model, dataset, device):
    model.eval()
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        kind = PROMPT_TYPES[index % len(PROMPT_TYPES)]
        prompts = simulate_prompt(item["target"], kind, dataset.seed + index, item["valid"])[None].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(item["query"][None].to(device), query_prompts=prompts)
        records.append(
            {
                "query_id": item["query_id"],
                "region": item["region"],
                "control": "interactive_probe",
                "prompt_type": kind,
                **score(
                    logits,
                    item["target"][None].to(device),
                    item["valid"][None].to(device),
                ),
            }
        )
    return records
