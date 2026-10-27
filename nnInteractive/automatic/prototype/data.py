"""Explicit nnU-Net b2nd reader, hierarchical targets, and disjoint episodes."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import blosc2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def region_mask(segmentation: torch.Tensor, values: list[int]) -> torch.Tensor:
    return torch.isin(segmentation, torch.as_tensor(values, device=segmentation.device)).float()


def prepare(args) -> None:
    """Resize complete volumes, without query-label-dependent cropping or normalization."""
    root, output = Path(args.dataset), Path(args.output)
    if (output / "manifest.json").exists():
        raise FileExistsError("cache already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "dataset.json").read_text())
    regions = {k: v if isinstance(v, list) else [v] for k, v in metadata["labels"].items() if k != "background"}
    split_path = Path(args.split) if getattr(args, "split", None) else root / "splits_final.json"
    definition = json.loads(split_path.read_text())
    split = definition if getattr(args, "split", None) else definition[args.fold]
    original = json.loads((root / "splits_final.json").read_text())[args.fold]
    for key in ("train", "val"):
        if len(set(split[key])) != len(split[key]) or not set(split[key]) <= set(original[key]):
            raise ValueError("selected cases must be unique subsets of their original partitions")
    if set(split["train"]) & set(split["val"]):
        raise ValueError("source split contains overlapping cases")
    generator = np.random.default_rng(args.seed)
    selected = {}
    for key, limit in (("train", args.train_cases), ("val", args.val_cases)):
        cases = sorted(split[key])
        selected[key] = sorted(generator.permutation(cases).tolist()[:limit] if limit else cases)
    case_ids = selected["train"] + selected["val"]
    if args.patient_map:
        patient_map = json.loads(Path(args.patient_map).read_text())
        patients = {case: patient_map[case] for case in case_ids}
    else:
        # Dataset supplies case IDs only. This assumption is recorded, never claimed as verified identity.
        patients = {case: case for case in case_ids}
    if {patients[c] for c in selected["train"]} & {patients[c] for c in selected["val"]}:
        raise ValueError("patient leakage between train and validation")
    manifest = {
        "version": 1,
        "dataset": str(root.resolve()),
        "configuration": args.configuration,
        "fold": args.fold,
        "seed": args.seed,
        "shape": args.shape,
        "regions": regions,
        "splits": selected,
        "cases": {},
        "patient_identity": "provided" if args.patient_map else "case_id_assumption",
        "dataset_json_sha256": hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest(),
        "splits_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "protocol": "whole preprocessed volume resized to fixed grid; no target-dependent query crop",
        "normalization": getattr(args, "normalization", "none"),
    }
    blosc2.set_nthreads(args.threads)
    for offset in range(0, len(case_ids), args.shard_size):
        shard = {}
        filename = f"shard_{offset // args.shard_size:05d}.pt"
        for case in case_ids[offset : offset + args.shard_size]:
            folder = Path(args.array_dir) if getattr(args, "array_dir", None) else root / args.configuration
            image_array = blosc2.open(str(folder / f"{case}.b2nd"), mode="r", dparams={"nthreads": 1})
            seg_array = blosc2.open(str(folder / f"{case}_seg.b2nd"), mode="r", dparams={"nthreads": 1})
            image = torch.from_numpy(image_array[:]).float()
            seg = torch.from_numpy(seg_array[:].astype(np.int16))
            if image.ndim != 4 or image.shape[0] != 1 or seg.shape != image.shape:
                raise ValueError(f"expected single-channel 3D data and categorical segmentation: {case}")
            if not torch.isfinite(image).all() or seg.min() < -1 or seg.max() > 255:
                raise ValueError(f"invalid image or categorical label values: {case}")
            if manifest["normalization"] == "zscore":
                image = (image - image.mean()) / image.std().clamp_min(1e-6)
            resized_image = F.interpolate(image[None], size=args.shape, mode="trilinear", align_corners=False)[0]
            resized_seg = F.interpolate(seg[None].float(), size=args.shape, mode="nearest-exact")[0].short()
            # Keep -1 as invalid/padding, distinct from labeled background.
            shard[case] = {"image": resized_image.half(), "seg": resized_seg}
            observed = set(np.unique(resized_seg.numpy()).tolist())
            present = [name for name, values in regions.items() if observed.intersection(values)]
            manifest["cases"][case] = {
                "shard": filename,
                "patient": patients[case],
                "present": present,
            }
        torch.save(shard, output / filename)
        print(
            json.dumps(
                {
                    "prepared": min(offset + args.shard_size, len(case_ids)),
                    "total": len(case_ids),
                }
            ),
            flush=True,
        )
    encoded = json.dumps(manifest, sort_keys=True, indent=2)
    (output / "manifest.json").write_text(encoded + "\n")


class VolumeStore:
    def __init__(self, root: str | Path, cache_shards: int = 2):
        self.manifest = json.loads((Path(root) / "manifest.json").read_text())
        self.root = Path(os.environ.get("NNI_COARSE_CACHE", root))
        if json.loads((self.root / "manifest.json").read_text()) != self.manifest:
            raise ValueError("node-local coarse cache does not match the source manifest")
        self.cache_shards = cache_shards
        self.cache = OrderedDict()

    def load(self, case: str) -> dict:
        filename = self.manifest["cases"][case]["shard"]
        if filename not in self.cache:
            self.cache[filename] = torch.load(self.root / filename, map_location="cpu", weights_only=True, mmap=True)
            if len(self.cache) > self.cache_shards:
                self.cache.popitem(last=False)
        self.cache.move_to_end(filename)
        return self.cache[filename][case]

    def sample(self, case, rng, target=None):
        return self.load(case)


class Episodes(Dataset):
    """Deterministic per-index episodes; validation support always comes from train.

    Training samples targets uniformly; queries include target-absent cases naturally.
    Validation covers every selected query/target, including target-absent volumes.
    No target label, target presence, or query segmentation is passed to the model.
    """

    def __init__(
        self,
        root,
        split,
        regions,
        shots,
        episodes=1000,
        seed=0,
        augment=False,
        retrieval=None,
        store=None,
    ):
        self.store = VolumeStore(root) if store is None else store
        self.retriever = None
        if retrieval is not None:
            from .retrieval import SupportRetriever

            self.retriever = SupportRetriever(retrieval, Path(root) / "manifest.json")
        manifest = self.store.manifest
        if split not in {"train", "val"} or shots < 1 or episodes < 1:
            raise ValueError("invalid split, shots, or episode count")
        if not regions or len(set(regions)) != len(regions):
            raise ValueError("regions must be a nonempty unique list")
        for name in regions:
            if name not in manifest["regions"]:
                raise ValueError(f"unknown region: {name}")
        self.split, self.regions, self.shots = split, regions, shots
        self.episodes, self.seed, self.augment = episodes, seed, augment
        self.queries = manifest["splits"][split]
        self.bank = {
            name: [c for c in manifest["splits"]["train"] if name in manifest["cases"][c]["present"]]
            for name in regions
        }
        for name, cases in self.bank.items():
            patients = {manifest["cases"][c]["patient"] for c in cases}
            required = shots + (split == "train")
            if len(patients) < required:
                raise ValueError(f"{name} needs at least {required} distinct support patients; found {len(patients)}")
        self.fixed = [(case, name) for case in self.queries for name in regions]

    def __len__(self):
        return self.episodes if self.split == "train" else len(self.fixed)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = np.random.default_rng(self.seed + index)
        if self.split == "train":
            name = self.regions[int(rng.integers(len(self.regions)))]
            query = self.queries[int(rng.integers(len(self.queries)))]
        else:
            query, name = self.fixed[index]
        manifest = self.store.manifest
        used = {manifest["cases"][query]["patient"]}
        supports = []
        if self.retriever is None:
            candidates = rng.permutation(self.bank[name])
        else:
            patients = {case: value["patient"] for case, value in manifest["cases"].items()}
            candidates = self.retriever.select(query, self.bank[name], patients, self.shots)
        for case in candidates:
            patient = manifest["cases"][case]["patient"]
            if patient not in used:
                supports.append(str(case))
                used.add(patient)
                if len(supports) == self.shots:
                    break
        if len(supports) != self.shots:
            raise ValueError("not enough distinct support patients")
        images, masks, valid, segments, starts = [], [], [], [], []
        for case in [query, *supports]:
            record = self.store.sample(case, rng, target=name if case != query else None)
            segments.append(record["seg"])
            starts.append(record.get("start", (0, 0, 0)))
            image = record["image"].float()
            mask = region_mask(record["seg"], manifest["regions"][name])
            known = record["seg"] >= 0
            if self.augment:
                # Never mirror anatomy: left/right regions have distinct semantics.
                image = image * float(rng.uniform(0.9, 1.1)) + float(rng.normal(0, 0.05))
            images.append(image)
            masks.append(mask if case == query else torch.where(known, mask, -1.0))
            valid.append(known)
        return {
            "query": images[0],
            "target": masks[0],
            "valid": valid[0],
            "supports": torch.stack(images[1:]),
            "support_masks": torch.stack(masks[1:]),
            "query_id": query,
            "support_ids": supports,
            "region": name,
            "query_seg": segments[0],
            "support_segs": torch.stack(segments[1:]),
            "crop_starts": torch.tensor(starts),
        }


class MixedEpisodes(Episodes):
    """Shared prompts for single-volume interaction and prompted 3D support memory."""

    def __getitem__(self, index):
        from .prompts import PROMPT_TYPES, simulate_prompt

        item = super().__getitem__(index)
        # Every example carries a local prompt; the trainer chooses whether to expose it.
        kind = PROMPT_TYPES[index % len(PROMPT_TYPES)]
        item["query_prompts"] = simulate_prompt(item["target"], kind, self.seed + index, item["valid"])
        item["prompt_type"] = kind
        prompts, dense = [], []
        for support_index, mask in enumerate(item["support_masks"]):
            support_kind = PROMPT_TYPES[(index + support_index) % len(PROMPT_TYPES)]
            prompts.append(simulate_prompt(mask, support_kind, self.seed + index + support_index + 1))
            dense.append((index + support_index) % 2 == 0)
        item["support_prompts"] = torch.stack(prompts)
        item["dense_support"] = torch.tensor(dense)
        return item
