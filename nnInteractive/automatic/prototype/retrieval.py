"""Label-free volumetric feature retrieval with patient-excluding MMR selection."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpoint import load_model
from ..backbone import sha256 as file_sha256


class SupportRetriever:
    def __init__(self, path, manifest_path):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache["manifest_sha256"] != file_sha256(manifest_path):
            raise ValueError("retrieval features do not match this dataset manifest")
        self.checkpoint_sha256 = cache["checkpoint_sha256"]
        self.indices = {case: index for index, case in enumerate(cache["case_ids"])}
        self.features = F.normalize(cache["features"].float(), dim=1)
        if len(self.indices) != len(cache["case_ids"]) or len(self.features) != len(
            self.indices
        ):
            raise ValueError("invalid retrieval case IDs or feature count")

    def select(self, query, candidates, patients, shots, relevance_weight=0.7):
        if not 0 <= relevance_weight <= 1 or shots < 1:
            raise ValueError("invalid MMR weight or shot count")
        eligible = sorted(c for c in candidates if patients[c] != patients[query])
        if len({patients[c] for c in eligible}) < shots:
            raise ValueError("not enough distinct eligible support patients")
        features = self.features[[self.indices[c] for c in eligible]]
        similarity = features @ self.features[self.indices[query]]
        selected, used = [], {patients[query]}
        redundancy = torch.full_like(similarity, -torch.inf)
        for _ in range(shots):
            score = (
                similarity
                if not selected
                else relevance_weight * similarity - (1 - relevance_weight) * redundancy
            )
            score = score.clone()
            for index, case in enumerate(eligible):
                if patients[case] in used:
                    score[index] = -torch.inf
            index = int(score.argmax())
            selected.append(eligible[index])
            used.add(patients[eligible[index]])
            redundancy = torch.maximum(redundancy, features @ features[index])
        return selected


@torch.inference_mode()
def build_retrieval(args):
    from .data import VolumeStore

    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    model, _ = load_model(args.checkpoint, device)
    model.eval()
    store = VolumeStore(args.cache)
    cases = sorted(store.manifest["cases"])
    features = []
    for case in cases:
        # The segmentation is never read by feature extraction.
        image = store.load(case)["image"].float()[None].to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            feature = model.backbone.encoder(model._input(image))[-2]
            pooled = F.adaptive_avg_pool3d(feature.float(), (args.grid,) * 3).flatten(1)
        features.append(F.normalize(pooled, dim=1)[0].cpu())
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a new retrieval output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "case_ids": cases,
            "features": torch.stack(features),
            "grid": args.grid,
            "manifest_sha256": file_sha256(Path(args.cache) / "manifest.json"),
            "checkpoint_sha256": file_sha256(args.checkpoint),
        },
        output,
    )
    print(
        f"Saved label-free 3D retrieval features for {len(cases)} cases to {output}",
        flush=True,
    )
