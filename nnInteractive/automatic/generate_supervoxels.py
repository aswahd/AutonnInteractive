"""Cache native nnInteractive proposals as independent, training-grid binary masks."""

import argparse
import fcntl
import hashlib
import json
import os
import pickle
from pathlib import Path

import blosc2
import numpy as np
from scipy import ndimage

from .backbone import sha256


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def select_cases(manifest, limit=None):
    cases = manifest["splits"]["train"]
    if len(cases) != len(set(cases)) or set(cases) & set(manifest["splits"]["val"]):
        raise ValueError("training split must be unique and disjoint from validation")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    return cases[:limit]


def align_mask(mask, properties, transpose, shape, configuration):
    """Apply the original nnU-Net transpose, crop, and categorical resampling."""
    mask = np.transpose(mask, transpose)
    if tuple(mask.shape) != tuple(properties["shape_before_cropping"]):
        raise ValueError("native mask disagrees with recorded preprocessing geometry")
    crop = tuple(slice(a, b) for a, b in properties["bbox_used_for_cropping"])
    mask = mask[crop][None].astype(np.uint8)
    if tuple(mask.shape[1:]) != tuple(properties["shape_after_cropping_and_before_resampling"]):
        raise ValueError("invalid recorded preprocessing crop")
    spacing = [properties["spacing"][axis] for axis in transpose]
    mask = configuration.resampling_fn_seg(mask, shape, spacing, configuration.spacing)
    if tuple(mask.shape) != (1, *shape) or not np.isin(mask, [0, 1]).all():
        raise ValueError("invalid aligned binary mask")
    return mask[0].astype(bool)


def seed_component(mask, point):
    components, _ = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    component = int(components[tuple(point)])
    return components == component if component else np.zeros_like(mask, dtype=bool)


def mask_iou(first, second):
    return int(np.count_nonzero(first & second)) / max(1, int(np.count_nonzero(first | second)))


def stability_point(mask, point, spacing, radius, rng):
    """Choose a different nearby interior seed, measuring distance in millimeters."""
    extent = np.ceil(radius / np.asarray(spacing)).astype(int)
    start = np.maximum(0, np.asarray(point) - extent)
    stop = np.minimum(mask.shape, np.asarray(point) + extent + 1)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    candidates = np.argwhere(ndimage.binary_erosion(mask[crop])) + start
    distance = np.sum(((candidates - point) * spacing) ** 2, axis=1)
    candidates = candidates[(distance > 0) & (distance <= radius**2)]
    return tuple(int(x) for x in candidates[rng.integers(len(candidates))]) if len(candidates) else None


def predict_region(session, target, point):
    session.reset_interactions()
    session.add_point_interaction(point, True)
    return seed_component(target > 0, point)


def body_mask(image, labels, modality="CT"):
    threshold = -500
    if modality == "MRI":
        from skimage.filters import threshold_otsu

        threshold = threshold_otsu(image)
    return ndimage.binary_fill_holes(image > threshold) | (labels > 0)


def overlap(mask, labels, label_counts, accepted):
    count = int(mask.sum())
    intersections = np.bincount(labels[mask].clip(0), minlength=len(label_counts))[: len(label_counts)]
    ious = intersections / np.maximum(1, count + label_counts - intersections)
    ious[0] = 0
    duplicate = max((mask_iou(mask, previous) for previous in accepted), default=0.0)
    return {"voxel_count": count, "max_real_label_iou": float(ious.max()), "real_label": int(ious.argmax())}, duplicate


def finalize_manifest(manifest_path, output, cases, provenance):
    manifest = json.loads(Path(manifest_path).read_text())
    if not set(cases) <= set(select_cases(manifest)) or len(cases) != len(set(cases)):
        raise ValueError("only unique training cases can be finalized")
    records = {}
    for case in cases:
        record = json.loads((output / case / "metadata.json").read_text())
        if record.get("provenance") != provenance or record.get("case") != case or not record.get("complete"):
            raise ValueError(f"incomplete or incompatible generation for {case}")
        for mask in record["records"]:
            array = blosc2.open(mask["path"], mode="r", dparams={"nthreads": 1})
            if tuple(array.shape) != (1, *manifest["cases"][case]["shape"]):
                raise ValueError(f"cached mask geometry mismatch for {case}")
        records[case] = record["records"]
    if not any(records.values()):
        raise ValueError("generation produced no usable masks")
    manifest["supervoxels"] = {"cases": records, "provenance": provenance}
    path = output / "manifest.json"
    write_json(path, manifest)
    return path


def generate_case(args, manifest, case, session, configuration, transpose, provenance):
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIO

    folder = args.output / case
    folder.mkdir(exist_ok=True)
    image_path = args.raw_dataset / "imagesTr" / f"{case}_0000.nii.gz"
    label_path = args.raw_dataset / "labelsTr" / f"{case}.nii.gz"
    source_hashes = {"image": sha256(image_path), "labels": sha256(label_path)}
    complete = folder / "metadata.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous.get("provenance") != provenance or previous.get("source_sha256") != source_hashes:
            raise ValueError(f"cached generation provenance changed for {case}")
        if not previous.get("complete") or previous.get("case") != case:
            raise ValueError(f"invalid completion record for {case}")
        for record in previous["records"]:
            if not Path(record["path"]).is_file():
                raise ValueError(f"missing cached mask for {case}")
        print(json.dumps({"case": case, "resumed": True, "masks": len(previous["records"])}), flush=True)
        return
    reader = NibabelIO()
    image, image_properties = reader.read_images([str(image_path)])
    labels, label_properties = reader.read_seg(str(label_path))
    if image.shape != labels.shape or not np.isfinite(image).all():
        raise ValueError(f"invalid raw image or label shape for {case}")
    if not np.allclose(
        image_properties["nibabel_stuff"]["original_affine"], label_properties["nibabel_stuff"]["original_affine"]
    ):
        raise ValueError(f"raw image/label affine mismatch for {case}")
    arrays = Path(manifest["dataset"]) / manifest["configuration"]
    with (arrays / f"{case}.pkl").open("rb") as handle:
        properties = pickle.load(handle)
    if not np.allclose(image_properties["spacing"], properties["spacing"]) or not np.allclose(
        image_properties["nibabel_stuff"]["original_affine"], properties["nibabel_stuff"]["original_affine"]
    ):
        raise ValueError(f"raw/preprocessed geometry mismatch for {case}")
    real = blosc2.open(str(arrays / f"{case}_seg.b2nd"), mode="r", dparams={"nthreads": 1})[0].astype(np.int64)
    shape = tuple(manifest["cases"][case]["shape"])
    if real.shape != shape:
        raise ValueError(f"manifest/label shape mismatch for {case}")
    label_counts = np.bincount(real.clip(0).ravel())
    body = body_mask(image[0], labels[0], getattr(args, "modality", "CT"))
    pools = [np.flatnonzero(body), np.flatnonzero(body & (labels[0] == 0))]
    if not len(pools[0]):
        raise ValueError(f"no body seed candidates for {case}")
    rng = np.random.default_rng([args.seed, int.from_bytes(hashlib.sha256(case.encode()).digest()[:4], "little")])
    target = np.zeros(image.shape[1:], dtype=np.uint8)
    session.set_image(image, image_properties)
    session.set_target_buffer(target)
    accepted, records = [], []
    rejected = dict.fromkeys(("seed_missing", "size", "duplicate", "known_label", "no_stability_seed", "unstable"), 0)
    for attempt in range(args.attempts_per_case):
        if attempt % 10 == 0:
            print(
                json.dumps({"case": case, "attempts_completed": attempt, "masks": len(records), "rejected": rejected}),
                flush=True,
            )
        pool = pools[attempt % 2] if len(pools[attempt % 2]) else pools[0]
        point = tuple(int(x) for x in np.unravel_index(rng.choice(pool), target.shape))
        native = predict_region(session, target, point)
        native_count = int(native.sum())
        if not native_count:
            rejected["seed_missing"] += 1
            continue
        volume_mm3 = float(native_count * np.prod(image_properties["spacing"]))
        body_fraction = native_count / int(body.sum())
        if volume_mm3 < args.min_volume_mm3 or body_fraction > args.max_body_fraction:
            rejected["size"] += 1
            continue
        mask = align_mask(native, properties, transpose, shape, configuration)
        qa, duplicate = overlap(mask, real, label_counts, accepted)
        if qa["voxel_count"] < args.min_voxels:
            rejected["size"] += 1
            continue
        if duplicate >= args.duplicate_iou:
            rejected["duplicate"] += 1
            continue
        if qa["max_real_label_iou"] > args.max_real_label_iou:
            rejected["known_label"] += 1
            continue
        second_point = stability_point(native, point, image_properties["spacing"], args.stability_radius_mm, rng)
        if second_point is None:
            rejected["no_stability_seed"] += 1
            continue
        stability = mask_iou(native, predict_region(session, target, second_point))
        if stability < args.stability_iou:
            rejected["unstable"] += 1
            continue
        qa.update(
            native_voxel_count=native_count,
            volume_mm3=volume_mm3,
            body_fraction=body_fraction,
            stability_iou=stability,
            stability_seed_native=list(second_point),
        )
        path = folder / f"mask_{len(records):03d}.b2nd"
        temporary = path.with_name(f".{path.name}.tmp")
        blosc2.asarray(
            mask[None].astype(np.uint8),
            urlpath=str(temporary),
            mode="w",
            cparams={"codec": blosc2.Codec.ZSTD, "clevel": 5, "nthreads": args.threads},
        )
        temporary.replace(path)
        foreground = np.flatnonzero(mask)
        sampled = rng.choice(foreground, min(100, len(foreground)), replace=False)
        locations = np.stack(np.unravel_index(sampled, shape), axis=1).tolist()
        records.append(
            {"path": str(path), "locations": locations, "shape": list(shape), "seed_native": list(point), **qa}
        )
        accepted.append(mask)
        print(json.dumps({"case": case, "attempt": attempt + 1, "masks": len(records), **qa}), flush=True)
        if len(records) == args.masks_per_case:
            break
    write_json(
        complete,
        {
            "case": case,
            "complete": True,
            "provenance": provenance,
            "source_sha256": source_hashes,
            "records": records,
            "attempts": attempt + 1,
            "rejected": rejected,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "raw-dataset", "model-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--modality", choices=("CT", "MRI"), default="CT")
    parser.add_argument("--masks-per-case", type=int, default=20)
    parser.add_argument("--attempts-per-case", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--min-voxels", type=int, default=64)
    parser.add_argument("--min-volume-mm3", type=float, default=100)
    parser.add_argument("--max-body-fraction", type=float, default=0.5)
    parser.add_argument("--duplicate-iou", type=float, default=0.9)
    parser.add_argument("--max-real-label-iou", type=float, default=1.0)
    parser.add_argument("--stability-iou", type=float, default=0.8)
    parser.add_argument("--stability-radius-mm", type=float, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if min(args.masks_per_case, args.attempts_per_case, args.min_voxels, args.threads, args.shards) < 1:
        parser.error("counts and threads must be positive")
    if (
        not 0 <= args.shard_index < args.shards
        or not 0 < args.duplicate_iou <= 1
        or not 0 < args.max_body_fraction <= 1
    ):
        parser.error("invalid shard or filtering threshold")
    if not 0 < args.stability_iou <= 1 or not 0 <= args.max_real_label_iou <= 1:
        parser.error("invalid consistency or real-label overlap threshold")
    if min(args.min_volume_mm3, args.stability_radius_mm) <= 0 or args.seed < 0:
        parser.error("physical thresholds must be positive and seed nonnegative")
    for name in ("manifest", "raw_dataset", "model_dir", "output"):
        setattr(args, name, getattr(args, name).resolve())
    manifest = json.loads(args.manifest.read_text())
    cases = select_cases(manifest, args.limit)
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    matches = [
        (path, name)
        for path in Path(manifest["dataset"]).glob("*Plans*.json")
        for name, config in json.loads(path.read_text())["configurations"].items()
        if config.get("data_identifier") == manifest["configuration"]
    ]
    if len(matches) != 1:
        raise ValueError("need exactly one preprocessing configuration")
    plans_path, name = matches[0]
    plans = PlansManager(str(plans_path))
    provenance = {
        "format_version": 1,
        "generator_sha256": sha256(Path(__file__)),
        "manifest_sha256": sha256(args.manifest),
        "plans_sha256": sha256(plans_path),
        "checkpoint_sha256": sha256(args.model_dir / f"fold_{args.fold}" / "checkpoint_final.pth"),
        "model_metadata_sha256": {path.name: sha256(path) for path in args.model_dir.glob("*.json")},
        "raw_dataset": str(args.raw_dataset),
        "model_dir": str(args.model_dir),
        "fold": args.fold,
        "modality": args.modality,
        **{
            key: getattr(args, key)
            for key in (
                "masks_per_case",
                "attempts_per_case",
                "seed",
                "min_voxels",
                "min_volume_mm3",
                "max_body_fraction",
                "duplicate_iou",
                "max_real_label_iou",
                "stability_iou",
                "stability_radius_mm",
            )
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".provenance.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = args.output / "provenance.json"
        if path.exists() and json.loads(path.read_text()) != provenance:
            raise ValueError("output directory belongs to a different generation recipe")
        write_json(path, provenance)
    if args.finalize:
        print(finalize_manifest(args.manifest, args.output, cases, provenance), flush=True)
        return
    import torch

    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    session = nnInteractiveInferenceSession(
        device=torch.device("cuda"), torch_n_threads=args.threads, enable_undo=False, do_autozoom=True
    )
    try:
        session.initialize_from_trained_model_folder(str(args.model_dir), use_fold=args.fold)
        session.network.eval().requires_grad_(False)
        with torch.inference_mode():
            for case in cases[args.shard_index :: args.shards]:
                (args.output / case).mkdir(exist_ok=True)
                with (args.output / case / ".lock").open("w") as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    generate_case(
                        args,
                        manifest,
                        case,
                        session,
                        plans.get_configuration(name),
                        plans.transpose_forward,
                        provenance,
                    )
    finally:
        session.executor.shutdown()


if __name__ == "__main__":
    main()
