"""Create portable offline viewers; only CT display intensities and surface tessellation are reduced."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes


def encode(array):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(array).tobytes(), compresslevel=6, mtime=0)).decode()


def foreground_center(mask):
    points = np.argwhere(mask)
    return (np.median(points, axis=0).astype(int) if len(points) else np.array(mask.shape) // 2).tolist()


def volume(array, window=None, surface=True):
    array = np.asarray(array)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("Expected finite DHW volume")
    item = dict(shape=list(array.shape), bits=window is None)
    if window is not None:
        display = np.rint(np.clip((array - window[0]) / (window[1] - window[0]), 0, 1) * 255).astype(np.uint8)
        item["data"] = encode(display)
        assert len(gzip.decompress(base64.b64decode(item["data"]))) == array.size
    else:
        if not np.isin(array, [0, 1]).all():
            raise ValueError("Viewer masks must be binary")
        item["data"] = encode(np.packbits(array.astype(bool).ravel(), bitorder="little"))
        decoded = np.frombuffer(gzip.decompress(base64.b64decode(item["data"])), np.uint8)
        assert np.array_equal(np.unpackbits(decoded, bitorder="little")[: array.size], array.ravel())
        if surface and array.any():
            # Original voxel coordinates; step_size affects only the displayed mesh.
            padded = np.pad(array, 3).astype(np.float32)
            if padded[::3, ::3, ::3].any():
                vertices, faces, _, _ = marching_cubes(padded, 0.5, step_size=3)
                item["mesh"] = dict(vertices=encode((vertices - 3).astype("<f4")), faces=encode(faces.astype("<u4")))
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("patches", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.data / "report.json").read_text())
    patches = json.loads((args.patches / "patches.json").read_text())
    if hashlib.sha256((args.data / "report.json").read_bytes()).hexdigest() != patches["source_report"]:
        raise ValueError("Patch predictions and full-volume report have different provenance")
    query = np.load(args.data / "query.npy")[0]
    window = np.percentile(query, [0.5, 99.5]).tolist()
    image = volume(query, window)
    methods = list(dict.fromkeys(r["method"] for r in patches["records"]))
    template = Path(__file__).with_name("viewer.html").read_text()
    for name in ["three", "fflate"]:
        template = template.replace(f"__{name.upper()}__", (args.assets / f"{name}.min.js").read_text())
    manifest = dict(
        query=report["query_id"],
        files=[],
        patches_per_target=len(patches["origins"]),
        checks="Binary masks round-trip exactly; CT voxel grid unchanged; query and support shapes verified.",
    )
    for target in report["targets"]:
        truth = np.load(args.data / f"{target}_truth.npy")
        assert truth.shape == query.shape
        bank = np.load(args.data / f"{target}_supports.npz")
        assert bank["images"].shape == bank["masks"].shape == (4, 1, 192, 192, 192)
        payload = dict(
            target=target,
            targets=report["targets"],
            case=report["query_id"],
            window=window,
            methods=methods,
            checkpoints=report["checkpoints"],
            native_sha256=report["native_source"]["checkpoint_sha256"],
            origins=patches["origins"],
            support_metadata=report["supports"][target],
            volumes={"image": image, "reference": volume(truth)},
            metrics={},
            patch_metrics={},
        )
        volumes = payload["volumes"]
        payload["focus"] = foreground_center(np.load(args.data / f"{target}_{methods[0]}_k4.npy"))
        payload["support_focus"] = [foreground_center(mask[0]) for mask in bank["masks"]]
        payload["default_patch"] = int(np.square(np.array(patches["origins"]) + 96 - payload["focus"]).sum(1).argmin())
        volumes["valid"] = volume(np.load(args.patches / "query_valid.npy"), surface=False)
        support_valid = np.load(args.patches / f"{target}_valid.npz")["masks"]
        for index in range(4):
            volumes[f"support_image_{index}"] = volume(bank["images"][index, 0], window)
            volumes[f"support_prompt_{index}"] = volume(bank["masks"][index, 0])
            volumes[f"support_valid_{index}"] = volume(support_valid[index], surface=False)
        for method in methods:
            for count in range(1, 5):
                for suffix in ["", "_autozoom"]:
                    key = f"{method}{suffix}_k{count}"
                    mask = np.load(args.data / f"{target}_{key}.npy")
                    assert mask.shape == query.shape
                    volumes[key] = volume(mask)
                    payload["metrics"][key] = next(
                        r
                        for r in report["records"]
                        if r["target"] == target and r["method"] == method + suffix and r["supports"] == count
                    )
            record = next(r for r in patches["records"] if r["target"] == target and r["method"] == method)
            payload["patch_metrics"][method] = record
            with np.load(args.patches / f"{target}_{method}.npz") as raw:
                for phase in ["before", "after"]:
                    masks = raw[phase]
                    assert masks.shape == (len(patches["origins"]), 192, 192, 192)
                    for index, mask in enumerate(masks):
                        volumes[f"{method}_{phase}_p{index}"] = volume(mask)
                for index, mask in enumerate(raw["support_prediction"]):
                    volumes[f"{method}_support_{index}"] = volume(mask)
        content = template.replace("__DATA__", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
        output = args.output / f"{target}.html"
        output.write_text(content)
        manifest["files"].append(
            dict(
                name=output.name,
                bytes=output.stat().st_size,
                sha256=hashlib.sha256(content.encode()).hexdigest(),
                volume_count=len(volumes),
            )
        )
        print(json.dumps(manifest["files"][-1]), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
