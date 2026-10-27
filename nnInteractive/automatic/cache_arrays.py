"""Decode a dataset once into a shared directory, suitable for a job-local tmpfs."""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blosc2
import numpy as np


def cache_arrays(manifest, output, workers=8):
    manifest = json.loads(Path(manifest).read_text())
    source = (Path(manifest["dataset"]) / manifest["configuration"]).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def decode(case):
        for suffix in ("", "_seg"):
            array = blosc2.open(str(source / f"{case}{suffix}.b2nd"), mode="r", dparams={"nthreads": 1})
            np.save(output / f"{case}{suffix}.npy", array[:], allow_pickle=False)
        shutil.copyfile(source / f"{case}.pkl", output / f"{case}.pkl")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(decode, manifest["cases"]), 1):
            if index % 100 == 0:
                print(f"Cached {index}/{len(manifest['cases'])} volumes", flush=True)
    # Readers only accept completed caches belonging to their source dataset.
    (output / "cache.json").write_text(json.dumps({"source": str(source), "cases": list(manifest["cases"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cache_arrays(args.manifest, args.output, args.workers)
