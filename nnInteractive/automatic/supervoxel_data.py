"""Case-local generated regions, independent of semantic-label episode sampling."""

from pathlib import Path

import blosc2
import numpy as np

from .data import ArrayStore, EpisodeDataset


class SupervoxelArrayStore(ArrayStore):
    def __init__(self, manifest):
        # A separate source manifest must not inherit another dataset's array override.
        super().__init__(manifest, use_array_override=False)

    @staticmethod
    def open_array(path):
        return blosc2.open(str(path), mode="r", dparams={"nthreads": 1})

    def foreground_locations(self, case, target):
        return np.asarray(target["locations"])

    def has_target(self, case, target):
        return True

    def valid_labels(self, labels):
        return labels >= 0

    def foreground(self, labels, target):
        if not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError("supervoxel masks must be binary")
        return labels == 1

    def crop(self, case, start, size, target, *, volume=None):
        if volume is None:
            volume = (self.open_array(self.folder / f"{case}.b2nd"), self.open_array(target["path"]))
        return super().crop(case, start, size, target, volume=volume)


class SupervoxelEpisodeDataset(EpisodeDataset):
    """One generated region supplies independently transformed query and support views."""

    store_class = SupervoxelArrayStore

    def _initialize(self, targets):
        if self.split != "train" or targets is not None:
            raise ValueError("supervoxel episodes are training-only and do not accept semantic targets")
        generated = self.manifest.get("supervoxels", {})
        records = generated.get("cases", {})
        if not set(records) <= set(self.manifest["splits"]["train"]) or set(records) & set(
            self.manifest["splits"].get("val", [])
        ):
            raise ValueError("supervoxel sources must belong exclusively to the training split")
        self.provenance = generated.get("provenance", {})
        self.supervoxels = {case: masks for case, masks in records.items() if masks}
        if not self.supervoxels:
            raise ValueError("supervoxel sampling requires nonempty generated training regions")
        self.targets = []
        for case, masks in self.supervoxels.items():
            shape = (1, *self.manifest["cases"][case]["shape"])
            if tuple(self.store.open_array(self.store.folder / f"{case}.b2nd").shape) != shape:
                raise ValueError("supervoxel source image shape does not match its manifest")
            for record in masks:
                path, locations = Path(record["path"]), np.asarray(record["locations"])
                if not path.is_absolute() or not path.is_file():
                    raise ValueError("supervoxel masks require existing absolute paths")
                mask = self.store.open_array(path)
                if tuple(mask.shape) != shape or mask.dtype not in (np.dtype("uint8"), np.dtype("bool")):
                    raise ValueError("supervoxel mask must be uint8/bool with the preprocessed image shape")
                if (
                    locations.ndim != 2
                    or locations.shape[1:] != (3,)
                    or not len(locations)
                    or not np.issubdtype(locations.dtype, np.integer)
                    or (locations < 0).any()
                    or (locations >= shape[1:]).any()
                ):
                    raise ValueError("supervoxel masks require nonempty valid foreground locations")
                if mask[(0, *locations[0].tolist())] != 1:
                    raise ValueError("supervoxel foreground location does not contain foreground")

    def _select_episode(self, rng, count, target, query_id, query_kind):
        if target is not None or query_id is not None or (query_kind is not None and not self.mixed):
            raise ValueError("supervoxel episodes select their own source region and positive query crop")
        query = str(rng.choice(list(self.supervoxels)))
        index = int(rng.integers(len(self.supervoxels[query])))
        region = self.supervoxels[query][index]
        return (
            query,
            region,
            [query] * count,
            query_kind or "positive",
            {
                "target": f"supervoxel:{query}:{index}",
                "episode_type": "supervoxel",
                "supervoxel_path": region["path"],
            },
        )
