"""Optional native nnInteractive refinement of a complete ICL volume prediction."""

from pathlib import Path

import numpy as np
import torch

from .backbone import sha256


class NativeAutoZoomRefiner:
    """Run the original frozen network's actual AutoZoom engine on the supplied grid."""

    def __init__(self, source, device="cuda", threads=4):
        from nnInteractive.inference.inference_session import (
            nnInteractiveInferenceSession,
        )

        checkpoint = Path(source["checkpoint_file"])
        if sha256(checkpoint) != source["checkpoint_sha256"]:
            raise ValueError("original nnInteractive checkpoint hash changed")
        for name, expected in source["source_metadata_sha256"].items():
            if sha256(Path(source["checkpoint_dir"]) / name) != expected:
                raise ValueError(f"original nnInteractive metadata changed: {name}")
        previous_threads, benchmark = (
            torch.get_num_threads(),
            torch.backends.cudnn.benchmark,
        )
        try:
            self.session = nnInteractiveInferenceSession(
                device=torch.device(device),
                torch_n_threads=threads,
                do_autozoom=True,
                use_torch_compile=False,
                enable_undo=False,
            )
            with torch.random.fork_rng(devices=[]):
                artifacts = self.session._load_model_artifacts_from_disk(
                    source["checkpoint_dir"], source["fold"], source["checkpoint_name"]
                )
            self.session.initialize_from_loaded_artifacts(artifacts)
            if (
                self.session.channel_mapping != source["channel_mapping"]
                or self.session.num_interaction_channels != source["prompt_channels"]
                or list(self.session.configuration_manager.patch_size) != source["native_patch_size"]
            ):
                raise ValueError("native refiner input contract differs from the student checkpoint")
            self.session.network.eval().requires_grad_(False)
        except BaseException:
            if hasattr(self, "session"):
                self.close()
            raise
        finally:
            torch.set_num_threads(previous_threads)
            torch.backends.cudnn.benchmark = benchmark

    @torch.inference_mode()
    def refine(self, preprocessed_query, initial_mask):
        """Return (hard mask, network calls); the student mask supplies the entire prompt."""
        query = torch.as_tensor(preprocessed_query, device="cpu", dtype=torch.float32)
        initial = np.asarray(initial_mask)
        if query.ndim != 4 or query.shape[0] != 1 or not torch.isfinite(query).all():
            raise ValueError("query must be finite preprocessed 1DHW")
        if initial.shape != tuple(query.shape[1:]) or not np.isin(initial, [0, 1]).all():
            raise ValueError("initial segmentation must be binary and match the supplied query grid")
        result = initial.astype(np.uint8, copy=True)
        if not result.any():
            return (
                result,
                0,
            )  # No target location is available; preserve the empty student result.
        session, calls = self.session, []
        session._reset_session()
        session.new_interaction_centers.clear()
        session.new_interaction_zoom_out_factors.clear()
        # set_image() normalizes raw CT. These images are already preprocessed, so bind
        # their existing grid directly, then use the unmodified native interaction engine.
        session.preprocessed_image, session.original_image_shape = (
            query,
            tuple(query.shape),
        )
        hook = session.network.register_forward_hook(lambda module, inputs, output: calls.append(1))
        try:
            session._initialize_interactions(query)
            session.set_target_buffer(result)
            session.add_initial_seg_interaction(initial.astype(np.uint8, copy=True), run_prediction=True)
            return result, len(calls)
        finally:
            hook.remove()
            session._reset_session()
            session._interactions_read_buffer = None
            session.new_interaction_centers.clear()
            session.new_interaction_zoom_out_factors.clear()

    def close(self):
        self.session.executor.shutdown(wait=True)
        self.session._reset_session()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
